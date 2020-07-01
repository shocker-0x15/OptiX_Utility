#pragma once

#include "uber_shared.h"

#define M_PI 3.14159265

using namespace Shared;

extern "C" RT_CONSTANT_MEMORY PipelineLaunchParameters plp;



struct SearchRayPayload {
    float3 alpha;
    float3 contribution;
    float3 origin;
    float3 direction;
    struct {
        unsigned int pathLength : 30;
        bool specularBounce : 1;
        bool terminate : 1;
    };
};

#define SearchRayPayloadSignature OptixTraversableHandle, PCG32RNG, SearchRayPayload*

struct VisibilityRayPayload {
    float visibility;
};

#define VisibilityRayPayloadSignature float



RT_PROGRAM void RT_RG_NAME(pathtracing)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.rngBuffer[launchIndex];

    float x = static_cast<float>(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    OptixTraversableHandle traversable = plp.travHandles[plp.travIndex];

    SearchRayPayload payload;
    payload.alpha = make_float3(1.0f, 1.0f, 1.0f);
    payload.contribution = make_float3(0.0f, 0.0f, 0.0f);
    payload.pathLength = 1;
    payload.terminate = false;
    SearchRayPayload* payloadPtr = &payload;
    while (true) {
        // JP: ペイロードとともにトレースを呼び出す。
        //     ペイロード数は最大で8DWだが、
        //     3つ目のペイロードをそのまま渡すと収まりきらないのでポインターとして渡す。
        // EN: Trace call with payloads.
        //     The maximum number of payloads is 8 dwords in total.
        //     However pass the third payload as pointer because its direct size cannot fit in.
        optixu::trace<SearchRayPayloadSignature>(
            traversable, origin, direction,
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            traversable, rng, payloadPtr);
        if (payload.terminate || payload.pathLength >= 10)
            break;

        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    plp.rngBuffer[launchIndex] = rng;
    float3 accResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 1) {
        float4 accResultF4 = plp.accumBuffer[launchIndex];
        accResult = make_float3(accResultF4.x, accResultF4.y, accResultF4.z);
    }
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
    plp.accumBuffer.write(launchIndex, make_float4(accResult + payload.contribution, 1.0f));
#else
    plp.accumBuffer[launchIndex] = make_float4(accResult + payload.contribution, 1.0f);
#endif
}

RT_PROGRAM void RT_MS_NAME(searchRay)() {
    // JP: getPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     しかしここでは最初のペイロードが不要なためnullポインターを最初のペイロードに渡す。
    // EN: The signature used in getPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     However pass the null pointer as the first payload because we don't need the first payload here.
    SearchRayPayload* payload;
    optixu::getPayloads<SearchRayPayloadSignature>(nullptr, nullptr, &payload);
    payload->contribution = payload->contribution + payload->alpha * make_float3(0.01f, 0.01f, 0.01f);
    payload->terminate = true;
}



using ProgSampleTexture = optixu::DirectCallableProgramID<float3(uint32_t, float2)>;

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(sampleTexture)(uint32_t texID, float2 texCoord) {
    CUtexObject texture = plp.textures[texID];
    float4 texValue = tex2DLod<float4>(texture, texCoord.x, texCoord.y, 0.0f);
    return make_float3(texValue.x, texValue.y, texValue.z);
}

RT_CALLABLE_PROGRAM void RT_DC_NAME(decodeHitPointTriangle)(const HitPointParameter &hitPointParam, const GeometryData &geom,
                                                            float3* p, float3* sn, float2* texCoord) {
    const Triangle &tri = geom.triangleBuffer[hitPointParam.primIndex];
    const Vertex &v0 = geom.vertexBuffer[tri.index0];
    const Vertex &v1 = geom.vertexBuffer[tri.index1];
    const Vertex &v2 = geom.vertexBuffer[tri.index2];
    float b0 = hitPointParam.b0;
    float b1 = hitPointParam.b1;
    float b2 = 1 - (b0 + b1);
    *p = b0 * v0.position + b1 * v1.position + b2 * v2.position;
    *sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
    *texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
}



RT_PROGRAM void RT_CH_NAME(shading_diffuse)() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    const MaterialData &mat = plp.materialData[sbtr.materialData];
    const GeometryData &geom = plp.geomInstData[sbtr.geomInstData];

    // JP: getPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     対応するtrace/getPayloads/setPayloadsのテンプレート引数に同じ型を明示的に渡して
    //     不一致を検出できるようにすることを推奨する。
    // EN: The signature used in getPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     It is recommended to explicitly pass the same template arguments to 
    //     corresponding trace/getPayloads/setPayloads to notice mismatch.
    OptixTraversableHandle traversable;
    PCG32RNG rng;
    SearchRayPayload* payload;
    optixu::getPayloads<SearchRayPayloadSignature>(&traversable, &rng, &payload);

    float3 p;
    float3 sn;
    float2 texCoord;
    geom.decodeHitPoint(HitPointParameter::get(), &p, &sn, &texCoord);

    float3 vOut = -optixGetWorldRayDirection();
    bool isFrontFace = dot(vOut, sn) > 0;
    if (!isFrontFace)
        sn = -sn;
    p = p + sn * 0.001f;

    //// Visualize normal
    //payload->contribution = 0.5f * sn + make_float3(0.5f, 0.5f, 0.5f);
    //payload->terminate = true;
    //return;

    float3 albedo = mat.albedo;
    if (mat.misc != 0xFFFFFFFF) {
        // Demonstrate how to use texture sampling and direct callable program.
        ProgSampleTexture sampleTexture(mat.program);
        albedo = sampleTexture(mat.texID, texCoord);
    }

    const float3 LightRadiance = make_float3(20, 20, 20);
    // Hard-coded directly visible light
    if (sbtr.materialData == plp.matLightIndex &&
        isFrontFace &&
        (payload->pathLength == 1 || payload->specularBounce)) {
        payload->contribution = payload->contribution + payload->alpha * LightRadiance;
    }

    // Next Event Estimation
    {
        // Use hard-coded area light for simplicity.
        float3 lp = make_float3(-0.25f, 0.99f, -0.25f) +
            rng.getFloat0cTo1o() * make_float3(0.5f, 0, 0) +
            rng.getFloat0cTo1o() * make_float3(0, 0, 0.5f);
        float areaPDF = 4.0f;
        float3 lpn = make_float3(0, -1, 0);

        float3 shadowRayDir = lp - p;
        float dist2 = dot(shadowRayDir, shadowRayDir);
        float dist = std::sqrt(dist2);
        shadowRayDir = shadowRayDir / dist;
        float cosLight = dot(lpn, -shadowRayDir);
        float3 Le = cosLight > 0 ? LightRadiance : make_float3(0, 0, 0);

        float visibility = 1.0f;
        optixu::trace<VisibilityRayPayloadSignature>(
            traversable, p, shadowRayDir, 0.0f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);

        float cosSP = dot(sn, shadowRayDir);
        float G = visibility * std::fabs(cosSP) * std::fabs(cosLight) / dist2;
        float3 fs = cosSP > 0 ? albedo / M_PI : make_float3(0, 0, 0);
        float3 contribution = payload->alpha * fs * G * Le / areaPDF;
        payload->contribution = payload->contribution + contribution;
    }

    const auto makeCoordinateSystem = [](const float3 &n, float3* s, float3* t) {
        float sign = n.z >= 0 ? 1 : -1;
        float a = -1 / (sign + n.z);
        float b = n.x * n.y * a;
        *s = make_float3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *t = make_float3(b, sign + n.y * n.y * a, -n.y);
    };

    float3 s;
    float3 t;
    makeCoordinateSystem(sn, &s, &t);

    // Sampling incoming direction.
    float phi = 2 * M_PI * rng.getFloat0cTo1o();
    float theta = std::asin(std::sqrt(rng.getFloat0cTo1o()));
    float sinTheta = std::sin(theta);
    float3 vIn = make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta));
    vIn = make_float3(dot(make_float3(s.x, t.x, sn.x), vIn),
                      dot(make_float3(s.y, t.y, sn.y), vIn),
                      dot(make_float3(s.z, t.z, sn.z), vIn));
    payload->alpha = payload->alpha * albedo;
    payload->origin = p;
    payload->direction = vIn;
    payload->specularBounce = false;
    payload->terminate = false;

    // JP: setPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     しかしここでは書き換えていないペイロードに関してはnullポインターを渡す。
    // EN: The signature used in setPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     However pass the null pointers for the payloads which were read only.
    optixu::setPayloads<SearchRayPayloadSignature>(nullptr, &rng, nullptr);
}

// JP: それなりの規模のパストレーシングを実装する場合はプログラムは基本的に共通化して
//     差異のある部分をCallable Programなどで呼び分けるほうが実用的だが、
//     ここではデモ目的であえて別のプログラムとする。
// EN: When implementing a moderately complex path tracing,
//     it appears better to basically use a common program and callable programs for different behaviors,
//     but here define another program on purpose for demonstration.
RT_PROGRAM void RT_CH_NAME(shading_specular)() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    const MaterialData &mat = plp.materialData[sbtr.materialData];
    const GeometryData &geom = plp.geomInstData[sbtr.geomInstData];

    // JP: getPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     しかしここでは最初のペイロードが不要なためnullポインターを最初のペイロードに渡す。
    // EN: The signature used in getPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     However pass the null pointer as the first payload because we don't need the first payload here.
    SearchRayPayload* payload;
    optixu::getPayloads<SearchRayPayloadSignature>(nullptr, nullptr, &payload);

    float3 p;
    float3 sn;
    float2 texCoord;
    geom.decodeHitPoint(HitPointParameter::get(), &p, &sn, &texCoord);

    p = p + sn * 0.001f;

    //// Visualize normal
    //payload->contribution = 0.5f * sn + make_float3(0.5f, 0.5f, 0.5f);
    //payload->terminate = true;
    //return;

    float3 albedo = mat.albedo;
    if (mat.misc != 0xFFFFFFFF) {
        // Demonstrate how to use texture sampling and direct callable program.
        albedo = optixDirectCall<float3>(mat.program, mat.texID, texCoord);
    }

    float3 vOut = -optixGetWorldRayDirection();

    // Sampling incoming direction (delta distribution).
    float3 vIn = normalize(2 * dot(vOut, sn) * sn - vOut);
    payload->alpha = payload->alpha * albedo;
    payload->origin = p;
    payload->direction = vIn;
    payload->specularBounce = true;
    payload->terminate = false;
}

RT_PROGRAM void RT_AH_NAME(visibility)() {
    // JP: setPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    // EN: The signature used in setPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);

    optixTerminateRay();
}



RT_PROGRAM void RT_IS_NAME(custom_primitive)() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    const GeometryData &geom = plp.geomInstData[sbtr.geomInstData];
    uint32_t primIndex = optixGetPrimitiveIndex();
    const SphereParameter &param = geom.paramBuffer[primIndex];
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();

    float3 nDir = normalize(rayDir);

    float3 co = rayOrg - param.center;
    float b = dot(nDir, co);

    float D = b * b - (sqLength(co) - param.radius * param.radius);
    if (D < 0)
        return;

    float sqrtD = std::sqrt(D);
    float t0 = -b - sqrtD;
    float t1 = -b + sqrtD;
    bool isFront = t0 >= 0;
    float t = isFront ? t0 : t1;
    if (t < 0)
        return;

    float3 np = normalize(co + t * nDir);
    float theta = std::acos(std::fmin(std::fmax(np.z, -1.0f), 1.0f));
    float phi = std::fmod(std::atan2(np.y, np.x) + 2 * Pi, 2 * Pi);

    optixu::reportIntersection(t, isFront ? 0 : 1, theta, phi);
}

RT_CALLABLE_PROGRAM void RT_DC_NAME(decodeHitPointSphere)(const HitPointParameter &hitPointParam, const GeometryData &geom,
                                                          float3* p, float3* sn, float2* texCoord) {
    const SphereParameter &param = geom.paramBuffer[hitPointParam.primIndex];
    float theta = hitPointParam.b0;
    float phi = hitPointParam.b1;
    float sinTheta = std::sin(theta);
    float3 np = make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta));
    *p = param.center + np * param.radius;
    *sn = np;
    *texCoord = make_float2(theta / Pi, phi / (2 * Pi)) * param.texCoordMultiplier;
}



RT_PROGRAM void RT_EX_NAME(print)() {
    uint3 launchIndex = optixGetLaunchIndex();
    int32_t code = optixGetExceptionCode();
    printf("(%u, %u, %u): Exception: %u\n", launchIndex.x, launchIndex.y, launchIndex.z, code);
}
