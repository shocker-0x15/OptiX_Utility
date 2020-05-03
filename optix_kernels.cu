#pragma once

#include "shared.h"

#define M_PI 3.14159265

namespace Sample {

using namespace Shared;

extern "C" __constant__ PipelineLaunchParameters plp;



struct SearchRayPayload {
    float3 alpha;
    float3 contribution;
    float3 origin;
    float3 direction;
    struct {
        uint32_t pathLength;
        bool specularBounce : 1;
        bool terminate : 1;
    };
};

#define SearchRayPayloadSignature OptixTraversableHandle, PCG32RNG, SearchRayPayload*

struct VisibilityRayPayload {
    float visibility;
};

#define VisibilityRayPayloadSignature float

struct HitPointParameter {
    float b0, b1;
    int32_t primIndex;

    RT_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        if (optixIsTriangleHit()) {
            float2 bc = optixGetTriangleBarycentrics();
            ret.b0 = 1 - bc.x - bc.y;
            ret.b1 = bc.x;
        }
        else {
            ret.b0 = __uint_as_float(optixGetAttribute_0());
            ret.b1 = __uint_as_float(optixGetAttribute_1());
        }
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};



RT_PROGRAM void __raygen__pathtracing() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);
    int32_t index = plp.imageSize.x * launchIndex.y + launchIndex.x;

    PCG32RNG rng = plp.rngBuffer[index];

    float x = (float)(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = (float)(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
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
            0.0f, INFINITY, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            traversable, rng, payloadPtr);
        if (payload.terminate || payload.pathLength >= 10)
            break;

        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    plp.rngBuffer[index] = rng;
    float3 accResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 1) {
#if defined(USE_BUFFER2D)
        float4 accResultF4 = plp.accumBuffer[launchIndex];
#else
        float4 accResultF4 = plp.accumBuffer[index];
#endif
        accResult = make_float3(accResultF4.x, accResultF4.y, accResultF4.z);
    }
#if defined(USE_BUFFER2D)
    plp.accumBuffer.write(launchIndex, make_float4(accResult + payload.contribution, 1.0f));
#else
    plp.accumBuffer[index] = make_float4(accResult + payload.contribution, 1.0f);
#endif
}

RT_PROGRAM void __miss__searchRay() {
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

RT_CALLABLE_PROGRAM float3 __direct_callable__sampleTexture(uint32_t texID, float2 texCoord) {
    CUtexObject texture = plp.textures[texID];
    float4 texValue = tex2D<float4>(texture, texCoord.x, texCoord.y);
    return make_float3(texValue.x, texValue.y, texValue.z);
}

RT_PROGRAM void __closesthit__shading_diffuse() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    auto matData = reinterpret_cast<const MaterialData*>(plp.materialData);
    auto geomInstData = reinterpret_cast<const GeometryData*>(plp.geomInstData);

    const MaterialData &mat = matData[sbtr.materialData];
    const GeometryData &geom = geomInstData[sbtr.geomInstData];

    auto hitPointParam = HitPointParameter::get();

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

    const Triangle &tri = geom.triangleBuffer[hitPointParam.primIndex];
    const Vertex &v0 = geom.vertexBuffer[tri.index0];
    const Vertex &v1 = geom.vertexBuffer[tri.index1];
    const Vertex &v2 = geom.vertexBuffer[tri.index2];
    float b0 = hitPointParam.b0;
    float b1 = hitPointParam.b1;
    float b2 = 1 - (b0 + b1);
    float3 p = optixTransformPointFromObjectToWorldSpace(b0 * v0.position + b1 * v1.position + b2 * v2.position);
    float3 sn = normalize(optixTransformNormalFromObjectToWorldSpace(b0 * v0.normal + b1 * v1.normal + b2 * v2.normal));

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
        float2 texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
        albedo = optixDirectCall<float3>(mat.program, mat.texID, texCoord);
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
RT_PROGRAM void __closesthit__shading_specular() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    auto matData = reinterpret_cast<const MaterialData*>(plp.materialData);
    auto geomInstData = reinterpret_cast<const GeometryData*>(plp.geomInstData);

    const MaterialData &mat = matData[sbtr.materialData];
    const GeometryData &geom = geomInstData[sbtr.geomInstData];

    auto hitPointParam = HitPointParameter::get();

    // JP: getPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     しかしここでは最初のペイロードが不要なためnullポインターを最初のペイロードに渡す。
    // EN: The signature used in getPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     However pass the null pointer as the first payload because we don't need the first payload here.
    SearchRayPayload* payload;
    optixu::getPayloads<SearchRayPayloadSignature>(nullptr, nullptr, &payload);

    const Triangle &tri = geom.triangleBuffer[hitPointParam.primIndex];
    const Vertex &v0 = geom.vertexBuffer[tri.index0];
    const Vertex &v1 = geom.vertexBuffer[tri.index1];
    const Vertex &v2 = geom.vertexBuffer[tri.index2];
    float b0 = hitPointParam.b0;
    float b1 = hitPointParam.b1;
    float b2 = 1 - (b0 + b1);
    float3 p = optixTransformPointFromObjectToWorldSpace(b0 * v0.position + b1 * v1.position + b2 * v2.position);
    float3 sn = normalize(optixTransformNormalFromObjectToWorldSpace(b0 * v0.normal + b1 * v1.normal + b2 * v2.normal));
    p = p + sn * 0.001f;

    float3 albedo = mat.albedo;
    if (mat.misc != 0xFFFFFFFF) {
        // Demonstrate how to use texture sampling and direct callable program.
        float2 texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
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

RT_PROGRAM void __anyhit__visibility() {
    // JP: setPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    // EN: The signature used in setPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);

    optixTerminateRay();
}

RT_PROGRAM void __exception__print() {
    uint3 launchIndex = optixGetLaunchIndex();
    int32_t code = optixGetExceptionCode();
    printf("(%u, %u, %u): Exception: %u\n", launchIndex.x, launchIndex.y, launchIndex.z, code);
}

}
