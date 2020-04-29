#pragma once

#include "shared.h"

#define M_PI 3.14159265

namespace Sample {

using namespace Shared;

extern "C" __constant__ PipelineLaunchParameters plp;



// JP: このクラスのようにシステマティックにuint32_t&にせずに、
//     個別に適切なペイロードの渡し方を考えたほうが性能は良いかもしれない。
// EN: It is possibly better to individually tune how to pass a payload
//     unlike this class which systematically uses uint32_t &.
template <typename PayloadType>
union PayloadAccessor {
    PayloadType raw;
    uint32_t asUInt[(sizeof(PayloadType) + 3) / 4];
    static_assert(sizeof(PayloadType) <= 8 * 4, "sizeof(PayloadType) must be within 8 DWords.");

    RT_FUNCTION PayloadAccessor() {
        for (int i = 0; i < sizeof(asUInt) / 4; ++i)
            asUInt[i] = optixUndefinedValue();
    }

    RT_FUNCTION uint32_t &operator[](uint32_t idx) {
        return asUInt[idx];
    }

    RT_FUNCTION void getAll() {
        constexpr uint32_t numSlots = sizeof(asUInt) / 4;
        if (numSlots > 0)
            asUInt[0] = optixGetPayload_0();
        if (numSlots > 1)
            asUInt[1] = optixGetPayload_1();
        if (numSlots > 2)
            asUInt[2] = optixGetPayload_2();
        if (numSlots > 3)
            asUInt[3] = optixGetPayload_3();
        if (numSlots > 4)
            asUInt[4] = optixGetPayload_4();
        if (numSlots > 5)
            asUInt[5] = optixGetPayload_5();
        if (numSlots > 6)
            asUInt[6] = optixGetPayload_6();
        if (numSlots > 7)
            asUInt[7] = optixGetPayload_7();
    }

    RT_FUNCTION void setAll() const {
        constexpr uint32_t numSlots = sizeof(asUInt) / 4;
        if (numSlots > 0)
            optixSetPayload_0(asUInt[0]);
        if (numSlots > 1)
            optixSetPayload_1(asUInt[1]);
        if (numSlots > 2)
            optixSetPayload_2(asUInt[2]);
        if (numSlots > 3)
            optixSetPayload_3(asUInt[3]);
        if (numSlots > 4)
            optixSetPayload_4(asUInt[4]);
        if (numSlots > 5)
            optixSetPayload_5(asUInt[5]);
        if (numSlots > 6)
            optixSetPayload_6(asUInt[6]);
        if (numSlots > 7)
            optixSetPayload_7(asUInt[7]);
    }
};



struct Ray {
    float3 origin;
    float3 direction;
    float tmin;
    float tmax;
    float time;

    RT_FUNCTION static Ray getWorld() {
        Ray ret;
        ret.origin = optixGetWorldRayOrigin();
        ret.direction = optixGetWorldRayDirection();
        ret.tmin = optixGetRayTmin();
        ret.tmax = optixGetRayTmax();
        ret.time = optixGetRayTime();
        return ret;
    }
    RT_FUNCTION static Ray getObject() {
        Ray ret;
        ret.origin = optixGetObjectRayOrigin();
        ret.direction = optixGetObjectRayDirection();
        ret.tmin = optixGetRayTmin();
        ret.tmax = optixGetRayTmax();
        ret.time = optixGetRayTime();
        return ret;
    }
};

struct SearchRayPayload {
    PCG32RNG rng;
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

struct VisibilityRayPayload {
    float visibility;
};

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

    OptixTraversableHandle topGroup = plp.travHandles[plp.travIndex];

    PayloadAccessor<SearchRayPayload*> payloadPtr;
    SearchRayPayload payload;
    payload.alpha = make_float3(1.0f, 1.0f, 1.0f);
    payload.contribution = make_float3(0.0f, 0.0f, 0.0f);
    payload.rng = rng;
    payload.pathLength = 1;
    payload.terminate = false;
    payloadPtr.raw = &payload;
    while (true) {
        optixTrace(topGroup, origin, direction, 0.0f, INFINITY, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                   RayType_Search, NumRayTypes, RayType_Search,
                   payloadPtr[0], payloadPtr[1]);
        if (payload.terminate || payload.pathLength >= 10)
            break;

        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    plp.rngBuffer[index] = payload.rng;
    float3 cumResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 1) {
#if defined(USE_BUFFER2D)
        float4 cumResultF4 = plp.accumBuffer[launchIndex];
#else
        float4 cumResultF4 = plp.accumBuffer[index];
#endif
        cumResult = make_float3(cumResultF4.x, cumResultF4.y, cumResultF4.z);
    }
#if defined(USE_BUFFER2D)
    plp.accumBuffer.write(launchIndex, make_float4(cumResult + payload.contribution, 1.0f));
#else
    plp.accumBuffer[index] = make_float4(cumResult + payload.contribution, 1.0f);
#endif
}

RT_PROGRAM void __miss__searchRay() {
    PayloadAccessor<SearchRayPayload*> payloadPtr;
    payloadPtr.getAll();
    SearchRayPayload &payload = *payloadPtr.raw;

    payload.contribution = payload.contribution + payload.alpha * make_float3(0.01f, 0.01f, 0.01f);
    payload.terminate = true;

    //payload.setAll();
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

    OptixTraversableHandle topGroup = plp.travHandles[plp.travIndex];

    auto hitPointParam = HitPointParameter::get();

    PayloadAccessor<SearchRayPayload*> payloadPtr;
    payloadPtr.getAll();
    SearchRayPayload &payload = *payloadPtr.raw;

    PCG32RNG &rng = payload.rng;

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
    //payload.contribution = 0.5f * sn + make_float3(0.5f, 0.5f, 0.5f);
    //payload.terminate = true;
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
        (payload.pathLength == 1 || payload.specularBounce)) {
        payload.contribution = payload.contribution + payload.alpha * LightRadiance;
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

        PayloadAccessor<VisibilityRayPayload> shadowPayload;
        shadowPayload.raw.visibility = 1.0f;
        optixTrace(topGroup, p, shadowRayDir, 0.0f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
                   RayType_Visibility, NumRayTypes, RayType_Visibility,
                   shadowPayload[0]);

        float cosSP = dot(sn, shadowRayDir);
        float G = shadowPayload.raw.visibility * std::fabs(cosSP) * std::fabs(cosLight) / dist2;
        float3 fs = cosSP > 0 ? albedo / M_PI : make_float3(0, 0, 0);
        float3 contribution = payload.alpha * fs * G * Le / areaPDF;
        payload.contribution = payload.contribution + contribution;
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
    payload.alpha = payload.alpha * albedo;
    payload.origin = p;
    payload.direction = vIn;
    payload.specularBounce = false;
    payload.terminate = false;

    //payload.setAll();
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

    PayloadAccessor<SearchRayPayload*> payloadPtr;
    payloadPtr.getAll();
    SearchRayPayload &payload = *payloadPtr.raw;

    PCG32RNG &rng = payload.rng;

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
    payload.alpha = payload.alpha * albedo;
    payload.origin = p;
    payload.direction = vIn;
    payload.specularBounce = true;
    payload.terminate = false;

    //payload.setAll();
}

RT_PROGRAM void __anyhit__visibility() {
    PayloadAccessor<VisibilityRayPayload> payload;

    payload.raw.visibility = 0.0f;
    payload.setAll();

    optixTerminateRay();
}

RT_PROGRAM void __exception__print() {
    uint3 launchIndex = optixGetLaunchIndex();
    int32_t code = optixGetExceptionCode();
    printf("(%u, %u, %u): Exception: %u\n", launchIndex.x, launchIndex.y, launchIndex.z, code);
}

}
