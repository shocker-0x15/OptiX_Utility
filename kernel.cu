#pragma once

#include "shared.h"

#define M_PI 3.14159265

namespace Sample {

using namespace Shared;

extern "C" __constant__ PipelineLaunchParameters plp;



// JP: このクラスのようにシステマティックにuint32_t&にせずに、
//     個別に適切なペイロードの渡し方を考えたほうが性能は良いかもしれない。
template <typename PayloadType>
union PayloadAccessor {
    PayloadType raw;
    uint32_t asUInt[(sizeof(PayloadType) + 3) / 4];

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
    float3 contribution;
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



RT_PROGRAM void __raygen__fill() {
    uint3 launchIndex = optixGetLaunchIndex();
    int32_t index = plp.imageSize.x * launchIndex.y + launchIndex.x;

    PCG32RNG rng = plp.rngBuffer[index];

    float x = (float)(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = (float)(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = make_float3(0, 0, 3);
    float3 direction = normalize(make_float3(vw * (x - 0.5f), vh * (0.5f - y), -1));

    OptixTraversableHandle topGroup = plp.baseParams.handles[plp.topGroupIndex];

    PayloadAccessor<SearchRayPayload> payload;
    payload.raw.rng = rng;
    optixTrace(topGroup, origin, direction, 0.0f, INFINITY, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
               RayType_Search, NumRayTypes, RayType_Search,
               payload[0], payload[1], payload[2], payload[3], payload[4]);

    plp.rngBuffer[index] = rng;
    float3 cumResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 1) {
        float4 cumResultF4 = plp.accumBuffer[index];
        cumResult = make_float3(cumResultF4.x, cumResultF4.y, cumResultF4.z);
    }
    plp.accumBuffer[index] = make_float4(cumResult + payload.raw.contribution, 1.0f);
}

RT_PROGRAM void __miss__searchRay() {
    PayloadAccessor<SearchRayPayload> payload;

    payload.raw.contribution = make_float3(0.0f, 0.0f, 0.05f);
    
    payload.setAll();
}

RT_PROGRAM void __closesthit__shading() {
    auto sbtr = optix::getHitGroupSBTRecordData();
    auto matData = reinterpret_cast<const MaterialData*>(plp.baseParams.materialData);
    auto geomInstData = reinterpret_cast<const GeometryData*>(plp.baseParams.geomInstData);

    const MaterialData &mat = matData[sbtr.materialDataIndex];
    const GeometryData &geom = geomInstData[sbtr.geomInstDataIndex];

    OptixTraversableHandle topGroup = plp.baseParams.handles[plp.topGroupIndex];

    auto hitPointParam = HitPointParameter::get();

    PayloadAccessor<SearchRayPayload> payload;
    payload.getAll();

    PCG32RNG &rng = payload.raw.rng;

    const Triangle &tri = geom.triangleBuffer[hitPointParam.primIndex];
    float3 p0 = geom.vertexBuffer[tri.index0].position;
    float3 p1 = geom.vertexBuffer[tri.index1].position;
    float3 p2 = geom.vertexBuffer[tri.index2].position;
    float3 n0 = geom.vertexBuffer[tri.index0].normal;
    float3 n1 = geom.vertexBuffer[tri.index1].normal;
    float3 n2 = geom.vertexBuffer[tri.index2].normal;
    float b0 = hitPointParam.b0;
    float b1 = hitPointParam.b1;
    float b2 = 1 - (b0 + b1);
    float3 p = b0 * p0 + b1 * p1 + b2 * p2;
    float3 sn = normalize(b0 * n0 + b1 * n1 + b2 * n2);
    p = p + sn * 0.001f;

    float3 lp = make_float3(-0.5f, 0.99f, -0.5f) +
        rng.getFloat0cTo1o() * make_float3(1, 0, 0) + 
        rng.getFloat0cTo1o() * make_float3(0, 0, 1);
    float areaPDF = 1.0f;
    float3 lpn = make_float3(0, -1, 0);

    float3 shadowRayDir = lp - p;
    float dist2 = dot(shadowRayDir, shadowRayDir);
    float dist = std::sqrt(dist2);
    shadowRayDir = shadowRayDir / dist;
    float cosLight = dot(lpn, -shadowRayDir);
    float3 Le = cosLight > 0 ? make_float3(5, 5, 5) : make_float3(0, 0, 0);

    PayloadAccessor<VisibilityRayPayload> shadowPayload;
    shadowPayload.raw.visibility = 1.0f;
    optixTrace(topGroup, p, shadowRayDir, 0.0f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
               RayType_Visibility, NumRayTypes, RayType_Visibility,
               shadowPayload[0]);

    float G = shadowPayload.raw.visibility * dot(sn, shadowRayDir) * cosLight / dist2;
    float3 contribution = (mat.albedo / M_PI) * G * Le / areaPDF;
    payload.raw.contribution = contribution;

    payload.setAll();
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
