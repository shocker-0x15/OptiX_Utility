#pragma once

#include "sphere_primitive_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    union {
        float b1;
        float secondDistance;
    };
    float b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        HitPointParameter ret;
        OptixPrimitiveType primType = optixGetPrimitiveType();
        if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
            float2 bc = optixGetTriangleBarycentrics();
            ret.b1 = bc.x;
            ret.b2 = bc.y;
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_SPHERE) {
            // When a ray hits a sphere twice, the attribute 0 contains the second distance.
            uint32_t attr0 = optixGetAttribute_0();
            ret.secondDistance = __uint_as_float(attr0);
        }
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcSphereSurfaceNormal(
    const GeometryData &geom, uint32_t primIndex, const float3 &hp) {
    float3 center;
    if constexpr (useEmbeddedVertexData) {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        uint32_t sbtGasIndex = optixGetSbtGASIndex();
        float4 centerAndRadius;
        optixGetSphereData(gasHandle, primIndex, sbtGasIndex, 0.0f, &centerAndRadius);
        center = make_float3(centerAndRadius);
    }
    else {
        const SphereParameter &param = geom.sphereParamBuffer[primIndex];
        center = param.center;
    }

    float3 sn = normalize(hp - center);

    return sn;
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    float3 color;
    MyPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer[launchIndex] = make_float4(color, 1.0f);
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    MyPayloadSignature::set(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;

    auto hpParam = HitPointParameter::get();

    float3 sn;
    OptixPrimitiveType primType = optixGetPrimitiveType();
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        const Triangle &triangle = geom.triangleBuffer[hpParam.primIndex];
        const Vertex &v0 = geom.vertexBuffer[triangle.index0];
        const Vertex &v1 = geom.vertexBuffer[triangle.index1];
        const Vertex &v2 = geom.vertexBuffer[triangle.index2];

        float b0 = 1 - (hpParam.b1 + hpParam.b2);
        sn = b0 * v0.normal + hpParam.b1 * v1.normal + hpParam.b2 * v2.normal;
    }
    else {
        uint32_t primIndex = optixGetPrimitiveIndex();
        float3 hp = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        hp = optixTransformPointFromWorldToObjectSpace(hp);

        sn = calcSphereSurfaceNormal(geom, primIndex, hp);
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    MyPayloadSignature::set(&color);
}
