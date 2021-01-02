﻿#pragma once

#include "pick_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS RenderPipelineLaunchParameters plp;



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    MaterialData matData;
    GeometryData geomData;
    GASData gasData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



#define PayloadSignature float3

CUDA_DEVICE_KERNEL void RT_RG_NAME(perspectiveRaygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(plp.imageSize.y - launchIndex.y - 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.perspCamera.fovY * 0.5f);
    float vw = plp.perspCamera.aspect * vh;

    float3 origin = plp.position;
    float3 direction = normalize(plp.orientation * make_float3(vw * (0.5f - x), vh * (y - 0.5f), 1));

    float3 color;
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer.write(launchIndex, make_float4(color, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(equirectangularRaygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + 0.5f) / plp.imageSize.y;
    float phi = plp.equirecCamera.horizentalExtent * (x - 0.5f) + 0.5f * Pi;
    float theta = plp.equirecCamera.verticalExtent * (y - 0.5f) + 0.5f * Pi;

    float3 origin = plp.position;
    float3 direction = normalize(plp.orientation *
                                 make_float3(std::cos(phi) * std::sin(theta),
                                             std::cos(theta),
                                             std::sin(phi) * std::sin(theta)));

    float3 color;
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer.write(launchIndex, make_float4(color, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    optixu::setPayloads<PayloadSignature>(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const MaterialData &mat = sbtr.matData;
    const GeometryData &geom = sbtr.geomData;
    const GASData &gas = sbtr.gasData;
    auto hp = HitPointParameter::get();

    const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geom.vertexBuffer[triangle.index0];
    const Vertex &v1 = geom.vertexBuffer[triangle.index1];
    const Vertex &v2 = geom.vertexBuffer[triangle.index2];

    float b0 = 1 - (hp.b1 + hp.b2);
    float3 sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    float3 color = 0.5f * mat.color + 0.5f * (0.5f * sn + make_float3(0.5f));
    const PickInfo &pickInfo = *plp.pickInfo;
    if (pickInfo.hit &&
        optixGetInstanceIndex() == pickInfo.instanceIndex &&
        optixGetSbtGASIndex() == pickInfo.matIndex &&
        optixGetPrimitiveIndex() == pickInfo.primIndex)
        color = 0.5f * color;
    optixu::setPayloads<PayloadSignature>(&color);
}
