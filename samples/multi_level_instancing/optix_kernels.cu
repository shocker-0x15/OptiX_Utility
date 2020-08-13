#pragma once

#include "multi_level_instancing_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



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



#define PayloadSignature float3

CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.rngBuffer[launchIndex];

    float x = static_cast<float>(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    // JP: モーションブラーの効果をわかりやすくするために画像中の位置ごとに時間幅を変える。
    // EN: Use different duration depending on position in the image to make it easy to see
    //     the motion blur effect.
    float timeRange = plp.timeEnd - plp.timeBegin;
    timeRange *= ((y < 0.5f ? 2 : 0) + (x > 0.5f ? 1 : 0)) / 3.0f;
    float time = plp.timeBegin + timeRange * rng.getFloat0cTo1o();

    float3 color;
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, time, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.rngBuffer[launchIndex] = rng;

    float3 curResult = color;
    float curWeight = 1.0f / (1 + plp.numAccumFrames);
    float3 prevResult = getXYZ(plp.accumBuffer[launchIndex]);
    curResult = (1 - curWeight) * prevResult + curWeight * curResult;
    plp.accumBuffer[launchIndex] = make_float4(curResult, 1.0f);
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    optixu::setPayloads<PayloadSignature>(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    const GeometryData &geom = plp.geomInstData[sbtr.geomInstData];
    HitPointParameter hp = HitPointParameter::get();

    const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geom.vertexBuffer[triangle.index0];
    const Vertex &v1 = geom.vertexBuffer[triangle.index1];
    const Vertex &v2 = geom.vertexBuffer[triangle.index2];

    float b0 = 1 - (hp.b1 + hp.b2);
    float3 sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;

    sn = normalize(sn);

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    optixu::setPayloads<PayloadSignature>(&color);
}
