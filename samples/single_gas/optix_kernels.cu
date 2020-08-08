#pragma once

#include "single_gas_shared.h"

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

CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen0)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    float3 color;
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer[launchIndex] = make_float4(color, 1.0f);
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss0)() {
    float3 color = make_float3(0, 0, 0.1f);
    optixu::setPayloads<PayloadSignature>(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit0)() {
    auto sbtr = optixu::getHitGroupSBTRecordData();
    const GeometryData &geom = plp.geomInstData[sbtr.geomInstData];
    HitPointParameter hp = HitPointParameter::get();

    const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geom.vertexBuffer[triangle.index0];
    const Vertex &v1 = geom.vertexBuffer[triangle.index1];
    const Vertex &v2 = geom.vertexBuffer[triangle.index2];

    float b0 = 1 - (hp.b1 + hp.b2);
    float3 sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;

    // JP: GeometryInstanceからGAS空間への変換は自前で実装する必要がある。
    //     ただしGASのビルド設定でRandom Vertex Accessを有効にしている場合はoptixGetTriangleVertexData()
    //     を呼ぶことで位置に関しては変換後の値を取得することができる。
    // EN: Transform from GeometryInstance to GAS space should be manually implemented by the user.
    //     However, it is possible to get post-transformed values using optixGetTriangleVertexData()
    //     only for positions if random vertex access is enabled for GAS build configuration.
    const GeometryPreTransform &preTransform = plp.geomPreTransforms[optixGetSbtGASIndex()];
    sn = normalize(preTransform.transformNormal(sn));

    // JP: 法線を可視化。
    //     このサンプルでは単一のGASしか使っていないためオブジェクト空間からワールド空間への変換は無い。
    // EN: Visualize the normal.
    //     There is no object to world space transform since this sample uses only a single GAS.
    float3 color = 0.5f * sn + make_float3(0.5f);
    optixu::setPayloads<PayloadSignature>(&color);
}
