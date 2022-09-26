#pragma once

#include "single_level_instancing_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
        HitPointParameter ret;
        float2 bc = optixGetTriangleBarycentrics();
        ret.b1 = bc.x;
        ret.b2 = bc.y;
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

/*
JP: CH/AH/ISプログラムにてoptixGetSbtDataPointer()で取得できるポインターの位置に
    GeometryInstanceAccelerationStructureのsetUserData(), setChildUserData(),
    GeometryInstanceのsetUserData(), MaterialのsetUserData()
    で設定したデータが順番に並んでいる(各データの相対的な開始位置は指定したアラインメントに従う)。
    各データの開始位置は前方のデータのサイズによって変わるので、例えば同じGASに属していても
    GASの子ごとのデータサイズが異なればGeometryInstanceのデータの開始位置は異なる可能性があることに注意。
    このサンプルではGASとGASの子達、Materialにはユーザーデータは設定していない。
EN: Data set by each of
    GeometryInstanceAccelerationStructure's setUserData() and setChildUserData(),
    GeometryInstance's setUserData(), Material's setUserData()
    line up in the order (Each relative offset follows the specified alignment)
    at the position pointed by optixGetSbtDataPointer() called in CH/AH/IS programs.
    Note that the start position of each data changes depending on the sizes of forward data.
    Therefore for example, the start positions of GeometryInstance's data are different
    if data sizes of GAS children are different even if those belong to the same GAS.
    This sample does not set user data to GAS, GAS's child and Material.
*/
struct HitGroupSBTRecordData {
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



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
    auto hp = HitPointParameter::get();

    const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geom.vertexBuffer[triangle.index0];
    const Vertex &v1 = geom.vertexBuffer[triangle.index1];
    const Vertex &v2 = geom.vertexBuffer[triangle.index2];

    float b0 = 1 - (hp.b1 + hp.b2);
    float3 sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;

    /*
    JP: GeometryInstanceからGAS空間への変換とは違って、GAS空間からインスタンス空間
        (1段階インスタンシングの場合はワールド空間に相当)への変換は組み込み関数が用意されている。
    EN: There is an intrinsic function to transform from GAS space to instance space
        (corresponds to world space in single-level instancing case)
        unlike the transform from GeometryInstance to GAS space.
    */
    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線の可視化。
    // EN: Display normal visualization.
    float3 color = 0.5f * sn + make_float3(0.5f);
    MyPayloadSignature::set(&color);
}
