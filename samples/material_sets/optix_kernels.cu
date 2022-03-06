#pragma once

#include "material_sets_shared.h"

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

// JP: CH/AH/ISプログラムにてoptixGetSbtDataPointer()で取得できるポインターの位置に
//     GeometryInstanceAccelerationStructureのsetUserData(), setChildUserData(),
//     GeometryInstanceのsetUserData(), MaterialのsetUserData()
//     で設定したデータが順番に並んでいる(各データの相対的な開始位置は指定したアラインメントに従う)。
//     各データの開始位置は前方のデータのサイズによって変わるので、例えば同じGASに属していても
//     GASの子ごとのデータサイズが異なればGeometryInstanceのデータの開始位置は異なる可能性があることに注意。
//     このサンプルではGAS、GASの子達にはユーザーデータは設定していない。
// EN: Data set by each of
//     GeometryInstanceAccelerationStructure's setUserData() and setChildUserData(),
//     GeometryInstance's setUserData(), Material's setUserData()
//     line up in the order (Each relative offset follows the specified alignment)
//     at the position pointed by optixGetSbtDataPointer() called in CH/AH/IS programs.
//     Note that the start position of each data changes depending on the sizes of forward data.
//     Therefore for example, the start positions of GeometryInstance's data are different
//     if data sizes of GAS children are different even if those belong to the same GAS.
//     This sample does not set user data to GAS and GAS's children.
struct HitGroupSBTRecordData {
    GeometryData geomData;
    MaterialData matData;

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
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer[launchIndex] = make_float4(color, 1.0f);
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    PayloadSignature::set(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    PayloadSignature::set(&sbtr.matData.color);
}
