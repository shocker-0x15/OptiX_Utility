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

// JP: optixGetSbtDataPointer()で取得できるポインターの位置に
//     Material, GeometryInstance, GeometryInstanceAccelerationStructureのsetUserData()
//     で設定したデータが順番に並んでいる(各データの相対的な開始位置は指定したアラインメントに従う)。
//     各データの開始位置は前方のデータのサイズによって変わるので、例えば同じGeometryInstanceに属していても
//     マテリアルが異なればGeometryInstanceのデータの開始位置は異なる可能性があることに注意。
//     このサンプルではMaterialとGASにはユーザーデータは設定していない。
// EN: Data set by each of Material, GeometryInstance, GeometryInstanceAccelerationStructure's setUserData()
//     line up in the order (Each relative offset follows the specified alignment)
//     at the position pointed by optixGetSbtDataPointer().
//     Note that the start position of each data changes depending on the sizes of forward data.
//     Therefore for example, the start positions of GeometryInstance's data are possibly different
//     if materials are different even if those belong to the same GeometryInstance.
//     This sample did not set user data to Material and GAS.
struct HitGroupSBTRecordData {
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
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
    // JP: ペイロードとともにトレースを呼び出す。
    //     ペイロード数は最大で合計8DW。
    // EN: Trace call with payloads.
    //     The maximum number of payloads is 8 dwords in total.
    optixu::trace<PayloadSignature>(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer[launchIndex] = make_float4(color, 1.0f);
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss0)() {
    float3 color = make_float3(0, 0, 0.1f);

    // JP: setPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     対応するtrace/getPayloads/setPayloadsのテンプレート引数に同じ型を明示的に渡して
    //     型の不一致を検出できるようにすることを推奨する。
    //     書き換えていないペイロードに関してはnullポインターを渡しても良い。
    // EN: The signature used in setPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     It is recommended to explicitly pass the same template arguments to 
    //     corresponding trace/getPayloads/setPayloads to notice type mismatch.
    //     Passing the null pointers is possible for the payloads which were read only.
    optixu::setPayloads<PayloadSignature>(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit0)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    HitPointParameter hp = HitPointParameter::get();

    Triangle triangle;
    if (geom.triangleBuffer)
        triangle = geom.triangleBuffer[hp.primIndex];
    else // triangle soup
        triangle = Triangle{ 3 * hp.primIndex + 0, 3 * hp.primIndex + 1, 3 * hp.primIndex + 2 };
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
    sn = normalize(geom.transformNormal(sn));

    // JP: 法線を可視化。
    //     このサンプルでは単一のGASしか使っていないためオブジェクト空間からワールド空間への変換は無い。
    // EN: Visualize the normal.
    //     There is no object to world space transform since this sample uses only a single GAS.
    float3 color = 0.5f * sn + make_float3(0.5f);

    // JP: setPayloads()のシグネチャーはoptixu::trace()におけるペイロード部を
    //     ポインターとしたものに一致しなければならない。
    //     対応するtrace/getPayloads/setPayloadsのテンプレート引数に同じ型を明示的に渡して
    //     型の不一致を検出できるようにすることを推奨する。
    //     書き換えていないペイロードに関してはnullポインターを渡しても良い。
    // EN: The signature used in setPayloads() must match the one replacing the part of payloads
    //     in optixu::trace() to pointer types.
    //     It is recommended to explicitly pass the same template arguments to 
    //     corresponding trace/getPayloads/setPayloads to notice type mismatch.
    //     Passing the null pointers is possible for the payloads which were read only.
    optixu::setPayloads<PayloadSignature>(&color);
}
