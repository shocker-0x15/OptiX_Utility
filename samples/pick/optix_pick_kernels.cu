#include "pick_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PickPipelineLaunchParameters plp;



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
// EN: Data set by each of
//     GeometryInstanceAccelerationStructure's setUserData() and setChildUserData(),
//     GeometryInstance's setUserData(), Material's setUserData()
//     line up in the order (Each relative offset follows the specified alignment)
//     at the position pointed by optixGetSbtDataPointer() called in CH/AH/IS programs.
//     Note that the start position of each data changes depending on the sizes of forward data.
//     Therefore for example, the start positions of GeometryInstance's data are different
//     if data sizes of GAS children are different even if those belong to the same GAS.
struct HitGroupSBTRecordData {
    GASData gasData;
    GASChildData gasChildData;
    GeometryData geomData;
    MaterialData matData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_RG_NAME(perspectiveRaygen)() {
    float x = static_cast<float>(plp.mousePosition.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(plp.imageSize.y - plp.mousePosition.y - 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.perspCamera.fovY * 0.5f);
    float vw = plp.perspCamera.aspect * vh;

    float3 origin = plp.position;
    float3 direction = normalize(plp.orientation * make_float3(vw * (0.5f - x), vh * (y - 0.5f), 1));

    PickInfo info;
    PickPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        PickRayType_Primary, NumPickRayTypes, PickRayType_Primary,
        info);

    *plp.pickInfo = info;
    if (plp.mousePosition.x < 0 || plp.mousePosition.x >= plp.imageSize.x ||
        plp.mousePosition.y < 0 || plp.mousePosition.y >= plp.imageSize.y)
        plp.pickInfo->hit = false;
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(equirectangularRaygen)() {
    float x = static_cast<float>(plp.mousePosition.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(plp.mousePosition.y + 0.5f) / plp.imageSize.y;
    float phi = plp.equirecCamera.horizentalExtent * (x - 0.5f) + 0.5f * Pi;
    float theta = plp.equirecCamera.verticalExtent * (y - 0.5f) + 0.5f * Pi;

    float3 origin = plp.position;
    float3 direction = normalize(plp.orientation *
                                 make_float3(std::cos(phi) * std::sin(theta),
                                             std::cos(theta),
                                             std::sin(phi) * std::sin(theta)));

    PickInfo info;
    PickPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        PickRayType_Primary, NumPickRayTypes, PickRayType_Primary,
        info);

    *plp.pickInfo = info;
    if (plp.mousePosition.x < 0 || plp.mousePosition.x >= plp.imageSize.x ||
        plp.mousePosition.y < 0 || plp.mousePosition.y >= plp.imageSize.y)
        plp.pickInfo->hit = false;
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    PickInfo info = {};
    info.hit = false;
    PickPayloadSignature::set(&info);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const MaterialData &mat = sbtr.matData;
    const GeometryData &geom = sbtr.geomData;
    const GASChildData &gasChild = sbtr.gasChildData;
    const GASData &gas = sbtr.gasData;
    auto hp = HitPointParameter::get();

    PickInfo info;
    info.hit = true;
    info.instanceIndex = optixGetInstanceIndex();
    info.matIndex = optixGetSbtGASIndex();
    info.primIndex = optixGetPrimitiveIndex();
    info.instanceID = optixGetInstanceId();
    info.gasID = gas.gasID;
    info.gasChildID = gasChild.gasChildID;
    info.geomID = geom.geomID;
    info.matID = mat.matID;
    PickPayloadSignature::set(&info);
}
