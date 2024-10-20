#include "scene_edit_shared.h"

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

struct HitGroupSBTRecordData {
    GASChildData gasChildData;
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + 0.5f) / plp.imageSize.x;
    float y = static_cast<float>(plp.imageSize.y - launchIndex.y - 0.5f) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (y - 0.5f), 1));

    float3 color;
    MyPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color);

    plp.resultBuffer.write(launchIndex, make_float4(color, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    MyPayloadSignature::set(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    const GASChildData &gasChild = sbtr.gasChildData;
    auto hp = HitPointParameter::get();

    const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geom.vertexBuffer[triangle.index0];
    const Vertex &v1 = geom.vertexBuffer[triangle.index1];
    const Vertex &v2 = geom.vertexBuffer[triangle.index2];

    float b0 = 1 - (hp.b1 + hp.b2);
    float3 sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;

    sn = gasChild.transformNormalFromObjectToWorld(sn);
    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    MyPayloadSignature::set(&color);
}
