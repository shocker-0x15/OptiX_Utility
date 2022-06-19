#pragma once

#include "curve_primitive_shared.h"
#include "../common/curve_evaluator.h"

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
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



template <OptixPrimitiveType curveType>
CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcCurveSurfaceNormal(
    const GeometryData &geom, uint32_t primIndex, float curveParam, const float3 &hp) {
    constexpr uint32_t numControlPoints = curve::getNumControlPoints<curveType>();
    float4 controlPoints[numControlPoints];
    if constexpr (useEmbeddedVertexData) {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        uint32_t sbtGasIndex = optixGetSbtGASIndex();
        if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
            optixGetLinearCurveVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
            optixGetQuadraticBSplineVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
            optixGetCubicBSplineVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM)
            optixGetCatmullRomVertexData(gasHandle, primIndex, sbtGasIndex, 0.0f, controlPoints);
    }
    else {
        uint32_t baseIndex = geom.segmentIndexBuffer[primIndex];
#pragma unroll
        for (int i = 0; i < numControlPoints; ++i) {
            const CurveVertex &v = geom.curveVertexBuffer[baseIndex + i];
            controlPoints[i] = make_float4(v.position, v.width);
        }
    }

    curve::Evaluator<curveType> ce(controlPoints);
    float3 sn = ce.calcNormal(curveParam, hp);

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
    PayloadSignature::trace(
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
    const GeometryData &geom = sbtr.geomData;

    float3 sn;
    OptixPrimitiveType primType = optixGetPrimitiveType();
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        auto hp = HitPointParameter::get();
        const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geom.vertexBuffer[triangle.index0];
        const Vertex &v1 = geom.vertexBuffer[triangle.index1];
        const Vertex &v2 = geom.vertexBuffer[triangle.index2];

        float b0 = 1 - (hp.b1 + hp.b2);
        sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;
    }
    else {
        uint32_t primIndex = optixGetPrimitiveIndex();
        float curveParam = optixGetCurveParameter();
        float3 hp = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        hp = optixTransformPointFromWorldToObjectSpace(hp);

        if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR>(geom, primIndex, curveParam, hp);
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE>(geom, primIndex, curveParam, hp);
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>(geom, primIndex, curveParam, hp);
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM)
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM>(geom, primIndex, curveParam, hp);
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    PayloadSignature::set(&color);
}
