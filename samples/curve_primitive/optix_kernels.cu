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
        // OptiX Programming Guide
        // > This data fetch of the currently intersected curve does not require
        // > building the corresponding geometry acceleration structure with flag
        // > OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS.
        // JP: 直近に交差した曲線の制御点取得だけであればGASビルド時に
        //     OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESSを指定する必要はない。
        // EN: It is not necessary to specify OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
        //     when building the GAS just to get the control points of the most recently intersected curve.
        if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
            optixGetLinearCurveVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
            optixGetQuadraticBSplineVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS)
            optixGetQuadraticBSplineRocapsVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
            optixGetCubicBSplineVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS)
            optixGetCubicBSplineRocapsVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM)
            optixGetCatmullRomVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS)
            optixGetCatmullRomRocapsVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
            optixGetCubicBezierVertexData(controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS)
            optixGetCubicBezierRocapsVertexData(controlPoints);
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

    float x = static_cast<float>(launchIndex.x + plp.subPixelOffset.x) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + plp.subPixelOffset.y) / plp.imageSize.y;
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

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.sampleIndex > 0)
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.sampleIndex);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * color;
    plp.colorAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 color = make_float3(0, 0, 0.1f);
    MyPayloadSignature::set(&color);
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
    else if (primType == OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE) {
        float2 ribbonParam = optixGetRibbonParameters();
        sn = optixGetRibbonNormal(ribbonParam);
    }
    else {
        float3 rayDir = optixGetWorldRayDirection();
        float3 hp = optixGetWorldRayOrigin() + optixGetRayTmax() * rayDir;
        float3 hpInObj = optixTransformPointFromWorldToObjectSpace(hp);
        float curveParam = optixGetCurveParameter();
        if (plp.enableRocapsRefinement) {
            // Enable this tuning only when normal vectors exhibit artifacts for rocaps curves
            // and the artifacts cannot be accepted.
            if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS ||
                primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS ||
                primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS ||
                primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS) {
                float3 rayDirInObj = optixTransformVectorFromWorldToObjectSpace(rayDir);
                curveParam = curve::tuneParameterForRocaps(rayDirInObj, hpInObj, primType, curveParam);
            }
        }

        uint32_t primIndex = optixGetPrimitiveIndex();
        if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE_ROCAPS>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE_ROCAPS>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM_ROCAPS>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER>(
                geom, primIndex, curveParam, hpInObj);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS) {
            sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER_ROCAPS>(
                geom, primIndex, curveParam, hpInObj);
        }
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    MyPayloadSignature::set(&color);
}
