#pragma once

#include "deformation_blur_shared.h"
#include "../common/curve_evaluator.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    float b1, b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION static HitPointParameter get() {
        HitPointParameter ret;
        OptixPrimitiveType primType = optixGetPrimitiveType();
        // JP: 衝突したプリミティブのタイプを組み込み関数によって取得できる。
        // EN: The primitive type hit can be obtained using the intrinsic function.
        if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
            float2 bc = optixGetTriangleBarycentrics();
            ret.b1 = bc.x;
            ret.b2 = bc.y;
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE) {
            ret.b1 = optixGetCurveParameter();
            ret.b2 = NAN;
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
            // JP: Intersection Programで設定したアトリビュート変数は optixu::getAttributes() で取得できる。
            // EN: Attribute variables set in the intersection program can be obtained using optixu::getAttributes().
            optixu::getAttributes<SphereAttributeSignature>(&ret.b1, &ret.b2);
        }
        else {
            optixuAssert_ShouldNotBeCalled();
        }
        ret.primIndex = optixGetPrimitiveIndex();
        return ret;
    }
};

struct HitGroupSBTRecordData {
    GeometryData geomData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



template <OptixPrimitiveType curveType>
CUDA_DEVICE_FUNCTION float3 calcCurveSurfaceNormal(const GeometryData &geom, const HitPointParameter &hpParam, float rayTime, const float3 &hp) {
    constexpr uint32_t numControlPoints = curve::getNumControlPoints<curveType>();
    float4 controlPoints[numControlPoints];
#if defined(USE_EMBEDDED_DATA)
    OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
    uint32_t sbtGasIndex = optixGetSbtGASIndex();
    if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
        optixGetLinearCurveVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
    else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
        optixGetQuadraticBSplineVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
    else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
        optixGetCubicBSplineVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
#else
    OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
    float timeBegin = optixGetGASMotionTimeBegin(gasHandle);
    float timeEnd = optixGetGASMotionTimeEnd(gasHandle);
    float normTime = std::fmin(std::fmax((optixGetRayTime() - timeBegin) / (timeEnd - timeBegin), 0.0f), 1.0f);

    uint32_t baseIndex = geom.segmentIndexBuffer[hpParam.primIndex];
    float stepF = (geom.numMotionSteps - 1) * normTime;
    uint32_t step = static_cast<uint32_t>(stepF);
    float stepWidth = 1.0f / (geom.numMotionSteps - 1);
    float p = stepF - step;
    for (int i = 0; i < numControlPoints; ++i) {
        const CurveVertex &vA = geom.curveVertexBuffers[step][baseIndex + i];
        const CurveVertex &vB = geom.curveVertexBuffers[step + 1][baseIndex + i];
        controlPoints[i] = (1 - p) * make_float4(vA.position, vA.width) + p * make_float4(vB.position, vB.width);
    }
#endif

    curve::Evaluator<curveType> ce(controlPoints);
    float3 sn = ce.calcNormal(hpParam.b1, hp);

    return sn;
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(intersectSphere)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    uint32_t primIndex = optixGetPrimitiveIndex();
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();

    float3 nDir = normalize(rayDir);

    const SphereParameter &param0 = geom.paramBuffers[0][primIndex];
    const SphereParameter &param1 = geom.paramBuffers[1][primIndex];

    OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
    float timeBegin = optixGetGASMotionTimeBegin(gasHandle);
    float timeEnd = optixGetGASMotionTimeEnd(gasHandle);
    float time = std::fmin(std::fmax((optixGetRayTime() - timeBegin) / (timeEnd - timeBegin), 0.0f), 1.0f);
    float3 center = (1 - time) * param0.center + time * param1.center;
    float radius = (1 - time) * param0.radius + time * param1.radius;

    float3 co = rayOrg - center;
    float b = dot(nDir, co);

    float D = b * b - (sqLength(co) - radius * radius);
    if (D < 0)
        return;

    float sqrtD = std::sqrt(D);
    float t0 = -b - sqrtD;
    float t1 = -b + sqrtD;
    bool isFront = t0 >= 0;
    float t = isFront ? t0 : t1;
    if (t < 0)
        return;

    float3 np = normalize(co + t * nDir);
    float theta = std::acos(std::fmin(std::fmax(np.z, -1.0f), 1.0f));
    float phi = std::fmod(std::atan2(np.y, np.x) + 2 * Pi, 2 * Pi);

    // JP: ペイロードと同様に、対応するreportIntersection()/getAttributes()で
    //     明示的にテンプレート引数を渡すことで型の不一致を検出できるようにすることを推奨する。
    // EN: It is recommended to explicitly pass template arguments to corresponding
    //     reportIntersection()/getAttributes() to detect type mismatch similar to payloads.
    optixu::reportIntersection<SphereAttributeSignature>(t, isFront ? 0 : 1, theta, phi);
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.rngBuffer[launchIndex];

    float x = static_cast<float>(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    float time = plp.timeBegin + (plp.timeEnd - plp.timeBegin) * rng.getFloat0cTo1o();

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
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    auto hp = HitPointParameter::get();

    float rayTime = optixGetRayTime();

    float3 sn;
    OptixPrimitiveType primType = optixGetPrimitiveType();
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geom.vertexBuffers[0][triangle.index0];
        const Vertex &v1 = geom.vertexBuffers[0][triangle.index1];
        const Vertex &v2 = geom.vertexBuffers[0][triangle.index2];

        float b0 = 1 - (hp.b1 + hp.b2);
        sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE) {
        float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
        pos = optixTransformPointFromWorldToObjectSpace(pos);

        sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>(
            geom, hp, rayTime, pos);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
        float theta = hp.b1;
        float phi = hp.b2;
        float sinTheta = std::sin(theta);
        sn = make_float3(std::cos(phi) * sinTheta,
                         std::sin(phi) * sinTheta,
                         std::cos(theta));
    }
    else {
        optixuAssert_ShouldNotBeCalled();
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    optixu::setPayloads<PayloadSignature>(&color);
}
