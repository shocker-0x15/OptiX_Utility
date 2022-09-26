#pragma once

#include "deformation_blur_shared.h"
#include "../common/curve_evaluator.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    union {
        float b1;
        float secondDistance;
    };
    float b2;
    int32_t primIndex;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get() {
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
        else if (primType == OPTIX_PRIMITIVE_TYPE_SPHERE) {
            // When a ray hits a sphere twice, the attribute 0 contains the second distance.
            uint32_t attr0 = optixGetAttribute_0();
            ret.secondDistance = __uint_as_float(attr0);
        }
        else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
            // JP: Intersection Programでアトリビュートシグネチャー型を通じて設定したアトリビュート変数は
            //     対応する AttributeSignature<...>::reportIntersection() を通じて取得できる。
            // EN: Attribute variables set via an attribute signature type in the intersection program
            //     can be obtained using the corresponding AttributeSignature<...>::reportIntersection().
            PartialSphereAttributeSignature::get(&ret.b1, &ret.b2);
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

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



template <OptixPrimitiveType curveType>
CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcCurveSurfaceNormal(
    const GeometryData &geom, const HitPointParameter &hpParam, float rayTime, const float3 &hp) {
    constexpr uint32_t numControlPoints = curve::getNumControlPoints<curveType>();
    float4 controlPoints[numControlPoints];
    if constexpr (useEmbeddedVertexData) {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        uint32_t sbtGasIndex = optixGetSbtGASIndex();
        if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
            optixGetLinearCurveVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
            optixGetQuadraticBSplineVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE)
            optixGetCubicBSplineVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
        else if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM)
            optixGetCatmullRomVertexData(gasHandle, hpParam.primIndex, sbtGasIndex, rayTime, controlPoints);
    }
    else {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        float timeBegin = optixGetGASMotionTimeBegin(gasHandle);
        float timeEnd = optixGetGASMotionTimeEnd(gasHandle);
        float normTime = std::fmin(std::fmax((optixGetRayTime() - timeBegin) / (timeEnd - timeBegin), 0.0f), 1.0f);

        uint32_t baseIndex = geom.segmentIndexBuffer[hpParam.primIndex];
        float stepF = (geom.numMotionSteps - 1) * normTime;
        uint32_t step = static_cast<uint32_t>(stepF);
        float p = stepF - step;
#pragma unroll
        for (int i = 0; i < numControlPoints; ++i) {
            const CurveVertex &vA = geom.curveVertexBuffers[step][baseIndex + i];
            const CurveVertex &vB = geom.curveVertexBuffers[step + 1][baseIndex + i];
            controlPoints[i] = (1 - p) * make_float4(vA.position, vA.width) + p * make_float4(vB.position, vB.width);
        }
    }

    curve::Evaluator<curveType> ce(controlPoints);
    float3 sn = ce.calcNormal(hpParam.b1, hp);

    return sn;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcSphereSurfaceNormal(
    const GeometryData &geom, uint32_t primIndex, float rayTime, const float3 &hp) {
    float3 center;
    if constexpr (useEmbeddedVertexData) {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        uint32_t sbtGasIndex = optixGetSbtGASIndex();
        float4 centerAndRadius;
        optixGetSphereData(gasHandle, primIndex, sbtGasIndex, rayTime, &centerAndRadius);
        center = make_float3(centerAndRadius);
    }
    else {
        OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
        float timeBegin = optixGetGASMotionTimeBegin(gasHandle);
        float timeEnd = optixGetGASMotionTimeEnd(gasHandle);
        float normTime = std::fmin(std::fmax((optixGetRayTime() - timeBegin) / (timeEnd - timeBegin), 0.0f), 1.0f);

        float stepF = (geom.numMotionSteps - 1) * normTime;
        uint32_t step = static_cast<uint32_t>(stepF);
        float p = stepF - step;
        const Sphere &sphA = geom.sphereBuffers[step][primIndex];
        const Sphere &sphB = geom.sphereBuffers[step + 1][primIndex];
        center = (1 - p) * sphA.center + p * sphB.center;
    }

    float3 sn = normalize(hp - center);

    return sn;
}



CUDA_DEVICE_KERNEL void RT_IS_NAME(partialSphere)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    uint32_t primIndex = optixGetPrimitiveIndex();
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();

    float3 nDir = normalize(rayDir);

    const PartialSphereParameter &param0 = geom.partialSphereParamBuffers[0][primIndex];
    const PartialSphereParameter &param1 = geom.partialSphereParamBuffers[1][primIndex];

    OptixTraversableHandle gasHandle = optixGetGASTraversableHandle();
    float timeBegin = optixGetGASMotionTimeBegin(gasHandle);
    float timeEnd = optixGetGASMotionTimeEnd(gasHandle);
    float time = std::fmin(std::fmax((optixGetRayTime() - timeBegin) / (timeEnd - timeBegin), 0.0f), 1.0f);
    float3 center = (1 - time) * param0.center + time * param1.center;
    float radius = (1 - time) * param0.radius + time * param1.radius;
    float minTheta = (1 - time) * param0.minTheta + time * param1.minTheta;
    float maxTheta = (1 - time) * param0.maxTheta + time * param1.maxTheta;
    float minPhi = (1 - time) * param0.minPhi + time * param1.minPhi;
    float maxPhi = (1 - time) * param0.maxPhi + time * param1.maxPhi;

    float3 co = rayOrg - center;
    float b = dot(nDir, co);

    float D = b * b - (sqLength(co) - radius * radius);
    if (D < 0)
        return;

    float sqrtD = std::sqrt(D);
    float t0 = -b - sqrtD;
    float t1 = -b + sqrtD;

    float3 np0 = normalize(co + t0 * nDir);
    float theta0 = std::acos(std::fmin(std::fmax(np0.z, -1.0f), 1.0f));
    float phi0 = std::fmod(std::atan2(np0.y, np0.x) + 2 * Pi, 2 * Pi);
    if (theta0 < minTheta || theta0 >= maxTheta ||
        phi0 < minPhi || phi0 >= maxPhi)
        t0 = -1; // discard

    float3 np1 = normalize(co + t1 * nDir);
    float theta1 = std::acos(std::fmin(std::fmax(np1.z, -1.0f), 1.0f));
    float phi1 = std::fmod(std::atan2(np1.y, np1.x) + 2 * Pi, 2 * Pi);
    if (theta1 < minTheta || theta1 >= maxTheta ||
        phi1 < minPhi || phi1 >= maxPhi)
        t1 = -1; // discard

    bool isFront = t0 >= 0;
    float t;
    float theta, phi;
    if (isFront) {
        t = t0;
        theta = theta0;
        phi = phi0;
    }
    else {
        t = t1;
        theta = theta1;
        phi = phi1;
    }
    if (t < 0)
        return;

    // JP: アトリビュートシグネチャー型を通じて、アトリビュートとともに交叉が有効であることを報告する。
    // EN: report that the intersection is valid with attributes via the attribute signature type.
    PartialSphereAttributeSignature::reportIntersection(t, isFront ? 0 : 1, theta, phi);
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
    MyPayloadSignature::trace(
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
    MyPayloadSignature::set(&color);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    auto hp = HitPointParameter::get();

    float3 pos = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
    pos = optixTransformPointFromWorldToObjectSpace(pos);
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
        sn = calcCurveSurfaceNormal<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>(
            geom, hp, rayTime, pos);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_SPHERE) {
        sn = calcSphereSurfaceNormal(geom, hp.primIndex, rayTime, pos);
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
    MyPayloadSignature::set(&color);
}
