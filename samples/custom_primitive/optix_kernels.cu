#pragma once

#include "custom_primitive_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    float b1, b2;
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
        else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
            // JP: Intersection Programでアトリビュートシグネチャー型を通じて設定したアトリビュート変数は
            //     対応する AttributeSignature<...>::reportIntersection() を通じて取得できる。
            // EN: Attribute variables set via an attribute signature type in the intersection program
            //     can be obtained using the corresponding AttributeSignature<...>::reportIntersection().
            PartialSphereAttributeSignature::get(&ret.b1, &ret.b2);
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



// JP: レイとカスタムプリミティブとの衝突判定はIntersection Programで記述する。
// EN: Intersection program is used to describe the intersection between a ray vs a custom primitive.
CUDA_DEVICE_KERNEL void RT_IS_NAME(partialSphere)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    uint32_t primIndex = optixGetPrimitiveIndex();
    const PartialSphereParameter &param = geom.paramBuffer[primIndex];
    const float3 rayOrg = optixGetObjectRayOrigin();
    const float3 rayDir = optixGetObjectRayDirection();

    float3 nDir = normalize(rayDir);

    float3 co = rayOrg - param.center;
    float b = dot(nDir, co);

    float D = b * b - (sqLength(co) - param.radius * param.radius);
    if (D < 0)
        return;

    float sqrtD = std::sqrt(D);
    float t0 = -b - sqrtD;
    float t1 = -b + sqrtD;

    float3 np0 = normalize(co + t0 * nDir);
    float theta0 = std::acos(std::fmin(std::fmax(np0.z, -1.0f), 1.0f));
    float phi0 = std::fmod(std::atan2(np0.y, np0.x) + 2 * Pi, 2 * Pi);
    if (theta0 < param.minTheta || theta0 >= param.maxTheta ||
        phi0 < param.minPhi || phi0 >= param.maxPhi)
        t0 = -1; // discard

    float3 np1 = normalize(co + t1 * nDir);
    float theta1 = std::acos(std::fmin(std::fmax(np1.z, -1.0f), 1.0f));
    float phi1 = std::fmod(std::atan2(np1.y, np1.x) + 2 * Pi, 2 * Pi);
    if (theta1 < param.minTheta || theta1 >= param.maxTheta ||
        phi1 < param.minPhi || phi1 >= param.maxPhi)
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

    float3 sn;
    OptixPrimitiveType primType = optixGetPrimitiveType();
    if (primType == OPTIX_PRIMITIVE_TYPE_TRIANGLE) {
        const Triangle &triangle = geom.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geom.vertexBuffer[triangle.index0];
        const Vertex &v1 = geom.vertexBuffer[triangle.index1];
        const Vertex &v2 = geom.vertexBuffer[triangle.index2];

        float b0 = 1 - (hp.b1 + hp.b2);
        sn = b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;
        sn = normalize(sn);
    }
    else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
        //const PartialSphereParameter &param = geom.paramBuffer[hp.primIndex];
        float theta = hp.b1;
        float phi = hp.b2;
        float sinTheta = std::sin(theta);
        sn = make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta));
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    // EN: Visualize the normal.
    float3 color = 0.5f * sn + make_float3(0.5f);
    MyPayloadSignature::set(&color);
}
