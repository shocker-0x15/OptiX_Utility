#pragma once

#include "custom_primitive_shared.h"

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
        else if (primType == OPTIX_PRIMITIVE_TYPE_CUSTOM) {
            // JP: Intersection Programで設定したアトリビュート変数は optixu::getAttributes() で取得できる。
            // EN: Attribute variables set in the intersection program can be obtained using optixu::getAttributes().
            optixu::getAttributes<SphereAttributeSignature>(&ret.b1, &ret.b2);
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



// JP: レイとカスタムプリミティブとの衝突判定はIntersection Programで記述する。
// EN: Intersection program is used to describe the intersection between a ray vs a custom primitive.
CUDA_DEVICE_KERNEL void RT_IS_NAME(intersectSphere)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryData &geom = sbtr.geomData;
    uint32_t primIndex = optixGetPrimitiveIndex();
    const SphereParameter &param = geom.paramBuffer[primIndex];
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
    optixu::setPayloads<PayloadSignature>(&color);
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
        //const SphereParameter &param = geom.paramBuffer[hp.primIndex];
        float theta = hp.b1;
        float phi = hp.b2;
        float sinTheta = std::sin(theta);
        sn = make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta));
    }

    sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

    // JP: 法線を可視化。
    //     このサンプルでは単一のGASしか使っていないためオブジェクト空間からワールド空間への変換は無い。
    // EN: Visualize the normal.
    //     There is no object to world space transform since this sample uses only a single GAS.
    float3 color = 0.5f * sn + make_float3(0.5f);
    optixu::setPayloads<PayloadSignature>(&color);
}
