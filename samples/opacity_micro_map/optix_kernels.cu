#pragma once

#include "opacity_micro_map_shared.h"

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
    GeometryInstanceData geomInstData;

    CUDA_DEVICE_FUNCTION CUDA_INLINE static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_FUNCTION CUDA_INLINE float3 calcFalseColor(
    int32_t value, int32_t minValue, int32_t maxValue) {
    float t = static_cast<float>(value - minValue) / (maxValue - minValue);
    t = min(max(t, 0.0f), 1.0f);
    constexpr float3 R = { 1.0f, 0.0f, 0.0f };
    constexpr float3 G = { 0.0f, 1.0f, 0.0f };
    constexpr float3 B = { 0.0f, 0.0f, 1.0f };
    float3 ret;
    if (t < 0.5f) {
        t = (t - 0.0f) / 0.5f;
        ret = B * (1 - t) + G * t;
    }
    else {
        t = (t - 0.5f) / 0.5f;
        ret = G * (1 - t) + R * t;
    }
    return ret;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float fetchAlpha(
    const GeometryInstanceData &geomInst, const HitPointParameter &hp) {
    if (!geomInst.texture)
        return 1.0f;
    const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
    const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
    const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
    const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
    float b1 = hp.b1;
    float b2 = hp.b2;
    float b0 = 1 - (b1 + b2);
    float2 texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;
    float4 texValue = tex2DLod<float4>(geomInst.texture, texCoord.x, texCoord.y, 0.0f);
    return texValue.w;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float isTransparent(float alpha) {
    return alpha < 0.5f;
}

CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    uint32_t superSampleSize = plp.superSampleSizeMinus1 + 1;
    float dx = (static_cast<float>(plp.sampleIndex % superSampleSize) + 0.5f) / superSampleSize;
    float dy = (static_cast<float>(plp.sampleIndex / superSampleSize) + 0.5f) / superSampleSize;
    float x = static_cast<float>(launchIndex.x + dx) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + dy) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    float3 color;
    uint32_t numAnyHitCalls = 0;
    PrimaryRayPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color, numAnyHitCalls);

    if (plp.visualizationMode != VisualizationMode_Final) {
        const uint32_t maxValue = plp.visualizationMode == VisualizationMode_NumPrimaryAnyHits ? 20 : 10;
        float3 falseColor = calcFalseColor(numAnyHitCalls, 0, maxValue);
        color = 0.2f * color + 0.8f * falseColor;
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.sampleIndex > 0)
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.sampleIndex);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * color;
    plp.colorAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(primary)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    uint32_t numAnyHitCalls;
    PrimaryRayPayloadSignature::get(nullptr, &numAnyHitCalls);
    ++numAnyHitCalls;
    PrimaryRayPayloadSignature::set(nullptr, &numAnyHitCalls);

    float alpha = fetchAlpha(geomInst, HitPointParameter::get());
    if (isTransparent(alpha))
        optixIgnoreIntersection();
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 contribution = plp.envRadiance;
    uint32_t numAnyHitCalls = 0;
    PrimaryRayPayloadSignature::set(
        &contribution,
        plp.visualizationMode == VisualizationMode_NumShadowAnyHits ? &numAnyHitCalls : nullptr);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(shading)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    auto hp = HitPointParameter::get();
    float3 p;
    float3 sn;
    float2 texCoord;
    {
        const Triangle &tri = geomInst.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geomInst.vertexBuffer[tri.index0];
        const Vertex &v1 = geomInst.vertexBuffer[tri.index1];
        const Vertex &v2 = geomInst.vertexBuffer[tri.index2];
        float b1 = hp.b1;
        float b2 = hp.b2;
        float b0 = 1 - (b1 + b2);
        p = b0 * v0.position + b1 * v1.position + b2 * v2.position;
        sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;
        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        p = optixTransformPointFromObjectToWorldSpace(p);
        sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));
    }

    float3 vOut = -optixGetWorldRayDirection();
    bool isFrontFace = dot(vOut, sn) > 0;
    if (!isFrontFace)
        sn = -sn;
    p = p + sn * 0.001f;

    float3 albedo;
    if (geomInst.texture)
        albedo = getXYZ(tex2DLod<float4>(geomInst.texture, texCoord.x, texCoord.y, 0.0f));
    else
        albedo = geomInst.albedo;

    float3 result = plp.envRadiance * albedo;

    float3 shadowRayDir = plp.lightDirection;

    float visibility = 1.0f;
    uint32_t numAnyHitCalls = 0;
    VisibilityRayPayloadSignature::trace(
        plp.travHandle, p, shadowRayDir,
        0.0f, 1e+10f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Visibility, NumRayTypes, RayType_Visibility,
        visibility, numAnyHitCalls);

    float cosSP = dot(sn, shadowRayDir);
    float G = visibility * std::fabs(cosSP);
    float3 fs = cosSP > 0 ? albedo / Pi : make_float3(0, 0, 0);
    result += fs * G * plp.lightRadiance;

    PrimaryRayPayloadSignature::set(
        &result,
        plp.visualizationMode == VisualizationMode_NumShadowAnyHits ? &numAnyHitCalls : nullptr);
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    uint32_t numAnyHitCalls;
    VisibilityRayPayloadSignature::get(nullptr, &numAnyHitCalls);
    ++numAnyHitCalls;
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility, &numAnyHitCalls);
    optixTerminateRay();
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibilityWithAlpha)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    uint32_t numAnyHitCalls;
    VisibilityRayPayloadSignature::get(nullptr, &numAnyHitCalls);
    ++numAnyHitCalls;
    VisibilityRayPayloadSignature::set(nullptr, &numAnyHitCalls);

    float alpha = fetchAlpha(geomInst, HitPointParameter::get());
    if (isTransparent(alpha)) {
        optixIgnoreIntersection();
    }
    else {
        float visibility = 0.0f;
        VisibilityRayPayloadSignature::set(&visibility, nullptr);
        optixTerminateRay();
    }
}

/*
JP: アルファテクスチャーの貼られた物体におけるシャドウレイでは通常Any-Hit Programを使用するが、
    Opacity Micro-Mapを適用すると遮蔽があると判定されるFully-OpaqueなMicro Triangleに関しても
    Any-Hitが呼ばれなくなるため、Closest-Hitでも可視性をゼロにセットする処理が必要になる。
EN: Shadow rays for alpha-textured object usually use an any-hit program.
    However when using opacity micro-map, the any-hit program will never be called even for fully-opaque
    micro triangle where it should find occulusion. Therefore, a closest-hit program to set
    the visibility to zero is required.
*/
CUDA_DEVICE_KERNEL void RT_CH_NAME(visibilityWithAlpha)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility, nullptr);
}
