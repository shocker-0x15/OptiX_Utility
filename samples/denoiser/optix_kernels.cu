#pragma once

#include "denoiser_shared.h"

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

struct HitGroupSBTRecordData {
    GeometryData geomData;
    MaterialData matData;

    CUDA_DEVICE_FUNCTION static const HitGroupSBTRecordData &get() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
};



CUDA_DEVICE_KERNEL void RT_RG_NAME(pathTracing)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    PCG32RNG rng = plp.rngBuffer[launchIndex];

    float x = static_cast<float>(launchIndex.x + rng.getFloat0cTo1o()) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + rng.getFloat0cTo1o()) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    SearchRayPayload payload;
    payload.alpha = make_float3(1.0f, 1.0f, 1.0f);
    payload.contribution = make_float3(0.0f, 0.0f, 0.0f);
    payload.pathLength = 1;
    payload.terminate = false;
    SearchRayPayload* payloadPtr = &payload;
    float3 firstHitAlbedo = make_float3(0.0f, 0.0f, 0.0f);
    float3 firstHitNormal = make_float3(0.0f, 0.0f, 0.0f);
    float3* firstHitAlbedoPtr = &firstHitAlbedo;
    float3* firstHitNormalPtr = &firstHitNormal;
    while (true) {
        optixu::trace<SearchRayPayloadSignature>(
            plp.travHandle, origin, direction,
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            rng, payloadPtr, firstHitAlbedoPtr, firstHitNormalPtr);
        if (payload.terminate || payload.pathLength >= 10)
            break;

        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    plp.rngBuffer[launchIndex] = rng;

    // Normal input to the denoiser should be in camera space (right handed, looking down the negative Z-axis).
    firstHitNormal = transpose(plp.camera.orientation) * firstHitNormal;
    firstHitNormal.x *= -1;

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevAlbedoResult = make_float3(0.0f, 0.0f, 0.0f);
    float3 prevNormalResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 0) {
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
        prevAlbedoResult = getXYZ(plp.albedoAccumBuffer.read(launchIndex));
        prevNormalResult = getXYZ(plp.normalAccumBuffer.read(launchIndex));
    }
    float curWeight = 1.0f / (1 + plp.numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * payload.contribution;
    float3 albedoResult = (1 - curWeight) * prevAlbedoResult + curWeight * firstHitAlbedo;
    float3 normalResult = (1 - curWeight) * prevNormalResult + curWeight * firstHitNormal;
    plp.colorAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
    plp.albedoAccumBuffer.write(launchIndex, make_float4(albedoResult, 1.0f));
    plp.normalAccumBuffer.write(launchIndex, make_float4(normalResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    SearchRayPayload* payload;
    float3* albedo;
    float3* normal;
    optixu::getPayloads<SearchRayPayloadSignature>(nullptr, &payload, &albedo, &normal);
    payload->contribution += payload->alpha * make_float3(0.01f, 0.01f, 0.01f);
    payload->terminate = true;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(shading)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const MaterialData &mat = sbtr.matData;
    const GeometryData &geom = sbtr.geomData;

    PCG32RNG rng;
    SearchRayPayload* payload;
    float3* firstHitAlbedo;
    float3* firstHitNormal;
    optixu::getPayloads<SearchRayPayloadSignature>(&rng, &payload, &firstHitAlbedo, &firstHitNormal);

    auto hp = HitPointParameter::get();
    float3 p;
    float3 sn;
    float2 texCoord;
    {
        const Triangle &tri = geom.triangleBuffer[hp.primIndex];
        const Vertex &v0 = geom.vertexBuffer[tri.index0];
        const Vertex &v1 = geom.vertexBuffer[tri.index1];
        const Vertex &v2 = geom.vertexBuffer[tri.index2];
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
    if (mat.texture)
        albedo = getXYZ(tex2DLod<float4>(mat.texture, texCoord.x, texCoord.y, 0.0f));
    else
        albedo = mat.albedo;

    if (payload->pathLength == 1) {
        *firstHitAlbedo = albedo;
        *firstHitNormal = sn;
    }

    const float3 LightRadiance = make_float3(30, 30, 30);
    // Hard-coded directly visible light
    if (mat.isEmitter && isFrontFace && payload->pathLength == 1)
        payload->contribution += payload->alpha * LightRadiance;

    // Next Event Estimation
    {
        // Use hard-coded area light for simplicity.
        float3 lp = make_float3(-0.25f, 0.9f, -0.25f) +
            rng.getFloat0cTo1o() * make_float3(0.5f, 0, 0) +
            rng.getFloat0cTo1o() * make_float3(0, 0, 0.5f);
        float areaPDF = 4.0f;
        float3 lpn = make_float3(0, -1, 0);

        float3 shadowRayDir = lp - p;
        float dist2 = dot(shadowRayDir, shadowRayDir);
        float dist = std::sqrt(dist2);
        shadowRayDir = shadowRayDir / dist;
        float cosLight = dot(lpn, -shadowRayDir);
        float3 Le = cosLight > 0 ? LightRadiance : make_float3(0, 0, 0);

        float visibility = 1.0f;
        optixu::trace<VisibilityRayPayloadSignature>(
            plp.travHandle, p, shadowRayDir,
            0.0f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);

        float cosSP = dot(sn, shadowRayDir);
        float G = visibility * std::fabs(cosSP) * std::fabs(cosLight) / dist2;
        float3 fs = cosSP > 0 ? albedo / Pi : make_float3(0, 0, 0);
        float3 contribution = payload->alpha * fs * G * Le / areaPDF;
        payload->contribution += contribution;
    }

    const auto makeCoordinateSystem = [](const float3 &n, float3* s, float3* t) {
        float sign = n.z >= 0 ? 1 : -1;
        float a = -1 / (sign + n.z);
        float b = n.x * n.y * a;
        *s = make_float3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        *t = make_float3(b, sign + n.y * n.y * a, -n.y);
    };

    float3 s;
    float3 t;
    makeCoordinateSystem(sn, &s, &t);

    // Sampling incoming direction.
    float phi = 2 * Pi * rng.getFloat0cTo1o();
    float theta = std::asin(std::sqrt(rng.getFloat0cTo1o()));
    float sinTheta = std::sin(theta);
    float3 vIn = make_float3(std::cos(phi) * sinTheta, std::sin(phi) * sinTheta, std::cos(theta));
    vIn = make_float3(dot(make_float3(s.x, t.x, sn.x), vIn),
                      dot(make_float3(s.y, t.y, sn.y), vIn),
                      dot(make_float3(s.z, t.z, sn.z), vIn));
    payload->alpha = payload->alpha * albedo;
    payload->origin = p;
    payload->direction = vIn;
    payload->terminate = false;

    optixu::setPayloads<SearchRayPayloadSignature>(&rng, nullptr, nullptr, nullptr);
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    optixu::setPayloads<VisibilityRayPayloadSignature>(&visibility);

    optixTerminateRay();
}
