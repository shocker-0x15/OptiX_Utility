#pragma once

#include "callable_program_shared.h"

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



struct ReferenceFrame {
    float3 tangent;
    float3 bitangent;
    float3 normal;
    
    CUDA_DEVICE_FUNCTION CUDA_INLINE ReferenceFrame(const float3 n) {
        normal = n;
        float sign = n.z >= 0 ? 1 : -1;
        float a = -1 / (sign + n.z);
        float b = n.x * n.y * a;
        tangent = make_float3(1 + sign * n.x * n.x * a, sign * b, -sign * n.x);
        bitangent = make_float3(b, sign + n.y * n.y * a, -n.y);
    }

    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 toLocal(const float3 dir) const {
        return make_float3(
            dot(tangent, dir),
            dot(bitangent, dir),
            dot(normal, dir));
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 fromLocal(const float3 dir) const {
        return make_float3(
            dot(make_float3(tangent.x, bitangent.x, normal.x), dir),
            dot(make_float3(tangent.y, bitangent.y, normal.y), dir),
            dot(make_float3(tangent.z, bitangent.z, normal.z), dir));
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
    while (true) {
        SearchRayPayloadSignature::trace(
            plp.travHandle, origin, direction,
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            rng, payloadPtr);
        if (payload.terminate || payload.pathLength >= 10)
            break;

        origin = payload.origin;
        direction = payload.direction;
        ++payload.pathLength;
    }

    plp.rngBuffer[launchIndex] = rng;

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.numAccumFrames > 0)
        prevColorResult = getXYZ(plp.accumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.numAccumFrames);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * payload.contribution;
    plp.accumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    SearchRayPayload* payload;
    SearchRayPayloadSignature::get(nullptr, &payload);
    payload->contribution += payload->alpha * make_float3(0.01f, 0.01f, 0.01f);
    payload->terminate = true;
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(shading)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;
    // JP: このサンプルではインスタンスIDフィールドをマテリアルバッファー中のオフセットとして使用している。
    // EN: This sample uses the instance ID field as an offset in the material buffer.
    const uint32_t matIdxOffset = optixGetInstanceId();
    const MaterialData &mat = plp.materialBuffer[geomInst.matIndex + matIdxOffset];

    PCG32RNG rng;
    SearchRayPayload* payload;
    SearchRayPayloadSignature::get(&rng, &payload);

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

    // JP: Callable Programを呼び出し、マテリアルとテクスチャー座標からBSDFデータを構築する。
    // EN: Call a callable program to build BSDF data from the material and the texture coordinates.
    BSDFData bsdfData;
    mat.setUpBSDF(mat, texCoord, &bsdfData);

    const float3 LightRadiance = make_float3(30, 30, 30);
    // Hard-coded directly visible light
    if (mat.isEmitter && isFrontFace && (payload->pathLength == 1 || payload->deltaSampled))
        payload->contribution += payload->alpha * LightRadiance;

    ReferenceFrame frame(sn);
    float3 vOutLocal = frame.toLocal(vOut);

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
        VisibilityRayPayloadSignature::trace(
            plp.travHandle, p, shadowRayDir,
            0.001f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);

        float3 shadowRayLocalDir = frame.toLocal(shadowRayDir);

        float cosSP = shadowRayLocalDir.z;
        float G = visibility * std::fabs(cosSP) * std::fabs(cosLight) / dist2;
        // JP: Callable Programを呼び出し、BSDFの値を評価する。
        // EN: Call a callable program to evaluate a BSDF value.
        float3 fs = mat.evaluateF(bsdfData, vOutLocal, shadowRayLocalDir);
        float3 contribution = payload->alpha * fs * G * Le / areaPDF;
        payload->contribution += contribution;
    }

    // Sampling incoming direction.
    float3 vInLocal;
    float dirProbDens;
    bool deltaSampled;
    float uDir[2] = { rng.getFloat0cTo1o(), rng.getFloat0cTo1o() };
    // JP: Callable Programを呼び出し、BSDFをサンプルする。
    // EN: Call a callable program to sample the BSDF.
    float3 fs = mat.sampleF(bsdfData, vOutLocal, uDir, &vInLocal, &dirProbDens, &deltaSampled);
    if (dirProbDens > 0.0f) {
        payload->alpha = payload->alpha * (fs * std::fabs(vInLocal.z) / dirProbDens);
        payload->origin = p = p + sn * (vInLocal.z > 0 ? 0.001f : -0.001f);
        payload->direction = frame.fromLocal(vInLocal);
        payload->terminate = false;
        payload->deltaSampled = deltaSampled;
    }
    else {
        payload->terminate = true;
    }

    SearchRayPayloadSignature::set(&rng, nullptr);
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);

    optixTerminateRay();
}
