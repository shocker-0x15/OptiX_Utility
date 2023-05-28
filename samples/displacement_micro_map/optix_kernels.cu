#pragma once

#include "displacement_micro_map_shared.h"
#include <optix_micromap.h>

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



CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + plp.subPixelOffset.x) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + plp.subPixelOffset.y) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    float3 color;
    HitPointFlags hpFlags = {};
    PrimaryRayPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        color, hpFlags);
    if (hpFlags.nearBaseTriEdge && plp.drawBaseEdges)
        color *= 0.01f;

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.sampleIndex > 0)
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.sampleIndex);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * color;
    plp.colorAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    float3 contribution = plp.envRadiance;
    PrimaryRayPayloadSignature::set(&contribution, nullptr);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(shading)() {
    auto sbtr = HitGroupSBTRecordData::get();
    const GeometryInstanceData &geomInst = sbtr.geomInstData;

    HitPointFlags hpFlags = {};

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
        if (b0 < 0.01f || b1 < 0.01f || b2 < 0.01f) {
            hpFlags.nearBaseTriEdge = true;
            PrimaryRayPayloadSignature::set(nullptr, &hpFlags);
        }

        if (plp.visualizationMode == VisualizationMode_Barycentric) {
            float3 result = make_float3(b0, b1, b2);
            PrimaryRayPayloadSignature::set(&result, nullptr);
            return;
        }

        texCoord = b0 * v0.texCoord + b1 * v1.texCoord + b2 * v2.texCoord;

        if (optixIsTriangleHit()) { // Vanilla Triangle
            p = b0 * v0.position + b1 * v1.position + b2 * v2.position;
            sn = b0 * v0.normal + b1 * v1.normal + b2 * v2.normal;

            if (plp.visualizationMode == VisualizationMode_MicroBarycentric) {
                float3 result = make_float3(b0, b1, b2);
                PrimaryRayPayloadSignature::set(&result, nullptr);
                return;
            }
        }
        else /*if (optixIsDisplacedMicromeshTriangleHit())*/ { // Displaced Micro-Mesh Triangle
            float3 microTriPositions[3];
            optixGetMicroTriangleVertexData(microTriPositions);
            float2 microTriBcCoords[3];
            optixGetMicroTriangleBarycentricsData(microTriBcCoords);

            float2 bcInMicroTri = optixBaseBarycentricsToMicroBarycentrics(make_float2(b1, b2), microTriBcCoords);
            float mb1 = bcInMicroTri.x;
            float mb2 = bcInMicroTri.y;
            float mb0 = 1 - (mb1 + mb2);

            if (plp.visualizationMode == VisualizationMode_MicroBarycentric) {
                float3 result = make_float3(mb0, mb1, mb2);
                PrimaryRayPayloadSignature::set(&result, nullptr);
                return;
            }

            p = mb0 * microTriPositions[0] + mb1 * microTriPositions[1] + mb2 * microTriPositions[2];
            sn = cross(microTriPositions[1] - microTriPositions[0],
                       microTriPositions[2] - microTriPositions[0]);
        }

        p = optixTransformPointFromObjectToWorldSpace(p);
        sn = normalize(optixTransformNormalFromObjectToWorldSpace(sn));

        if (plp.visualizationMode == VisualizationMode_Normal) {
            float3 result = 0.5f * sn + make_float3(0.5f);
            PrimaryRayPayloadSignature::set(&result, nullptr);
            return;
        }
        else if (plp.visualizationMode == VisualizationMode_SubdivLevel) {
            OptixDisplacementMicromapDesc dmmDesc;
            if (geomInst.dmmDescBuffer) {
                uint32_t dmmIndex = hp.primIndex;
                if (geomInst.dmmIndexSize) {
                    if (geomInst.dmmIndexSize == 2)
                        dmmIndex = reinterpret_cast<uint16_t*>(geomInst.dmmIndexBuffer)[hp.primIndex];
                    else if (geomInst.dmmIndexSize == 4)
                        dmmIndex = reinterpret_cast<uint32_t*>(geomInst.dmmIndexBuffer)[hp.primIndex];
                }
                dmmDesc = geomInst.dmmDescBuffer[dmmIndex];
            }
            float3 result = make_float3(dmmDesc.subdivisionLevel / 5.0f);
            PrimaryRayPayloadSignature::set(&result, nullptr);
            return;
        }
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
    VisibilityRayPayloadSignature::trace(
        plp.travHandle, p, shadowRayDir,
        0.0f, 1e+10f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Visibility, NumRayTypes, RayType_Visibility,
        visibility);

    float cosSP = dot(sn, shadowRayDir);
    float G = visibility * std::fabs(cosSP);
    float3 fs = cosSP > 0 ? albedo / Pi : make_float3(0, 0, 0);
    result += fs * G * plp.lightRadiance;

    PrimaryRayPayloadSignature::set(&result, nullptr);
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
    optixTerminateRay();
}
