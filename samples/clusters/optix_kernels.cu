#include "clusters_shared.h"

using namespace Shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters plp;



struct HitPointParameter {
    float b1, b2;
    uint32_t primIndex;

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



CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    float x = static_cast<float>(launchIndex.x + plp.subPixelOffset.x) / plp.imageSize.x;
    float y = static_cast<float>(launchIndex.y + plp.subPixelOffset.y) / plp.imageSize.y;
    float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    float vw = plp.camera.aspect * vh;

    float3 origin = plp.camera.position;
    float3 direction = normalize(plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    uint32_t instIndex;
    uint32_t clusterId;
    uint32_t primIdx;
    float2 barycentrics;
    float3 shadingNormal;
    float3 geomNormal;
    MyPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        instIndex, clusterId, primIdx, barycentrics, shadingNormal, geomNormal);

    if (launchIndex == plp.mousePosition) {
        plp.pickInfo->instanceIndex = instIndex;
        plp.pickInfo->clusterId = clusterId;
        plp.pickInfo->primitiveIndex = primIdx;
        plp.pickInfo->barycentrics = barycentrics;
        if (clusterId != OPTIX_CLUSTER_ID_INVALID) {
            plp.pickInfo->cluster = plp.clusters[clusterId];
        }
        else {
            plp.pickInfo->cluster.level = 0;
            plp.pickInfo->cluster.vertexCount = 0;
            plp.pickInfo->cluster.triangleCount = 0;
        }
    }

    bool hit = geomNormal != make_float3(0, 0, 0);
    float3 color = make_float3(0.0f, 0.0f, 0.1f);
    if (hit) {
        if (plp.visMode == VisualizationMode_ShadingNormal) {
            color = 0.5f * shadingNormal + make_float3(0.5f);
        }
        else if (plp.visMode == VisualizationMode_GeometricNormal) {
            color = 0.5f * geomNormal + make_float3(0.5f);
        }
        else if (plp.visMode == VisualizationMode_Cluster) {
            if (clusterId == OPTIX_CLUSTER_ID_INVALID) {
                color = make_float3(0.0f, 0.0f, 0.0f);
            }
            else {
                const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
                const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
                color = HSVtoRGB(
                    std::fmod((GoldenAngle * clusterId) / (2 * pi_v<float>), 1.0f),
                    1.0f, 1.0f);
            }
        }
        else if (plp.visMode == VisualizationMode_Level) {
            if (clusterId == OPTIX_CLUSTER_ID_INVALID) {
                color = make_float3(0.0f, 0.0f, 0.0f);
            }
            else {
                color = calcFalseColor(plp.clusters[clusterId].level, 0, 10);
            }
        }
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.sampleIndex > 0)
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
    float curWeight = 1.0f / (1 + plp.sampleIndex);
    float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * color;
    plp.colorAccumBuffer.write(launchIndex, make_float4(colorResult, 1.0f));
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(miss)() {
    constexpr uint32_t instIndex = 0xFFFF'FFFF;
    constexpr uint32_t clusterId = OPTIX_CLUSTER_ID_INVALID;
    constexpr uint32_t primIdx = 0xFFFF'FFFF;
    constexpr float2 barycentrics = { 0.0f, 0.0f };
    float3 shadingNormal = make_float3(0, 0, 0);
    float3 geomNormal = make_float3(0, 0, 0);
    MyPayloadSignature::set(&instIndex, &clusterId, &primIdx, &barycentrics, &shadingNormal, &geomNormal);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(closesthit)() {
    const uint32_t clusterId = optixGetClusterId();
    const auto hp = HitPointParameter::get();

    const Cluster &cluster = plp.clusters[clusterId];
    const LocalTriangle &tri = plp.trianglePool[cluster.triPoolStartIndex + hp.primIndex];
    const Vertex (&vs)[] = {
        plp.vertexPool[cluster.vertPoolStartIndex + tri.index0],
        plp.vertexPool[cluster.vertPoolStartIndex + tri.index1],
        plp.vertexPool[cluster.vertPoolStartIndex + tri.index2],
    };

    float bcB = hp.b1;
    float bcC = hp.b2;
    float bcA = 1.0f - bcB - bcC;
    float3 shadingNormal =
        bcA * vs[0].normal + bcB * vs[1].normal + bcC * vs[2].normal;
    shadingNormal = normalize(optixTransformNormalFromObjectToWorldSpace(shadingNormal));

    //float3 positions[3];
    //optixGetTriangleVertexData(positions);
    //float3 geomNormal = normalize(cross(
    //    positions[1] - positions[0],
    //    positions[2] - positions[0]));
    float3 geomNormal = normalize(cross(
        vs[1].position - vs[0].position,
        vs[2].position - vs[0].position));
    geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));

    uint32_t instIndex = optixGetInstanceIndex();
    float2 barycentrics = make_float2(hp.b1, hp.b2);
    MyPayloadSignature::set(&instIndex, &clusterId, &hp.primIndex, &barycentrics, &shadingNormal, &geomNormal);
}
