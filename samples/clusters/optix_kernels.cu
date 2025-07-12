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

    HitInfo hitInfo;
    MyPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        hitInfo);

    if (launchIndex == plp.mousePosition) {
        plp.pickInfo->instanceIndex = hitInfo.instIndex;
        plp.pickInfo->clusterId = hitInfo.clusterId;
        plp.pickInfo->primitiveIndex = hitInfo.primIndex;
        plp.pickInfo->barycentrics = hitInfo.barycentrics;
        if (hitInfo.clusterId != OPTIX_CLUSTER_ID_INVALID) {
            plp.pickInfo->cluster = plp.clusters[hitInfo.clusterId];
        }
        else {
            plp.pickInfo->cluster.level = 0;
            plp.pickInfo->cluster.vertexCount = 0;
            plp.pickInfo->cluster.triangleCount = 0;
        }
    }

    bool hit = hitInfo.geomNormal != make_float3(0, 0, 0);
    float3 color = make_float3(0.0f, 0.0f, 0.1f);
    if (hit) {
        const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
        const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
        if (plp.visMode == VisualizationMode_ShadingNormal) {
            color = 0.5f * hitInfo.shadingNormal + make_float3(0.5f);
        }
        else if (plp.visMode == VisualizationMode_GeometricNormal) {
            color = 0.5f * hitInfo.geomNormal + make_float3(0.5f);
        }
        else if (plp.visMode == VisualizationMode_Cluster) {
            if (hitInfo.clusterId == OPTIX_CLUSTER_ID_INVALID) {
                color = make_float3(0.0f, 0.0f, 0.0f);
            }
            else {
                color = HSVtoRGB(
                    std::fmod((GoldenAngle * hitInfo.clusterId) / (2 * pi_v<float>), 1.0f),
                    1.0f, 1.0f);
            }
        }
        else if (plp.visMode == VisualizationMode_Level) {
            if (hitInfo.clusterId == OPTIX_CLUSTER_ID_INVALID) {
                color = make_float3(0.0f, 0.0f, 0.0f);
            }
            else {
                color = calcFalseColor(plp.clusters[hitInfo.clusterId].level, 0, 10);
            }
        }
        else if (plp.visMode == VisualizationMode_Triangle) {
            color = HSVtoRGB(
                std::fmod((GoldenAngle * hitInfo.primIndex) / (2 * pi_v<float>), 1.0f),
                1.0f, 1.0f);
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
    HitInfo missInfo = {};
    missInfo.instIndex = 0xFFFF'FFFF;
    missInfo.clusterId = OPTIX_CLUSTER_ID_INVALID;
    missInfo.primIndex = 0xFFFF'FFFF;
    missInfo.barycentrics = { 0.0f, 0.0f };
    missInfo.shadingNormal = make_float3(0, 0, 0);
    missInfo.geomNormal = make_float3(0, 0, 0);
    MyPayloadSignature::set(&missInfo);
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

    HitInfo hitInfo = {};

    const float bcB = hp.b1;
    const float bcC = hp.b2;
    const float bcA = 1.0f - bcB - bcC;
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

    hitInfo.instIndex = optixGetInstanceIndex();
    hitInfo.clusterId = clusterId;
    hitInfo.primIndex = hp.primIndex;
    hitInfo.barycentrics = make_float2(hp.b1, hp.b2);
    hitInfo.shadingNormal = shadingNormal;
    hitInfo.geomNormal = geomNormal;
    MyPayloadSignature::set(&hitInfo);
}
