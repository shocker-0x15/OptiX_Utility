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

CUDA_DEVICE_FUNCTION CUDA_INLINE static const NormalMeshData &getNormalMeshData() {
    return *reinterpret_cast<NormalMeshData*>(optixGetSbtDataPointer());
}

CUDA_DEVICE_FUNCTION CUDA_INLINE static const HierarchicalMeshData &getHierarchicalMeshData() {
    return *reinterpret_cast<HierarchicalMeshData*>(optixGetSbtDataPointer());
}



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
            plp.pickInfo->cluster = hitInfo.hiMeshData->clusters[hitInfo.clusterId];
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
                color = make_float3(0.25f, 0.0f, 0.5f);
            }
            else {
                color = HSVtoRGB(
                    std::fmod((GoldenAngle * hitInfo.clusterId) / (2 * pi_v<float>), 1.0f),
                    1.0f, 1.0f);
            }
        }
        else if (plp.visMode == VisualizationMode_Level) {
            if (hitInfo.clusterId == OPTIX_CLUSTER_ID_INVALID) {
                color = make_float3(0.25f, 0.0f, 0.5f);
            }
            else {
                color = calcFalseColor(hitInfo.hiMeshData->clusters[hitInfo.clusterId].level, 0, 10);
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
    const auto hp = HitPointParameter::get();

    HitInfo hitInfo = {};
    Vertex vs[3];
    const uint32_t clusterId = optixGetClusterId();
    if (clusterId == OPTIX_CLUSTER_ID_INVALID) {
        const NormalMeshData &meshData = getNormalMeshData();
        const Triangle &tri = meshData.triangles[hp.primIndex];
        vs[0] = meshData.vertices[tri.index0];
        vs[1] = meshData.vertices[tri.index1];
        vs[2] = meshData.vertices[tri.index2];
    }
    else {
        const HierarchicalMeshData &meshData = getHierarchicalMeshData();
        const Cluster &cluster = meshData.clusters[clusterId];
        const LocalTriangle &tri = meshData.trianglePool[cluster.triPoolStartIndex + hp.primIndex];
        vs[0] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index0];
        vs[1] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index1];
        vs[2] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index2];

        hitInfo.hiMeshData = &meshData;
    }

    const float bcB = hp.b1;
    const float bcC = hp.b2;
    const float bcA = 1.0f - bcB - bcC;
    float3 shadingNormal =
        bcA * vs[0].normal + bcB * vs[1].normal + bcC * vs[2].normal;
    shadingNormal = normalize(optixTransformNormalFromObjectToWorldSpace(shadingNormal));

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
