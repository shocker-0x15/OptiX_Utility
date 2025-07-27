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

/*
JP: CH/AH/ISプログラムにてoptixGetSbtDataPointer()で取得できるポインターの位置に
    CGAS SetのsetUserData(), CLAS SetのsetUserData(), MaterialのsetUserData()
    で設定したデータが順番に並んでいる(各データの相対的な開始位置は指定したアラインメントに従う)。
EN: Data set by each of
    CGAS Set's setUserData(), CLAS Set's setUserData(), Material's setUserData()
    line up in the order (Each relative offset follows the specified alignment)
    at the position pointed by optixGetSbtDataPointer() called in CH/AH/IS programs.
*/
CUDA_DEVICE_FUNCTION CUDA_INLINE static const ClusteredMeshData &getClusteredMeshData() {
    return *reinterpret_cast<ClusteredMeshData*>(optixGetSbtDataPointer());
}



CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352d;
    x ^= x >> 15;
    x *= 0x846ca68b;
    x ^= x >> 16;
    return x;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE static float hash(const float3 &p) {
    const uint32_t h = hash_u32(
        __float_as_uint(p.x) ^
        hash_u32(__float_as_uint(p.y)) ^
        hash_u32(__float_as_uint(p.z)));
    return static_cast<float>(h) / static_cast<float>(0xFFFF'FFFF);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE static float noise(const float3 &p) {
    const float3 i = floor(p);
    const float3 f = fract(p);

    const float3 u = f * f * (make_float3(3.0f) - 2.0f * f);

    const float n000 = hash(i + make_float3(0.0f, 0.0f, 0.0f));
    const float n100 = hash(i + make_float3(1.0f, 0.0f, 0.0f));
    const float n010 = hash(i + make_float3(0.0f, 1.0f, 0.0f));
    const float n110 = hash(i + make_float3(1.0f, 1.0f, 0.0f));
    const float n001 = hash(i + make_float3(0.0f, 0.0f, 1.0f));
    const float n101 = hash(i + make_float3(1.0f, 0.0f, 1.0f));
    const float n011 = hash(i + make_float3(0.0f, 1.0f, 1.0f));
    const float n111 = hash(i + make_float3(1.0f, 1.0f, 1.0f));

    const float nx00 = lerp(n000, n100, u.x);
    const float nx10 = lerp(n010, n110, u.x);
    const float nx01 = lerp(n001, n101, u.x);
    const float nx11 = lerp(n011, n111, u.x);

    const float nxy0 = lerp(nx00, nx10, u.y);
    const float nxy1 = lerp(nx01, nx11, u.y);

    return lerp(nxy0, nxy1, u.z);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE static float fbm(float3 p) {
    float f = 0.0f;
    float amp = 0.5f;
    for (uint32_t i = 0; i < 5; ++i) {
        f += amp * noise(p);
        p = 2.0f * p;
        amp *= 0.5f;
    }
    return f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE static float3 evalWood(const float3 &npos) {
    const float3 p = 2.0f * npos - make_float3(1.0f);
    const float r = length(make_float2(p.x, p.z));
    const float grain = 1.5f * fbm(5.0f * p);
    const float rings = fract((4.0f * r + grain));

    const auto dark = sRGB_degamma(float3{ 0.396f, 0.263f, 0.129f });
    const auto light = sRGB_degamma(float3{ 0.804f, 0.522f, 0.247f });
    return lerp(dark, light, make_float3(std::sqrt(rings)));
}



CUDA_DEVICE_KERNEL void RT_RG_NAME(raygen)() {
    uint2 launchIndex = make_uint2(optixGetLaunchIndex().x, optixGetLaunchIndex().y);

    const float x = static_cast<float>(launchIndex.x + plp.subPixelOffset.x) / plp.imageSize.x;
    const float y = static_cast<float>(launchIndex.y + plp.subPixelOffset.y) / plp.imageSize.y;
    const float vh = 2 * std::tan(plp.camera.fovY * 0.5f);
    const float vw = plp.camera.aspect * vh;

    const float3 origin = plp.camera.position;
    const float3 direction = normalize(
        plp.camera.orientation * make_float3(vw * (0.5f - x), vh * (0.5f - y), 1));

    HitInfo hitInfo;
    float3 shadedColor;
    MyPayloadSignature::trace(
        plp.travHandle, origin, direction,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
        RayType_Primary, NumRayTypes, RayType_Primary,
        hitInfo, shadedColor);

    if (launchIndex == plp.mousePosition) {
        plp.pickInfo->instanceIndex = hitInfo.instIndex;
        plp.pickInfo->clusterId = hitInfo.clusterId;
        plp.pickInfo->primitiveIndex = hitInfo.primIndex;
        plp.pickInfo->barycentrics = hitInfo.barycentrics;
        if (hitInfo.clusterId != OPTIX_CLUSTER_ID_INVALID) {
            plp.pickInfo->cluster = hitInfo.cMeshData->clusters[hitInfo.clusterId];
        }
        else {
            plp.pickInfo->cluster.level = 0;
            plp.pickInfo->cluster.vertexCount = 0;
            plp.pickInfo->cluster.triangleCount = 0;
        }
    }

    const bool hit = hitInfo.geomNormal != make_float3(0, 0, 0);
    float3 color = make_float3(0.0f, 0.0f, 0.1f);
    if (plp.visMode == VisualizationMode_Final) {
        color = shadedColor;
    }
    else {
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
                    color = calcFalseColor(hitInfo.cMeshData->clusters[hitInfo.clusterId].level, 0, 10);
                }
            }
            else if (plp.visMode == VisualizationMode_Triangle) {
                color = HSVtoRGB(
                    std::fmod((GoldenAngle * hitInfo.primIndex) / (2 * pi_v<float>), 1.0f),
                    1.0f, 1.0f);
            }
        }
    }

    float3 prevColorResult = make_float3(0.0f, 0.0f, 0.0f);
    if (plp.sampleIndex > 0)
        prevColorResult = getXYZ(plp.colorAccumBuffer.read(launchIndex));
    const float curWeight = 1.0f / (1 + plp.sampleIndex);
    const float3 colorResult = (1 - curWeight) * prevColorResult + curWeight * color;
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
    const float3 shadedColor = plp.envRadiance;
    MyPayloadSignature::set(&missInfo, &shadedColor);
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
        const ClusteredMeshData &meshData = getClusteredMeshData();
        const Cluster &cluster = meshData.clusters[clusterId];
        const LocalTriangle &tri = meshData.trianglePool[cluster.triPoolStartIndex + hp.primIndex];
        vs[0] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index0];
        vs[1] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index1];
        vs[2] = meshData.vertexPool[cluster.vertPoolStartIndex + tri.index2];

        hitInfo.cMeshData = &meshData;
    }

    for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t bitMask = ~((1u << plp.posTruncateBitWidth) - 1);
        vs[i].position.x = __uint_as_float(__float_as_uint(vs[i].position.x) & bitMask);
        vs[i].position.y = __uint_as_float(__float_as_uint(vs[i].position.y) & bitMask);
        vs[i].position.z = __uint_as_float(__float_as_uint(vs[i].position.z) & bitMask);
    }

    const float bcB = hp.b1;
    const float bcC = hp.b2;
    const float bcA = 1.0f - bcB - bcC;

    const float3 localPos = bcA * vs[0].position + bcB * vs[1].position + bcC * vs[2].position;
    const float3 position = optixTransformPointFromObjectToWorldSpace(localPos);

    float3 shadingNormal = bcA * vs[0].normal + bcB * vs[1].normal + bcC * vs[2].normal;
    shadingNormal = normalize(optixTransformNormalFromObjectToWorldSpace(shadingNormal));

    float3 geomNormal = cross(
        vs[1].position - vs[0].position,
        vs[2].position - vs[0].position);
    geomNormal = normalize(optixTransformNormalFromObjectToWorldSpace(geomNormal));

    float3 albedo = make_float3(0.5f, 0.5f, 0.5f);
    if (clusterId != OPTIX_CLUSTER_ID_INVALID) {
        // JP: クラスター化メッシュに適当なテクスチャーを設定する。
        // EN: Set some texture for clustered mesh.
        const ClusteredMeshData &meshData = getClusteredMeshData();
        const float3 npos = meshData.bbox.calcNormalizedPos(localPos);
        //albedo = npos;
        albedo = evalWood(npos);
    }
    float3 shadedColor = plp.envRadiance * albedo;

    float visibility = 1.0f;
    VisibilityRayPayloadSignature::trace(
        plp.travHandle,
        position + 1e-4f * geomNormal,
        plp.lightDirection,
        0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        RayType_Visibility, NumRayTypes, RayType_Visibility,
        visibility);
    if (visibility > 0.0f) {
        const float cosTerm = dot(shadingNormal, plp.lightDirection);
        const float3 fs = cosTerm > 0 ? albedo / Pi : make_float3(0, 0, 0);
        shadedColor += fs * cosTerm * plp.lightRadiance;
    }

    hitInfo.instIndex = optixGetInstanceIndex();
    hitInfo.clusterId = clusterId;
    hitInfo.primIndex = hp.primIndex;
    hitInfo.barycentrics = make_float2(hp.b1, hp.b2);
    hitInfo.shadingNormal = shadingNormal;
    hitInfo.geomNormal = geomNormal;

    MyPayloadSignature::set(&hitInfo, &shadedColor);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);
}
