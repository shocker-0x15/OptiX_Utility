#include "clusters_shared.h"

using namespace Shared;

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t identifyLevel(
    const uint32_t* levelStartClusterIndices, const uint32_t levelCount,
    const uint32_t clusterIdx)
{
    uint32_t level = 0;
    for (uint32_t d = nextPowerOf2(levelCount) >> 1; d >= 1; d >>= 1) {
        if (level + d >= levelCount)
            continue;
        if (levelStartClusterIndices[level + d] <= clusterIdx)
            level += d;
    }
    Assert(level < levelCount, "Invalid level.");

    return level;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float estimateClusterErrorInNormalizedScreen(
    const Sphere &bounds, const float errorInWorld,
    const float3 &cameraPosition, const Matrix3x3 &/*cameraOrientation*/,
    const float cameraFovY)
{
    const float3 diff = bounds.center - cameraPosition;
    const float dist = max(length(diff) - bounds.radius, 0.0f);
    const float screenHeightInWorld = 2.0f * dist * std::tan(0.5f * cameraFovY);
    return errorInWorld / screenHeightInWorld;
}

CUDA_DEVICE_KERNEL void emitClusterArgsArray(
    const LoDMode lodMode, const uint32_t manualUniformLevel,
    const float3 cameraPosition, const Matrix3x3 cameraOrientation,
    const float cameraFovY, const uint32_t imageHeight,
    const InstanceTransform instTransform,
    const Cluster* clusters, const OptixClusterAccelBuildInputTrianglesArgs* const srcClusterArgsArray,
    const uint32_t clusterCount,
    const uint32_t* const levelStartClusterIndices, const uint32_t levelCount,
    ClusterSetInfo* const clusterSetInfo,
    ClusterGasInfo* const clusterGasInfo)
{
    const uint32_t clusterIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t usedFlagsBinIdx = clusterIdx / 32;
    const uint32_t usedFlagIdxInBin = clusterIdx % 32;

    bool emit = false;
    bool alreadyEmitted = false;
    if (clusterIdx < clusterCount) {
        alreadyEmitted = ((clusterSetInfo->usedFlags[usedFlagsBinIdx] >> usedFlagIdxInBin) & 0b1) != 0;

        const Cluster &cluster = clusters[clusterIdx];
        if (lodMode == LoDMode_ViewAdaptive) {
            const float onePixelInNS = 1.0f / imageHeight;
            const float threshold = 0.5f * onePixelInNS;

            const Matrix3x3 matRot = instTransform.orientation.toMatrix3x3();
            Sphere selfBounds = cluster.bounds;
            selfBounds.center = matRot * selfBounds.center + instTransform.position;
            selfBounds.radius *= instTransform.scale;
            Sphere parentBounds = cluster.parentBounds;
            parentBounds.center = matRot * parentBounds.center + instTransform.position;
            parentBounds.radius *= instTransform.scale;

            const float selfErrorInNS = estimateClusterErrorInNormalizedScreen(
                selfBounds, instTransform.scale * cluster.error,
                cameraPosition, cameraOrientation, cameraFovY);
            const float parentErrorInNS = estimateClusterErrorInNormalizedScreen(
                parentBounds, instTransform.scale * cluster.parentError,
                cameraPosition, cameraOrientation, cameraFovY);
            emit = selfErrorInNS <= threshold && parentErrorInNS > threshold;
        }
        else if (lodMode == LoDMode_ManualUniform) {
            emit = cluster.level == min(manualUniformLevel, levelCount - 1);
        }
    }

    uint32_t clasBuildIdx;
    {
        const uint32_t waveNewEmitFlags = __ballot_sync(0xFFFF'FFFF, emit && !alreadyEmitted);
        uint32_t clusterBuildBaseIdx = 0;
        if (threadIdx.x == 0) {
            atomicOr(&clusterSetInfo->usedFlags[usedFlagsBinIdx], waveNewEmitFlags);
            clusterBuildBaseIdx = atomicAdd(&clusterSetInfo->argsCountToBuild, popcnt(waveNewEmitFlags));
        }
        clasBuildIdx = __shfl_sync(0xFFFF'FFFF, clusterBuildBaseIdx, 0) +
            popcnt(waveNewEmitFlags & ((1u << threadIdx.x) - 1));
    }

    if (emit) {
        if (alreadyEmitted) {
            clasBuildIdx = clusterSetInfo->indexMapClusterToClasBuild[clusterIdx];
        }
        else {
            clusterSetInfo->argsArray[clasBuildIdx] = srcClusterArgsArray[clusterIdx];
            clusterSetInfo->indexMapClusterToClasBuild[clusterIdx] = clasBuildIdx;
        }
    }

    uint32_t clasHandleIdx;
    {
        const uint32_t waveEmitFlags = __ballot_sync(0xFFFF'FFFF, emit);
        uint32_t compactClusterBaseIdx = 0;
        if (threadIdx.x == 0)
            compactClusterBaseIdx = atomicAdd(&clusterGasInfo->clasHandleCount, popcnt(waveEmitFlags));
        clasHandleIdx = __shfl_sync(0xFFFF'FFFF, compactClusterBaseIdx, 0) +
            popcnt(waveEmitFlags & ((1u << threadIdx.x) - 1));
    }

    if (emit)
        clusterGasInfo->indexMapClasHandleToClasBuild[clasHandleIdx] = clasBuildIdx;
}



CUDA_DEVICE_KERNEL void copyClasHandles(
    ClusterSetInfo* const clusterSetInfo,
    ClusterGasInfo* const clusterGasInfo)
{
    const uint32_t clasHandleIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (clasHandleIdx >= clusterGasInfo->clasHandleCount)
        return;

    const uint32_t clasBuildIdx = clusterGasInfo->indexMapClasHandleToClasBuild[clasHandleIdx];
    clusterGasInfo->clasHandles[clasHandleIdx] = clusterSetInfo->clasHandles[clasBuildIdx];
}
