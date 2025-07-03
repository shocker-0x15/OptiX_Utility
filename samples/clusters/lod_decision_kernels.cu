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
    const Vertex* const vertexPool, const LocalTriangle* const trianglePool,
    const Cluster* clusters, const OptixClusterAccelBuildInputTrianglesArgs* const srcClusterArgsArray,
    const uint32_t clusterCount,
    const uint32_t* const levelStartClusterIndices, const uint32_t levelCount,
    OptixClusterAccelBuildInputTrianglesArgs* const dstClusterArgsArray,
    uint32_t* const emittedClusterCount)
{
    const uint32_t clusterIdx = blockDim.x * blockIdx.x + threadIdx.x;
    bool emit = false;
    if (clusterIdx < clusterCount) {
        if (lodMode == LoDMode_ViewAdaptive) {
            const Cluster &cluster = clusters[clusterIdx];
            const float onePixelInNS = 1.0f / imageHeight;
            const float threshold = 0.5f * onePixelInNS;
            const float selfErrorInNS = estimateClusterErrorInNormalizedScreen(
                cluster.bounds, cluster.error,
                cameraPosition, cameraOrientation, cameraFovY);
            const float parentErrorInNS = estimateClusterErrorInNormalizedScreen(
                cluster.parentBounds, cluster.parentError,
                cameraPosition, cameraOrientation, cameraFovY);
            emit = selfErrorInNS <= threshold && parentErrorInNS > threshold;
        }
        else if (lodMode == LoDMode_ManualUniform) {
            const uint32_t level = identifyLevel(levelStartClusterIndices, levelCount, clusterIdx);
            emit = level == min(manualUniformLevel, levelCount - 1);
        }
    }

    uint32_t clusterArgsIdx;
    {
        const uint32_t waveEmitFlags = __ballot_sync(0xFFFF'FFFF, emit);
        uint32_t clusterArgsBaseIdx = 0;
        if (threadIdx.x == 0)
            clusterArgsBaseIdx = atomicAdd(emittedClusterCount, popcnt(waveEmitFlags));
        clusterArgsIdx = __shfl_sync(0xFFFF'FFFF, clusterArgsBaseIdx, 0) +
            popcnt(waveEmitFlags & ((1u << threadIdx.x) - 1));
    }

    if (emit) {
        dstClusterArgsArray[clusterArgsIdx] = srcClusterArgsArray[clusterIdx];
    }
}



CUDA_DEVICE_KERNEL void emitCgasArgsArray(
    const CUdeviceptr clasHandles, const uint32_t clasHandleStride,
    const uint32_t* const clusterCount,
    OptixClusterAccelBuildInputClustersArgs* const cgasArgsArray,
    uint32_t* const emittedCgasCount)
{
    if (threadIdx.x > 0)
        return;

    OptixClusterAccelBuildInputClustersArgs cgasArgs = {};
    cgasArgs.clusterHandlesBuffer = clasHandles;
    cgasArgs.clusterHandlesCount = *clusterCount;
    cgasArgs.clusterHandlesBufferStrideInBytes = clasHandleStride;

    cgasArgsArray[0] = cgasArgs;
    *emittedCgasCount = 1;
}
