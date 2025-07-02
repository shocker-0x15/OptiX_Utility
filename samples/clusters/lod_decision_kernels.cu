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
    const float screenHeightInWorld = 2.0f * dist * tanf(0.5f * cameraFovY);
    return errorInWorld / screenHeightInWorld;
}

CUDA_DEVICE_KERNEL void emitClusterArgsArray(
    const LoDMode lodMode, const uint32_t manualUniformLevel,
    const float3 cameraPosition, const Matrix3x3 cameraOrientation,
    const float cameraFovY, const uint32_t imageHeight,
    const Vertex* const vertexPool, const LocalTriangle* const trianglePool,
    const Cluster* const clusters, const uint32_t clusterCount,
    const uint32_t* const levelStartClusterIndices, const uint32_t levelCount,
    OptixClusterAccelBuildInputTrianglesArgs* const clusterArgsArray,
    uint32_t* const emittedClusterCount)
{
    const uint32_t clusterIdx = blockDim.x * blockIdx.x + threadIdx.x;
    bool emit = false;
    if (clusterIdx < clusterCount) {
        const uint32_t level = identifyLevel(levelStartClusterIndices, levelCount, clusterIdx);
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
            emit = level == manualUniformLevel;
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
        const Cluster &cluster = clusters[clusterIdx];

        OptixClusterAccelBuildInputTrianglesArgs args = {};
        args.clusterId = clusterIdx;
        args.clusterFlags = OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE;
        args.triangleCount = cluster.triangleCount;
        args.vertexCount = cluster.vertexCount;
        args.positionTruncateBitCount = 0;
        args.indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_8BIT;
        args.opacityMicromapIndexFormat = 0; // not used in this sample
        args.basePrimitiveInfo.sbtIndex = 0;
        args.basePrimitiveInfo.primitiveFlags = OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_NONE;
        args.indexBufferStrideInBytes = sizeof(uint8_t);
        args.vertexBufferStrideInBytes = sizeof(Shared::Vertex);
        args.primitiveInfoBufferStrideInBytes = 0; // not used in this sample
        args.opacityMicromapIndexBufferStrideInBytes = 0; // not used in this sample
        args.indexBuffer = reinterpret_cast<CUdeviceptr>(&trianglePool[cluster.triPoolStartIndex]);
        args.vertexBuffer = reinterpret_cast<CUdeviceptr>(&vertexPool[cluster.vertPoolStartIndex]);
        args.primitiveInfoBuffer = 0; // not used in this sample
        args.opacityMicromapArray = 0; // not used in this sample
        args.opacityMicromapIndexBuffer = 0; // not used in this sample
        args.instantiationBoundingBoxLimit = 0; // ignored for this arg type

        clusterArgsArray[clusterArgsIdx] = args;
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
