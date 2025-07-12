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

CUDA_DEVICE_KERNEL void emitClasArgsArray(
    const LoDMode lodMode, const uint32_t manualUniformLevel,
    const float3 cameraPosition, const Matrix3x3 cameraOrientation,
    const float cameraFovY, const uint32_t imageHeight,
    const Cluster* clusters, const OptixClusterAccelBuildInputTrianglesArgs* const srcClusterArgsArray,
    const uint32_t meshTotalClusterCount,
    const uint32_t* const levelStartClusterIndices, const uint32_t levelCount,
    ClusterSetInfo* const clusterSetInfo,
    ClusterGasInstanceInfo* const clusterGasInstInfos, const uint32_t instCount)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instClusterCountStride = (meshTotalClusterCount + 31) / 32 * 32;
    const uint32_t instIdx = globalThreadIdx / instClusterCountStride;
    const uint32_t clusterIdx = globalThreadIdx % instClusterCountStride;
    const bool isValidThread =
        instIdx < instCount &&
        clusterIdx < meshTotalClusterCount;

    const uint32_t usedFlagsBinIdx = isValidThread ? clusterIdx / 32 : 0;
    const uint32_t usedFlagIdxInBin = isValidThread ? clusterIdx % 32 : 0;

    ClusterGasInstanceInfo &clusterGasInstInfo = clusterGasInstInfos[isValidThread ? instIdx : 0];

    // JP: クラスターを描画すべきかどうかを決定する。
    // EN: 
    bool emit = false;
    if (isValidThread) {
        const Cluster &cluster = clusters[clusterIdx];
        if (lodMode == LoDMode_ViewAdaptive) {
            const float onePixelInNS = 1.0f / imageHeight;
            const float threshold = 1.0f * onePixelInNS;

            const InstanceTransform &xfm = clusterGasInstInfo.transform;
            const Matrix3x3 matRot = xfm.orientation.toMatrix3x3();
            Sphere selfBounds = cluster.bounds;
            selfBounds.center = matRot * selfBounds.center + xfm.position;
            selfBounds.radius *= xfm.scale;
            Sphere parentBounds = cluster.parentBounds;
            parentBounds.center = matRot * parentBounds.center + xfm.position;
            parentBounds.radius *= xfm.scale;

            const float selfErrorInNS = estimateClusterErrorInNormalizedScreen(
                selfBounds, xfm.scale * cluster.error,
                cameraPosition, cameraOrientation, cameraFovY);
            const float parentErrorInNS = estimateClusterErrorInNormalizedScreen(
                parentBounds, xfm.scale * cluster.parentError,
                cameraPosition, cameraOrientation, cameraFovY);
            emit = selfErrorInNS <= threshold && parentErrorInNS > threshold;
        }
        else if (lodMode == LoDMode_ManualUniform) {
            emit = cluster.level == min(manualUniformLevel, levelCount - 1);
        }
    }

    uint32_t clasBuildIdx;
    bool isNewEmit = false;
    {
        // JP: クラスターを描画する場合、使用フラグをセットする。
        //     また、このスレッドがクラスターの設定を初めて行うスレッドかを判定する。
        // EN: 
        const uint32_t waveEmitFlags = __ballot_sync(0xFFFF'FFFF, emit);
        uint32_t oldWaveEmitFlags = 0;
        if (threadIdx.x == 0)
            oldWaveEmitFlags = atomicOr(&clusterSetInfo->usedFlags[usedFlagsBinIdx], waveEmitFlags);
        oldWaveEmitFlags = __shfl_sync(0xFFFF'FFFF, oldWaveEmitFlags, 0);
        isNewEmit = emit && ((oldWaveEmitFlags >> usedFlagIdxInBin) & 0b1) == 0;

        // JP: 新規クラスターのCLASバッチビルド中のインデックスを決定する。
        // EN:
        const uint32_t waveNewEmitFlags = __ballot_sync(0xFFFF'FFFF, isNewEmit);
        uint32_t clusterBuildBaseIdx = 0;
        if (threadIdx.x == 0)
            clusterBuildBaseIdx = atomicAdd(&clusterSetInfo->argsCountToBuild, popcnt(waveNewEmitFlags));
        clasBuildIdx = __shfl_sync(0xFFFF'FFFF, clusterBuildBaseIdx, 0) +
            popcnt(waveNewEmitFlags & ((1u << threadIdx.x) - 1));
    }

    /*
    JP: 新規クラスターの設定を行う。
        CLASバッチビルドの出力ハンドル列を、複数のCluster GASそれぞれの入力である
        CLASハンドルバッファーにコピーできるよう、クラスターインデックスからCLASバッチビルド中のインデックス
        へのマップを作成する。
    EN: 
    */
    if (isNewEmit) {
        clusterSetInfo->argsArray[clasBuildIdx] = srcClusterArgsArray[clusterIdx];
        clusterSetInfo->indexMapClusterToClasBuild[clusterIdx] = clasBuildIdx;
    }

    // JP: Cluster GASで使用するCLAS数をカウントする。
    // EN: 
    uint32_t clasHandleIdx;
    {
        const uint32_t waveEmitFlags = __ballot_sync(0xFFFF'FFFF, emit);
        uint32_t compactClusterBaseIdx = 0;
        if (threadIdx.x == 0)
            compactClusterBaseIdx = atomicAdd(&clusterGasInstInfo.clasHandleCount, popcnt(waveEmitFlags));
        clasHandleIdx = __shfl_sync(0xFFFF'FFFF, compactClusterBaseIdx, 0) +
            popcnt(waveEmitFlags & ((1u << threadIdx.x) - 1));
    }

    // JP: Cluster GASの入力CLASハンドルバッファーに、CLASバッチビルドの出力ハンドル列をコピーできるよう、
    //     入力ハンドルインデックスからクラスターインデックスへのマップを作成する。
    // EN: 
    if (emit)
        clusterGasInstInfo.indexMapClasHandleToCluster[clasHandleIdx] = clusterIdx;
}



CUDA_DEVICE_KERNEL void copyClasHandles(
    const uint32_t maxClusterCountPerInst, ClusterSetInfo* const clusterSetInfo,
    ClusterGasInstanceInfo* const clusterGasInstInfos, const uint32_t instCount)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instIdx = globalThreadIdx / maxClusterCountPerInst;
    const uint32_t clasHandleIdx = globalThreadIdx % maxClusterCountPerInst;
    const bool isValidThread =
        instIdx < instCount &&
        clasHandleIdx < maxClusterCountPerInst;

    // JP: メッシュのCLASバッチビルドの出力ハンドル列を、Cluster GASの入力ハンドルバッファーにコピーする。
    ClusterGasInstanceInfo &clusterGasInstInfo = clusterGasInstInfos[isValidThread ? instIdx : 0];
    const uint32_t srcClusterIdx =
        clusterGasInstInfo.indexMapClasHandleToCluster[isValidThread ? clasHandleIdx : 0];
    const uint32_t clasBuildIdx =
        clusterSetInfo->indexMapClusterToClasBuild[srcClusterIdx];
    if (isValidThread)
        clusterGasInstInfo.clasHandles[clasHandleIdx] = clusterSetInfo->clasHandles[clasBuildIdx];
}



CUDA_DEVICE_KERNEL void emitClusterGasArgsArray(
    ClusterGasInstanceInfo* const clusterGasInstInfos, const uint32_t instCount,
    OptixClusterAccelBuildInputClustersArgs* const cgasArgsArray)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instIdx = globalThreadIdx;
    if (instIdx >= instCount)
        return;

    ClusterGasInstanceInfo &clusterGasInstInfo = clusterGasInstInfos[instIdx];
    OptixClusterAccelBuildInputClustersArgs cgasArgs = {};
    cgasArgs.clusterHandlesBuffer = reinterpret_cast<CUdeviceptr>(clusterGasInstInfo.clasHandles);
    cgasArgs.clusterHandlesBufferStrideInBytes = sizeof(CUdeviceptr);
    cgasArgs.clusterHandlesCount = clusterGasInstInfo.clasHandleCount;

    cgasArgsArray[instIdx] = cgasArgs;
}
