#include "clusters_shared.h"

#if !defined(__INTELLISENSE__)
#include <cub/cub.cuh>
#endif

using namespace Shared;



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
    const GeometryConfig geomConfig,
    const float3 cameraPosition, const Matrix3x3 cameraOrientation,
    const float cameraFovY, const uint32_t imageHeight,
    const Cluster* clusters, const OptixClusterAccelBuildInputTrianglesArgs* const srcClusterArgsArray,
    const uint32_t meshTotalClusterCount, const uint32_t levelCount,
    ClusterSetInfo* const clusterSetInfo,
    const InstanceStaticInfo* const instStaticInfos,
    InstanceDynamicInfo* const instDynamicInfos,
    const uint32_t instCount, bool invalidate, const uint32_t bufIdx)
{
    const uint32_t prevBufIdx = (bufIdx + 1) % 2;

    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instClusterCountStride = (meshTotalClusterCount + 31) / 32 * 32;
    const uint32_t instIdx = globalThreadIdx / instClusterCountStride;
    const uint32_t clusterIdx = globalThreadIdx % instClusterCountStride;
    const bool isValidThread =
        instIdx < instCount &&
        clusterIdx < meshTotalClusterCount;

    const uint32_t flagsBinIdx = isValidThread ? clusterIdx / 32 : 0;
    const uint32_t flagIdxInBin = isValidThread ? clusterIdx % 32 : 0;

    const InstanceStaticInfo &instStaticInfo = instStaticInfos[isValidThread ? instIdx : 0];
    InstanceDynamicInfo &instDynamicInfo = instDynamicInfos[isValidThread ? instIdx : 0];

    // JP: クラスターを描画すべきかどうかを決定する。
    // EN: Determine whether to draw the cluster or not.
    bool clusterIsUsed = false;
    if (isValidThread) {
        const Cluster &cluster = clusters[clusterIdx];
        if (geomConfig.lodMode == LoDMode_ViewAdaptive) {
            const float onePixelInNS = 1.0f / imageHeight;
            const float threshold = 2.0f * onePixelInNS;

            const InstanceTransform &xfm = instDynamicInfo.transform;
            const Matrix3x3 matSR = xfm.scale * xfm.orientation.toMatrix3x3();
            Sphere selfBounds = cluster.bounds;
            selfBounds.center = matSR * selfBounds.center + xfm.position;
            selfBounds.radius *= xfm.scale;
            Sphere parentBounds = cluster.parentBounds;
            parentBounds.center = matSR * parentBounds.center + xfm.position;
            parentBounds.radius *= xfm.scale;

            // Reference:
            // - Nanite: A Deep Dive
            // - meshoptimizer
            const float errorScale = xfm.scale * clusterSetInfo->meshErrorScale;
            const float selfErrorInNS = estimateClusterErrorInNormalizedScreen(
                selfBounds, errorScale * cluster.error,
                cameraPosition, cameraOrientation, cameraFovY);
            const float parentErrorInNS = estimateClusterErrorInNormalizedScreen(
                parentBounds, errorScale * cluster.parentError,
                cameraPosition, cameraOrientation, cameraFovY);
            clusterIsUsed = selfErrorInNS <= threshold && parentErrorInNS > threshold;
        }
        else if (geomConfig.lodMode == LoDMode_ManualUniform) {
            clusterIsUsed = cluster.level == min(geomConfig.manualUniformLevel, levelCount - 1);
        }
    }

    uint32_t* const curUsedFlagArray = clusterSetInfo->usedFlagArrays[bufIdx];
    uint32_t* const curBuildFlagArray = clusterSetInfo->buildFlagArrays[bufIdx];
    uint32_t* const prevUsedFlagArray = clusterSetInfo->usedFlagArrays[prevBufIdx];
    uint32_t* const prevBuildFlagArray = clusterSetInfo->buildFlagArrays[prevBufIdx];

    bool isFirstThread = false;
    {
        // JP: クラスターを描画する場合、使用フラグをセットする。
        //     また、このスレッドがクラスターの設定を初めて行うスレッドかを判定する。
        // EN: If the cluster is to be drawn, set the used flag.
        //     Also determine if this thread is the first thread to set the cluster.
        const uint32_t waveUsedFlags = __ballot_sync(0xFFFF'FFFF, clusterIsUsed);
        uint32_t oldWaveUsedFlags = 0;
        if (threadIdx.x == 0)
            oldWaveUsedFlags = atomicOr(&curUsedFlagArray[flagsBinIdx], waveUsedFlags);
        oldWaveUsedFlags = __shfl_sync(0xFFFF'FFFF, oldWaveUsedFlags, 0);
        isFirstThread = clusterIsUsed && ((oldWaveUsedFlags >> flagIdxInBin) & 0b1) == 0;
    }

    enum class ClusterState : uint32_t {
        Unused = 0,
        CopyFromLastBuild,
        CopyFromLastReuse,
        Build,
    };

    ClusterState state = ClusterState::Unused;
    if (clusterIsUsed) {
        bool const prevUsedFlag = isValidThread ?
            ((prevUsedFlagArray[flagsBinIdx] >> flagIdxInBin) & 0b1) != 0 : false;
        bool const prevBuildFlag = isValidThread ?
            ((prevBuildFlagArray[flagsBinIdx] >> flagIdxInBin) & 0b1) != 0 : false;
        Assert(!prevBuildFlag || prevUsedFlag, "The previous build flag is set but the previous used flag is not set.");
        if (prevUsedFlag && prevBuildFlag)
            state = ClusterState::CopyFromLastBuild;
        else if (prevUsedFlag && !prevBuildFlag)
            state = ClusterState::CopyFromLastReuse;
        else
            state = ClusterState::Build;

        if (invalidate)
            state = ClusterState::Build;
    }

    uint32_t clasBuildOrReuseIdx = 0xFFFF'FFFF;

    // state == ClusterState::Build
    {
        const bool isSubject = state == ClusterState::Build;

        // JP: クラスターをビルドする場合、ビルドフラグのセットと
        //     CLASバッチビルド中のインデックス決定を行い、新規クラスターの設定を行う。
        // EN: If the cluster is to be built, set the build flag, determine the index in the CLAS batch build,
        //     and set up the new cluster.
        const uint32_t waveBuildFlags = __ballot_sync(0xFFFF'FFFF, isSubject && isFirstThread);
        uint32_t clusterBuildIdxBase = 0;
        if (threadIdx.x == 0) {
            atomicOr(&curBuildFlagArray[flagsBinIdx], waveBuildFlags);
            clusterBuildIdxBase = atomicAdd(&clusterSetInfo->buildCount, popcnt(waveBuildFlags));
        }
        clusterBuildIdxBase = __shfl_sync(0xFFFF'FFFF, clusterBuildIdxBase, 0);
        if (isSubject && isFirstThread) {
            clasBuildOrReuseIdx = clusterBuildIdxBase + popcnt(waveBuildFlags & ((1u << threadIdx.x) - 1));
            OptixClusterAccelBuildInputTrianglesArgs args = srcClusterArgsArray[clusterIdx];
            args.positionTruncateBitCount = geomConfig.positionTruncateBitWidth;
            clusterSetInfo->buildArgsArray[clasBuildOrReuseIdx] = args;
        }
    }

    // state == ClusterState::CopyFromLastBuild || state == ClusterState::CopyFromLastReuse
    {
        const bool isSubject =
            (state == ClusterState::CopyFromLastBuild || state == ClusterState::CopyFromLastReuse);

        uint32_t prevBuildOrReuseIdx = isSubject ?
            clusterSetInfo->indexMapClusterToBuildOrReuse[clusterIdx] : 0xFFFF'FFFF;
        Assert(
            prevBuildOrReuseIdx != 0xFFFF'FFFF || !isSubject,
            "The previous CLAS build/reuse index is invalid but the thread is subject.");
        const bool fromLastReuse = (prevBuildOrReuseIdx >> 31) != 0;
        prevBuildOrReuseIdx &= 0x7FFF'FFFF;

        CUdeviceptr clasAddress = 0;
        uint32_t clasSize = 0;
        if (isSubject && isFirstThread) {
            if (fromLastReuse) {
                clasAddress = clusterSetInfo->reuseHandleArrays[prevBufIdx][prevBuildOrReuseIdx];
                clasSize = clusterSetInfo->reuseSizeArrays[prevBufIdx][prevBuildOrReuseIdx];
            }
            else {
                clasAddress = clusterSetInfo->buildHandleArray[prevBuildOrReuseIdx];
                clasSize = clusterSetInfo->buildSizeArray[prevBuildOrReuseIdx];
            }
        }

        // JP: クラスターを再利用する場合、CLASサブセットコピー中のインデックス決定を行う。
        // EN: If the cluster is to be reused, determine the index in the CLAS subset copy.
        const uint32_t waveReuseFlags = __ballot_sync(0xFFFF'FFFF, isSubject && isFirstThread);
        uint32_t clusterReuseIdxBase = 0;
        if (threadIdx.x == 0)
            clusterReuseIdxBase = atomicAdd(&clusterSetInfo->reuseCounts[bufIdx], popcnt(waveReuseFlags));
        clusterReuseIdxBase = __shfl_sync(0xFFFF'FFFF, clusterReuseIdxBase, 0);
        if (isSubject && isFirstThread) {
            clasBuildOrReuseIdx = clusterReuseIdxBase + popcnt(waveReuseFlags & ((1u << threadIdx.x) - 1));
            clusterSetInfo->reuseHandleArrays[bufIdx][clasBuildOrReuseIdx] = clasAddress;
            clusterSetInfo->reuseSizeArrays[bufIdx][clasBuildOrReuseIdx] = clasSize;
            clasBuildOrReuseIdx |= 0x8000'0000; // Set the MSB to indicate that this is a reuse index.
        }
    }

    if (isFirstThread)
        clusterSetInfo->indexMapClusterToBuildOrReuse[clusterIdx] = clasBuildOrReuseIdx;

    // JP: Cluster GASで使用するCLAS数をカウントする。
    //     Cluster GASの入力CLASハンドルバッファーに、CLASバッチビルドの出力ハンドル列をコピーできるよう、
    //     入力ハンドルインデックスからクラスターインデックスへのマップを作成する。
    // EN: Count the number of CLASes to use in the Cluster GAS.
    //     Create a map from the input handle index to the cluster index, so that
    //     the output handle array of the CLAS batch build can be copied to
    //     the input CLAS handle buffer of the Cluster GAS.
    {
        const uint32_t waveUsedFlags = __ballot_sync(0xFFFF'FFFF, clusterIsUsed);
        uint32_t compactClusterIdxBase = 0;
        if (threadIdx.x == 0)
            compactClusterIdxBase = atomicAdd(&instDynamicInfo.cgas.clasHandleCount, popcnt(waveUsedFlags));
        compactClusterIdxBase = __shfl_sync(0xFFFF'FFFF, compactClusterIdxBase, 0);
        if (clusterIsUsed) {
            const uint32_t clasHandleIdx = compactClusterIdxBase +
                popcnt(waveUsedFlags & ((1u << threadIdx.x) - 1));
            instStaticInfo.cgas.indexMapClasHandleToCluster[clasHandleIdx] = clusterIdx;
        }
    }
}



CUDA_DEVICE_KERNEL void scanReuseSizeArray(
    ClusterSetInfo const* const clusterSetInfo,
    void* const commonOpScratchMem, size_t commonOpScratchMemSize,
    const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalThreadIdx >= 1)
        return;

    const uint32_t reuseCount = clusterSetInfo->reuseCounts[bufIdx];

    cub::DeviceScan::ExclusiveSum(
        commonOpScratchMem, commonOpScratchMemSize,
        clusterSetInfo->reuseSizeArrays[bufIdx], clusterSetInfo->reuseOffsetArray,
        reuseCount + 1);
}



CUDA_DEVICE_KERNEL void __copyReusedClases(
    ClusterSetInfo const* const clusterSetInfo,
    const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    constexpr uint32_t byteCountPerThread = 16;
    const uint32_t dstByteOffset = byteCountPerThread * globalThreadIdx;

    const uint32_t reuseCount = clusterSetInfo->reuseCounts[bufIdx];
    const uint32_t* const reuseOffsetArray = clusterSetInfo->reuseOffsetArray;

    uint32_t clasIdx;
    {
        uint32_t lo = 0;
        uint32_t hi = reuseCount;
        while (lo + 1 < hi) {
            const uint32_t mid = lo + (hi - lo) / 2;

            if (reuseOffsetArray[mid] <= dstByteOffset)
                lo = mid;
            else
                hi = mid;
        }
        clasIdx = lo;
    }

    const CUdeviceptr srcAddr = clusterSetInfo->reuseHandleArrays[bufIdx][clasIdx];
    const uint32_t srcSize = clusterSetInfo->reuseSizeArrays[bufIdx][clasIdx];
    const uint32_t byteOffsetInClas = dstByteOffset - reuseOffsetArray[clasIdx];
    Assert(
        (srcSize & (OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT - 1)) == 0,
        "CLAS size is not aligned to the required alignment.");
    if (byteOffsetInClas < srcSize) {
        const auto data = *reinterpret_cast<const uint4*>(srcAddr + byteOffsetInClas);
        *reinterpret_cast<uint4*>(clusterSetInfo->reuseMems[bufIdx] + dstByteOffset) = data;
    }
}

CUDA_DEVICE_KERNEL void copyReusedClases(
    ClusterSetInfo const* const clusterSetInfo,
    const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalThreadIdx >= 1)
        return;

    const uint32_t reuseCount = clusterSetInfo->reuseCounts[bufIdx];
    const uint32_t totalCopySize = clusterSetInfo->reuseOffsetArray[reuseCount];
    //printf("Total copy size: %u bytes\n", totalCopySize);

    constexpr uint32_t byteCountPerThread = 16;
    const uint32_t threadCount = (totalCopySize + byteCountPerThread - 1) / byteCountPerThread;
    constexpr uint32_t blockSize = 32;
    const uint32_t blockCount = (threadCount + blockSize - 1) / blockSize;
    if (blockCount == 0)
        return;
    
    __copyReusedClases<<<blockCount, blockSize>>>(
        clusterSetInfo, bufIdx);
}



CUDA_DEVICE_KERNEL void __updateReuseClasHandles(
    ClusterSetInfo const* const clusterSetInfo,
    const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;

    const uint32_t clasIdx = globalThreadIdx;
    const uint32_t reuseCount = clusterSetInfo->reuseCounts[bufIdx];
    if (clasIdx >= reuseCount)
        return;

    const uint32_t* const reuseOffsetArray = clusterSetInfo->reuseOffsetArray;
    clusterSetInfo->reuseHandleArrays[bufIdx][clasIdx] =
        clusterSetInfo->reuseMems[bufIdx] + reuseOffsetArray[clasIdx];
}

CUDA_DEVICE_KERNEL void updateReuseClasHandles(
    ClusterSetInfo const* const clusterSetInfo,
    const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (globalThreadIdx >= 1)
        return;

    const uint32_t reuseCount = clusterSetInfo->reuseCounts[bufIdx];
    constexpr uint32_t blockSize = 32;
    const uint32_t blockCount = (reuseCount + blockSize - 1) / blockSize;
    if (blockCount == 0)
        return;

    __updateReuseClasHandles<<<blockCount, blockSize>>>(
        clusterSetInfo, bufIdx);
}



CUDA_DEVICE_KERNEL void copyClasHandles(
    const uint32_t maxClusterCountPerInst, const ClusterSetInfo* const clusterSetInfo,
    const InstanceStaticInfo* const instStaticInfos,
    const uint32_t instCount, const uint32_t bufIdx)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instIdx = globalThreadIdx / maxClusterCountPerInst;
    const uint32_t clasHandleIdx = globalThreadIdx % maxClusterCountPerInst;
    const bool isValidThread =
        instIdx < instCount &&
        clasHandleIdx < maxClusterCountPerInst;

    // JP: メッシュのCLASバッチビルド/再利用の出力ハンドル列を、Cluster GASの入力ハンドルバッファーにコピーする。
    // EN: Copy the output handle array of the mesh CLAS batch build/reuse
    //     to the input handle buffer of the Cluster GAS.
    const InstanceStaticInfo &instStaticInfo = instStaticInfos[isValidThread ? instIdx : 0];
    const uint32_t srcClusterIdx =
        instStaticInfo.cgas.indexMapClasHandleToCluster[isValidThread ? clasHandleIdx : 0];
    uint32_t clasBuildOrReuseIdx =
        clusterSetInfo->indexMapClusterToBuildOrReuse[srcClusterIdx];
    const bool fromReuse = (clasBuildOrReuseIdx >> 31) != 0;
    clasBuildOrReuseIdx &= 0x7FFF'FFFF;
    if (isValidThread) {
        instStaticInfo.cgas.clasHandles[clasHandleIdx] = fromReuse ?
            clusterSetInfo->reuseHandleArrays[bufIdx][clasBuildOrReuseIdx] :
            clusterSetInfo->buildHandleArray[clasBuildOrReuseIdx];
    }
}



CUDA_DEVICE_KERNEL void emitClusterGasArgsArray(
    const InstanceStaticInfo* const instStaticInfos,
    const InstanceDynamicInfo* const instDynamicInfos,
    const uint32_t instCount,
    OptixClusterAccelBuildInputClustersArgs* const cgasArgsArray)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instIdx = globalThreadIdx;
    if (instIdx >= instCount)
        return;

    const InstanceStaticInfo &instStaticInfo = instStaticInfos[instIdx];
    const InstanceDynamicInfo &instDynamicInfo = instDynamicInfos[instIdx];
    OptixClusterAccelBuildInputClustersArgs cgasArgs = {};
    cgasArgs.clusterHandlesBuffer = reinterpret_cast<CUdeviceptr>(instStaticInfo.cgas.clasHandles);
    cgasArgs.clusterHandlesBufferStrideInBytes = sizeof(CUdeviceptr);
    cgasArgs.clusterHandlesCount = instDynamicInfo.cgas.clasHandleCount;

    cgasArgsArray[instIdx] = cgasArgs;
}



CUDA_DEVICE_KERNEL void copyDataForCpu(
    const ClusterSetInfo* const clusterSetInfo,
    const InstanceDynamicInfo* const instDynamicInfos, const uint32_t instCount,
    uint32_t* clasBuildCounts)
{
    const uint32_t globalThreadIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const uint32_t instIdx = globalThreadIdx;
    if (instIdx >= instCount)
        return;

    if (instIdx == 0)
        clasBuildCounts[0] = clusterSetInfo->buildCount;

    clasBuildCounts[1 + instIdx] = instDynamicInfos[instIdx].cgas.clasHandleCount;
}
