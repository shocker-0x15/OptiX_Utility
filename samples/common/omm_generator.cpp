#include "omm_generator.h"
#include "../../ext/cubd/cubd.h"

static CUmodule s_ommModule;
static cudau::Kernel s_countOMMFormats;
static cudau::Kernel s_createOMMDescriptors;
static cudau::Kernel s_evaluateMicroTriangleTransparencies;

size_t getScratchMemSizeForOMMGeneration(uint32_t maxNumTriangles) {
    static bool isInitialized = false;
    if (!isInitialized) {
        CUDADRV_CHECK(cuModuleLoad(
            &s_ommModule,
            (getExecutableDirectory() / "opacity_micro_map/ptxes/omm_kernels.ptx").string().c_str()));
        s_countOMMFormats.set(
            s_ommModule, "countOMMFormats", cudau::dim3(32), 0);
        s_createOMMDescriptors.set(
            s_ommModule, "createOMMDescriptors", cudau::dim3(32), 0);
        s_evaluateMicroTriangleTransparencies.set(
            s_ommModule, "evaluateMicroTriangleTransparencies", cudau::dim3(32), 0);
        isInitialized = true;
    }

    size_t size = 0;
    // ommSizes / ommOffsets
    size += sizeof(uint64_t) * (maxNumTriangles + 1);
    // perTriInfos
    size += sizeof(uint32_t) * ((maxNumTriangles + 3) / 4);
    // counter
    size += sizeof(uint32_t);
    // ommFormatCounts
    size += sizeof(uint32_t) * shared::NumOMMFormats;
    // scratchMemForScan
    size_t sizeOfScratchMemForScan;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, sizeOfScratchMemForScan,
        nullptr, nullptr, maxNumTriangles + 1);
    size += sizeOfScratchMemForScan;

    return size;
}

void countOMMFormats(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    const cudau::Buffer &scratchMem,
    uint32_t ommFormatCounts[shared::NumOMMFormats], uint64_t* rawOmmArraySize) {
    CUstream stream = 0;

    CUdeviceptr scratchMemBase = scratchMem.getCUdeviceptr();
    uint64_t curScratchMemOffset = 0;
    CUdeviceptr ommSizes = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint64_t) * (numTriangles + 1);
    CUdeviceptr perTriInfos = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * ((numTriangles + 3) / 4);
    CUdeviceptr counter = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t);
    CUdeviceptr ommFormatCountsOnDevice = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * shared::NumOMMFormats;
    CUdeviceptr scratchMemForScan = scratchMemBase + curScratchMemOffset;
    size_t sizeOfScratchMemForScan;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, sizeOfScratchMemForScan,
        nullptr, nullptr, numTriangles + 1);
    //curScratchMemOffset += sizeOfScratchMemForScan;

    CUDADRV_CHECK(cuMemsetD32Async(perTriInfos, 0, (numTriangles + 3) / 4, stream));
    CUDADRV_CHECK(cuMemsetD32Async(ommFormatCountsOnDevice, 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));

    maxSubdivLevel = std::max(minSubdivLevel, maxSubdivLevel);
    s_countOMMFormats(
        stream, cudau::dim3(1024),
        texCoords, vertexStride, triangles, triangleStride, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        minSubdivLevel, maxSubdivLevel, subdivLevelBias,
        counter,
        ommFormatCountsOnDevice, perTriInfos, ommSizes);

    cubd::DeviceScan::ExclusiveSum(
        reinterpret_cast<void*>(scratchMemForScan), sizeOfScratchMemForScan,
        reinterpret_cast<uint64_t*>(ommSizes), reinterpret_cast<uint64_t*>(ommSizes),
        numTriangles + 1, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    CUDADRV_CHECK(cuMemcpyDtoH(
        ommFormatCounts, ommFormatCountsOnDevice, sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        rawOmmArraySize, ommSizes + sizeof(uint64_t) * numTriangles, sizeof(uint64_t)));
}

void generateOMMArray(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::Buffer &scratchMem,
    const cudau::Buffer &ommArray, const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer, uint32_t ommIndexSize) {
    CUstream stream = 0;

    CUdeviceptr scratchMemBase = scratchMem.getCUdeviceptr();
    uint64_t curScratchMemOffset = 0;
    CUdeviceptr ommOffsets = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint64_t) * (numTriangles + 1);
    CUdeviceptr perTriInfos = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * ((numTriangles + 3) / 4);
    CUdeviceptr counter = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t);
    CUdeviceptr ommFormatCountsOnDevice = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * shared::NumOMMFormats;
    CUdeviceptr scratchMemForScan = scratchMemBase + curScratchMemOffset;
    size_t sizeOfScratchMemForScan;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, sizeOfScratchMemForScan,
        nullptr, nullptr, numTriangles + 1);
    //curScratchMemOffset += sizeOfScratchMemForScan;

    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));
    s_createOMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        perTriInfos, ommOffsets, numTriangles,
        counter,
        ommDescs, ommIndexBuffer, ommIndexSize);
    //CUDADRV_CHECK(cuStreamSynchronize(stream));
    //uint32_t counterOnHost;
    //counter.read(&counterOnHost, 1);
    //std::vector<OptixOpacityMicromapDesc> ommDescsOnHost = ommDescs;
    //std::vector<int16_t> indicesOnHost;
    //ommIndexBuffer.read(indicesOnHost);

    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        texCoords, vertexStride, triangles, triangleStride, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        perTriInfos, ommOffsets, counter, ommArray);

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
