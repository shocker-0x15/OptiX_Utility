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
    const OMMGeneratorContext &context,
    uint32_t ommFormatCounts[shared::NumOMMFormats], uint64_t* rawOmmArraySize) {
    CUstream stream = 0;

    CUdeviceptr scratchMemBase = context.scratchMem;
    uint64_t curScratchMemOffset = 0;
    CUdeviceptr ommSizes = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint64_t) * (context.numTriangles + 1);
    CUdeviceptr perTriInfos = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * ((context.numTriangles + 3) / 4);
    CUdeviceptr counter = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t);
    CUdeviceptr ommFormatCountsOnDevice = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * shared::NumOMMFormats;
    CUdeviceptr scratchMemForScan = scratchMemBase + curScratchMemOffset;
    size_t sizeOfScratchMemForScan;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, sizeOfScratchMemForScan,
        nullptr, nullptr, context.numTriangles + 1);
    //curScratchMemOffset += sizeOfScratchMemForScan;

    CUDADRV_CHECK(cuMemsetD32Async(perTriInfos, 0, (context.numTriangles + 3) / 4, stream));
    CUDADRV_CHECK(cuMemsetD32Async(ommFormatCountsOnDevice, 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));

    auto maxSubdivLevel =
        static_cast<shared::OMMFormat>(std::max(context.minSubdivLevel, context.maxSubdivLevel));
    s_countOMMFormats(
        stream, cudau::dim3(1024),
        context.texCoords, context.vertexStride,
        context.triangles, context.triangleStride, context.numTriangles,
        context.texture, context.texSize, context.numChannels, context.alphaChannelIndex,
        context.minSubdivLevel, maxSubdivLevel, context.subdivLevelBias,
        static_cast<bool>(context.useIndexBuffer), counter,
        ommFormatCountsOnDevice, perTriInfos, ommSizes);

    cubd::DeviceScan::ExclusiveSum(
        reinterpret_cast<void*>(scratchMemForScan), sizeOfScratchMemForScan,
        reinterpret_cast<uint64_t*>(ommSizes), reinterpret_cast<uint64_t*>(ommSizes),
        context.numTriangles + 1, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    CUDADRV_CHECK(cuMemcpyDtoH(
        ommFormatCounts, ommFormatCountsOnDevice, sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        rawOmmArraySize, ommSizes + sizeof(uint64_t) * context.numTriangles, sizeof(uint64_t)));
}

void generateOMMArray(
    const OMMGeneratorContext &context,
    const cudau::Buffer &ommArray,
    const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer) {
    CUstream stream = 0;

    CUdeviceptr scratchMemBase = context.scratchMem;
    uint64_t curScratchMemOffset = 0;
    CUdeviceptr ommOffsets = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint64_t) * (context.numTriangles + 1);
    CUdeviceptr perTriInfos = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * ((context.numTriangles + 3) / 4);
    CUdeviceptr counter = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t);
    CUdeviceptr ommFormatCountsOnDevice = scratchMemBase + curScratchMemOffset;
    curScratchMemOffset += sizeof(uint32_t) * shared::NumOMMFormats;
    CUdeviceptr scratchMemForScan = scratchMemBase + curScratchMemOffset;
    size_t sizeOfScratchMemForScan;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, sizeOfScratchMemForScan,
        nullptr, nullptr, context.numTriangles + 1);
    //curScratchMemOffset += sizeOfScratchMemForScan;

    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));
    s_createOMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(context.numTriangles),
        perTriInfos, ommOffsets, context.numTriangles,
        static_cast<bool>(context.useIndexBuffer), counter,
        ommDescs, ommIndexBuffer, context.indexSize);

    CUDADRV_CHECK(cuMemsetD32Async(counter, 0, 1, stream));
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        context.texCoords, context.vertexStride,
        context.triangles, context.triangleStride, context.numTriangles,
        context.texture, context.texSize, context.numChannels, context.alphaChannelIndex,
        perTriInfos, ommOffsets, counter, ommArray);

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
