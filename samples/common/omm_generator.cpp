#include "omm_generator.h"
#include "../../ext/cubd/cubd.h"

static CUmodule s_ommModule;
static cudau::Kernel s_countOMMFormats;
static cudau::Kernel s_createOMMDescriptors;
static cudau::Kernel s_evaluateMicroTriangleTransparencies;

void countOMMFormats(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &scratchMemForScan,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommOffsets) {
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

    CUstream stream = 0;

    counter.fill(0, stream);
    ommFormatCounts.fill(0, stream);

    maxSubdivLevel = std::max(minSubdivLevel, maxSubdivLevel);
    s_countOMMFormats(
        stream, cudau::dim3(1024),
        texCoords, vertexStride, triangles, triangleStride, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        minSubdivLevel, maxSubdivLevel, subdivLevelBias,
        counter,
        ommFormatCounts, ommOffsets);

    size_t sizeOfScratchMemForScan = scratchMemForScan.sizeInBytes();
    cubd::DeviceScan::ExclusiveSum(
        scratchMemForScan.getDevicePointer(), sizeOfScratchMemForScan,
        ommOffsets.getDevicePointer(), ommOffsets.getDevicePointer(),
        numTriangles + 1, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}

void generateOMMArray(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint64_t> &ommOffsets,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &ommArray, const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer, uint32_t ommIndexSize) {
    CUstream stream = 0;

    counter.fill(0, stream);
    s_createOMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        ommOffsets, numTriangles,
        counter,
        ommDescs, ommIndexBuffer, ommIndexSize);

    counter.fill(0, stream);
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        texCoords, vertexStride, triangles, triangleStride, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        ommOffsets, counter, ommArray);

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
