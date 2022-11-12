#include "omm_generator.h"
#include "../../ext/cubd/cubd.h"

static CUmodule s_ommModule;
static cudau::Kernel s_evaluateTriangleTransparencies;
static cudau::Kernel s_createOMMDescriptors;
static cudau::Kernel s_evaluateMicroTriangleTransparencies;

void evaluatePerTriangleStates(
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &scratchMemForScan,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommOffsets) {
    static bool isInitialized = false;
    if (!isInitialized) {
        CUDADRV_CHECK(cuModuleLoad(
            &s_ommModule,
            (getExecutableDirectory() / "opacity_micro_map/ptxes/omm_kernels.ptx").string().c_str()));
        s_evaluateTriangleTransparencies.set(
            s_ommModule, "evaluateTriangleTransparencies", cudau::dim3(32), 0);
        s_createOMMDescriptors.set(
            s_ommModule, "createOMMDescriptors", cudau::dim3(32), 0);
        s_evaluateMicroTriangleTransparencies.set(
            s_ommModule, "evaluateMicroTriangleTransparencies", cudau::dim3(32), 0);
        isInitialized = true;
    }

    CUstream stream = 0;

    counter.fill(0, stream);
    ommFormatCounts.fill(0, stream);

    s_evaluateTriangleTransparencies(
        stream, cudau::dim3(1024),
        vertices, triangles, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
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
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
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
    //CUDADRV_CHECK(cuStreamSynchronize(stream));
    //std::vector<OptixOpacityMicromapDesc> ommDescsOnHost = ommDescs;
    //std::vector<uint16_t> ommIndicesOnHost(numTriangles);
    //ommIndexBuffer.read(ommIndicesOnHost);
    //uint32_t numOmms;
    //counter.read(&numOmms, 1u);

    counter.fill(0, stream);
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        vertices, triangles, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        ommOffsets, counter, ommArray);

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
