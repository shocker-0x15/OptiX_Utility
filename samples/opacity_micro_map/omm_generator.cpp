#include "omm_generator.h"

static CUmodule s_ommModule;
static cudau::Kernel s_evaluateTriangleTransparencies;
static cudau::Kernel s_countOMMFormats;

void evaluatePerTriangleStates(
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint32_t> &transparentCounts,
    const cudau::TypedBuffer<uint32_t> &numPixelsValues,
    const cudau::TypedBuffer<uint32_t> &numFetchedTriangles,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommSizes,
    std::vector<uint32_t>* triStates,
    uint32_t ommFormatCountsOnHost[Shared::NumOMMFormats]) {
    static bool isInitialized = false;
    if (!isInitialized) {
        CUDADRV_CHECK(cuModuleLoad(
            &s_ommModule,
            (getExecutableDirectory() / "opacity_micro_map/ptxes/omm_kernels.ptx").string().c_str()));
        s_evaluateTriangleTransparencies.set(
            s_ommModule, "evaluateTriangleTransparencies", cudau::dim3(32), 0);
        s_countOMMFormats.set(
            s_ommModule, "countOMMFormats", cudau::dim3(32), 0);
        isInitialized = true;
    }

    CUstream stream = 0;

    transparentCounts.fill(0, stream);
    numFetchedTriangles.fill(0, stream);
    ommFormatCounts.fill(0, stream);

    s_evaluateTriangleTransparencies(
        stream, cudau::dim3(1024),
        vertices, triangles, numTriangles,
        texture, texSize, numChannels, alphaChannelIndex,
        numFetchedTriangles,
        transparentCounts, numPixelsValues);

    s_countOMMFormats.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        transparentCounts, numPixelsValues, numTriangles,
        ommFormatCounts, ommSizes);

    /*
    ommSizesをスキャン、三角形ごとのOMMオフセットがわかる。
    三角形ごとのOMM Descが作成できる。
    三角形ごとにOMMを作成する。
    */

    CUDADRV_CHECK(cuStreamSynchronize(stream));
    std::vector<uint32_t> transparentCountsOnHost = transparentCounts;
    std::vector<uint32_t> numPixelsValuesOnHost = numPixelsValues;
    //std::vector<uint32_t> numFetchedTrianglesOnHost = numFetchedTriangles;
    transparentCountsOnHost.resize(numTriangles);
    numPixelsValuesOnHost.resize(numTriangles);
    triStates->resize(numTriangles);
    for (int i = 0; i < numTriangles; ++i) {
        uint32_t trCount = transparentCountsOnHost[i];
        uint32_t numPixels = numPixelsValuesOnHost[i];
        float trRatio = static_cast<float>(trCount) / numPixels;
        //hpprintf(
        //    "%5u: %6u / %6u, %4.1f%%\n", i,
        //    transparentCountsOnHost[i], numPixelsValuesOnHost[i],
        //    100 * trRatio);

        // JP: OptiXのenumを流用する。
        // EN: Reuse OptiX's enum.
        if (trCount == 0)
            (*triStates)[i] = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
        else if (trCount == numPixels)
            (*triStates)[i] = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
        else if (trRatio < 0.5f)
            (*triStates)[i] = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
        else
            (*triStates)[i] = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
    }

    ommFormatCounts.read(ommFormatCountsOnHost, Shared::NumOMMFormats, stream);
}
