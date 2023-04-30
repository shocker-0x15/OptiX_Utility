#include "dmm_generator_private.h"
#include "../../ext/cubd/cubd.h"

static CUmodule s_mmModule;
static CUmodule s_ommModule;
static cudau::Kernel s_extractTexCoords;
static cudau::Kernel s_testIfTCTupleIsUnique;
static cudau::Kernel s_countOMMFormats;
static cudau::Kernel s_fillNonUniqueEntries;
static cudau::Kernel s_createOMMDescriptors;
static cudau::Kernel s_evaluateMicroTriangleTransparencies;
static cudau::Kernel s_copyOpacityMicroMaps;

static bool enableDebugPrint = false;

template <typename T = void>
inline auto allocate(uintptr_t &curOffset, size_t numElems = 1, size_t alignment = alignof(T))
-> std::conditional_t<std::is_same_v<T, void>, uintptr_t, T*> {
    uint64_t mask = alignment - 1;
    uintptr_t ret = (curOffset + mask) & ~mask;
    if constexpr (std::is_same_v<T, void>) {
        curOffset = ret + numElems;
        return ret;
    }
    else {
        curOffset = ret + sizeof(T) * numElems;
        return reinterpret_cast<T*>(ret);
    }
}

size_t getScratchMemSizeForDMMGenerator(uint32_t numTriangles) {
    static bool isInitialized = false;
    if (!isInitialized) {
        CUDADRV_CHECK(cuModuleLoad(
            &s_mmModule,
            (getExecutableDirectory() / "opacity_micro_map/ptxes/micro_map_kernels.ptx").string().c_str()));
        s_extractTexCoords.set(
            s_mmModule, "extractTexCoords", cudau::dim3(32), 0);
        s_testIfTCTupleIsUnique.set(
            s_mmModule, "testIfTCTupleIsUnique", cudau::dim3(32), 0);

        CUDADRV_CHECK(cuModuleLoad(
            &s_ommModule,
            (getExecutableDirectory() / "opacity_micro_map/ptxes/omm_kernels.ptx").string().c_str()));
        s_countOMMFormats.set(
            s_ommModule, "countOMMFormats", cudau::dim3(32), 0);
        s_fillNonUniqueEntries.set(
            s_ommModule, "fillNonUniqueEntries", cudau::dim3(32), 0);
        s_createOMMDescriptors.set(
            s_ommModule, "createOMMDescriptors", cudau::dim3(32), 0);
        s_evaluateMicroTriangleTransparencies.set(
            s_ommModule, "evaluateMicroTriangleTransparencies", cudau::dim3(32), 0);
        s_copyOpacityMicroMaps.set(
            s_ommModule, "copyOpacityMicroMaps", cudau::dim3(32), 0);
        isInitialized = true;
    }

    size_t curOffset = 0;

    // triTcTuples
    allocate<shared::TriTexCoordTuple>(curOffset, numTriangles);
    // triIndices
    allocate<uint32_t>(curOffset, numTriangles);
    size_t memSizeForSortTuples = __getScratchMemSizeForSortTriTexCoordTuples(numTriangles);
    // memForSortTuples
    allocate(curOffset, memSizeForSortTuples, alignof(uint64_t));
    // refTupleIndices
    allocate<uint32_t>(curOffset, numTriangles);
    size_t memSizeForScanRefTupleIndices;
    cubd::DeviceScan::InclusiveMax<const uint32_t*, uint32_t*>(
        nullptr, memSizeForScanRefTupleIndices,
        nullptr, nullptr, numTriangles);
    // memForScanRefTupleIndices
    allocate(curOffset, memSizeForScanRefTupleIndices, alignof(uint64_t));

    // ommSizes
    allocate<uint64_t>(curOffset, numTriangles + 1);
    // hasOmmFlags
    allocate<uint32_t>(curOffset, numTriangles);
    // perTriInfos
    allocate<uint32_t>(curOffset, (numTriangles + 3) / 4);
    // counter
    allocate<uint32_t>(curOffset);
    // histInOmmArray
    allocate<uint32_t>(curOffset, shared::NumOMMFormats);
    // histInMesh
    allocate<uint32_t>(curOffset, shared::NumOMMFormats);
    // scratchMemForScan
    size_t memSizeForScanOmmSizes;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, memSizeForScanOmmSizes,
        nullptr, nullptr, numTriangles + 1);
    size_t memSizeForScanHasOmmFlags;
    cubd::DeviceScan::ExclusiveSum<const uint32_t*, uint32_t*>(
        nullptr, memSizeForScanHasOmmFlags,
        nullptr, nullptr, numTriangles);
    allocate(curOffset, std::max(memSizeForScanOmmSizes, memSizeForScanHasOmmFlags), alignof(uint64_t));

    return curOffset;
}

void initializeDMMGeneratorContext(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIndex,
    shared::DMMFormat minSubdivLevel, shared::DMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    DMMGeneratorContext* context) {
    context->internalState.resize(sizeof(Context), 0);
    auto &_context = *reinterpret_cast<Context*>(context->internalState.data());
    _context.texCoords = texCoords;
    _context.vertexStride = vertexStride;
    _context.triangles = triangles;
    _context.triangleStride = triangleStride;
    _context.numTriangles = numTriangles;

    _context.texture = texture;
    _context.texSize = texSize;
    _context.numChannels = numChannels;
    _context.alphaChannelIndex = heightChannelIndex;

    _context.minSubdivLevel = minSubdivLevel;
    _context.maxSubdivLevel = maxSubdivLevel;
    _context.subdivLevelBias = subdivLevelBias;

    _context.useIndexBuffer = useIndexBuffer;
    _context.indexSize = indexSize;

    _context.scratchMem = scratchMem;
    _context.scratchMemSize = scratchMemSize;

    size_t curScratchMemHead = scratchMem;
}
