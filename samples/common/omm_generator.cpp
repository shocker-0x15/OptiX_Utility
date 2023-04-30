#include "omm_generator_private.h"
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

size_t getScratchMemSizeForOMMGenerator(uint32_t numTriangles) {
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

void initializeOMMGeneratorContext(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    OMMGeneratorContext* context) {
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
    _context.alphaChannelIndex = alphaChannelIndex;

    _context.minSubdivLevel = minSubdivLevel;
    _context.maxSubdivLevel = maxSubdivLevel;
    _context.subdivLevelBias = subdivLevelBias;

    _context.useIndexBuffer = useIndexBuffer;
    _context.indexSize = indexSize;

    _context.scratchMem = scratchMem;
    _context.scratchMemSize = scratchMemSize;

    size_t curScratchMemHead = scratchMem;

    _context.triTcTuples = allocate<shared::TriTexCoordTuple>(curScratchMemHead, numTriangles);
    _context.triIndices = allocate<uint32_t>(curScratchMemHead, numTriangles);
    _context.memSizeForSortTuples = __getScratchMemSizeForSortTriTexCoordTuples(numTriangles);
    _context.memForSortTuples = allocate(
        curScratchMemHead, _context.memSizeForSortTuples, alignof(uint64_t));
    _context.refTupleIndices = allocate<uint32_t>(curScratchMemHead, numTriangles);
    cubd::DeviceScan::InclusiveMax<const uint32_t*, uint32_t*>(
        nullptr, _context.memSizeForScanRefTupleIndices,
        nullptr, nullptr, numTriangles);
    _context.memForScanRefTupleIndices = allocate(
        curScratchMemHead, _context.memSizeForScanRefTupleIndices, alignof(uint64_t));

    _context.ommSizes = allocate<uint64_t>(curScratchMemHead, numTriangles + 1);
    _context.hasOmmFlags = allocate<uint32_t>(curScratchMemHead, numTriangles);
    _context.perTriInfos = allocate<uint32_t>(curScratchMemHead, (numTriangles + 3) / 4);
    _context.counter = allocate<uint32_t>(curScratchMemHead);
    _context.histInOmmArray = allocate<uint32_t>(curScratchMemHead, shared::NumOMMFormats);
    _context.histInMesh = allocate<uint32_t>(curScratchMemHead, shared::NumOMMFormats);
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, _context.memSizeForScanOmmSizes,
        nullptr, nullptr, numTriangles + 1);
    cubd::DeviceScan::ExclusiveSum<const uint32_t*, uint32_t*>(
        nullptr, _context.memSizeForScanHasOmmFlags,
        nullptr, nullptr, numTriangles);
    _context.memForScanOmmSizes = allocate(
        curScratchMemHead, std::max(_context.memForScanOmmSizes, _context.memSizeForScanHasOmmFlags),
        alignof(uint64_t));
    _context.memForScanHasOmmFlags = _context.memForScanOmmSizes;
}

void countOMMFormats(
    const OMMGeneratorContext &context,
    uint32_t histInOmmArray[shared::NumOMMFormats],
    uint32_t histInMesh[shared::NumOMMFormats],
    uint64_t* rawOmmArraySize) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());
    size_t cubScratchMemSize;

    // JP: 三角形ごと3頂点のテクスチャー座標を抽出する。
    // EN: Extract texture coordinates of three vertices for each triangle.
    s_extractTexCoords.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.texCoords, _context.vertexStride,
        _context.triangles, _context.triangleStride, _context.numTriangles,
        _context.triTcTuples, _context.triIndices);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(_context.numTriangles);
        std::vector<uint32_t> triIndices(_context.numTriangles);
        CUDADRV_CHECK(cuMemcpyDtoH(
            triTcTuples.data(), reinterpret_cast<CUdeviceptr>(_context.triTcTuples),
            sizeof(shared::TriTexCoordTuple) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            triIndices.data(), reinterpret_cast<CUdeviceptr>(_context.triIndices),
            sizeof(uint32_t) * _context.numTriangles));
        printf("");
    }

    // JP: テクスチャー座標と三角形インデックスの配列をソートする。
    // EN: Sort the arrays of texture coordinates and triangle indices.
    cubScratchMemSize = _context.memSizeForSortTuples;
    __sortTriTexCoordTuples(
        _context.triTcTuples, _context.triIndices, _context.numTriangles,
        reinterpret_cast<void*>(_context.memForSortTuples), _context.memSizeForSortTuples);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(_context.numTriangles);
        std::vector<uint32_t> triIndices(_context.numTriangles);
        CUDADRV_CHECK(cuMemcpyDtoH(
            triTcTuples.data(), reinterpret_cast<CUdeviceptr>(_context.triTcTuples),
            sizeof(shared::TriTexCoordTuple) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            triIndices.data(), reinterpret_cast<CUdeviceptr>(_context.triIndices),
            sizeof(uint32_t) * _context.numTriangles));
        printf("");
    }

    // JP: 同じテクスチャー座標を持つ連続する要素の中から先頭の要素を見つける。
    // EN: Find the head elements in the consecutive elements with the same texture coordinates.
    s_testIfTCTupleIsUnique.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.triTcTuples, _context.refTupleIndices, _context.numTriangles);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(_context.numTriangles);
        std::vector<uint32_t> triIndices(_context.numTriangles);
        std::vector<uint32_t> refTupleIndices(_context.numTriangles);
        CUDADRV_CHECK(cuMemcpyDtoH(
            triTcTuples.data(), reinterpret_cast<CUdeviceptr>(_context.triTcTuples),
            sizeof(shared::TriTexCoordTuple) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            triIndices.data(), reinterpret_cast<CUdeviceptr>(_context.triIndices),
            sizeof(uint32_t) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            refTupleIndices.data(), reinterpret_cast<CUdeviceptr>(_context.refTupleIndices),
            sizeof(uint32_t) * _context.numTriangles));
        printf("");
    }

    // JP: 各三角形が同じテクスチャー座標を持つ先頭要素を発見できるようにインデックスを生成する。
    // EN: Generate indices so that each triangle can find the head element with the same texture coordinates.
    cubScratchMemSize = _context.memSizeForScanRefTupleIndices;
    cubd::DeviceScan::InclusiveMax(
        reinterpret_cast<void*>(_context.memForScanRefTupleIndices), cubScratchMemSize,
        _context.refTupleIndices, _context.refTupleIndices,
        _context.numTriangles, stream);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(_context.numTriangles);
        std::vector<uint32_t> triIndices(_context.numTriangles);
        std::vector<uint32_t> refTupleIndices(_context.numTriangles);
        CUDADRV_CHECK(cuMemcpyDtoH(
            triTcTuples.data(), reinterpret_cast<CUdeviceptr>(_context.triTcTuples),
            sizeof(shared::TriTexCoordTuple) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            triIndices.data(), reinterpret_cast<CUdeviceptr>(_context.triIndices),
            sizeof(uint32_t) * _context.numTriangles));
        CUDADRV_CHECK(cuMemcpyDtoH(
            refTupleIndices.data(), reinterpret_cast<CUdeviceptr>(_context.refTupleIndices),
            sizeof(uint32_t) * _context.numTriangles));
        printf("");
    }

    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.perTriInfos), 0, (_context.numTriangles + 3) / 4, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInOmmArray), 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInMesh), 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));

    // JP: 先頭である三角形においてラスタライズを行いOMMのメタデータを決定する。
    //     2つのヒストグラムのカウントも行う。
    // EN: Rasterize each of head triangles and determine the meta data.
    //     Count for the two histograms as well.
    auto maxSubdivLevel =
        static_cast<shared::OMMFormat>(std::max(_context.minSubdivLevel, _context.maxSubdivLevel));
    s_countOMMFormats(
        stream, cudau::dim3(1024),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        _context.numTriangles,
        _context.texture, _context.texSize, _context.numChannels, _context.alphaChannelIndex,
        _context.minSubdivLevel, maxSubdivLevel, _context.subdivLevelBias,
        static_cast<bool>(_context.useIndexBuffer), _context.counter,
        _context.histInOmmArray, _context.histInMesh,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes);

    // JP: 先頭ではない三角形においてメタデータのコピーを行う。
    // EN: Copy meta data for non-head triangles.
    s_fillNonUniqueEntries.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        _context.numTriangles, static_cast<bool>(_context.useIndexBuffer),
        _context.histInOmmArray, _context.histInMesh,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<uint32_t> histInOmmArray(shared::NumOMMFormats);
        std::vector<uint32_t> histInMesh(shared::NumOMMFormats);
        std::vector<uint32_t> perTriInfos((_context.numTriangles + 3) / 4);
        std::vector<uint32_t> hasOmmFlags(_context.numTriangles);
        std::vector<uint64_t> ommSizes(_context.numTriangles + 1);
        CUDADRV_CHECK(cuMemcpyDtoH(
            histInOmmArray.data(), reinterpret_cast<CUdeviceptr>(_context.histInOmmArray),
            histInOmmArray.size() * sizeof(histInOmmArray[0])));
        CUDADRV_CHECK(cuMemcpyDtoH(
            histInMesh.data(), reinterpret_cast<CUdeviceptr>(_context.histInMesh),
            histInMesh.size() * sizeof(histInMesh[0])));
        CUDADRV_CHECK(cuMemcpyDtoH(
            perTriInfos.data(), reinterpret_cast<CUdeviceptr>(_context.perTriInfos),
            perTriInfos.size() * sizeof(perTriInfos[0])));
        CUDADRV_CHECK(cuMemcpyDtoH(
            hasOmmFlags.data(), reinterpret_cast<CUdeviceptr>(_context.hasOmmFlags),
            hasOmmFlags.size() * sizeof(hasOmmFlags[0])));
        CUDADRV_CHECK(cuMemcpyDtoH(
            ommSizes.data(), reinterpret_cast<CUdeviceptr>(_context.ommSizes),
            ommSizes.size() * sizeof(ommSizes[0])));
        static bool printPerTriInfos = false;
        if (printPerTriInfos) {
            for (uint32_t triIdx = 0; triIdx < _context.numTriangles; ++triIdx) {
                const uint32_t triInfoBinIdx = triIdx / 4;
                const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
                shared::PerTriInfo triInfo = {};
                triInfo.asUInt = (perTriInfos[triInfoBinIdx] >> offsetInTriInfoBin) & 0xFF;
                hpprintf("%5u: %u, %u\n", triIdx, triInfo.state, triInfo.level);
            }
        }
        printf("");
    }

    // JP: 各要素のOMMサイズとユニークなOMMを持つか否かのフラグをスキャンして
    //     各OMM・デスクリプターのアドレスを計算する。
    // EN: Scan OMM sizes and flags indicating whether an element has a unique OMM or not
    //     to compute the addresses of each OMM and descriptor.
    cubScratchMemSize = _context.memSizeForScanOmmSizes;
    cubd::DeviceScan::ExclusiveSum(
        reinterpret_cast<void*>(_context.memForScanOmmSizes), cubScratchMemSize,
        _context.ommSizes, _context.ommSizes,
        _context.numTriangles + 1, stream);
    cubScratchMemSize = _context.memSizeForScanHasOmmFlags;
    cubd::DeviceScan::ExclusiveSum(
      reinterpret_cast<void*>(_context.memForScanHasOmmFlags), cubScratchMemSize,
      _context.hasOmmFlags, _context.hasOmmFlags,
      _context.numTriangles, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    CUDADRV_CHECK(cuMemcpyDtoH(
        histInOmmArray, reinterpret_cast<uintptr_t>(_context.histInOmmArray),
        sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        histInMesh, reinterpret_cast<uintptr_t>(_context.histInMesh),
        sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        rawOmmArraySize, reinterpret_cast<uintptr_t>(&_context.ommSizes[_context.numTriangles]),
        sizeof(uint64_t)));
}

void generateOMMArray(
    const OMMGeneratorContext &context,
    const cudau::Buffer &ommArray,
    const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());

    // JP: OMMデスクリプターと各三角形のOMMインデックスを計算する。
    // EN: Compute the OMM descriptors and the OMM index for each triangle.
    s_createOMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.refTupleIndices, _context.triIndices,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes, _context.numTriangles,
        static_cast<bool>(_context.useIndexBuffer),
        ommDescs, ommIndexBuffer, _context.indexSize);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<OptixOpacityMicromapDesc> ommDescsOnHost = ommDescs;
        if (_context.indexSize == 4) {
            std::vector<int32_t> ommIndices(_context.numTriangles);
            if (_context.useIndexBuffer)
                CUDADRV_CHECK(cuMemcpyDtoH(
                    ommIndices.data(), ommIndexBuffer.getCUdeviceptr(), ommIndexBuffer.sizeInBytes()));
        }
        else if (_context.indexSize == 2) {
            std::vector<int16_t> ommIndices(_context.numTriangles);
            if (_context.useIndexBuffer)
                CUDADRV_CHECK(cuMemcpyDtoH(
                    ommIndices.data(), ommIndexBuffer.getCUdeviceptr(), ommIndexBuffer.sizeInBytes()));
        }
        printf("");
    }

    // JP: 先頭である三角形においてMicro Triangleレベルでラスタライズを行いOMMを生成する。
    // EN: Rasterize each head triangle in micro triangle level to compute the OMM.
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        _context.numTriangles,
        _context.texture, _context.texSize, _context.numChannels, _context.alphaChannelIndex,
        _context.perTriInfos, _context.ommSizes, _context.counter,
        ommArray);

    if (!_context.useIndexBuffer) {
        // JP: 先頭ではない三角形においてOMMのコピーを行う。
        // EN: Copy the OMMs for non-head triangles.
        CUDADRV_CHECK(cuMemsetD32Async(
            reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
        s_copyOpacityMicroMaps(
            stream, cudau::dim3(1024),
            _context.refTupleIndices, _context.triIndices, _context.ommSizes,
            _context.numTriangles, _context.counter,
            ommArray);
    }

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
