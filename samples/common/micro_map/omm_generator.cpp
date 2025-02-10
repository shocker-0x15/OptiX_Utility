﻿#include "omm_generator_private.h"
#include "../../../ext/cubd/cubd.h"

extern cudau::Kernel g_extractTexCoords;
extern cudau::Kernel g_testIfTCTupleIsUnique;

static CUmodule s_ommModule;
static cudau::Kernel s_countOMMFormats;
static cudau::Kernel s_fillNonUniqueEntries;
static cudau::Kernel s_createOMMDescriptors;
static cudau::Kernel s_evaluateMicroTriangleTransparencies;
static cudau::Kernel s_copyOpacityMicroMaps;

static bool enableDebugPrint = false;

// TODO: Overlap scratch memory allocations that do not overlap others in life time.
size_t getScratchMemSizeForOMMGenerator(uint32_t numTriangles) {
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
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr texCoords, uint32_t vertexStride, uint32_t numVertices,
    CUdeviceptr triangles, uint32_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    OMMGeneratorContext* context) {
    static bool isInitialized = false;
    if (!isInitialized) {
        initializeMicroMapGeneratorKernels(ptxDirPath);

        CUDADRV_CHECK(cuModuleLoad(
            &s_ommModule,
            (ptxDirPath / "omm_kernels.ptx").string().c_str()));
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

    context->internalState.resize(sizeof(Context), 0);
    auto &_context = *reinterpret_cast<Context*>(context->internalState.data());
    _context.texCoords = shared::StridedBuffer<float2>(texCoords, numVertices, vertexStride);
    _context.triangles = shared::StridedBuffer<shared::Triangle>(triangles, numTriangles, triangleStride);

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

    const uint32_t numTriangles = _context.triangles.numElements;

    // JP: 三角形ごと3頂点のテクスチャー座標を抽出する。
    // EN: Extract texture coordinates of three vertices for each triangle.
    g_extractTexCoords.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.texCoords, _context.triangles,
        _context.triTcTuples, _context.triIndices);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        read(triTcTuples, _context.triTcTuples);
        read(triIndices, _context.triIndices);
        printf("");
    }

    // JP: テクスチャー座標と三角形インデックスの配列をソートする。
    // EN: Sort the arrays of texture coordinates and triangle indices.
    cubScratchMemSize = _context.memSizeForSortTuples;
    __sortTriTexCoordTuples(
        _context.triTcTuples, _context.triIndices, numTriangles,
        reinterpret_cast<void*>(_context.memForSortTuples), _context.memSizeForSortTuples);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        read(triTcTuples, _context.triTcTuples);
        read(triIndices, _context.triIndices);
        printf("");
    }

    // JP: 同じテクスチャー座標を持つ連続する要素の中から先頭の要素を見つける。
    // EN: Find the head elements in the consecutive elements with the same texture coordinates.
    g_testIfTCTupleIsUnique.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.triTcTuples, _context.refTupleIndices, numTriangles);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        std::vector<uint32_t> refTupleIndices(numTriangles);
        read(triTcTuples, _context.triTcTuples);
        read(triIndices, _context.triIndices);
        read(refTupleIndices, _context.refTupleIndices);
        printf("");
    }

    // JP: 各三角形が同じテクスチャー座標を持つ先頭要素を発見できるようにインデックスを生成する。
    // EN: Generate indices so that each triangle can find the head element with the same texture coordinates.
    cubScratchMemSize = _context.memSizeForScanRefTupleIndices;
    cubd::DeviceScan::InclusiveMax(
        reinterpret_cast<void*>(_context.memForScanRefTupleIndices), cubScratchMemSize,
        _context.refTupleIndices, _context.refTupleIndices,
        numTriangles, stream);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriTexCoordTuple> triTcTuples(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        std::vector<uint32_t> refTupleIndices(numTriangles);
        read(triTcTuples, _context.triTcTuples);
        read(triIndices, _context.triIndices);
        read(refTupleIndices, _context.refTupleIndices);
        printf("");
    }

    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.perTriInfos), 0, (numTriangles + 3) / 4, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInOmmArray), 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInMesh), 0, shared::NumOMMFormats, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));

    // JP: 先頭である三角形においてラスタライズを行いOMMのメタデータを決定する。
    //     2つのヒストグラムのカウントも行う。
    // EN: Rasterize each of head triangles and determine the OMM meta data.
    //     Count for the two histograms as well.
    auto maxSubdivLevel =
        static_cast<shared::OMMFormat>(std::max(_context.minSubdivLevel, _context.maxSubdivLevel));
    s_countOMMFormats(
        stream, cudau::dim3(1024),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        numTriangles,
        _context.texture, _context.texSize, _context.numChannels, _context.alphaChannelIndex,
        _context.minSubdivLevel, maxSubdivLevel, _context.subdivLevelBias,
        _context.useIndexBuffer, _context.counter,
        _context.histInOmmArray, _context.histInMesh,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes);

    // JP: 先頭ではない三角形においてメタデータのコピーを行う。
    // EN: Copy meta data for non-head triangles.
    s_fillNonUniqueEntries.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        numTriangles, _context.useIndexBuffer,
        _context.histInOmmArray, _context.histInMesh,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<uint32_t> histInOmmArray(shared::NumOMMFormats);
        std::vector<uint32_t> histInMesh(shared::NumOMMFormats);
        std::vector<uint32_t> perTriInfos((numTriangles + 3) / 4);
        std::vector<uint32_t> hasOmmFlags(numTriangles);
        std::vector<uint64_t> ommSizes(numTriangles + 1);
        read(histInOmmArray, _context.histInOmmArray);
        read(histInMesh, _context.histInMesh);
        read(perTriInfos, _context.perTriInfos);
        read(hasOmmFlags, _context.hasOmmFlags);
        read(ommSizes, _context.ommSizes);
        static bool printPerTriInfos = false;
        if (printPerTriInfos) {
            for (uint32_t triIdx = 0; triIdx < numTriangles; ++triIdx) {
                const uint32_t triInfoBinIdx = triIdx / 4;
                const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
                shared::PerTriInfo triInfo = {};
                triInfo.asUInt = (perTriInfos[triInfoBinIdx] >> offsetInTriInfoBin) & 0xFF;
                hpprintf("%5u: state %u, level %u\n", triIdx, triInfo.state, triInfo.level);
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
        numTriangles + 1, stream);
    cubScratchMemSize = _context.memSizeForScanHasOmmFlags;
    cubd::DeviceScan::ExclusiveSum(
      reinterpret_cast<void*>(_context.memForScanHasOmmFlags), cubScratchMemSize,
      _context.hasOmmFlags, _context.hasOmmFlags,
      numTriangles, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    CUDADRV_CHECK(cuMemcpyDtoH(
        histInOmmArray, reinterpret_cast<uintptr_t>(_context.histInOmmArray),
        sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        histInMesh, reinterpret_cast<uintptr_t>(_context.histInMesh),
        sizeof(uint32_t) * shared::NumOMMFormats));
    CUDADRV_CHECK(cuMemcpyDtoH(
        rawOmmArraySize, reinterpret_cast<uintptr_t>(&_context.ommSizes[numTriangles]),
        sizeof(uint64_t)));
}

void generateOMMArray(
    const OMMGeneratorContext &context,
    const optixu::BufferView &ommArray,
    const optixu::BufferView &ommDescs,
    const optixu::BufferView &ommIndexBuffer) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());

    const uint32_t numTriangles = _context.triangles.numElements;

    // JP: OMMデスクリプターと各三角形のOMMインデックスを計算する。
    // EN: Compute the OMM descriptor and the OMM index for each triangle.
    s_createOMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.refTupleIndices, _context.triIndices,
        _context.perTriInfos, _context.hasOmmFlags, _context.ommSizes, numTriangles,
        _context.useIndexBuffer,
        shared::StridedBuffer<OptixOpacityMicromapDesc>(
            ommDescs.getCUdeviceptr(),
            static_cast<uint32_t>(ommDescs.numElements()),
            ommDescs.stride()),
        ommIndexBuffer.getCUdeviceptr(), _context.indexSize);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));

        std::vector<OptixOpacityMicromapDesc> ommDescsOnHost(ommDescs.numElements());
        {
            std::vector<uint8_t> stridedOmmDescs(ommDescs.sizeInBytes());
            CUDADRV_CHECK(cuMemcpyDtoH(
                stridedOmmDescs.data(),
                ommDescs.getCUdeviceptr(),
                stridedOmmDescs.size()));
            for (uint32_t i = 0; i < ommDescs.numElements(); ++i) {
                ommDescsOnHost[i] = reinterpret_cast<OptixOpacityMicromapDesc &>(
                    stridedOmmDescs[ommDescs.stride() * i]);
            }
        }

        if (_context.useIndexBuffer) {
            std::vector<uint8_t> stridedOmmIndices(ommIndexBuffer.sizeInBytes());
            CUDADRV_CHECK(cuMemcpyDtoH(
                stridedOmmIndices.data(),
                ommIndexBuffer.getCUdeviceptr(),
                stridedOmmIndices.size()));
            if (_context.indexSize == 4) {
                std::vector<int32_t> ommIndices(numTriangles);
                for (uint32_t i = 0; i < numTriangles; ++i)
                    ommIndices[i] = reinterpret_cast<int32_t &>(stridedOmmIndices[ommIndexBuffer.stride() * i]);
                hpprintf("");
            }
            else if (_context.indexSize == 2) {
                std::vector<int16_t> ommIndices(numTriangles);
                for (uint32_t i = 0; i < numTriangles; ++i)
                    ommIndices[i] = reinterpret_cast<int16_t &>(stridedOmmIndices[ommIndexBuffer.stride() * i]);
                hpprintf("");
            }
            hpprintf("");
        }
    }

    // JP: 先頭である三角形においてMicro Triangleレベルでラスタライズを行いOMMを生成する。
    // EN: Rasterize each head triangle in micro triangle level to compute the OMM.
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
    s_evaluateMicroTriangleTransparencies(
        stream, cudau::dim3(1024),
        _context.triTcTuples, _context.refTupleIndices, _context.triIndices,
        numTriangles,
        _context.texture, _context.texSize, _context.numChannels, _context.alphaChannelIndex,
        _context.perTriInfos, _context.ommSizes, _context.counter,
        ommArray.getCUdeviceptr());

    if (!_context.useIndexBuffer) {
        // JP: 先頭ではない三角形においてOMMのコピーを行う。
        // EN: Copy the OMMs for non-head triangles.
        CUDADRV_CHECK(cuMemsetD32Async(
            reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
        s_copyOpacityMicroMaps(
            stream, cudau::dim3(1024),
            _context.refTupleIndices, _context.triIndices, _context.ommSizes,
            numTriangles, _context.counter,
            ommArray.getCUdeviceptr());
    }

    CUDADRV_CHECK(cuStreamSynchronize(stream));
}
