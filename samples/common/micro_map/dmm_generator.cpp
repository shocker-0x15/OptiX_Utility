#include "dmm_generator_private.h"
#include "../../ext/cubd/cubd.h"

#define VISUALIZE_MICRO_VERTICES_WITH_VDB 0
#if VISUALIZE_MICRO_VERTICES_WITH_VDB
#   include "../vdb_interface.h"
#endif

extern cudau::Kernel g_computeMeshAABB;
extern cudau::Kernel g_finalizeMeshAABB;
extern cudau::Kernel g_initializeHalfEdges;
extern cudau::Kernel g_findTwinHalfEdges;
extern cudau::Kernel g_findTriangleNeighbors;
extern cudau::Kernel g_testIfMicroMapKeyIsUnique;

static CUmodule s_dmmModule;
static cudau::Kernel s_determineTargetSubdivLevels;
static cudau::Kernel s_adjustSubdivLevels;
static cudau::Kernel s_finalizeMicroMapFormats;
static cudau::Kernel s_countDMMFormats;
static cudau::Kernel s_fillNonUniqueEntries;
static cudau::Kernel s_createDMMDescriptors;
static cudau::Kernel s_evaluateMicroVertexHeights;
static cudau::Kernel s_copyDisplacementMicroMaps;

static bool enableDebugPrint = false;

// TODO: Overlap scratch memory allocations that do not overlap others in life time.
size_t getScratchMemSizeForDMMGenerator(uint32_t numTriangles) {
    const uint32_t numHalfEdges = 3 * numTriangles;
    size_t curOffset = 0;

    // directedEdges
    allocate<shared::DirectedEdge>(curOffset, numHalfEdges);
    // halfEdgeIndices
    allocate<uint32_t>(curOffset, numHalfEdges);
    // halfEdges
    allocate<shared::HalfEdge>(curOffset, numHalfEdges);
    size_t memSizeForSortDirectedEdges = __getScratchMemSizeForSortDirectedEdges(numHalfEdges);
    // memForSortDirectedEdges
    allocate(curOffset, memSizeForSortDirectedEdges, alignof(uint64_t));
    // triNeighborLists
    allocate<shared::TriNeighborList>(curOffset, numTriangles);

    // meshAabbAsOrderedInt
    allocate<AABBAsOrderedInt>(curOffset);
    // meshAabb
    allocate<AABB>(curOffset);
    // meshAabbArea
    allocate<float>(curOffset);
    // microMapKeys
    allocate<shared::MicroMapKey>(curOffset, numTriangles);
    // microMapFormats
    allocate<shared::MicroMapFormat>(curOffset, numTriangles);
    // triIndices
    allocate<uint32_t>(curOffset, numTriangles);
    size_t memSizeForSortMicroMapKeys = __getScratchMemSizeForSortMicroMapKeys(numTriangles);
    // memForSortMicroMapKeys
    allocate(curOffset, memSizeForSortMicroMapKeys, alignof(uint64_t));
    // refKeyIndices
    allocate<uint32_t>(curOffset, numTriangles);
    size_t memSizeForScanRefKeyIndices;
    cubd::DeviceScan::InclusiveMax<const uint32_t*, uint32_t*>(
        nullptr, memSizeForScanRefKeyIndices,
        nullptr, nullptr, numTriangles);
    // memForScanRefKeyIndices
    allocate(curOffset, memSizeForScanRefKeyIndices, alignof(uint64_t));

    // dmmSizes
    allocate<uint64_t>(curOffset, numTriangles + 1);
    // hasDmmFlags
    allocate<uint32_t>(curOffset, numTriangles);
    // histInDmmArray
    allocate<uint32_t>(curOffset, shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels);
    // histInMesh
    allocate<uint32_t>(curOffset, shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels);
    // scratchMemForScan
    size_t memSizeForScanDmmSizes;
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, memSizeForScanDmmSizes,
        nullptr, nullptr, numTriangles + 1);
    size_t memSizeForScanHasDmmFlags;
    cubd::DeviceScan::ExclusiveSum<const uint32_t*, uint32_t*>(
        nullptr, memSizeForScanHasDmmFlags,
        nullptr, nullptr, numTriangles);
    allocate(curOffset, std::max(memSizeForScanDmmSizes, memSizeForScanHasDmmFlags), alignof(uint64_t));
    // counter
    allocate<uint32_t>(curOffset);

    return curOffset;
}

void initializeDMMGeneratorContext(
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr positions, CUdeviceptr texCoords, uint32_t vertexStride, uint32_t numVertices,
    CUdeviceptr triangles, uint32_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIndex,
    shared::DMMEncoding maxCompressedFormat,
    shared::DMMSubdivLevel minSubdivLevel, shared::DMMSubdivLevel maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    DMMGeneratorContext* context) {
    static bool isInitialized = false;
    if (!isInitialized) {
        initializeMicroMapGeneratorKernels(ptxDirPath);

        CUDADRV_CHECK(cuModuleLoad(
            &s_dmmModule,
            (ptxDirPath / "dmm_kernels.ptx").string().c_str()));
        s_determineTargetSubdivLevels.set(
            s_dmmModule, "determineTargetSubdivLevels", cudau::dim3(32), 0);
        s_adjustSubdivLevels.set(
            s_dmmModule, "adjustSubdivLevels", cudau::dim3(32), 0);
        s_finalizeMicroMapFormats.set(
            s_dmmModule, "finalizeMicroMapFormats", cudau::dim3(32), 0);
        s_countDMMFormats.set(
            s_dmmModule, "countDMMFormats", cudau::dim3(32), 0);
        s_fillNonUniqueEntries.set(
            s_dmmModule, "fillNonUniqueEntries", cudau::dim3(32), 0);
        s_createDMMDescriptors.set(
            s_dmmModule, "createDMMDescriptors", cudau::dim3(32), 0);
        s_evaluateMicroVertexHeights.set(
            s_dmmModule, "evaluateMicroVertexHeights", cudau::dim3(32), 0);
        s_copyDisplacementMicroMaps.set(
            s_dmmModule, "copyDisplacementMicroMaps", cudau::dim3(32), 0);

        isInitialized = true;
    }

    context->internalState.resize(sizeof(Context), 0);
    auto &_context = *reinterpret_cast<Context*>(context->internalState.data());
    _context.positions = shared::StridedBuffer<float3>(positions, numVertices, vertexStride);
    _context.texCoords = shared::StridedBuffer<float2>(texCoords, numVertices, vertexStride);
    _context.triangles = shared::StridedBuffer<shared::Triangle>(triangles, numTriangles, triangleStride);

    _context.texture = texture;
    _context.texSize = texSize;
    _context.numChannels = numChannels;
    _context.alphaChannelIndex = heightChannelIndex;

    _context.maxCompressedFormat = maxCompressedFormat;
    _context.minSubdivLevel = minSubdivLevel;
    _context.maxSubdivLevel = maxSubdivLevel;
    _context.subdivLevelBias = subdivLevelBias;

    _context.useIndexBuffer = useIndexBuffer;
    _context.indexSize = indexSize;

    _context.scratchMem = scratchMem;
    _context.scratchMemSize = scratchMemSize;

    const uint32_t numHalfEdges = 3 * numTriangles;
    size_t curScratchMemHead = scratchMem;

    _context.directedEdges = allocate<shared::DirectedEdge>(curScratchMemHead, numHalfEdges);
    _context.halfEdgeIndices = allocate<uint32_t>(curScratchMemHead, numHalfEdges);
    _context.halfEdges = allocate<shared::HalfEdge>(curScratchMemHead, numHalfEdges);
    _context.memSizeForSortDirectedEdges = __getScratchMemSizeForSortDirectedEdges(numHalfEdges);
    _context.memForSortDirectedEdges = allocate(
        curScratchMemHead, _context.memSizeForSortDirectedEdges, alignof(uint64_t));
    _context.triNeighborLists = allocate<shared::TriNeighborList>(curScratchMemHead, numTriangles);

    _context.meshAabbAsOrderedInt = allocate<AABBAsOrderedInt>(curScratchMemHead);
    _context.meshAabb = allocate<AABB>(curScratchMemHead);
    _context.meshAabbArea = allocate<float>(curScratchMemHead);
    _context.microMapKeys = allocate<shared::MicroMapKey>(curScratchMemHead, numTriangles);
    _context.microMapFormats = allocate<shared::MicroMapFormat>(curScratchMemHead, numTriangles);
    _context.triIndices = allocate<uint32_t>(curScratchMemHead, numTriangles);
    _context.memSizeForSortMicroMapKeys = __getScratchMemSizeForSortMicroMapKeys(numTriangles);
    _context.memForSortMicroMapKeys = allocate(
        curScratchMemHead, _context.memSizeForSortMicroMapKeys, alignof(uint64_t));
    _context.refKeyIndices = allocate<uint32_t>(curScratchMemHead, numTriangles);
    cubd::DeviceScan::InclusiveMax<const uint32_t*, uint32_t*>(
        nullptr, _context.memSizeForScanRefKeyIndices,
        nullptr, nullptr, numTriangles);
    _context.memForScanRefKeyIndices = allocate(
        curScratchMemHead, _context.memSizeForScanRefKeyIndices, alignof(uint64_t));

    _context.dmmSizes = allocate<uint64_t>(curScratchMemHead, numTriangles + 1);
    _context.hasDmmFlags = allocate<uint32_t>(curScratchMemHead, numTriangles);
    _context.histInDmmArray = allocate<uint32_t>(
        curScratchMemHead, shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels);
    _context.histInMesh = allocate<uint32_t>(
        curScratchMemHead, shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels);
    cubd::DeviceScan::ExclusiveSum<const uint64_t*, uint64_t*>(
        nullptr, _context.memSizeForScanDmmSizes,
        nullptr, nullptr, numTriangles + 1);
    cubd::DeviceScan::ExclusiveSum<const uint32_t*, uint32_t*>(
        nullptr, _context.memSizeForScanHasDmmFlags,
        nullptr, nullptr, numTriangles);
    _context.memForScanDmmSizes = allocate(
        curScratchMemHead, std::max(_context.memForScanDmmSizes, _context.memSizeForScanHasDmmFlags),
        alignof(uint64_t));
    _context.memForScanHasDmmFlags = _context.memForScanDmmSizes;
    _context.counter = allocate<uint32_t>(curScratchMemHead);
}

void countDMMFormats(
    const DMMGeneratorContext &context,
    uint32_t histInDmmArray[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels],
    uint32_t histInMesh[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels],
    uint64_t* rawDmmArraySize) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());
    size_t cubScratchMemSize;

    const uint32_t numTriangles = _context.triangles.numElements;
    const uint32_t numHalfEdges = 3 * numTriangles;

    // JP: ハーフエッジ構造を初期化する。
    // EN: Initialize half-edge data structures.
    g_initializeHalfEdges.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.triangles,
        _context.directedEdges, _context.halfEdgeIndices, _context.halfEdges);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::DirectedEdge> directedEdges(numHalfEdges);
        std::vector<uint32_t> halfEdgeIndices(numHalfEdges);
        std::vector<shared::HalfEdge> halfEdges(numHalfEdges);
        read(directedEdges, _context.directedEdges);
        read(halfEdgeIndices, _context.halfEdgeIndices);
        read(halfEdges, _context.halfEdges);
        hpprintf("");
    }

    // JP: エッジとハーフエッジインデックスの配列をソートする。
    // EN: Sort the arrays of edges and half edge indices.
    cubScratchMemSize = _context.memSizeForSortDirectedEdges;
    __sortDirectedEdges(
        _context.directedEdges, _context.halfEdgeIndices, numHalfEdges,
        reinterpret_cast<void*>(_context.memForSortDirectedEdges), _context.memSizeForSortDirectedEdges);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::DirectedEdge> directedEdges(numHalfEdges);
        std::vector<uint32_t> halfEdgeIndices(numHalfEdges);
        read(directedEdges, _context.directedEdges);
        read(halfEdgeIndices, _context.halfEdgeIndices);
        hpprintf("");
    }

    // JP: 双子のハーフエッジを特定する。
    // EN: Find the twin for each half edge.
    g_findTwinHalfEdges.launchWithThreadDim(
        stream, cudau::dim3(numHalfEdges),
        _context.directedEdges, _context.halfEdgeIndices,
        _context.halfEdges, numHalfEdges);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::DirectedEdge> directedEdges(numHalfEdges);
        std::vector<uint32_t> halfEdgeIndices(numHalfEdges);
        std::vector<shared::HalfEdge> halfEdges(numHalfEdges);
        read(directedEdges, _context.directedEdges);
        read(halfEdgeIndices, _context.halfEdgeIndices);
        read(halfEdges, _context.halfEdges);
        hpprintf("");
    }

    // JP: 各三角形の隣接三角形を特定する。
    // EN: Find the neighbor triangles of each triangle.
    g_findTriangleNeighbors.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.halfEdges, _context.triNeighborLists, numTriangles);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriNeighborList> triNeighborLists(numTriangles);
        read(triNeighborLists, _context.triNeighborLists);
        hpprintf("");
    }

    // JP: メッシュのAABBを計算する。
    // EN: Compute the AABB of the mesh.
    AABBAsOrderedInt initAabb;
    CUDADRV_CHECK(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(_context.meshAabbAsOrderedInt),
        &initAabb, sizeof(initAabb), stream));
    g_computeMeshAABB.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.positions, _context.triangles,
        _context.meshAabbAsOrderedInt);

    // JP: メッシュのAABB計算を完了させる。
    // EN: Complete the mesh AABB computation.
    g_finalizeMeshAABB.launchWithThreadDim(
        stream, cudau::dim3(1),
        _context.meshAabbAsOrderedInt,
        _context.meshAabb, _context.meshAabbArea);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        AABBAsOrderedInt meshAabbAsOrderedInt;
        AABB meshAabb;
        float meshAabbArea;
        CUDADRV_CHECK(cuMemcpyDtoH(
            &meshAabbAsOrderedInt, reinterpret_cast<CUdeviceptr>(_context.meshAabbAsOrderedInt),
            sizeof(meshAabbAsOrderedInt)));
        CUDADRV_CHECK(cuMemcpyDtoH(
            &meshAabb, reinterpret_cast<CUdeviceptr>(_context.meshAabb),
            sizeof(meshAabb)));
        CUDADRV_CHECK(cuMemcpyDtoH(
            &meshAabbArea, reinterpret_cast<CUdeviceptr>(_context.meshAabbArea),
            sizeof(meshAabbArea)));
        hpprintf("");
    }

    // JP: 三角形ごとの目標分割レベルを計算する。
    // EN: Determine the target subdivision level for each triangle.
    auto maxSubdivLevel =
        static_cast<shared::DMMSubdivLevel>(std::max(_context.minSubdivLevel, _context.maxSubdivLevel));
    s_determineTargetSubdivLevels.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.meshAabbArea,
        _context.positions, _context.texCoords, _context.triangles,
        _context.texSize,
        _context.minSubdivLevel, maxSubdivLevel, _context.subdivLevelBias,
        _context.microMapKeys, _context.triIndices);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::MicroMapKey> microMapKeys(numTriangles);
        read(microMapKeys, _context.microMapKeys);
        static bool printPerTriInfos = false;
        if (printPerTriInfos) {
            for (uint32_t triIdx = 0; triIdx < numTriangles; ++triIdx) {
                const shared::MicroMapKey &mmKey = microMapKeys[triIdx];
                hpprintf("%5u: level %u\n", triIdx, mmKey.format.level);
            }
        }
        hpprintf("");
    }

    // JP: DMMでは隣り合う三角形の分割レベル差は1以内である必要があるため、
    //     それを満たすように分割レベルを調整する。
    // EN: Adjust the subdivision level of each triangle so that it meets the condition that
    //     the difference in subdivision levels of neighboring triangle must not exceed one.
    for (uint32_t i = 0; i < 4; ++i) {
        s_adjustSubdivLevels.launchWithThreadDim(
            stream, cudau::dim3(numTriangles),
            _context.triNeighborLists, numTriangles,
            _context.microMapKeys, _context.microMapFormats, i);
    }
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::MicroMapKey> microMapKeys(numTriangles);
        read(microMapKeys, _context.microMapKeys);
        static bool printPerTriInfos = false;
        if (printPerTriInfos) {
            for (uint32_t triIdx = 0; triIdx < numTriangles; ++triIdx) {
                const shared::MicroMapKey &mmKey = microMapKeys[triIdx];
                hpprintf("%5u: level %u\n", triIdx, mmKey.format.level);
            }
        }
        hpprintf("");
    }

    // JP: 各三角形のDMMフォーマットを確定させる。
    // EN: Finalize the DMM format of each triangle.
    s_finalizeMicroMapFormats.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.microMapKeys, _context.microMapFormats, numTriangles,
        _context.maxCompressedFormat);

    // JP: マイクロマップキーとキーインデックスの配列をソートする。
    // EN: Sort the arrays of micro map keys and key indices.
    cubScratchMemSize = _context.memSizeForSortMicroMapKeys;
    __sortMicroMapKeys(
        _context.microMapKeys, _context.triIndices, numTriangles,
        reinterpret_cast<void*>(_context.memForSortMicroMapKeys), _context.memSizeForSortMicroMapKeys);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::MicroMapKey> triMicroMapkeys(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        read(triMicroMapkeys, _context.microMapKeys);
        read(triIndices, _context.triIndices);
        hpprintf("");
    }

    // JP: 同じマイクロマップキーを持つ連続する要素の中から先頭の要素を見つける。
    // EN: Find the head elements in the consecutive elements with the same micro map key.
    g_testIfMicroMapKeyIsUnique.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.microMapKeys, _context.refKeyIndices, numTriangles);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::MicroMapKey> microMapKeys(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        std::vector<uint32_t> refKeyIndices(numTriangles);
        read(microMapKeys, _context.microMapKeys);
        read(triIndices, _context.triIndices);
        read(refKeyIndices, _context.refKeyIndices);
        hpprintf("");
    }

    // JP: 各三角形が同じマイクロマップキーを持つ先頭要素を発見できるようにインデックスを生成する。
    // EN: Generate indices so that each triangle can find the head element with the same micro map key.
    cubScratchMemSize = _context.memSizeForScanRefKeyIndices;
    cubd::DeviceScan::InclusiveMax(
        reinterpret_cast<void*>(_context.memForScanRefKeyIndices), cubScratchMemSize,
        _context.refKeyIndices, _context.refKeyIndices,
        numTriangles, stream);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::MicroMapKey> microMapKeys(numTriangles);
        std::vector<uint32_t> triIndices(numTriangles);
        std::vector<uint32_t> refKeyIndices(numTriangles);
        read(microMapKeys, _context.microMapKeys);
        read(triIndices, _context.triIndices);
        read(refKeyIndices, _context.refKeyIndices);
        hpprintf("");
    }

    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInDmmArray), 0,
        shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels, stream));
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.histInMesh), 0,
        shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels, stream));

    // JP: 先頭である三角形においてラスタライズを行いDMMのメタデータを決定する。
    //     2つのヒストグラムのカウントも行う。
    // EN: Rasterize each of head triangles and determine the DMM meta data.
    //     Count for the two histograms as well.
    s_countDMMFormats.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.microMapKeys, _context.refKeyIndices, _context.triIndices,
        numTriangles,
        _context.histInDmmArray, _context.histInMesh,
        _context.hasDmmFlags, _context.dmmSizes);

    // JP: 先頭ではない三角形においてメタデータのコピーを行う。
    // EN: Copy meta data for non-head triangles.
    s_fillNonUniqueEntries.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.microMapKeys, _context.refKeyIndices, _context.triIndices,
        numTriangles, _context.useIndexBuffer,
        _context.histInDmmArray, _context.histInMesh,
        _context.hasDmmFlags, _context.dmmSizes);

    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<uint32_t> hasDmmFlags(numTriangles);
        std::vector<uint64_t> dmmSizes(numTriangles + 1);
        CUDADRV_CHECK(cuMemcpyDtoH(
            histInDmmArray, reinterpret_cast<uintptr_t>(_context.histInDmmArray),
            sizeof(uint32_t) * shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels));
        CUDADRV_CHECK(cuMemcpyDtoH(
            histInMesh, reinterpret_cast<uintptr_t>(_context.histInMesh),
            sizeof(uint32_t) * shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels));
        read(hasDmmFlags, _context.hasDmmFlags);
        read(dmmSizes, _context.dmmSizes);
        hpprintf("");
    }

    // JP: 各要素のDMMサイズとユニークなDMMを持つか否かのフラグをスキャンして
    //     各DMM・デスクリプターのアドレスを計算する。
    // EN: Scan DMM sizes and flags indicating whether an element has a unique DMM or not
    //     to compute the addresses of each DMM and descriptor.
    cubScratchMemSize = _context.memSizeForScanDmmSizes;
    cubd::DeviceScan::ExclusiveSum(
        reinterpret_cast<void*>(_context.memForScanDmmSizes), cubScratchMemSize,
        _context.dmmSizes, _context.dmmSizes,
        numTriangles + 1, stream);
    cubScratchMemSize = _context.memSizeForScanHasDmmFlags;
    cubd::DeviceScan::ExclusiveSum(
        reinterpret_cast<void*>(_context.memForScanHasDmmFlags), cubScratchMemSize,
        _context.hasDmmFlags, _context.hasDmmFlags,
        numTriangles, stream);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    CUDADRV_CHECK(cuMemcpyDtoH(
        histInDmmArray, reinterpret_cast<uintptr_t>(_context.histInDmmArray),
        sizeof(uint32_t) * shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels));
    CUDADRV_CHECK(cuMemcpyDtoH(
        histInMesh, reinterpret_cast<uintptr_t>(_context.histInMesh),
        sizeof(uint32_t) * shared::NumDMMEncodingTypes * shared::NumDMMSubdivLevels));
    CUDADRV_CHECK(cuMemcpyDtoH(
        rawDmmArraySize, reinterpret_cast<uintptr_t>(&_context.dmmSizes[numTriangles]),
        sizeof(uint64_t)));
}

void generateDMMArray(
    const DMMGeneratorContext &context,
    const cudau::Buffer &dmmArray,
    const cudau::TypedBuffer<OptixDisplacementMicromapDesc> &dmmDescs,
    const cudau::Buffer &dmmIndexBuffer,
    const cudau::Buffer &dmmTriangleFlagsBuffer,
    const cudau::Buffer &debugSubdivLevelBuffer) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());

    const uint32_t numTriangles = _context.triangles.numElements;

    // JP: DMMデスクリプターと各三角形のDMMインデックス、フラグを計算する。
    // EN: Compute the DMM descriptor, DMM index and flags for each triangle.
    s_createDMMDescriptors.launchWithThreadDim(
        stream, cudau::dim3(numTriangles),
        _context.refKeyIndices, _context.triIndices,
        _context.hasDmmFlags, _context.dmmSizes, numTriangles,
        _context.useIndexBuffer,
        _context.microMapFormats, _context.triNeighborLists,
        dmmDescs, dmmIndexBuffer, _context.indexSize,
        shared::StridedBuffer<OptixDisplacementMicromapTriangleFlags>(
            dmmTriangleFlagsBuffer.getCUdeviceptr(),
            dmmTriangleFlagsBuffer.numElements(),
            dmmTriangleFlagsBuffer.stride()),
        debugSubdivLevelBuffer);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<OptixDisplacementMicromapDesc> dmmDescsOnHost = dmmDescs;
        std::vector<uint8_t> stridedTriFlags(dmmTriangleFlagsBuffer.stride() * numTriangles);
        CUDADRV_CHECK(cuMemcpyDtoH(
            stridedTriFlags.data(),
            dmmTriangleFlagsBuffer.getCUdeviceptr(),
            stridedTriFlags.size()));
        std::vector<OptixDisplacementMicromapTriangleFlags> triFlags(numTriangles);
        for (uint32_t i = 0; i < numTriangles; ++i)
            triFlags[i] = reinterpret_cast<OptixDisplacementMicromapTriangleFlags &>(
                stridedTriFlags[dmmTriangleFlagsBuffer.stride() * i]);
        if (_context.indexSize == 4) {
            std::vector<int32_t> dmmIndices(numTriangles);
            if (_context.useIndexBuffer)
                CUDADRV_CHECK(cuMemcpyDtoH(
                    dmmIndices.data(), dmmIndexBuffer.getCUdeviceptr(), dmmIndexBuffer.sizeInBytes()));
            hpprintf("");
        }
        else if (_context.indexSize == 2) {
            std::vector<int16_t> dmmIndices(numTriangles);
            if (_context.useIndexBuffer)
                CUDADRV_CHECK(cuMemcpyDtoH(
                    dmmIndices.data(), dmmIndexBuffer.getCUdeviceptr(), dmmIndexBuffer.sizeInBytes()));
            hpprintf("");
        }
        hpprintf("");
    }

    // JP: 先頭である三角形においてMicro-Vertex上でハイトマップの評価を行いDMMを計算する。
    // EN: Evaluate the height map at the micro-vertices of each head triangle to compute the DMM.
    CUDADRV_CHECK(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
    s_evaluateMicroVertexHeights(
        stream, cudau::dim3(1024),
        _context.microMapKeys, _context.refKeyIndices, _context.triIndices,
        numTriangles,
        _context.texture, _context.texSize, _context.numChannels, _context.alphaChannelIndex,
        _context.dmmSizes, _context.counter,
        dmmArray);

    /*
    TODO:
    JP: Watertightnessのために隣り合うサブ三角形やDMM間でシフト量を一致させる。
        これを考える場合マイクロマップキーには隣り合う三角形のエンコーディングと分割レベルも
        加えないといけないかもしれない。
        さらに頂点ごとのバイアスやスケールも考えるともはやジオメトリのまるごとコピーを除いて
        DMMの再利用、つまり意味のあるインデックスバッファーは難しいかもしれない。
    EN: Matching shift amounts between neighboring sub-triangles and DMMs for watertightness.
        It probably needs to take neighboring triangles' encodings and subdivision levels into account
        for micro map keys.
        Reusing DMMs, that is meaningful index buffer seems probably difficult if additionally
        considering per-vertex bias and scale except for copying the entire geometry.
    */

    if (!_context.useIndexBuffer) {
        // JP: 先頭ではない三角形においてDMMのコピーを行う。
        // EN: Copy the DMMs for non-head triangles.
        CUDADRV_CHECK(cuMemsetD32Async(
            reinterpret_cast<CUdeviceptr>(_context.counter), 0, 1, stream));
        s_copyDisplacementMicroMaps(
            stream, cudau::dim3(1024),
            _context.refKeyIndices, _context.triIndices, _context.dmmSizes,
            numTriangles, _context.counter,
            dmmArray);
    }

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    if (enableDebugPrint) {
        std::vector<OptixDisplacementMicromapDesc> dmmDescsOnHost = dmmDescs;
        std::vector<uint8_t> dmmArrayOnHost(dmmArray.sizeInBytes());
        CUDADRV_CHECK(cuMemcpyDtoH(
            dmmArrayOnHost.data(), dmmArray.getCUdeviceptr(),
            dmmArray.sizeInBytes()));
        for (uint32_t dmmIdx = 0; dmmIdx < dmmDescsOnHost.size(); ++dmmIdx) {
            OptixDisplacementMicromapDesc desc = dmmDescsOnHost[dmmIdx];
            if (desc.format == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES) {
                using DispBlock = shared::DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES>;
                auto dispBlocks = reinterpret_cast<const DispBlock*>(&dmmArrayOnHost[desc.byteOffset]);
                const uint32_t stSubdivLevel =
                    std::max<uint32_t>(desc.subdivisionLevel, DispBlock::maxSubdivLevel) - DispBlock::maxSubdivLevel;
                const uint32_t numSubTris = 1 << (2 * stSubdivLevel);
                const uint32_t subdivLevelInBlock =
                    std::min<uint32_t>(desc.subdivisionLevel, DispBlock::maxSubdivLevel);
                const uint32_t numEdgeVerticesInBlock = (1 << subdivLevelInBlock) + 1;
                const uint32_t numMicroVerticesInBlock = (1 + numEdgeVerticesInBlock) * numEdgeVerticesInBlock / 2;
                for (uint32_t subTriIdx = 0; subTriIdx < numSubTris; ++subTriIdx) {
                    const DispBlock &dispBlock = dispBlocks[subTriIdx];
                    for (uint32_t microVtxIdx = 0; microVtxIdx < numMicroVerticesInBlock; ++microVtxIdx) {
                        float height = dispBlock.getValue(microVtxIdx);
                        hpprintf("%4u-%2u-%4u: %g\n", dmmIdx, subTriIdx, microVtxIdx, height);
                    }
                    hpprintf("");
                }
            }
            else if (desc.format == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES) {

            }
            else if (desc.format == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES) {
                using DispBlock = shared::DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES>;
                auto dispBlocks = reinterpret_cast<const DispBlock*>(&dmmArrayOnHost[desc.byteOffset]);
                const uint32_t stSubdivLevel =
                    std::max<uint32_t>(desc.subdivisionLevel, DispBlock::maxSubdivLevel) - DispBlock::maxSubdivLevel;
                const uint32_t numSubTris = 1 << (2 * stSubdivLevel);
                const uint32_t subdivLevelInBlock =
                    std::min<uint32_t>(desc.subdivisionLevel, DispBlock::maxSubdivLevel);
                const uint32_t numEdgeVerticesInBlock = (1 << subdivLevelInBlock) + 1;
                const uint32_t numMicroVerticesInBlock = (1 + numEdgeVerticesInBlock) * numEdgeVerticesInBlock / 2;
                for (uint32_t subTriIdx = 0; subTriIdx < numSubTris; ++subTriIdx) {
                    const DispBlock &dispBlock = dispBlocks[subTriIdx];
                    hpprintf("%4u-%2u\n", dmmIdx, subTriIdx);
                    //hpprintf("anchor 0: %g\n", dispBlock.getAnchor(0));
                    //hpprintf("       1: %g\n", dispBlock.getAnchor(1));
                    //hpprintf("       2: %g\n", dispBlock.getAnchor(2));
                    hpprintf("");
                }
            }
        }
        hpprintf("");
    }
}



void printConstants() {
    const auto calcTriOrientation = []
    (uint32_t triIdx, uint32_t level) {
        constexpr uint32_t table[] = {
            0b00,
            0b01,
            0b00,
            0b10,
        };
        uint32_t flags = 0; // bit1: horizontal flip, bit0: vertical flip
        uint32_t b = triIdx;
        for (uint32_t l = 0; l < level; ++l) {
            flags ^= table[b & 0b11];
            b >>= 2;
        }
        return flags;
    };

    using shared::Triangle;
    using shared::MicroVertexInfo;

    struct EdgeInfo {
        uint32_t childVertexIndex;
        uint32_t edgeIdx : 2;
    };

    const auto makeEdgeKey = []
    (uint32_t vAIdx, uint32_t vBIdx) {
        return std::make_pair(std::min(vAIdx, vBIdx), std::max(vAIdx, vBIdx));
    };

    constexpr uint32_t maxLevel = 5;

    std::vector<uint2> vertices;
    std::vector<MicroVertexInfo> vertInfos;
    vertices.push_back(uint2(0, 0));
    vertices.push_back(uint2(1 << maxLevel, 0));
    vertices.push_back(uint2(0, 1 << maxLevel));
    vertInfos.push_back(MicroVertexInfo{ 0xFF, 0xFF, 0, 0 });
    vertInfos.push_back(MicroVertexInfo{ 0xFF, 0xFF, 0, 0 });
    vertInfos.push_back(MicroVertexInfo{ 0xFF, 0xFF, 0, 0 });
    std::vector<Triangle> triangles[2];
    triangles[0].push_back(Triangle(0, 1, 2));

    std::map<std::pair<uint32_t, uint32_t>, EdgeInfo> edgeInfos;
    edgeInfos[makeEdgeKey(0, 1)] = EdgeInfo{ 0xFFFFFFFF, 1 };
    edgeInfos[makeEdgeKey(1, 2)] = EdgeInfo{ 0xFFFFFFFF, 2 };
    edgeInfos[makeEdgeKey(2, 0)] = EdgeInfo{ 0xFFFFFFFF, 3 };

    uint32_t curBufIdx = 0;
    for (uint32_t level = 1; level <= maxLevel; ++level) {
        const std::vector<Triangle> &srcTriangles = triangles[curBufIdx];
        std::vector<Triangle> &dstTriangles = triangles[(curBufIdx + 1) % 2];
        const uint32_t numSubdivTris = 1 << (2 * (level - 1));
        dstTriangles.resize(numSubdivTris << 2);
        uint32_t curNumVertices = static_cast<uint32_t>(vertices.size());
        for (uint32_t triIdx = 0; triIdx < numSubdivTris; ++triIdx) {
            const Triangle &srcTri = srcTriangles[triIdx];
            const uint32_t triOriFlags = calcTriOrientation(triIdx, level - 1);
            const bool isUprightTri = (triOriFlags & 0b1) == 0;
            if (isUprightTri) {
                const uint2 vA = (vertices[srcTri.indices[0]] + vertices[srcTri.indices[2]]) / 2;
                const uint2 vB = (vertices[srcTri.indices[1]] + vertices[srcTri.indices[2]]) / 2;
                const uint2 vC = (vertices[srcTri.indices[0]] + vertices[srcTri.indices[1]]) / 2;
                const bool isNormal = (triOriFlags & 0b10) == 0;
                vertices.resize(curNumVertices + 3);
                vertInfos.resize(curNumVertices + 3);
                const uint32_t vAIdx = curNumVertices + (isNormal ? 0 : 1);
                const uint32_t vBIdx = curNumVertices + (isNormal ? 1 : 0);
                const uint32_t vCIdx = curNumVertices + 2;
                vertices[vAIdx] = vA;
                vertices[vBIdx] = vB;
                vertices[vCIdx] = vC;

                EdgeInfo &srcEdgeInfoA = edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[2]));
                EdgeInfo &srcEdgeInfoB = edgeInfos.at(makeEdgeKey(srcTri.indices[1], srcTri.indices[2]));
                EdgeInfo &srcEdgeInfoC = edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[1]));
                srcEdgeInfoA.childVertexIndex = vAIdx;
                srcEdgeInfoB.childVertexIndex = vBIdx;
                srcEdgeInfoC.childVertexIndex = vCIdx;

                vertInfos[vAIdx] = MicroVertexInfo{
                    srcTri.indices[0], srcTri.indices[2],
                    srcEdgeInfoA.edgeIdx, level, 0
                };
                vertInfos[vBIdx] = MicroVertexInfo{
                    srcTri.indices[1], srcTri.indices[2],
                    srcEdgeInfoB.edgeIdx, level, 0
                };
                vertInfos[vCIdx] = MicroVertexInfo{
                    srcTri.indices[0], srcTri.indices[1],
                    srcEdgeInfoC.edgeIdx, level, 0
                };
                dstTriangles[4 * triIdx + 0] = Triangle(srcTri.indices[0], vCIdx, vAIdx);
                dstTriangles[4 * triIdx + 1] = Triangle(vAIdx, vBIdx, vCIdx);
                dstTriangles[4 * triIdx + 2] = Triangle(vCIdx, srcTri.indices[1], vBIdx);
                dstTriangles[4 * triIdx + 3] = Triangle(vBIdx, vAIdx, srcTri.indices[2]);

                edgeInfos[makeEdgeKey(srcTri.indices[0], vAIdx)] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoA.edgeIdx };
                edgeInfos[makeEdgeKey(vAIdx, srcTri.indices[2])] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoA.edgeIdx };
                edgeInfos[makeEdgeKey(srcTri.indices[1], vBIdx)] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoB.edgeIdx };
                edgeInfos[makeEdgeKey(vBIdx, srcTri.indices[2])] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoB.edgeIdx };
                edgeInfos[makeEdgeKey(srcTri.indices[0], vCIdx)] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoC.edgeIdx };
                edgeInfos[makeEdgeKey(vCIdx, srcTri.indices[1])] = EdgeInfo{ 0xFFFFFFFF, srcEdgeInfoC.edgeIdx };
                edgeInfos[makeEdgeKey(vAIdx, vBIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vBIdx, vCIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vCIdx, vAIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };

                curNumVertices += 3;
            }
        }
        for (uint32_t triIdx = 0; triIdx < numSubdivTris; ++triIdx) {
            const Triangle &srcTri = srcTriangles[triIdx];
            const uint32_t triOriFlags = calcTriOrientation(triIdx, level - 1);
            const bool isUprightTri = (triOriFlags & 0b1) == 0;
            if (!isUprightTri) {
                const uint32_t vAIdx =
                    edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[2])).childVertexIndex;
                const uint32_t vBIdx =
                    edgeInfos.at(makeEdgeKey(srcTri.indices[1], srcTri.indices[2])).childVertexIndex;
                const uint32_t vCIdx =
                    edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[1])).childVertexIndex;

                const EdgeInfo &srcEdgeInfoA = edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[2]));
                const EdgeInfo &srcEdgeInfoB = edgeInfos.at(makeEdgeKey(srcTri.indices[1], srcTri.indices[2]));
                const EdgeInfo &srcEdgeInfoC = edgeInfos.at(makeEdgeKey(srcTri.indices[0], srcTri.indices[1]));

                dstTriangles[4 * triIdx + 0] = Triangle(srcTri.indices[0], vCIdx, vAIdx);
                dstTriangles[4 * triIdx + 1] = Triangle(vAIdx, vBIdx, vCIdx);
                dstTriangles[4 * triIdx + 2] = Triangle(vCIdx, srcTri.indices[1], vBIdx);
                dstTriangles[4 * triIdx + 3] = Triangle(vBIdx, vAIdx, srcTri.indices[2]);

                edgeInfos[makeEdgeKey(srcTri.indices[0], vAIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vAIdx, srcTri.indices[2])] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(srcTri.indices[1], vBIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vBIdx, srcTri.indices[2])] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(srcTri.indices[0], vCIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vCIdx, srcTri.indices[1])] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vAIdx, vBIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vBIdx, vCIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
                edgeInfos[makeEdgeKey(vCIdx, vAIdx)] = EdgeInfo{ 0xFFFFFFFF, 0 };
            }
        }
        curBufIdx = (curBufIdx + 1) % 2;
    }

    constexpr uint32_t normalizer = 1 << maxLevel;

#if VISUALIZE_MICRO_VERTICES_WITH_VDB
    const float3 pA = float3(-1.0f, 0.0f, 0.0f);
    const float3 pB = float3(1.0f, 0.0f, 0.0f);
    const float3 pC = float3(0.0f, 1.7f, 0.0f);
    vdb_frame();
    for (int i = 0; i < vertices.size(); ++i) {
        vdb_color(
            i % 3 == 0 ? 1 : 0,
            i % 3 == 1 ? 1 : 0,
            i % 3 == 2 ? 1 : 0);
        const uint2 v = vertices[i];
        const float bcB = static_cast<float>(v.x) / normalizer;
        const float bcC = static_cast<float>(v.y) / normalizer;
        const float bcA = 1 - (bcB + bcC);
        const float3 p = bcA * pA + bcB * pB + bcC * pC;
        vdb_point(p.x, p.y, p.z);

        //std::this_thread::sleep_for(std::chrono::microseconds(10));
        uint32_t temp = 0;
        while (temp++ < 2000000);
    }
#endif

    for (int i = 0; i < vertices.size(); i += 3) {
        hpprintf(
            "{ %2u, %2u }, { %2u, %2u }, { %2u, %2u },\n",
            vertices[i + 0].x, vertices[i + 0].y,
            vertices[i + 1].x, vertices[i + 1].y,
            vertices[i + 2].x, vertices[i + 2].y);
    }

    for (int i = 0; i < vertices.size(); i += 3) {
        const MicroVertexInfo &info0 = vertInfos[i + 0];
        const MicroVertexInfo &info1 = vertInfos[i + 1];
        const MicroVertexInfo &info2 = vertInfos[i + 2];
        hpprintf(
            "{ %3u, %3u, %u, %u }, { %3u, %3u, %u, %u }, { %3u, %3u, %u, %u },\n",
            info0.adjA, info0.adjB, info0.vtxType, info0.level,
            info1.adjA, info1.adjB, info1.vtxType, info1.level,
            info2.adjA, info2.adjB, info2.vtxType, info2.level);
    }

    printf("");
}
