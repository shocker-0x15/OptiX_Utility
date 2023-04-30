#include "dmm_generator_private.h"
#include "../../ext/cubd/cubd.h"

extern cudau::Kernel g_initializeHalfEdges;
extern cudau::Kernel g_findTwinHalfEdges;
extern cudau::Kernel g_findTriangleNeighbors;
extern cudau::Kernel g_extractTexCoords;
extern cudau::Kernel g_testIfTCTupleIsUnique;

static CUmodule s_dmmModule;

static bool enableDebugPrint = false;

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

    return curOffset;
}

void initializeDMMGeneratorContext(
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIndex,
    shared::DMMFormat minSubdivLevel, shared::DMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    DMMGeneratorContext* context) {
    static bool isInitialized = false;
    if (!isInitialized) {
        initializeMicroMapGeneratorKernels(ptxDirPath);

        isInitialized = true;
    }

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

    const uint32_t numHalfEdges = 3 * numTriangles;
    size_t curScratchMemHead = scratchMem;

    _context.directedEdges = allocate<shared::DirectedEdge>(curScratchMemHead, numHalfEdges);
    _context.halfEdgeIndices = allocate<uint32_t>(curScratchMemHead, numHalfEdges);
    _context.halfEdges = allocate<shared::HalfEdge>(curScratchMemHead, numHalfEdges);
    _context.memSizeForSortDirectedEdges = __getScratchMemSizeForSortDirectedEdges(numHalfEdges);
    _context.memForSortDirectedEdges = allocate(
        curScratchMemHead, _context.memSizeForSortDirectedEdges, alignof(uint64_t));
    _context.triNeighborLists = allocate<shared::TriNeighborList>(curScratchMemHead, numTriangles);
}

void countDMMFormats(
    const DMMGeneratorContext &context,
    uint32_t histInDmmArray[shared::NumDMMFormats],
    uint32_t histInMesh[shared::NumDMMFormats],
    uint64_t* rawDmmArraySize) {
    CUstream stream = 0;
    auto &_context = *reinterpret_cast<const Context*>(context.internalState.data());
    size_t cubScratchMemSize;

    const uint32_t numHalfEdges = 3 * _context.numTriangles;

    // JP: ハーフエッジ構造を初期化する。
    // EN: Initialize half-edge data structures.
    g_initializeHalfEdges.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.triangles, _context.triangleStride, _context.numTriangles,
        _context.directedEdges, _context.halfEdgeIndices, _context.halfEdges);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::DirectedEdge> directedEdges(numHalfEdges);
        std::vector<uint32_t> halfEdgeIndices(numHalfEdges);
        std::vector<shared::HalfEdge> halfEdges(numHalfEdges);
        read(directedEdges, _context.directedEdges);
        read(halfEdgeIndices, _context.halfEdgeIndices);
        read(halfEdges, _context.halfEdges);
        printf("");
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
        printf("");
    }

    // JP: 双子のハーフエッジを特定する。
    // EN: Find the twin half edge for each half edge.
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
        printf("");
    }

    // JP: 各三角形の隣接三角形を特定する。
    // EN: Find the neighbor triangles of each triangle.
    g_findTriangleNeighbors.launchWithThreadDim(
        stream, cudau::dim3(_context.numTriangles),
        _context.halfEdges, _context.triNeighborLists, _context.numTriangles);
    if (enableDebugPrint) {
        CUDADRV_CHECK(cuStreamSynchronize(stream));
        std::vector<shared::TriNeighborList> triNeighborLists(_context.numTriangles);
        read(triNeighborLists, _context.triNeighborLists);
        printf("");
    }
}
