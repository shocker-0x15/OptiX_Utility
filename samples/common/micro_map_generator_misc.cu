#include "micro_map_generator_private.h"
#if !defined(__INTELLISENSE__)
#include <cub/cub.cuh>
#endif

struct DirectedEdgeLessOp {
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool operator()(
        const shared::DirectedEdge &l, const shared::DirectedEdge &r) const {
        return l < r;
    }
};

size_t __getScratchMemSizeForSortDirectedEdges(uint32_t numEdges) {
    size_t size;
    cub::DeviceMergeSort::StableSortPairs<shared::DirectedEdge*, uint32_t*>(
        nullptr, size,
        nullptr, nullptr, numEdges, DirectedEdgeLessOp());
    return size;
}

// TODO: Use radix sort?
void __sortDirectedEdges(
    shared::DirectedEdge* edges, uint32_t* edgeIndices, uint32_t numEdges,
    void* scratchMem, size_t scratchMemSize) {
    cub::DeviceMergeSort::StableSortPairs<shared::DirectedEdge*, uint32_t*>(
        scratchMem, scratchMemSize,
        edges, edgeIndices, numEdges, DirectedEdgeLessOp());
}



struct TriTexCoordTupleLessOp {
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool operator()(
        const shared::TriTexCoordTuple &l, const shared::TriTexCoordTuple &r) const {
        return l < r;
    }
};

size_t __getScratchMemSizeForSortTriTexCoordTuples(uint32_t numTriangles) {
    size_t size;
    cub::DeviceMergeSort::StableSortPairs<shared::TriTexCoordTuple*, uint32_t*>(
        nullptr, size,
        nullptr, nullptr, numTriangles, TriTexCoordTupleLessOp());
    return size;
}

// TODO: Use radix sort?
void __sortTriTexCoordTuples(
    shared::TriTexCoordTuple* tuples, uint32_t* triIndices, uint32_t numTriangles,
    void* scratchMem, size_t scratchMemSize) {
    cub::DeviceMergeSort::StableSortPairs<shared::TriTexCoordTuple*, uint32_t*>(
        scratchMem, scratchMemSize,
        tuples, triIndices, numTriangles, TriTexCoordTupleLessOp());
}
