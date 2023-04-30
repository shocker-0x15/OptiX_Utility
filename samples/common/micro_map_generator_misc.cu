#include "micro_map_generator_private.h"
#if !defined(__INTELLISENSE__)
#include <cub/cub.cuh>
#endif

struct Less {
    CUDA_DEVICE_FUNCTION CUDA_INLINE bool operator()(
        const shared::TriTexCoordTuple &l, const shared::TriTexCoordTuple &r) const {
        return l < r;
    }
};

size_t __getScratchMemSizeForSortTriTexCoordTuples(uint32_t numTriangles) {
    size_t size;
    cub::DeviceMergeSort::StableSortPairs<shared::TriTexCoordTuple*, uint32_t*>(
        nullptr, size,
        nullptr, nullptr, numTriangles, Less());
    return size;
}

void __sortTriTexCoordTuples(
    shared::TriTexCoordTuple* tuples, uint32_t* triIndices, uint32_t numTriangles,
    void* scratchMem, size_t scratchMemSize) {
    cub::DeviceMergeSort::StableSortPairs<shared::TriTexCoordTuple*, uint32_t*>(
        scratchMem, scratchMemSize,
        tuples, triIndices, numTriangles, Less());
}
