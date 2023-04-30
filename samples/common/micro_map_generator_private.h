#pragma once

#include "omm_generator.h"

namespace shared {
    struct TriTexCoordTuple {
        float2 tcA;
        float2 tcB;
        float2 tcC;

        CUDA_DEVICE_FUNCTION bool operator==(const TriTexCoordTuple &r) const {
            return tcA == r.tcA && tcB == r.tcB && tcB == r.tcB;
        }
        CUDA_DEVICE_FUNCTION bool operator!=(const TriTexCoordTuple &r) const {
            return tcA != r.tcA || tcB != r.tcB || tcB != r.tcB;
        }

        CUDA_DEVICE_FUNCTION bool operator<(const TriTexCoordTuple &r) const {
            if (tcA.x < r.tcA.x)
                return true;
            if (tcA.x > r.tcA.x)
                return false;
            if (tcA.y < r.tcA.y)
                return true;
            if (tcA.y > r.tcA.y)
                return false;

            // tcA == r.tcA
            if (tcB.x < r.tcB.x)
                return true;
            if (tcB.x > r.tcB.x)
                return false;
            if (tcB.y < r.tcB.y)
                return true;
            if (tcB.y > r.tcB.y)
                return false;

            // tcA == r.tcA && tcB == r.tcB
            if (tcC.x < r.tcC.x)
                return true;
            if (tcC.x > r.tcC.x)
                return false;
            if (tcC.y < r.tcC.y)
                return true;
            if (tcC.y > r.tcC.y)
                return false;

            // *this == r
            return false;
        }
    };
}

#if !defined(__CUDA_ARCH__)

size_t __getScratchMemSizeForSortTriTexCoordTuples(uint32_t numTriangles);

void __sortTriTexCoordTuples(
    shared::TriTexCoordTuple* tuples, uint32_t* triIndices, uint32_t numTriangles,
    void* scratchMem, size_t scratchMemSize);

#endif
