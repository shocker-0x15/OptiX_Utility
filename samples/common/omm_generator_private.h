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

    union PerTriInfo {
        struct {
            uint32_t state : 2;
            uint32_t level : 4;
            uint32_t placeHolder : 26;
        };
        uint32_t asUInt;
    };
}

#if !defined(__CUDA_ARCH__)

struct Context {
    CUdeviceptr texCoords;
    size_t vertexStride;
    CUdeviceptr triangles;
    size_t triangleStride;
    uint32_t numTriangles;
    CUtexObject texture;
    uint2 texSize;
    uint32_t numChannels;
    uint32_t alphaChannelIndex;
    shared::OMMFormat minSubdivLevel;
    shared::OMMFormat maxSubdivLevel;
    uint32_t subdivLevelBias;
    bool useIndexBuffer;
    uint32_t indexSize;
    CUdeviceptr scratchMem;
    size_t scratchMemSize;

    shared::TriTexCoordTuple* triTcTuples;
    uint32_t* triIndices;
    CUdeviceptr memForSortTuples;
    size_t memSizeForSortTuples;
    uint32_t* refTupleIndices;
    CUdeviceptr memForScanRefTupleIndices;
    size_t memSizeForScanRefTupleIndices;

    uint64_t* ommSizes;
    uint32_t* hasOmmFlags;
    uint32_t* perTriInfos;
    uint32_t* counter;
    uint32_t* histInOmmArray;
    uint32_t* histInMesh;
    CUdeviceptr memForScanOmmSizes;
    size_t memSizeForScanOmmSizes;
    CUdeviceptr memForScanHasOmmFlags;
    size_t memSizeForScanHasOmmFlags;
};



size_t __getScratchMemSizeForSortTriTexCoordTuples(uint32_t numTriangles);

void __sortTriTexCoordTuples(
    shared::TriTexCoordTuple* tuples, uint32_t* triIndices, uint32_t numTriangles,
    void* scratchMem, size_t scratchMemSize);

#endif
