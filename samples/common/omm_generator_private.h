#pragma once

#include "micro_map_generator_private.h"
#include "omm_generator.h"

namespace shared {
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

#endif
