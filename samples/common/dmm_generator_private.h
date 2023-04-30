#pragma once

#include "micro_map_generator_private.h"
#include "dmm_generator.h"

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
    shared::DMMFormat minSubdivLevel;
    shared::DMMFormat maxSubdivLevel;
    uint32_t subdivLevelBias;
    bool useIndexBuffer;
    uint32_t indexSize;
    CUdeviceptr scratchMem;
    size_t scratchMemSize;

    shared::DirectedEdge* directedEdges;
    uint32_t* halfEdgeIndices;
    shared::HalfEdge* halfEdges;
    CUdeviceptr memForSortDirectedEdges;
    size_t memSizeForSortDirectedEdges;
    shared::TriNeighborList* triNeighborLists;
};

#endif
