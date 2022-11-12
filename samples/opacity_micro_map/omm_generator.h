#pragma once

#include "opacity_micro_map_shared.h"

void evaluatePerTriangleStates(
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &scratchMemForScan,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommOffsets);

void generateOMMArray(
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint64_t> &ommOffsets,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &ommArray, const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer, uint32_t ommIndexSize);
