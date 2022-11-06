#pragma once

#include "opacity_micro_map_shared.h"

void evaluatePerTriangleStates(
    const cudau::TypedBuffer<Shared::Vertex> &vertices,
    const cudau::TypedBuffer<Shared::Triangle> &triangles,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint32_t> &transparentCounts,
    const cudau::TypedBuffer<uint32_t> &numPixelsValues,
    const cudau::TypedBuffer<uint32_t> &numFetchedTriangles,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommSizes,
    std::vector<uint32_t>* triStates,
    uint32_t ommFormatCountsOnHost[Shared::NumOMMFormats]);
