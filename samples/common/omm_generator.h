#pragma once

#include "common.h"

namespace shared {
    enum OMMFormat : uint32_t {
        OMMFormat_None = 0, // TODO: Level 0は無視？
        OMMFormat_Level1, // 4 micro-tris,
        OMMFormat_Level2, // 16 micro-tris
        OMMFormat_Level3, // 64 micro-tris
        OMMFormat_Level4, // 256 micro-tris
        OMMFormat_Level5, // 1024 micro-tris
        OMMFormat_Level6, // 4096 micro-tris
        OMMFormat_Level7, // 16384 micro-tris
        OMMFormat_Level8, // 65536 micro-tris
        OMMFormat_Level9, // 262144 micro-tris
        OMMFormat_Level10, // 1048576 micro-tris
        OMMFormat_Level11, // 4194394 micro-tris
        OMMFormat_Level12, // 16777216 micro-tris
        NumOMMFormats
    };
}

#if !defined(__CUDA_ARCH__)

void countOMMFormats(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &scratchMemForScan,
    const cudau::TypedBuffer<uint32_t> &ommFormatCounts,
    const cudau::TypedBuffer<uint64_t> &ommOffsets);

void generateOMMArray(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::TypedBuffer<uint64_t> &ommOffsets,
    const cudau::TypedBuffer<uint32_t> &counter,
    const cudau::Buffer &ommArray, const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer, uint32_t ommIndexSize);

#endif // #if !defined(__CUDA_ARCH__)
