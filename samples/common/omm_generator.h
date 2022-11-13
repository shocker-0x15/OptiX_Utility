#pragma once

#include "common.h"

namespace shared {
    enum OMMFormat : uint32_t {
        OMMFormat_Level0 = 0, //        1 micro-tris
        OMMFormat_Level1,     //        4 micro-tris
        OMMFormat_Level2,     //       16 micro-tris
        OMMFormat_Level3,     //       64 micro-tris
        OMMFormat_Level4,     //      256 micro-tris
        OMMFormat_Level5,     //     1024 micro-tris
        OMMFormat_Level6,     //     4096 micro-tris
        OMMFormat_Level7,     //    16384 micro-tris
        OMMFormat_Level8,     //    65536 micro-tris
        OMMFormat_Level9,     //   262144 micro-tris
        OMMFormat_Level10,    //  1048576 micro-tris
        OMMFormat_Level11,    //  4194394 micro-tris
        OMMFormat_Level12,    // 16777216 micro-tris
        OMMFormat_None,
        NumOMMFormats
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

size_t getScratchMemSizeForOMMGeneration(uint32_t maxNumTriangles);

void countOMMFormats(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    const cudau::Buffer &scratchMem,
    uint32_t ommFormatCounts[shared::NumOMMFormats], uint64_t* rawOmmArraySize);

void generateOMMArray(
    CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    const cudau::Buffer &scratchMem,
    const cudau::Buffer &ommArray, const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer, uint32_t ommIndexSize);

#endif // #if !defined(__CUDA_ARCH__)
