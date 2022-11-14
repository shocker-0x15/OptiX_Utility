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

struct OMMGeneratorContext {
    CUdeviceptr texCoords;
    size_t vertexStride;
    CUdeviceptr triangles;
    size_t triangleStride;
    uint32_t numTriangles;
    CUtexObject texture;
    uint2 texSize;
    CUdeviceptr scratchMem;
    uint32_t numChannels : 3;
    uint32_t alphaChannelIndex : 2;
    uint32_t useIndexBuffer : 1;
    uint32_t indexSize : 3;
    uint32_t minSubdivLevel : 4;
    uint32_t maxSubdivLevel : 4;
    uint32_t subdivLevelBias : 4;
};

size_t getScratchMemSizeForOMMGeneration(uint32_t maxNumTriangles);

void countOMMFormats(
    const OMMGeneratorContext &context,
    uint32_t ommFormatCounts[shared::NumOMMFormats], uint64_t* rawOmmArraySize);

void generateOMMArray(
    const OMMGeneratorContext &context,
    const cudau::Buffer &ommArray,
    const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer);

#endif // #if !defined(__CUDA_ARCH__)
