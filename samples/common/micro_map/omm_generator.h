#pragma once

#include "../common.h"

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
}

#if !defined(__CUDA_ARCH__)

struct OMMGeneratorContext {
    std::vector<uint8_t> internalState; // TODO: use a more proper implementation.
};

size_t getScratchMemSizeForOMMGenerator(uint32_t numTriangles);

void initializeOMMGeneratorContext(
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr texCoords, uint32_t vertexStride, uint32_t numVertices,
    CUdeviceptr triangles, uint32_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIndex,
    shared::OMMFormat minSubdivLevel, shared::OMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    OMMGeneratorContext* context);

void countOMMFormats(
    const OMMGeneratorContext &context,
    uint32_t histInOmmArray[shared::NumOMMFormats],
    uint32_t histInMesh[shared::NumOMMFormats],
    uint64_t* rawOmmArraySize);

void generateOMMArray(
    const OMMGeneratorContext &context,
    const cudau::Buffer &ommArray,
    const cudau::TypedBuffer<OptixOpacityMicromapDesc> &ommDescs,
    const cudau::Buffer &ommIndexBuffer);

#endif // #if !defined(__CUDA_ARCH__)
