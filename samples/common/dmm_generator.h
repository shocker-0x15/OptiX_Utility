#pragma once

#include "common.h"

namespace shared {
    enum DMMFormat : uint32_t {
        DMMFormat_Level0 = 0, //    1 micro-tris
        DMMFormat_Level1,     //    4 micro-tris
        DMMFormat_Level2,     //   16 micro-tris
        DMMFormat_Level3,     //   64 micro-tris
        DMMFormat_Level4,     //  256 micro-tris
        DMMFormat_Level5,     // 1024 micro-tris
        DMMFormat_None,
        NumDMMFormats
    };
}

#if !defined(__CUDA_ARCH__)

struct DMMGeneratorContext {
    std::vector<uint8_t> internalState;
};

size_t getScratchMemSizeForDMMGenerator(uint32_t numTriangles);

void initializeDMMGeneratorContext(
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr positions, CUdeviceptr texCoords, size_t vertexStride,
    CUdeviceptr triangles, size_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIndex,
    shared::DMMFormat minSubdivLevel, shared::DMMFormat maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    DMMGeneratorContext* context);

void countDMMFormats(
    const DMMGeneratorContext &context,
    uint32_t histInDmmArray[shared::NumDMMFormats],
    uint32_t histInMesh[shared::NumDMMFormats],
    uint64_t* rawDmmArraySize);

#endif // #if !defined(__CUDA_ARCH__)
