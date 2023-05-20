#pragma once

#include "../common.h"

namespace shared {
    enum DMMEncoding : uint32_t {
        DMMEncoding_None = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_NONE,
        DMMEncoding_64B_per_64MicroTris = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES,
        DMMEncoding_128B_per_256MicroTris = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES,
        DMMEncoding_128B_per_1024MicroTris = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES,
        NumDMMEncodingTypes
    };

    enum DMMSubdivLevel : uint32_t {
        DMMSubdivLevel_0 = 0, //    1 micro-tris
        DMMSubdivLevel_1,     //    4 micro-tris
        DMMSubdivLevel_2,     //   16 micro-tris
        DMMSubdivLevel_3,     //   64 micro-tris
        DMMSubdivLevel_4,     //  256 micro-tris
        DMMSubdivLevel_5,     // 1024 micro-tris
        NumDMMSubdivLevels
    };
}

#if !defined(__CUDA_ARCH__)

struct DMMGeneratorContext {
    std::vector<uint8_t> internalState; // TODO: use a more proper implementation.
};

size_t getScratchMemSizeForDMMGenerator(uint32_t numTriangles);

void initializeDMMGeneratorContext(
    const std::filesystem::path &ptxDirPath,
    CUdeviceptr positions, CUdeviceptr texCoords, uint32_t vertexStride, uint32_t numVertices,
    CUdeviceptr triangles, uint32_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIndex,
    shared::DMMEncoding maxCompressedFormat,
    shared::DMMSubdivLevel minSubdivLevel, shared::DMMSubdivLevel maxSubdivLevel, uint32_t subdivLevelBias,
    bool useIndexBuffer, uint32_t indexSize,
    CUdeviceptr scratchMem, size_t scratchMemSize,
    DMMGeneratorContext* context);

void countDMMFormats(
    const DMMGeneratorContext &context,
    uint32_t histInDmmArray[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels],
    uint32_t histInMesh[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels],
    uint64_t* rawDmmArraySize);

void generateDMMArray(
    const DMMGeneratorContext &context,
    const cudau::Buffer &dmmArray,
    const cudau::TypedBuffer<OptixDisplacementMicromapDesc> &dmmDescs,
    const cudau::Buffer &dmmIndexBuffer,
    const cudau::Buffer &dmmTriangleFlagsBuffer,
    const cudau::Buffer &debugSubdivLevelBuffer = cudau::Buffer());



// Used for generating constants for CUDA kernels.
void printConstants();

#endif // #if !defined(__CUDA_ARCH__)
