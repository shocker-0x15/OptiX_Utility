#pragma once

#include "micro_map_generator_private.h"
#include "dmm_generator.h"

namespace shared {
    template <OptixDisplacementMicromapFormat encoding>
    struct DisplacementBlock;

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 3;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t maxNumEdgeVertices = (1 << maxSubdivLevel) + 1;
        static constexpr uint32_t maxNumMicroVertices = (1 + maxNumEdgeVertices) * maxNumEdgeVertices / 2;
        static constexpr uint32_t numBytes = 64;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        static constexpr uint32_t numBitsPerValue = 11;
        static constexpr uint32_t maxValue = (1 << numBitsPerValue) - 1;

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void setValue(uint32_t microVtxIdx, float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _numBitsPerValue = numBitsPerValue;
            constexpr uint32_t _maxValue = maxValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxValue * value), _maxValue);
            const uint32_t bitOffset = numBitsPerValue * microVtxIdx;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, _numBitsPerValue);
            atomicOr(&data[binIdx], (uiValue & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < numBitsPerValue)
                atomicOr(&data[binIdx + 1], uiValue >> numLowerBits);
        }
#endif

        float getValue(uint32_t microVtxIdx) const {
            const uint32_t bitOffset = numBitsPerValue * microVtxIdx;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, numBitsPerValue);
            uint32_t uiValue = 0;
            uiValue |= ((data[binIdx] >> bitOffsetInBin) & ((1 << numLowerBits) - 1));
            if (numLowerBits < numBitsPerValue)
                uiValue |= (data[binIdx + 1] & ((1 << (numBitsPerValue - numLowerBits)) - 1)) << numLowerBits;
            return static_cast<float>(uiValue) / maxValue;
        }
    };

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 4;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t numBytes = 128;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void setValue(uint32_t microVtxIdx, float value) {
            Assert_NotImplemented();
        }
#endif
    };

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 5;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t numBytes = 128;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION void setValue(uint32_t microVtxIdx, float value) {
            Assert_NotImplemented();
        }
#endif
    };
}

#if !defined(__CUDA_ARCH__)

struct Context {
    shared::StridedBuffer<float3> positions;
    shared::StridedBuffer<float2> texCoords;
    shared::StridedBuffer<shared::Triangle> triangles;
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

    AABBAsOrderedInt* meshAabbAsOrderedInt;
    AABB* meshAabb;
    float* meshAabbArea;
    shared::MicroMapKey* microMapKeys;
    shared::MicroMapFormat* microMapFormats;
    uint32_t* triIndices;
    CUdeviceptr memForSortMicroMapKeys;
    size_t memSizeForSortMicroMapKeys;
    uint32_t* refKeyIndices;
    CUdeviceptr memForScanRefKeyIndices;
    size_t memSizeForScanRefKeyIndices;

    uint64_t* dmmSizes;
    uint32_t* hasDmmFlags;
    uint32_t* histInDmmArray;
    uint32_t* histInMesh;
    CUdeviceptr memForScanDmmSizes;
    size_t memSizeForScanDmmSizes;
    CUdeviceptr memForScanHasDmmFlags;
    size_t memSizeForScanHasDmmFlags;
    uint32_t* counter;
};

#endif
