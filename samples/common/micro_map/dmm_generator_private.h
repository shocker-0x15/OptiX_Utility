﻿#pragma once

#include "micro_map_generator_private.h"
#include "dmm_generator.h"

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
struct DisplacementBlockLayoutInfo {
    uint32_t correctionBitOffsets[6];
    uint32_t correctionBitWidths[6];
    uint32_t shiftBitOffsets[6];
    uint32_t shiftBitWidths[6];
    uint32_t maxShifts[6];
};

CUDA_CONSTANT_MEM DisplacementBlockLayoutInfo displacementBlock256MicroTrisLayoutInfo = {
    { 0, 33, 66, 165, 465, 0xFFFFFFFF },
    { 11, 11, 11, 10, 5, 0xFFFFFFFF },
    { 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1018, 1006, 0xFFFFFFFF },
    { 0, 0, 0, 1, 3, 0xFFFFFFFF },
    { 0, 0, 0, 1, 6, 0xFFFFFFFF },
};
CUDA_CONSTANT_MEM DisplacementBlockLayoutInfo displacementBlock1024MicroTrisLayoutInfo = {
    { 0, 33, 66, 138, 258, 474 },
    { 11, 11, 8, 4, 2, 1 },
    { 0xFFFFFFFF, 0xFFFFFFFF, 1014, 1002, 986, 970 },
    { 0, 0, 2, 3, 4, 4 },
    { 0, 0, 3, 7, 9, 10 },
};

template <OptixDisplacementMicromapFormat encoding>
CUDA_DEVICE_FUNCTION const DisplacementBlockLayoutInfo &getDisplacementBlockLayoutInfo() {
    if constexpr (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES)
        return displacementBlock256MicroTrisLayoutInfo;
    else /*if constexpr (OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES)*/
        return displacementBlock1024MicroTrisLayoutInfo;
}
#endif

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

        static constexpr uint32_t anchorBitOffset = 0;
        static constexpr uint32_t anchorBitWidth = 11;
        static constexpr uint32_t maxAnchorValue = (1 << anchorBitWidth) - 1;

        static constexpr uint32_t correctionBitOffsets[] = {
            0, 33, 66, 165, 465, 0xFFFFFFFF
        };
        static constexpr uint32_t correctionBitWidths[] = {
            11, 11, 11, 10, 5, 0xFFFFFFFF
        };
        static constexpr uint32_t shiftBitOffsets[] = {
            0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 1018, 1006, 0xFFFFFFFF
        };
        static constexpr uint32_t shiftBitWidths[] = {
            0, 0, 0, 1, 3, 0xFFFFFFFF
        };
        static constexpr uint32_t maxShifts[] = {
            0, 0, 0, 1, 6, 0xFFFFFFFF
        };

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t quantize(float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _maxAnchorValue = maxAnchorValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxAnchorValue * value), _maxAnchorValue);
            return uiValue;
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t dequantize(uint32_t value) {
            Assert(value <= maxAnchorValue, "OOB quantized value: %u", value);
            const float fValue = static_cast<float>(value) / maxAnchorValue;
            return fValue;
        }

        CUDA_DEVICE_FUNCTION void setValue(uint32_t level, uint32_t microVtxIdxInLevel, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock256MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.correctionBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.correctionBitOffsets[level] + bitWidth * microVtxIdxInLevel;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }

        CUDA_DEVICE_FUNCTION void setShift(uint32_t level, uint32_t vtxType, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock256MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.shiftBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.shiftBitOffsets[level] + bitWidth * vtxType;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }
#endif
    };

    template <>
    struct DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES> {
        static constexpr uint32_t maxSubdivLevel = 5;
        static constexpr uint32_t maxNumMicroTris = 1 << (2 * maxSubdivLevel);
        static constexpr uint32_t numBytes = 128;
        static constexpr uint32_t numDwords = numBytes / sizeof(uint32_t);

        static constexpr uint32_t anchorBitOffset = 0;
        static constexpr uint32_t anchorBitWidth = 11;
        static constexpr uint32_t maxAnchorValue = (1 << anchorBitWidth) - 1;

        static constexpr uint32_t correctionBitOffsets[] = {
            0, 33, 66, 138, 258, 474
        };
        static constexpr uint32_t correctionBitWidths[] = {
            11, 11, 8, 4, 2, 1
        };
        static constexpr uint32_t shiftBitOffsets[] = {
            0xFFFFFFFF, 0xFFFFFFFF, 1014, 1002, 986, 970
        };
        static constexpr uint32_t shiftBitWidths[] = {
            0, 0, 2, 3, 4, 4
        };
        static constexpr uint32_t maxShifts[] = {
            0, 0, 3, 7, 9, 10
        };

        uint32_t data[numDwords];

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t quantize(float value) {
            Assert(value <= 1.0f, "Height value must be normalized: %g", value);
            constexpr uint32_t _maxAnchorValue = maxAnchorValue; // workaround for NVCC bug? (CUDA 11.7)
            const uint32_t uiValue = min(static_cast<uint32_t>(maxAnchorValue * value), _maxAnchorValue);
            return uiValue;
        }
        CUDA_DEVICE_FUNCTION CUDA_INLINE static uint32_t dequantize(uint32_t value) {
            Assert(value <= maxAnchorValue, "OOB quantized value: %u", value);
            const float fValue = static_cast<float>(value) / maxAnchorValue;
            return fValue;
        }

        CUDA_DEVICE_FUNCTION void setValue(uint32_t level, uint32_t microVtxIdxInLevel, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock1024MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.correctionBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.correctionBitOffsets[level] + bitWidth * microVtxIdxInLevel;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }

        CUDA_DEVICE_FUNCTION void setShift(uint32_t level, uint32_t vtxType, uint32_t value) {
            const DisplacementBlockLayoutInfo &layoutInfo = displacementBlock1024MicroTrisLayoutInfo;
            const uint32_t bitWidth = layoutInfo.shiftBitWidths[level];
            const uint32_t mask = (1 << bitWidth) - 1;
            value &= mask;
            const uint32_t bitOffset = layoutInfo.shiftBitOffsets[level] + bitWidth * vtxType;
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            atomicOr(&data[binIdx], (value & ((1 << numLowerBits) - 1)) << bitOffsetInBin);
            if (numLowerBits < bitWidth)
                atomicOr(&data[binIdx + 1], value >> numLowerBits);
        }
#endif

        CUDA_COMMON_FUNCTION void setBits(uint32_t value, uint32_t bitOffset, uint32_t bitWidth) {
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            const uint32_t lowerMask = (1 << numLowerBits) - 1;
            data[binIdx] &= ~(lowerMask << bitOffsetInBin);
            data[binIdx] |= (value & lowerMask) << bitOffsetInBin;
            if (numLowerBits < bitWidth) {
                const uint32_t higherMask = (1 << (bitWidth - numLowerBits)) - 1;
                data[binIdx + 1] &= ~higherMask;
                data[binIdx + 1] |= value >> numLowerBits;
            }
        }

        CUDA_COMMON_FUNCTION uint32_t getBits(uint32_t bitOffset, uint32_t bitWidth) const {
            const uint32_t binIdx = bitOffset / 32;
            const uint32_t bitOffsetInBin = bitOffset % 32;
            const uint32_t numLowerBits = min(32 - bitOffsetInBin, bitWidth);
            uint32_t value = 0;
            value |= ((data[binIdx] >> bitOffsetInBin) & ((1 << numLowerBits) - 1));
            if (numLowerBits < bitWidth)
                value |= (data[binIdx + 1] & ((1 << (bitWidth - numLowerBits)) - 1)) << numLowerBits;
            return value;
        }
    };



    struct MicroVertexInfo {
        uint32_t adjA : 8; // to support the max vertex index 152
        uint32_t adjB : 8;
        uint32_t vtxType : 2;
        uint32_t level : 3;
        uint32_t placeHolder : 11;
    };
}

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
struct QuantizedBarycentrics {
    uint16_t b : 8;
    uint16_t c : 8;
};

CUDA_CONSTANT_MEM constexpr QuantizedBarycentrics microVertexQuantizedBarycentricsList[] = {
    // Level 0
    {  0,  0 }, { 32,  0 }, {  0, 32 },
    // + Level 1
    {  0, 16 }, { 16, 16 }, { 16,  0 },
    // + Level 2
    {  0,  8 }, {  8,  8 }, {  8,  0 },
    { 16,  8 }, { 24,  8 }, { 24,  0 },
    {  0, 24 }, {  8, 24 }, {  8, 16 },
    // + Level 3
    {  0,  4 }, {  4,  4 }, {  4,  0 },
    {  8,  4 }, { 12,  4 }, { 12,  0 },
    {  0, 12 }, {  4, 12 }, {  4,  8 },
    {  8, 12 }, { 12, 12 }, { 12,  8 },
    { 16,  4 }, { 20,  4 }, { 20,  0 },
    { 24,  4 }, { 28,  4 }, { 28,  0 },
    { 16, 12 }, { 20, 12 }, { 20,  8 },
    {  8, 20 }, { 12, 20 }, { 12, 16 },
    {  0, 20 }, {  4, 20 }, {  4, 16 },
    {  0, 28 }, {  4, 28 }, {  4, 24 },
    // + Level 4
    {  0,  2 }, {  2,  2 }, {  2,  0 },
    {  4,  2 }, {  6,  2 }, {  6,  0 },
    {  0,  6 }, {  2,  6 }, {  2,  4 },
    {  4,  6 }, {  6,  6 }, {  6,  4 },
    {  8,  2 }, { 10,  2 }, { 10,  0 },
    { 12,  2 }, { 14,  2 }, { 14,  0 },
    {  8,  6 }, { 10,  6 }, { 10,  4 },
    {  4, 10 }, {  6, 10 }, {  6,  8 },
    {  0, 10 }, {  2, 10 }, {  2,  8 },
    {  0, 14 }, {  2, 14 }, {  2, 12 },
    {  4, 14 }, {  6, 14 }, {  6, 12 },
    {  8, 10 }, { 10, 10 }, { 10,  8 },
    { 12, 10 }, { 14, 10 }, { 14,  8 },
    {  8, 14 }, { 10, 14 }, { 10, 12 },
    { 12, 14 }, { 14, 14 }, { 14, 12 },
    { 12,  6 }, { 14,  6 }, { 14,  4 },
    { 16,  2 }, { 18,  2 }, { 18,  0 },
    { 20,  2 }, { 22,  2 }, { 22,  0 },
    { 16,  6 }, { 18,  6 }, { 18,  4 },
    { 20,  6 }, { 22,  6 }, { 22,  4 },
    { 24,  2 }, { 26,  2 }, { 26,  0 },
    { 28,  2 }, { 30,  2 }, { 30,  0 },
    { 24,  6 }, { 26,  6 }, { 26,  4 },
    { 20, 10 }, { 22, 10 }, { 22,  8 },
    { 16, 10 }, { 18, 10 }, { 18,  8 },
    { 16, 14 }, { 18, 14 }, { 18, 12 },
    { 12, 18 }, { 14, 18 }, { 14, 16 },
    {  8, 18 }, { 10, 18 }, { 10, 16 },
    {  8, 22 }, { 10, 22 }, { 10, 20 },
    {  4, 22 }, {  6, 22 }, {  6, 20 },
    {  4, 18 }, {  6, 18 }, {  6, 16 },
    {  0, 18 }, {  2, 18 }, {  2, 16 },
    {  0, 22 }, {  2, 22 }, {  2, 20 },
    {  0, 26 }, {  2, 26 }, {  2, 24 },
    {  4, 26 }, {  6, 26 }, {  6, 24 },
    {  0, 30 }, {  2, 30 }, {  2, 28 },
    // + Level 5
    {  0,  1 }, {  1,  1 }, {  1,  0 },
    {  2,  1 }, {  3,  1 }, {  3,  0 },
    {  0,  3 }, {  1,  3 }, {  1,  2 },
    {  2,  3 }, {  3,  3 }, {  3,  2 },
    {  4,  1 }, {  5,  1 }, {  5,  0 },
    {  6,  1 }, {  7,  1 }, {  7,  0 },
    {  4,  3 }, {  5,  3 }, {  5,  2 },
    {  2,  5 }, {  3,  5 }, {  3,  4 },
    {  0,  5 }, {  1,  5 }, {  1,  4 },
    {  0,  7 }, {  1,  7 }, {  1,  6 },
    {  2,  7 }, {  3,  7 }, {  3,  6 },
    {  4,  5 }, {  5,  5 }, {  5,  4 },
    {  6,  5 }, {  7,  5 }, {  7,  4 },
    {  4,  7 }, {  5,  7 }, {  5,  6 },
    {  6,  7 }, {  7,  7 }, {  7,  6 },
    {  6,  3 }, {  7,  3 }, {  7,  2 },
    {  8,  1 }, {  9,  1 }, {  9,  0 },
    { 10,  1 }, { 11,  1 }, { 11,  0 },
    {  8,  3 }, {  9,  3 }, {  9,  2 },
    { 10,  3 }, { 11,  3 }, { 11,  2 },
    { 12,  1 }, { 13,  1 }, { 13,  0 },
    { 14,  1 }, { 15,  1 }, { 15,  0 },
    { 12,  3 }, { 13,  3 }, { 13,  2 },
    { 10,  5 }, { 11,  5 }, { 11,  4 },
    {  8,  5 }, {  9,  5 }, {  9,  4 },
    {  8,  7 }, {  9,  7 }, {  9,  6 },
    {  6,  9 }, {  7,  9 }, {  7,  8 },
    {  4,  9 }, {  5,  9 }, {  5,  8 },
    {  4, 11 }, {  5, 11 }, {  5, 10 },
    {  2, 11 }, {  3, 11 }, {  3, 10 },
    {  2,  9 }, {  3,  9 }, {  3,  8 },
    {  0,  9 }, {  1,  9 }, {  1,  8 },
    {  0, 11 }, {  1, 11 }, {  1, 10 },
    {  0, 13 }, {  1, 13 }, {  1, 12 },
    {  2, 13 }, {  3, 13 }, {  3, 12 },
    {  0, 15 }, {  1, 15 }, {  1, 14 },
    {  2, 15 }, {  3, 15 }, {  3, 14 },
    {  4, 13 }, {  5, 13 }, {  5, 12 },
    {  6, 13 }, {  7, 13 }, {  7, 12 },
    {  4, 15 }, {  5, 15 }, {  5, 14 },
    {  6, 15 }, {  7, 15 }, {  7, 14 },
    {  6, 11 }, {  7, 11 }, {  7, 10 },
    {  8,  9 }, {  9,  9 }, {  9,  8 },
    { 10,  9 }, { 11,  9 }, { 11,  8 },
    {  8, 11 }, {  9, 11 }, {  9, 10 },
    { 10, 11 }, { 11, 11 }, { 11, 10 },
    { 12,  9 }, { 13,  9 }, { 13,  8 },
    { 14,  9 }, { 15,  9 }, { 15,  8 },
    { 12, 11 }, { 13, 11 }, { 13, 10 },
    { 10, 13 }, { 11, 13 }, { 11, 12 },
    {  8, 13 }, {  9, 13 }, {  9, 12 },
    {  8, 15 }, {  9, 15 }, {  9, 14 },
    { 10, 15 }, { 11, 15 }, { 11, 14 },
    { 12, 13 }, { 13, 13 }, { 13, 12 },
    { 14, 13 }, { 15, 13 }, { 15, 12 },
    { 12, 15 }, { 13, 15 }, { 13, 14 },
    { 14, 15 }, { 15, 15 }, { 15, 14 },
    { 14, 11 }, { 15, 11 }, { 15, 10 },
    { 14,  7 }, { 15,  7 }, { 15,  6 },
    { 14,  5 }, { 15,  5 }, { 15,  4 },
    { 12,  5 }, { 13,  5 }, { 13,  4 },
    { 12,  7 }, { 13,  7 }, { 13,  6 },
    { 10,  7 }, { 11,  7 }, { 11,  6 },
    { 14,  3 }, { 15,  3 }, { 15,  2 },
    { 16,  1 }, { 17,  1 }, { 17,  0 },
    { 18,  1 }, { 19,  1 }, { 19,  0 },
    { 16,  3 }, { 17,  3 }, { 17,  2 },
    { 18,  3 }, { 19,  3 }, { 19,  2 },
    { 20,  1 }, { 21,  1 }, { 21,  0 },
    { 22,  1 }, { 23,  1 }, { 23,  0 },
    { 20,  3 }, { 21,  3 }, { 21,  2 },
    { 18,  5 }, { 19,  5 }, { 19,  4 },
    { 16,  5 }, { 17,  5 }, { 17,  4 },
    { 16,  7 }, { 17,  7 }, { 17,  6 },
    { 18,  7 }, { 19,  7 }, { 19,  6 },
    { 20,  5 }, { 21,  5 }, { 21,  4 },
    { 22,  5 }, { 23,  5 }, { 23,  4 },
    { 20,  7 }, { 21,  7 }, { 21,  6 },
    { 22,  7 }, { 23,  7 }, { 23,  6 },
    { 22,  3 }, { 23,  3 }, { 23,  2 },
    { 24,  1 }, { 25,  1 }, { 25,  0 },
    { 26,  1 }, { 27,  1 }, { 27,  0 },
    { 24,  3 }, { 25,  3 }, { 25,  2 },
    { 26,  3 }, { 27,  3 }, { 27,  2 },
    { 28,  1 }, { 29,  1 }, { 29,  0 },
    { 30,  1 }, { 31,  1 }, { 31,  0 },
    { 28,  3 }, { 29,  3 }, { 29,  2 },
    { 26,  5 }, { 27,  5 }, { 27,  4 },
    { 24,  5 }, { 25,  5 }, { 25,  4 },
    { 24,  7 }, { 25,  7 }, { 25,  6 },
    { 22,  9 }, { 23,  9 }, { 23,  8 },
    { 20,  9 }, { 21,  9 }, { 21,  8 },
    { 20, 11 }, { 21, 11 }, { 21, 10 },
    { 18, 11 }, { 19, 11 }, { 19, 10 },
    { 18,  9 }, { 19,  9 }, { 19,  8 },
    { 16,  9 }, { 17,  9 }, { 17,  8 },
    { 16, 11 }, { 17, 11 }, { 17, 10 },
    { 16, 13 }, { 17, 13 }, { 17, 12 },
    { 18, 13 }, { 19, 13 }, { 19, 12 },
    { 16, 15 }, { 17, 15 }, { 17, 14 },
    { 14, 17 }, { 15, 17 }, { 15, 16 },
    { 12, 17 }, { 13, 17 }, { 13, 16 },
    { 12, 19 }, { 13, 19 }, { 13, 18 },
    { 10, 19 }, { 11, 19 }, { 11, 18 },
    { 10, 17 }, { 11, 17 }, { 11, 16 },
    {  8, 17 }, {  9, 17 }, {  9, 16 },
    {  8, 19 }, {  9, 19 }, {  9, 18 },
    {  8, 21 }, {  9, 21 }, {  9, 20 },
    { 10, 21 }, { 11, 21 }, { 11, 20 },
    {  8, 23 }, {  9, 23 }, {  9, 22 },
    {  6, 23 }, {  7, 23 }, {  7, 22 },
    {  6, 21 }, {  7, 21 }, {  7, 20 },
    {  4, 21 }, {  5, 21 }, {  5, 20 },
    {  4, 23 }, {  5, 23 }, {  5, 22 },
    {  2, 23 }, {  3, 23 }, {  3, 22 },
    {  6, 19 }, {  7, 19 }, {  7, 18 },
    {  6, 17 }, {  7, 17 }, {  7, 16 },
    {  4, 17 }, {  5, 17 }, {  5, 16 },
    {  4, 19 }, {  5, 19 }, {  5, 18 },
    {  2, 19 }, {  3, 19 }, {  3, 18 },
    {  2, 17 }, {  3, 17 }, {  3, 16 },
    {  0, 17 }, {  1, 17 }, {  1, 16 },
    {  0, 19 }, {  1, 19 }, {  1, 18 },
    {  0, 21 }, {  1, 21 }, {  1, 20 },
    {  2, 21 }, {  3, 21 }, {  3, 20 },
    {  0, 23 }, {  1, 23 }, {  1, 22 },
    {  0, 25 }, {  1, 25 }, {  1, 24 },
    {  2, 25 }, {  3, 25 }, {  3, 24 },
    {  0, 27 }, {  1, 27 }, {  1, 26 },
    {  2, 27 }, {  3, 27 }, {  3, 26 },
    {  4, 25 }, {  5, 25 }, {  5, 24 },
    {  6, 25 }, {  7, 25 }, {  7, 24 },
    {  4, 27 }, {  5, 27 }, {  5, 26 },
    {  2, 29 }, {  3, 29 }, {  3, 28 },
    {  0, 29 }, {  1, 29 }, {  1, 28 },
    {  0, 31 }, {  1, 31 }, {  1, 30 },
};

static constexpr float microBcNormalizer = 1.0f / (1 << 5);

CUDA_CONSTANT_MEM constexpr shared::MicroVertexInfo microVertexInfos[] = {
    // Level 0
    { 255, 255, 0, 0 }, { 255, 255, 0, 0 }, { 255, 255, 0, 0 },
    // + Level 1
    {   0,   2, 3, 1 }, {   1,   2, 2, 1 }, {   0,   1, 1, 1 },
    // + Level 2
    {   0,   3, 3, 2 }, {   5,   3, 0, 2 }, {   0,   5, 1, 2 },
    {   5,   4, 0, 2 }, {   1,   4, 2, 2 }, {   5,   1, 1, 2 },
    {   3,   2, 3, 2 }, {   4,   2, 2, 2 }, {   4,   3, 0, 2 },
    // + Level 3
    {   0,   6, 3, 3 }, {   8,   6, 0, 3 }, {   0,   8, 1, 3 },
    {   8,   7, 0, 3 }, {   5,   7, 0, 3 }, {   8,   5, 1, 3 },
    {   6,   3, 3, 3 }, {   7,   3, 0, 3 }, {   7,   6, 0, 3 },
    {   7,  14, 0, 3 }, {   9,  14, 0, 3 }, {   7,   9, 0, 3 },
    {   5,   9, 0, 3 }, {  11,   9, 0, 3 }, {   5,  11, 1, 3 },
    {  11,  10, 0, 3 }, {   1,  10, 2, 3 }, {  11,   1, 1, 3 },
    {   9,   4, 0, 3 }, {  10,   4, 2, 3 }, {  10,   9, 0, 3 },
    {  14,  13, 0, 3 }, {   4,  13, 2, 3 }, {   4,  14, 0, 3 },
    {   3,  12, 3, 3 }, {  14,  12, 0, 3 }, {  14,   3, 0, 3 },
    {  12,   2, 3, 3 }, {  13,   2, 2, 3 }, {  12,  13, 0, 3 },
    // + Level 4
    {   0,  15, 3, 4 }, {  17,  15, 0, 4 }, {   0,  17, 1, 4 },
    {  17,  16, 0, 4 }, {   8,  16, 0, 4 }, {  17,   8, 1, 4 },
    {  15,   6, 3, 4 }, {  16,   6, 0, 4 }, {  16,  15, 0, 4 },
    {  16,  23, 0, 4 }, {  18,  23, 0, 4 }, {  16,  18, 0, 4 },
    {   8,  18, 0, 4 }, {  20,  18, 0, 4 }, {   8,  20, 1, 4 },
    {  20,  19, 0, 4 }, {   5,  19, 0, 4 }, {  20,   5, 1, 4 },
    {  18,   7, 0, 4 }, {  19,   7, 0, 4 }, {  19,  18, 0, 4 },
    {  23,  22, 0, 4 }, {   7,  22, 0, 4 }, {   7,  23, 0, 4 },
    {   6,  21, 3, 4 }, {  23,  21, 0, 4 }, {  23,   6, 0, 4 },
    {  21,   3, 3, 4 }, {  22,   3, 0, 4 }, {  21,  22, 0, 4 },
    {  22,  41, 0, 4 }, {  24,  41, 0, 4 }, {  22,  24, 0, 4 },
    {   7,  24, 0, 4 }, {  26,  24, 0, 4 }, {   7,  26, 0, 4 },
    {  26,  25, 0, 4 }, {   9,  25, 0, 4 }, {  26,   9, 0, 4 },
    {  24,  14, 0, 4 }, {  25,  14, 0, 4 }, {  25,  24, 0, 4 },
    {  25,  38, 0, 4 }, {  33,  38, 0, 4 }, {  25,  33, 0, 4 },
    {  19,  26, 0, 4 }, {  27,  26, 0, 4 }, {  27,  19, 0, 4 },
    {   5,  27, 0, 4 }, {  29,  27, 0, 4 }, {   5,  29, 1, 4 },
    {  29,  28, 0, 4 }, {  11,  28, 0, 4 }, {  29,  11, 1, 4 },
    {  27,   9, 0, 4 }, {  28,   9, 0, 4 }, {  28,  27, 0, 4 },
    {  28,  35, 0, 4 }, {  30,  35, 0, 4 }, {  28,  30, 0, 4 },
    {  11,  30, 0, 4 }, {  32,  30, 0, 4 }, {  11,  32, 1, 4 },
    {  32,  31, 0, 4 }, {   1,  31, 2, 4 }, {  32,   1, 1, 4 },
    {  30,  10, 0, 4 }, {  31,  10, 2, 4 }, {  31,  30, 0, 4 },
    {  35,  34, 0, 4 }, {  10,  34, 2, 4 }, {  10,  35, 0, 4 },
    {   9,  33, 0, 4 }, {  35,  33, 0, 4 }, {  35,   9, 0, 4 },
    {  33,   4, 0, 4 }, {  34,   4, 2, 4 }, {  33,  34, 0, 4 },
    {  38,  37, 0, 4 }, {   4,  37, 2, 4 }, {   4,  38, 0, 4 },
    {  14,  36, 0, 4 }, {  38,  36, 0, 4 }, {  38,  14, 0, 4 },
    {  36,  13, 0, 4 }, {  37,  13, 2, 4 }, {  36,  37, 0, 4 },
    {  40,  44, 0, 4 }, {  36,  44, 0, 4 }, {  36,  40, 0, 4 },
    {  41,  40, 0, 4 }, {  14,  40, 0, 4 }, {  14,  41, 0, 4 },
    {   3,  39, 3, 4 }, {  41,  39, 0, 4 }, {  41,   3, 0, 4 },
    {  39,  12, 3, 4 }, {  40,  12, 0, 4 }, {  39,  40, 0, 4 },
    {  12,  42, 3, 4 }, {  44,  42, 0, 4 }, {  12,  44, 0, 4 },
    {  44,  43, 0, 4 }, {  13,  43, 2, 4 }, {  44,  13, 0, 4 },
    {  42,   2, 3, 4 }, {  43,   2, 2, 4 }, {  43,  42, 0, 4 },
    // + Level 5
    {   0,  45, 3, 5 }, {  47,  45, 0, 5 }, {   0,  47, 1, 5 },
    {  47,  46, 0, 5 }, {  17,  46, 0, 5 }, {  47,  17, 1, 5 },
    {  45,  15, 3, 5 }, {  46,  15, 0, 5 }, {  46,  45, 0, 5 },
    {  46,  53, 0, 5 }, {  48,  53, 0, 5 }, {  46,  48, 0, 5 },
    {  17,  48, 0, 5 }, {  50,  48, 0, 5 }, {  17,  50, 1, 5 },
    {  50,  49, 0, 5 }, {   8,  49, 0, 5 }, {  50,   8, 1, 5 },
    {  48,  16, 0, 5 }, {  49,  16, 0, 5 }, {  49,  48, 0, 5 },
    {  53,  52, 0, 5 }, {  16,  52, 0, 5 }, {  16,  53, 0, 5 },
    {  15,  51, 3, 5 }, {  53,  51, 0, 5 }, {  53,  15, 0, 5 },
    {  51,   6, 3, 5 }, {  52,   6, 0, 5 }, {  51,  52, 0, 5 },
    {  52,  71, 0, 5 }, {  54,  71, 0, 5 }, {  52,  54, 0, 5 },
    {  16,  54, 0, 5 }, {  56,  54, 0, 5 }, {  16,  56, 0, 5 },
    {  56,  55, 0, 5 }, {  18,  55, 0, 5 }, {  56,  18, 0, 5 },
    {  54,  23, 0, 5 }, {  55,  23, 0, 5 }, {  55,  54, 0, 5 },
    {  55,  68, 0, 5 }, {  63,  68, 0, 5 }, {  55,  63, 0, 5 },
    {  49,  56, 0, 5 }, {  57,  56, 0, 5 }, {  57,  49, 0, 5 },
    {   8,  57, 0, 5 }, {  59,  57, 0, 5 }, {   8,  59, 1, 5 },
    {  59,  58, 0, 5 }, {  20,  58, 0, 5 }, {  59,  20, 1, 5 },
    {  57,  18, 0, 5 }, {  58,  18, 0, 5 }, {  58,  57, 0, 5 },
    {  58,  65, 0, 5 }, {  60,  65, 0, 5 }, {  58,  60, 0, 5 },
    {  20,  60, 0, 5 }, {  62,  60, 0, 5 }, {  20,  62, 1, 5 },
    {  62,  61, 0, 5 }, {   5,  61, 0, 5 }, {  62,   5, 1, 5 },
    {  60,  19, 0, 5 }, {  61,  19, 0, 5 }, {  61,  60, 0, 5 },
    {  65,  64, 0, 5 }, {  19,  64, 0, 5 }, {  19,  65, 0, 5 },
    {  18,  63, 0, 5 }, {  65,  63, 0, 5 }, {  65,  18, 0, 5 },
    {  63,   7, 0, 5 }, {  64,   7, 0, 5 }, {  63,  64, 0, 5 },
    {  68,  67, 0, 5 }, {   7,  67, 0, 5 }, {   7,  68, 0, 5 },
    {  23,  66, 0, 5 }, {  68,  66, 0, 5 }, {  68,  23, 0, 5 },
    {  66,  22, 0, 5 }, {  67,  22, 0, 5 }, {  66,  67, 0, 5 },
    {  70,  74, 0, 5 }, {  66,  74, 0, 5 }, {  66,  70, 0, 5 },
    {  71,  70, 0, 5 }, {  23,  70, 0, 5 }, {  23,  71, 0, 5 },
    {   6,  69, 3, 5 }, {  71,  69, 0, 5 }, {  71,   6, 0, 5 },
    {  69,  21, 3, 5 }, {  70,  21, 0, 5 }, {  69,  70, 0, 5 },
    {  21,  72, 3, 5 }, {  74,  72, 0, 5 }, {  21,  74, 0, 5 },
    {  74,  73, 0, 5 }, {  22,  73, 0, 5 }, {  74,  22, 0, 5 },
    {  72,   3, 3, 5 }, {  73,   3, 0, 5 }, {  73,  72, 0, 5 },
    {  73, 140, 0, 5 }, {  75, 140, 0, 5 }, {  73,  75, 0, 5 },
    {  22,  75, 0, 5 }, {  77,  75, 0, 5 }, {  22,  77, 0, 5 },
    {  77,  76, 0, 5 }, {  24,  76, 0, 5 }, {  77,  24, 0, 5 },
    {  75,  41, 0, 5 }, {  76,  41, 0, 5 }, {  76,  75, 0, 5 },
    {  76, 137, 0, 5 }, {  84, 137, 0, 5 }, {  76,  84, 0, 5 },
    {  67,  77, 0, 5 }, {  78,  77, 0, 5 }, {  78,  67, 0, 5 },
    {   7,  78, 0, 5 }, {  80,  78, 0, 5 }, {   7,  80, 0, 5 },
    {  80,  79, 0, 5 }, {  26,  79, 0, 5 }, {  80,  26, 0, 5 },
    {  78,  24, 0, 5 }, {  79,  24, 0, 5 }, {  79,  78, 0, 5 },
    {  79,  86, 0, 5 }, {  81,  86, 0, 5 }, {  79,  81, 0, 5 },
    {  26,  81, 0, 5 }, {  83,  81, 0, 5 }, {  26,  83, 0, 5 },
    {  83,  82, 0, 5 }, {   9,  82, 0, 5 }, {  83,   9, 0, 5 },
    {  81,  25, 0, 5 }, {  82,  25, 0, 5 }, {  82,  81, 0, 5 },
    {  86,  85, 0, 5 }, {  25,  85, 0, 5 }, {  25,  86, 0, 5 },
    {  24,  84, 0, 5 }, {  86,  84, 0, 5 }, {  86,  24, 0, 5 },
    {  84,  14, 0, 5 }, {  85,  14, 0, 5 }, {  84,  85, 0, 5 },
    {  85, 128, 0, 5 }, {  87, 128, 0, 5 }, {  85,  87, 0, 5 },
    {  25,  87, 0, 5 }, {  89,  87, 0, 5 }, {  25,  89, 0, 5 },
    {  89,  88, 0, 5 }, {  33,  88, 0, 5 }, {  89,  33, 0, 5 },
    {  87,  38, 0, 5 }, {  88,  38, 0, 5 }, {  88,  87, 0, 5 },
    {  88, 125, 0, 5 }, { 120, 125, 0, 5 }, {  88, 120, 0, 5 },
    {  82,  89, 0, 5 }, { 117,  89, 0, 5 }, { 117,  82, 0, 5 },
    {  91,  83, 0, 5 }, {  99,  83, 0, 5 }, {  99,  91, 0, 5 },
    {  92,  91, 0, 5 }, {  27,  91, 0, 5 }, {  27,  92, 0, 5 },
    {  19,  90, 0, 5 }, {  92,  90, 0, 5 }, {  92,  19, 0, 5 },
    {  90,  26, 0, 5 }, {  91,  26, 0, 5 }, {  90,  91, 0, 5 },
    {  64,  80, 0, 5 }, {  90,  80, 0, 5 }, {  90,  64, 0, 5 },
    {  61,  92, 0, 5 }, {  93,  92, 0, 5 }, {  61,  93, 0, 5 },
    {   5,  93, 0, 5 }, {  95,  93, 0, 5 }, {   5,  95, 1, 5 },
    {  95,  94, 0, 5 }, {  29,  94, 0, 5 }, {  95,  29, 1, 5 },
    {  93,  27, 0, 5 }, {  94,  27, 0, 5 }, {  94,  93, 0, 5 },
    {  94, 101, 0, 5 }, {  96, 101, 0, 5 }, {  94,  96, 0, 5 },
    {  29,  96, 0, 5 }, {  98,  96, 0, 5 }, {  29,  98, 1, 5 },
    {  98,  97, 0, 5 }, {  11,  97, 0, 5 }, {  98,  11, 1, 5 },
    {  96,  28, 0, 5 }, {  97,  28, 0, 5 }, {  97,  96, 0, 5 },
    { 101, 100, 0, 5 }, {  28, 100, 0, 5 }, {  28, 101, 0, 5 },
    {  27,  99, 0, 5 }, { 101,  99, 0, 5 }, { 101,  27, 0, 5 },
    {  99,   9, 0, 5 }, { 100,   9, 0, 5 }, {  99, 100, 0, 5 },
    { 100, 119, 0, 5 }, { 102, 119, 0, 5 }, { 100, 102, 0, 5 },
    {  28, 102, 0, 5 }, { 104, 102, 0, 5 }, {  28, 104, 0, 5 },
    { 104, 103, 0, 5 }, {  30, 103, 0, 5 }, { 104,  30, 0, 5 },
    { 102,  35, 0, 5 }, { 103,  35, 0, 5 }, { 103, 102, 0, 5 },
    { 103, 116, 0, 5 }, { 111, 116, 0, 5 }, { 103, 111, 0, 5 },
    {  97, 104, 0, 5 }, { 105, 104, 0, 5 }, { 105,  97, 0, 5 },
    {  11, 105, 0, 5 }, { 107, 105, 0, 5 }, {  11, 107, 1, 5 },
    { 107, 106, 0, 5 }, {  32, 106, 0, 5 }, { 107,  32, 1, 5 },
    { 105,  30, 0, 5 }, { 106,  30, 0, 5 }, { 106, 105, 0, 5 },
    { 106, 113, 0, 5 }, { 108, 113, 0, 5 }, { 106, 108, 0, 5 },
    {  32, 108, 0, 5 }, { 110, 108, 0, 5 }, {  32, 110, 1, 5 },
    { 110, 109, 0, 5 }, {   1, 109, 2, 5 }, { 110,   1, 1, 5 },
    { 108,  31, 0, 5 }, { 109,  31, 2, 5 }, { 109, 108, 0, 5 },
    { 113, 112, 0, 5 }, {  31, 112, 2, 5 }, {  31, 113, 0, 5 },
    {  30, 111, 0, 5 }, { 113, 111, 0, 5 }, { 113,  30, 0, 5 },
    { 111,  10, 0, 5 }, { 112,  10, 2, 5 }, { 111, 112, 0, 5 },
    { 116, 115, 0, 5 }, {  10, 115, 2, 5 }, {  10, 116, 0, 5 },
    {  35, 114, 0, 5 }, { 116, 114, 0, 5 }, { 116,  35, 0, 5 },
    { 114,  34, 0, 5 }, { 115,  34, 2, 5 }, { 114, 115, 0, 5 },
    { 118, 122, 0, 5 }, { 114, 122, 0, 5 }, { 114, 118, 0, 5 },
    { 119, 118, 0, 5 }, {  35, 118, 0, 5 }, {  35, 119, 0, 5 },
    {   9, 117, 0, 5 }, { 119, 117, 0, 5 }, { 119,   9, 0, 5 },
    { 117,  33, 0, 5 }, { 118,  33, 0, 5 }, { 117, 118, 0, 5 },
    {  33, 120, 0, 5 }, { 122, 120, 0, 5 }, {  33, 122, 0, 5 },
    { 122, 121, 0, 5 }, {  34, 121, 2, 5 }, { 122,  34, 0, 5 },
    { 120,   4, 0, 5 }, { 121,   4, 2, 5 }, { 121, 120, 0, 5 },
    { 125, 124, 0, 5 }, {   4, 124, 2, 5 }, {   4, 125, 0, 5 },
    {  38, 123, 0, 5 }, { 125, 123, 0, 5 }, { 125,  38, 0, 5 },
    { 123,  37, 0, 5 }, { 124,  37, 2, 5 }, { 123, 124, 0, 5 },
    { 127, 131, 0, 5 }, { 123, 131, 0, 5 }, { 123, 127, 0, 5 },
    { 128, 127, 0, 5 }, {  38, 127, 0, 5 }, {  38, 128, 0, 5 },
    {  14, 126, 0, 5 }, { 128, 126, 0, 5 }, { 128,  14, 0, 5 },
    { 126,  36, 0, 5 }, { 127,  36, 0, 5 }, { 126, 127, 0, 5 },
    {  36, 129, 0, 5 }, { 131, 129, 0, 5 }, {  36, 131, 0, 5 },
    { 131, 130, 0, 5 }, {  37, 130, 2, 5 }, { 131,  37, 0, 5 },
    { 129,  13, 0, 5 }, { 130,  13, 2, 5 }, { 130, 129, 0, 5 },
    { 133, 149, 0, 5 }, { 129, 149, 0, 5 }, { 129, 133, 0, 5 },
    { 134, 133, 0, 5 }, {  36, 133, 0, 5 }, {  36, 134, 0, 5 },
    {  40, 132, 0, 5 }, { 134, 132, 0, 5 }, { 134,  40, 0, 5 },
    { 132,  44, 0, 5 }, { 133,  44, 0, 5 }, { 132, 133, 0, 5 },
    { 142, 146, 0, 5 }, { 132, 146, 0, 5 }, { 132, 142, 0, 5 },
    { 136, 134, 0, 5 }, { 126, 134, 0, 5 }, { 136, 126, 0, 5 },
    { 137, 136, 0, 5 }, {  14, 136, 0, 5 }, {  14, 137, 0, 5 },
    {  41, 135, 0, 5 }, { 137, 135, 0, 5 }, { 137,  41, 0, 5 },
    { 135,  40, 0, 5 }, { 136,  40, 0, 5 }, { 135, 136, 0, 5 },
    { 139, 143, 0, 5 }, { 135, 143, 0, 5 }, { 135, 139, 0, 5 },
    { 140, 139, 0, 5 }, {  41, 139, 0, 5 }, {  41, 140, 0, 5 },
    {   3, 138, 3, 5 }, { 140, 138, 0, 5 }, { 140,   3, 0, 5 },
    { 138,  39, 3, 5 }, { 139,  39, 0, 5 }, { 138, 139, 0, 5 },
    {  39, 141, 3, 5 }, { 143, 141, 0, 5 }, {  39, 143, 0, 5 },
    { 143, 142, 0, 5 }, {  40, 142, 0, 5 }, { 143,  40, 0, 5 },
    { 141,  12, 3, 5 }, { 142,  12, 0, 5 }, { 142, 141, 0, 5 },
    {  12, 144, 3, 5 }, { 146, 144, 0, 5 }, {  12, 146, 0, 5 },
    { 146, 145, 0, 5 }, {  44, 145, 0, 5 }, { 146,  44, 0, 5 },
    { 144,  42, 3, 5 }, { 145,  42, 0, 5 }, { 145, 144, 0, 5 },
    { 145, 152, 0, 5 }, { 147, 152, 0, 5 }, { 145, 147, 0, 5 },
    {  44, 147, 0, 5 }, { 149, 147, 0, 5 }, {  44, 149, 0, 5 },
    { 149, 148, 0, 5 }, {  13, 148, 2, 5 }, { 149,  13, 0, 5 },
    { 147,  43, 0, 5 }, { 148,  43, 2, 5 }, { 148, 147, 0, 5 },
    { 152, 151, 0, 5 }, {  43, 151, 2, 5 }, {  43, 152, 0, 5 },
    {  42, 150, 3, 5 }, { 152, 150, 0, 5 }, { 152,  42, 0, 5 },
    { 150,   2, 3, 5 }, { 151,   2, 2, 5 }, { 150, 151, 0, 5 },
};
#endif

#if !defined(__CUDA_ARCH__)

struct Context {
    shared::StridedBuffer<float3> positions;
    shared::StridedBuffer<float2> texCoords;
    shared::StridedBuffer<shared::Triangle> triangles;
    CUtexObject texture;
    uint2 texSize;
    uint32_t numChannels;
    uint32_t alphaChannelIndex;
    shared::DMMEncoding maxCompressedFormat;
    shared::DMMSubdivLevel minSubdivLevel;
    shared::DMMSubdivLevel maxSubdivLevel;
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
