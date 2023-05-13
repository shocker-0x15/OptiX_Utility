#include "dmm_generator_private.h"
#include "optix_micromap.h"

using namespace shared;

static constexpr uint32_t WarpSize = 32;

CUDA_DEVICE_FUNCTION CUDA_INLINE void getInfoForMicroMapEncoding(
    OptixDisplacementMicromapFormat encoding,
    uint32_t* maxNumMicroTris, uint32_t* numBytes) {
    if (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES) {
        using DispBlock = DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES>;
        *maxNumMicroTris = DispBlock::maxNumMicroTris;
        *numBytes = DispBlock::numBytes;
    }
    else if (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES) {
        using DispBlock = DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES>;
        *maxNumMicroTris = DispBlock::maxNumMicroTris;
        *numBytes = DispBlock::numBytes;
    }
    else /*if (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES)*/ {
        using DispBlock = DisplacementBlock<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES>;
        *maxNumMicroTris = DispBlock::maxNumMicroTris;
        *numBytes = DispBlock::numBytes;
    }
}



CUDA_DEVICE_KERNEL void determineTargetSubdivLevels(
    const float* meshAabbArea,
    StridedBuffer<float3> positions, StridedBuffer<float2> texCoords, StridedBuffer<Triangle> triangles,
    uint2 texSize,
    DMMSubdivLevel minSubdivLevel, DMMSubdivLevel maxSubdivLevel, int32_t subdivLevelBias,
    MicroMapKey* microMapKeys, uint32_t* triIndices) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= triangles.numElements)
        return;

    const Triangle &tri = triangles[triIdx];

    const float3 triPositions[] = {
        positions[tri.indices[0]],
        positions[tri.indices[1]],
        positions[tri.indices[2]],
    };
    const float normTriArea = 0.5f * length(cross(
        triPositions[2] - triPositions[0], triPositions[1] - triPositions[0])) / *meshAabbArea;

    const TriTexCoordTuple tcTuple{
        texCoords[tri.indices[0]],
        texCoords[tri.indices[1]],
        texCoords[tri.indices[2]],
    };
    const float2 texSizeF = make_float2(texSize.x, texSize.y);
    const float numTexelsF = 0.5f * fabsf(cross(
        texSizeF * (tcTuple.tcC - tcTuple.tcA),
        texSizeF * (tcTuple.tcB - tcTuple.tcA)));

    // JP: 三角形のテクセル数とメッシュ中の相対的な三角形の大きさから目標分割レベルを決定する。
    //     サンプルコード用の計算式であって熟考されたものではないことに注意。
    // EN: Determine the target subdivision level based on the number of texels and the relative size of
    //     the triangle in the mesh.
    //     Note that this formula is just for sample code and not a well considered one.
    const float targetSubdivLevelF =
        std::log(numTexelsF * normTriArea) / std::log(4.0f) + 4; // +4: ad-hoc offset
    //printf("Tri %u: tgt level: %g (%.1f texels)\n", triIdx, targetSubdivLevelF, numTexelsF);
    const int32_t minLevel = static_cast<int32_t>(minSubdivLevel);
    const int32_t maxLevel = static_cast<int32_t>(maxSubdivLevel);
    const int32_t targetSubdivLevel =
        min(max(static_cast<int32_t>(targetSubdivLevelF) + subdivLevelBias, minLevel), maxLevel);

    MicroMapKey mmKey = {};
    mmKey.tcTuple = tcTuple;
    mmKey.format.asUInt = 0;
    mmKey.format.level = targetSubdivLevel;
    microMapKeys[triIdx] = mmKey;
    triIndices[triIdx] = triIdx;
}



CUDA_DEVICE_KERNEL void adjustSubdivLevels(
    const TriNeighborList* triNeighborLists, uint32_t numTriangles,
    MicroMapKey* microMapKeys, MicroMapFormat* microMapFormatsShadow, uint32_t pass) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const bool srcIsShadow = pass % 2 == 1;

    // JP: 周囲の三角形の最大分割レベルを特定する。
    // EN: Determine the maximum subdivision level among the neighboring triangles.
    uint32_t maxLevelInNeighbors = 0;
    const TriNeighborList &triNeighborList = triNeighborLists[triIdx];
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t nbTriIdx = triNeighborList.neighbors[i];
        if (nbTriIdx == 0xFFFFFFFF)
            continue;

        const MicroMapFormat nbMmFormat = srcIsShadow ?
            microMapFormatsShadow[nbTriIdx] : microMapKeys[nbTriIdx].format;
        maxLevelInNeighbors = max(nbMmFormat.level, maxLevelInNeighbors);
    }

    // JP: 自分の分割レベルが周囲の最大値より2以上小さくならないようにする。
    // EN: Ensure own's subdivision level no more than 1 lower than the maximum in the neighbors.
    const MicroMapFormat &srcMmFormat = srcIsShadow ? microMapFormatsShadow[triIdx] : microMapKeys[triIdx].format;
    MicroMapFormat &dstMmFormat = srcIsShadow ? microMapKeys[triIdx].format : microMapFormatsShadow[triIdx];
    dstMmFormat.level = max(srcMmFormat.level, max(maxLevelInNeighbors, 1u) - 1);
}



CUDA_DEVICE_KERNEL void finalizeMicroMapFormats(
    MicroMapKey* microMapKeys, MicroMapFormat* microMapFormats, uint32_t numTriangles,
    DMMEncoding maxCompressedFormat) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    static_assert(
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES == 1 &&
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES == 2 &&
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES == 3,
        "Assumption for the enum values breaks.");

    MicroMapKey &mmKey = microMapKeys[triIdx];
    mmKey.format.encoding =
        mmKey.format.level == 5 ? OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES :
        mmKey.format.level == 4 ? OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES :
        OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
    if (maxCompressedFormat != DMMEncoding_None)
        mmKey.format.encoding = min(mmKey.format.encoding, static_cast<uint32_t>(maxCompressedFormat));
    microMapFormats[triIdx] = mmKey.format;
}



CUDA_DEVICE_KERNEL void countDMMFormats(
    const MicroMapKey* microMapKeys, const uint32_t* refKeyIndices, const uint32_t* triIndices,
    uint32_t numTriangles,
    uint32_t* histInDmmArray, uint32_t* histInMesh,
    uint32_t* hasDmmFlags, uint64_t* dmmSizes) {
    const uint32_t keyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (keyIdx >= numTriangles)
        return;

    const uint32_t refKeyIdx = refKeyIndices[keyIdx];
    if (keyIdx != refKeyIdx)
        return;

    const MicroMapKey &mmKey = microMapKeys[keyIdx];
    const uint32_t triIdx = triIndices[keyIdx];

    atomicAdd(&histInDmmArray[mmKey.format.encoding * NumDMMSubdivLevels + mmKey.format.level], 1u);
    atomicAdd(&histInMesh[mmKey.format.encoding * NumDMMSubdivLevels + mmKey.format.level], 1u);

    uint32_t numMicroTrisPerSubTri;
    uint32_t numBytesPerSubTri;
    getInfoForMicroMapEncoding(
        static_cast<OptixDisplacementMicromapFormat>(mmKey.format.encoding),
        &numMicroTrisPerSubTri, &numBytesPerSubTri);
    const uint32_t numMicroTris = 1 << (2 * mmKey.format.level);
    const uint32_t numSubTris = (numMicroTris + numMicroTrisPerSubTri - 1) / numMicroTrisPerSubTri;
    hasDmmFlags[triIdx] = 1;
    dmmSizes[triIdx] = numBytesPerSubTri * numSubTris;
}



CUDA_DEVICE_KERNEL void fillNonUniqueEntries(
    const MicroMapKey* microMapKeys, const uint32_t* refKeyIndices, const uint32_t* triIndices,
    uint32_t numTriangles, bool useIndexBuffer,
    uint32_t* histInDmmArray, uint32_t* histInMesh,
    uint32_t* hasDmmFlags, uint64_t* dmmSizes) {
    const uint32_t keyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (keyIdx >= numTriangles)
        return;

    const uint32_t refKeyIdx = refKeyIndices[keyIdx];
    if (keyIdx == refKeyIdx)
        return;

    const MicroMapKey &mmKey = microMapKeys[keyIdx];
    const uint32_t triIdx = triIndices[keyIdx];
    if (useIndexBuffer) {
        atomicAdd(&histInDmmArray[DMMEncoding_None], 1u);

        hasDmmFlags[triIdx] = 0;
        dmmSizes[triIdx] = 0;
    }
    else {
        atomicAdd(&histInDmmArray[mmKey.format.encoding * NumDMMSubdivLevels + mmKey.format.level], 1u);

        uint32_t numMicroTrisPerSubTri;
        uint32_t numBytesPerSubTri;
        getInfoForMicroMapEncoding(
            static_cast<OptixDisplacementMicromapFormat>(mmKey.format.encoding),
            &numMicroTrisPerSubTri, &numBytesPerSubTri);
        const uint32_t numMicroTris = 1 << (2 * mmKey.format.level);
        const uint32_t numSubTris = (numMicroTris + numMicroTrisPerSubTri - 1) / numMicroTrisPerSubTri;
        hasDmmFlags[triIdx] = 1;
        dmmSizes[triIdx] = numBytesPerSubTri * numSubTris;
    }

    atomicAdd(&histInMesh[mmKey.format.encoding * NumDMMSubdivLevels + mmKey.format.level], 1u);
}



CUDA_DEVICE_KERNEL void createDMMDescriptors(
    const uint32_t* refKeyIndices, const uint32_t* triIndices,
    const uint32_t* triToDmmMap, const uint64_t* dmmOffsets, uint32_t numTriangles,
    bool useIndexBuffer,
    const MicroMapFormat* microMapFormats, const TriNeighborList* triNeighborLists,
    OptixDisplacementMicromapDesc* dmmDescs, void* dmmIndices, uint32_t dmmIndexSize,
    StridedBuffer<OptixDisplacementMicromapTriangleFlags> triFlagsBuffer) {
    const uint32_t keyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (keyIdx >= numTriangles)
        return;

    const auto dmmIndices8 = reinterpret_cast<int8_t*>(dmmIndices);
    const auto dmmIndices16 = reinterpret_cast<int16_t*>(dmmIndices);
    const auto dmmIndices32 = reinterpret_cast<int32_t*>(dmmIndices);

    const uint32_t triIdx = triIndices[keyIdx];
    const uint32_t refKeyIdx = refKeyIndices[keyIdx];
    const uint32_t refTriIdx = triIndices[refKeyIdx];
    const MicroMapFormat &mmFormat = microMapFormats[triIdx];
    const int32_t dmmIdx = static_cast<int32_t>(triToDmmMap[useIndexBuffer ? refTriIdx : triIdx]);
    if (keyIdx == refKeyIdx || !useIndexBuffer) {
        OptixDisplacementMicromapDesc &dmmDesc = dmmDescs[dmmIdx];
        dmmDesc.byteOffset = dmmOffsets[triIdx];
        dmmDesc.format = mmFormat.encoding;
        dmmDesc.subdivisionLevel = mmFormat.level;
    }

    if (useIndexBuffer) {
        if (dmmIndexSize == 1)
            dmmIndices8[triIdx] = dmmIdx;
        else if (dmmIndexSize == 2)
            dmmIndices16[triIdx] = dmmIdx;
        else/* if (dmmIndexSize == 4)*/
            dmmIndices32[triIdx] = dmmIdx;
    }

    // JP: 隣接三角形の分割レベルが低い場合、対応するエッジのデシメートフラグを立てておく必要がある。
    // EN: Edge has to be marked for decimation when the corresponding neighboring triangle has lower
    //     subdivision level.
    static_assert(
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_01 == (1 << 0) &&
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_12 == (1 << 1) &&
        OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_DECIMATE_EDGE_20 == (1 << 2),
        "Assumption for the enum values breaks.");
    uint32_t flags = OPTIX_DISPLACEMENT_MICROMAP_TRIANGLE_FLAG_NONE;
    const TriNeighborList &triNeighborList = triNeighborLists[triIdx];
    for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t nbTriIdx = triNeighborList.neighbors[i];
        if (nbTriIdx == 0xFFFFFFFF)
            continue;
        const MicroMapFormat &nbMmFormat = microMapFormats[nbTriIdx];
        if (nbMmFormat.level < mmFormat.level)
            flags |= 1 << i;
    }
    triFlagsBuffer[triIdx] = static_cast<OptixDisplacementMicromapTriangleFlags>(flags);
}




CUDA_DEVICE_FUNCTION CUDA_INLINE float fetchHeight(
    CUtexObject texture, uint32_t numChannels, uint32_t heightChannelIdx,
    float2 texCoords) {
    union Alias {
        float4 f4;
        float2 f2;
        float f;
        float a[4];
        CUDA_DEVICE_FUNCTION Alias() {}
    } alias;

    float height;
    if (numChannels == 4) {
        alias.f4 = tex2DLod<float4>(texture, texCoords.x, texCoords.y, 0.0f);
        height = alias.a[heightChannelIdx];
    }
    else if (numChannels == 2) {
        alias.f2 = tex2DLod<float2>(texture, texCoords.x, texCoords.y, 0.0f);
        height = alias.a[heightChannelIdx];
    }
    else {
        height = tex2DLod<float>(texture, texCoords.x, texCoords.y, 0.0f);
    }
    return height;

    //const float dist = std::sqrt(pow2(texCoords.x - 0.5f) + pow2(texCoords.y - 0.5f));
    //return 0.5f * std::cos(2 * 3.14159265 * 6 * dist) + 0.5f;
}

template <OptixDisplacementMicromapFormat encoding>
CUDA_DEVICE_FUNCTION CUDA_INLINE void buildSingleDisplacementMicroMap(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIdx,
    uint32_t dmmIdx, const TriTexCoordTuple &tcTuple, uint32_t subdivLevel,
    uint8_t* const displacementMicroMap) {
    using DispBlock = DisplacementBlock<encoding>;
    auto displacementBlocks = reinterpret_cast<DispBlock*>(displacementMicroMap);

    const uint32_t stSubdivLevel = max(subdivLevel, DispBlock::maxSubdivLevel) - DispBlock::maxSubdivLevel;
    const uint32_t numSubTris = 1 << (2 * stSubdivLevel);

    CUDA_SHARED_MEM uint32_t b_mem[DispBlock::numDwords];
    DispBlock &b_dispBlock = reinterpret_cast<DispBlock &>(b_mem);

    for (uint32_t subTriIdx = 0; subTriIdx < numSubTris; ++subTriIdx) {
        // JP: シェアードメモリ上のDisplacementBlockをクリア。
        // EN: Clear the displacement block on the shared memory.
        for (uint32_t dwIdx = threadIdx.x; dwIdx < DispBlock::numDwords; dwIdx += WarpSize)
            b_dispBlock.data[dwIdx] = 0;
        __syncwarp();

        float2 stBcs[3];
        optixMicromapIndexToBaseBarycentrics(subTriIdx, stSubdivLevel, stBcs[0], stBcs[1], stBcs[2]);

        float2 stTcs[3];
        for (uint32_t i = 0; i < 3; ++i) {
            const float2 stBc = stBcs[i];
            stTcs[i] = (1 - (stBc.x + stBc.y)) * tcTuple.tcA + stBc.x * tcTuple.tcB + stBc.y * tcTuple.tcC;
        }

        constexpr bool enableDebugPrint = false;

        if constexpr (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES) {
            const uint32_t subdivLevelInBlock = min(subdivLevel, DispBlock::maxSubdivLevel);
            const uint32_t numEdgeVerticesInBlock = (1 << subdivLevelInBlock) + 1;
            const uint32_t numMicroVerticesInBlock = (1 + numEdgeVerticesInBlock) * numEdgeVerticesInBlock / 2;

            for (uint32_t microVtxIdx = threadIdx.x;
                 microVtxIdx < numMicroVerticesInBlock; microVtxIdx += WarpSize) {
                const float2 microVtxBc = microVertexBarycentrics[microVtxIdx];
                const float2 microVtxTc =
                    (1 - (microVtxBc.x + microVtxBc.y)) * stTcs[0]
                    + microVtxBc.x * stTcs[1]
                    + microVtxBc.y * stTcs[2];
                const float height = fetchHeight(texture, numChannels, heightChannelIdx, microVtxTc);
                if constexpr (enableDebugPrint) {
                    printf("%4u-%2u-%2u: %g\n", dmmIdx, subTriIdx, microVtxIdx, height);
                }
                b_dispBlock.setValue(microVtxIdx, height);
            }
        }
        else {
            const DisplacementBlockLayoutInfo &layoutInfo = getDisplacementBlockLayoutInfo<encoding>();

            constexpr uint32_t maxNumEdgeVerticesInBlock = (1 << DispBlock::maxSubdivLevel) + 1;
            constexpr uint32_t maxNumMicroVerticesInBlock =
                (1 + maxNumEdgeVerticesInBlock) * maxNumEdgeVerticesInBlock / 2;
            constexpr uint32_t anchorBitWidth = DispBlock::anchorBitWidth;
            constexpr uint32_t numBitsOfValueCache = anchorBitWidth * maxNumMicroVerticesInBlock;
            constexpr uint32_t numDwordsOfValueCache = (numBitsOfValueCache + 31) / 32;
            CUDA_SHARED_MEM uint32_t b_valueCache[numDwordsOfValueCache];
            CUDA_SHARED_MEM uint32_t b_shifts[DispBlock::maxSubdivLevel + 1];
            for (uint32_t dwIdx = threadIdx.x; dwIdx < numDwordsOfValueCache; dwIdx += WarpSize)
                b_valueCache[dwIdx] = 0;
            if (threadIdx.x < (DispBlock::maxSubdivLevel + 1))
                b_shifts[threadIdx.x] = 0;
            __syncwarp();

            const auto storeValue = [&]
            (uint32_t idx, uint32_t value) {
                const uint32_t bitOffset = anchorBitWidth * idx;
                const uint32_t binIdx = bitOffset / 32;
                const uint32_t bitOffsetInBin = bitOffset % 32;
                const uint32_t numLowerBits = min(32 - bitOffsetInBin, anchorBitWidth);
                const uint32_t lowerMask = (1 << numLowerBits) - 1;
                atomicAnd(&b_valueCache[binIdx], ~(lowerMask << bitOffsetInBin));
                atomicOr(&b_valueCache[binIdx], (value & lowerMask) << bitOffsetInBin);
                if (numLowerBits < anchorBitWidth) {
                    const uint32_t higherMask = (1 << (anchorBitWidth - numLowerBits)) - 1;
                    atomicAnd(&b_valueCache[binIdx + 1], ~higherMask);
                    atomicOr(&b_valueCache[binIdx + 1], value >> numLowerBits);
                }
            };
            const auto loadValue = [&]
            (uint32_t idx) {
                const uint32_t bitOffset = anchorBitWidth * idx;
                const uint32_t binIdx = bitOffset / 32;
                const uint32_t bitOffsetInBin = bitOffset % 32;
                const uint32_t numLowerBits = min(32 - bitOffsetInBin, anchorBitWidth);
                uint32_t value = 0;
                value |= ((b_valueCache[binIdx] >> bitOffsetInBin) & ((1 << numLowerBits) - 1));
                if (numLowerBits < anchorBitWidth) {
                    const uint32_t numRemBits = anchorBitWidth - numLowerBits;
                    value |= (b_valueCache[binIdx + 1] & ((1 << numRemBits) - 1)) << numLowerBits;
                }
                return value;
            };

            // Anchors
            {
                const uint32_t microVtxIdx = threadIdx.x;
                if (microVtxIdx < 3){
                    const float2 microVtxBc = microVertexBarycentrics[microVtxIdx];
                    const float2 microVtxTc =
                        (1 - (microVtxBc.x + microVtxBc.y)) * stTcs[0]
                        + microVtxBc.x * stTcs[1]
                        + microVtxBc.y * stTcs[2];
                    const float height = fetchHeight(texture, numChannels, heightChannelIdx, microVtxTc);
                    const uint32_t value = DispBlock::quantize(height);
                    storeValue(microVtxIdx, value);
                    b_dispBlock.setValue(0, microVtxIdx, value);

                    if constexpr (enableDebugPrint) {
                        printf(
                            "%4u-%2u-%4u (lv %u): %g (%4u)\n",
                            dmmIdx, subTriIdx, microVtxIdx, 0,
                            height, value);
                    }
                }
                __syncwarp();
            }

            // Corrections
            uint32_t microVtxBaseIdx = 3;
            for (uint32_t curLevel = 1; curLevel <= DispBlock::maxSubdivLevel; ++curLevel) {
                const uint32_t numEdgeVerticesForLevel = (1 << curLevel) + 1;
                const uint32_t numMicroVerticesForLevel =
                    (1 + numEdgeVerticesForLevel) * numEdgeVerticesForLevel / 2;

                // JP: まずは現在のレベル共通のシフト量を求める。
                // EN: First, determine the common shift amount for the current level.
                for (uint32_t microVtxIdx = microVtxBaseIdx + threadIdx.x;
                     microVtxIdx < numMicroVerticesForLevel; microVtxIdx += WarpSize) {
                    // JP: 実際のディスプレイスメント量を計算する。
                    //     後段のために値をキャッシュしておく。
                    // EN: Calculate the actual displacement amount.
                    //     Cache the value for the subsequent process.
                    const float2 microVtxBc = microVertexBarycentrics[microVtxIdx];
                    const float2 microVtxTc =
                        (1 - (microVtxBc.x + microVtxBc.y)) * stTcs[0]
                        + microVtxBc.x * stTcs[1]
                        + microVtxBc.y * stTcs[2];
                    const float height = fetchHeight(texture, numChannels, heightChannelIdx, microVtxTc);
                    const uint32_t value = DispBlock::quantize(height);
                    storeValue(microVtxIdx, value);

                    // JP: 下位レベルのディスプレイスメントからの予測値と実際の値の差分を求め、
                    //     シフト量を算出する。
                    // EN: Compute the difference between a predicted value from the lower level and
                    //     the actual value, then calculate the shift amount.
                    const MicroVertexInfo info = microVertexInfos[microVtxIdx];
                    const uint32_t adjValueA = loadValue(info.adjA);
                    const uint32_t adjValueB = loadValue(info.adjB);
                    const uint32_t predValue = (adjValueA + adjValueB + 1) / 2;
                    const int32_t correction = value - predValue;
                    const int32_t msSetPos = 31 - __clz(correction >= 0 ? correction : ~correction);
                    const int32_t reqShift = static_cast<uint32_t>(max(
                        msSetPos + 2 - static_cast<int32_t>(layoutInfo.correctionBitWidths[curLevel]), 0));
                    atomicMax(&b_shifts[curLevel],
                              min(reqShift, static_cast<int32_t>(layoutInfo.maxShifts[curLevel])));

                    if constexpr (enableDebugPrint) {
                        printf(
                            "%4u-%2u-%4u (lv %u): %g (%4u), pred: %4u (%4u, %4u), corr: %5d, reqShift: %u\n",
                            dmmIdx, subTriIdx, microVtxIdx, curLevel,
                            height, value, predValue, adjValueA, adjValueB, correction, reqShift);
                    }
                }
                __syncwarp();

                for (uint32_t microVtxIdx = microVtxBaseIdx + threadIdx.x;
                     microVtxIdx < numMicroVerticesForLevel; microVtxIdx += WarpSize) {
                    const uint32_t value = loadValue(microVtxIdx);

                    // JP: 下位レベルのディスプレイスメントからの予測値と実際の値の差分を求め、
                    //     シフト量を算出する。
                    // EN: Compute the difference between a predicted value from the lower level and
                    //     the actual value, then calculate the shift amount.
                    const MicroVertexInfo info = microVertexInfos[microVtxIdx];
                    const uint32_t adjValueA = loadValue(info.adjA);
                    const uint32_t adjValueB = loadValue(info.adjB);
                    const uint32_t predValue = (adjValueA + adjValueB + 1) / 2;
                    int32_t correction = value - predValue;
                    const int32_t msSetPos = 31 - __clz(correction >= 0 ? correction : ~correction);
                    const int32_t reqShift = static_cast<uint32_t>(max(
                        msSetPos + 2 - static_cast<int32_t>(layoutInfo.correctionBitWidths[curLevel]), 0));

                    // JP: 必要なシフト量が達成できない場合は可能な限り大きなcorrectionに設定する。
                    // EN: Set the correction to the maximum value as much as possible if
                    //     the required shift cannot be applied.
                    const uint32_t shift = b_shifts[curLevel];
                    if (reqShift > shift) {
                        correction = correction > 0 ?
                            ((1 << anchorBitWidth) - 1) :
                            ~((1 << (anchorBitWidth - 1)) - 1);
                    }
                    // JP: デコード値が負にならないようにcorrectionを丸める。
                    // EN: Round off the correction so that the decoded value will not be negative.
                    if (correction < 0) {
                        const uint32_t mask = (1 << shift) - 1;
                        correction = (correction + mask) & ~mask;
                    }

                    // JP: 補正値を記録する。
                    //     また、補正値を用いたディスプレイスメントのデコード値を求めて次のレベルの計算のために
                    //     キャッシュに記録する。
                    // EN: Store the correction value.
                    //     And store a decoded displacement value using the correction to the cache as well
                    //     for the next level computation.
                    correction >>= shift;
                    b_dispBlock.setValue(curLevel, microVtxIdx - microVtxBaseIdx, correction);
                    const uint32_t decodedValue = predValue + (correction << shift);
                    storeValue(microVtxIdx, decodedValue);

                    if constexpr (enableDebugPrint) {
                        const float relErr = (static_cast<float>(decodedValue) - value) / value * 100;
                        printf(
                            "%4u-%2u-%4u (lv %u): %4u (%6.2f%%), corr: %5d, shift: %u, OF: %u\n",
                            dmmIdx, subTriIdx, microVtxIdx, curLevel,
                            decodedValue, relErr, correction,
                            shift, reqShift > shift ? 1 : 0);
                    }
                }

                microVtxBaseIdx = numMicroVerticesForLevel;
                __syncwarp();
            }

            // Shifts
            if (threadIdx.x <= 5) {
                const uint32_t level = threadIdx.x;
                const uint32_t shiftBitOffset = layoutInfo.shiftBitOffsets[level];
                if (shiftBitOffset != 0xFFFFFFFF) {
                    const uint32_t shift = b_shifts[level];
                    if constexpr (enableDebugPrint) {
                        printf("Shift for level %u: %u\n", level, shift);
                    }
                    b_dispBlock.setShift(level, 0, shift);
                    b_dispBlock.setShift(level, 1, shift);
                    b_dispBlock.setShift(level, 2, shift);
                    b_dispBlock.setShift(level, 3, shift);
                }
            }
        }
        __syncwarp();

        // JP: シェアードメモリ上のDisplacementBlockをメモリに書き出す。
        // EN: Write out the displacement block on the shared memory to the memory.
        DispBlock &displacementBlock = displacementBlocks[subTriIdx];
        for (uint32_t dwIdx = threadIdx.x; dwIdx < DispBlock::numDwords; dwIdx += WarpSize)
            displacementBlock.data[dwIdx] = b_dispBlock.data[dwIdx];
    }
}

CUDA_DEVICE_KERNEL void evaluateMicroVertexHeights(
    const MicroMapKey* microMapKeys, const uint32_t* refKeyIndices, const uint32_t* triIndices,
    uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIdx,
    const uint64_t* dmmOffsets, volatile uint32_t* numFetchedTriangles,
    uint8_t* displacementMicroMaps) {
    while (true) {
        if (*numFetchedTriangles >= numTriangles)
            return;

        constexpr uint32_t numTrisPerFetch = 8;
        uint32_t baseKeyIdx;
        if (threadIdx.x == 0)
            baseKeyIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), numTrisPerFetch);
        baseKeyIdx = __shfl_sync(0xFFFFFFFF, baseKeyIdx, 0);

        for (uint32_t keySubIdx = 0; keySubIdx < numTrisPerFetch; ++keySubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t keyIdx = baseKeyIdx + keySubIdx;
            if (keyIdx >= numTriangles)
                return;

            // JP: 処理中の要素が同じマイクロマップキーを持つ連続要素の先頭ではない場合は終了する。
            // EN: Terminate if the current element is not the head of consecutive elements
            //     with the same micro map key.
            const uint32_t refKeyIdx = refKeyIndices[keyIdx];
            if (keyIdx != refKeyIdx)
                continue;

            const uint32_t triIdx = triIndices[keyIdx];
            const MicroMapKey &mmKey = microMapKeys[keyIdx];

            uint8_t* const displacementMicroMap = displacementMicroMaps + dmmOffsets[triIdx];

            if (mmKey.format.encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES)
                buildSingleDisplacementMicroMap<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES>(
                    texture, texSize, numChannels, heightChannelIdx,
                    triIdx, mmKey.tcTuple, mmKey.format.level, displacementMicroMap);
            else if (mmKey.format.encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES)
                buildSingleDisplacementMicroMap<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_256_MICRO_TRIS_128_BYTES>(
                    texture, texSize, numChannels, heightChannelIdx,
                    triIdx, mmKey.tcTuple, mmKey.format.level, displacementMicroMap);
            else /*if (mmKey.format.encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES)*/
                buildSingleDisplacementMicroMap<OPTIX_DISPLACEMENT_MICROMAP_FORMAT_1024_MICRO_TRIS_128_BYTES>(
                    texture, texSize, numChannels, heightChannelIdx,
                    triIdx, mmKey.tcTuple, mmKey.format.level, displacementMicroMap);
        }
    }
}



CUDA_DEVICE_KERNEL void copyDisplacementMicroMaps(
    const uint32_t* refKeyIndices, const uint32_t* triIndices, const uint64_t* dmmOffsets,
    uint32_t numTriangles, volatile uint32_t* numFetchedTriangles,
    uint8_t* displacementMicroMaps) {
    while (true) {
        if (*numFetchedTriangles >= numTriangles)
            return;

        constexpr uint32_t numTrisPerFetch = 8;
        uint32_t baseKeyIdx;
        if (threadIdx.x == 0)
            baseKeyIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), numTrisPerFetch);
        baseKeyIdx = __shfl_sync(0xFFFFFFFF, baseKeyIdx, 0);

        for (uint32_t keySubIdx = 0; keySubIdx < numTrisPerFetch; ++keySubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t keyIdx = baseKeyIdx + keySubIdx;
            if (keyIdx >= numTriangles)
                return;

            const uint32_t refKeyIdx = refKeyIndices[keyIdx];
            if (keyIdx == refKeyIdx)
                continue;

            const uint32_t triIdx = triIndices[keyIdx];
            const uint64_t dmmOffset = dmmOffsets[triIdx];
            const uint64_t dmmNextOffset = dmmOffsets[triIdx + 1];
            const uint64_t dmmSizeInDwords = (dmmNextOffset - dmmOffset) / sizeof(uint32_t);

            const uint32_t refTriIdx = triIndices[refKeyIdx];
            const uint64_t refDmmOffset = dmmOffsets[refTriIdx];

            const auto srcDisplacementMicroMap =
                reinterpret_cast<const uint32_t*>(displacementMicroMaps + refDmmOffset);
            const auto dstDisplacementMicroMap =
                reinterpret_cast<uint32_t*>(displacementMicroMaps + dmmOffset);
            for (uint32_t dwBaseIdx = 0; dwBaseIdx < dmmSizeInDwords; dwBaseIdx += WarpSize) {
                const uint32_t dwIdx = dwBaseIdx + threadIdx.x;
                if (dwIdx < dmmSizeInDwords)
                    dstDisplacementMicroMap[dwIdx] = srcDisplacementMicroMap[dwIdx];
            }
        }
    }
}
