#include "dmm_generator_private.h"
#include "optix_micromap.h"

using namespace shared;

static constexpr uint32_t WarpSize = 32;

CUDA_CONSTANT_MEM constexpr float2 microVertBarycentricsFor64MicroTris[] = {
    // Level 0
    float2{ 0.0f, 0.0f }, float2{ 1.0f, 0.0f }, float2{ 0.0f, 1.0f },
    // + Level 1
    float2{ 0.0f, 0.5f }, float2{ 0.5f, 0.5f }, float2{ 0.5f, 0.0f },
    // + Level 2
    float2{ 0.0f, 0.25f }, float2{ 0.25f, 0.25f }, float2{ 0.25f, 0.0f },
    float2{ 0.5f, 0.25f }, float2{ 0.75f, 0.25f }, float2{ 0.75f, 0.0f },
    float2{ 0.0f, 0.75f }, float2{ 0.25f, 0.75f }, float2{ 0.25f, 0.5f },
    // + Level 3
    float2{ 0.0f, 0.125f }, float2{ 0.125f, 0.125f }, float2{ 0.125f, 0.0f },
    float2{ 0.25f, 0.125f }, float2{ 0.375f, 0.125f }, float2{ 0.375f, 0.0f },
    float2{ 0.0f, 0.375f }, float2{ 0.125f, 0.375f }, float2{ 0.125f, 0.25f },
    float2{ 0.25f, 0.375f }, float2{ 0.375f, 0.375f }, float2{ 0.375f, 0.25f },
    float2{ 0.5f, 0.125f }, float2{ 0.625f, 0.125f }, float2{ 0.625f, 0.0f },
    float2{ 0.75f, 0.125f }, float2{ 0.875f, 0.125f }, float2{ 0.875f, 0.0f },
    float2{ 0.5f, 0.375f }, float2{ 0.625f, 0.375f }, float2{ 0.625f, 0.25f },
    float2{ 0.25f, 0.625f }, float2{ 0.375f, 0.625f }, float2{ 0.375f, 0.5f },
    float2{ 0.0f, 0.625f }, float2{ 0.125f, 0.625f }, float2{ 0.125f, 0.5f },
    float2{ 0.0f, 0.875f }, float2{ 0.125f, 0.875f }, float2{ 0.125f, 0.75f },
};

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
    DMMFormat minSubdivLevel, DMMFormat maxSubdivLevel, int32_t subdivLevelBias,
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

    const float targetSubdivLevelF = std::log(numTexelsF) / std::log(4.0f)
        + std::log(normTriArea) / std::log(4.0f) + 4; // +4: ad-hoc offset
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
    MicroMapKey* microMapKeys, MicroMapFormat* microMapFormats, uint32_t numTriangles) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    MicroMapKey &mmKey = microMapKeys[triIdx];
    mmKey.format.encoding = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
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

    atomicAdd(&histInDmmArray[mmKey.format.level], 1u);
    atomicAdd(&histInMesh[mmKey.format.level], 1u);

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
        atomicAdd(&histInDmmArray[DMMFormat_None], 1u);

        hasDmmFlags[triIdx] = 0;
        dmmSizes[triIdx] = 0;
    }
    else {
        atomicAdd(&histInDmmArray[mmKey.format.level], 1u);

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

    atomicAdd(&histInMesh[mmKey.format.level], 1u);
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
        dmmDesc.format = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
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
}

template <OptixDisplacementMicromapFormat encoding>
CUDA_DEVICE_FUNCTION CUDA_INLINE void buildSingleDisplacementMicroMap(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t heightChannelIdx,
    uint32_t /*dmmIdx*/, const TriTexCoordTuple &tcTuple, uint32_t subdivLevel,
    uint8_t* const displacementMicroMap) {
    Assert(
        encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES,
        "DMM format supported now is only 64-byte one.");
    using DispBlock = DisplacementBlock<encoding>;
    auto displacementBlocks = reinterpret_cast<DispBlock*>(displacementMicroMap);

    const uint32_t stSubdivLevel = max(subdivLevel, DispBlock::maxSubdivLevel) - DispBlock::maxSubdivLevel;
    const uint32_t numSubTris = 1 << (2 * stSubdivLevel);
    const uint32_t subdivLevelInBlock = min(subdivLevel, DispBlock::maxSubdivLevel);
    const uint32_t numEdgeVerticesInBlock = (1 << subdivLevelInBlock) + 1;
    const uint32_t numMicroVerticesInBlock = (1 + numEdgeVerticesInBlock) * numEdgeVerticesInBlock / 2;

    for (uint32_t subTriIdx = 0; subTriIdx < numSubTris; ++subTriIdx) {
        DispBlock &displacementBlock = displacementBlocks[subTriIdx];

        float2 stBcs[3];
        optixMicromapIndexToBaseBarycentrics(subTriIdx, stSubdivLevel, stBcs[0], stBcs[1], stBcs[2]);

        float2 stTcs[3];
        for (uint32_t i = 0; i < 3; ++i) {
            const float2 stBc = stBcs[i];
            stTcs[i] = (1 - (stBc.x + stBc.y)) * tcTuple.tcA + stBc.x * tcTuple.tcB + stBc.y * tcTuple.tcC;
        }

        const float2* microVertBarycentrics;
        if constexpr (encoding == OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES)
            microVertBarycentrics = microVertBarycentricsFor64MicroTris;

        for (uint32_t microVtxIdx = threadIdx.x; microVtxIdx < numMicroVerticesInBlock; microVtxIdx += WarpSize) {
            const float2 microVtxBc = microVertBarycentrics[microVtxIdx];
            const float2 microVtxTc =
                (1 - (microVtxBc.x + microVtxBc.y)) * stTcs[0]
                + microVtxBc.x * stTcs[1]
                + microVtxBc.y * stTcs[2];
            const float height = fetchHeight(texture, numChannels, heightChannelIdx, microVtxTc);
            //printf("%4u-%2u-%4u: %g\n", dmmIdx, subTriIdx, microVtxIdx, height);
            displacementBlock.setValue(microVtxIdx, height);
        }
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

            const uint64_t dmmOffset = dmmOffsets[triIdx];
            const uint32_t dmmSize = static_cast<uint32_t>(dmmOffsets[triIdx + 1] - dmmOffset);
            uint8_t* const displacementMicroMap = displacementMicroMaps + dmmOffset;

            // JP: Displacement Micro-Mapのクリア。
            // EN: Clear the displacement micro-map.
            const uint32_t numDwords = dmmSize / 4;
            for (uint32_t dwBaseIdx = 0; dwBaseIdx < numDwords; dwBaseIdx += WarpSize) {
                const uint32_t dwIdx = dwBaseIdx + threadIdx.x;
                if (dwIdx < numDwords)
                    reinterpret_cast<uint32_t*>(displacementMicroMap)[dwIdx] = 0;
            }

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
