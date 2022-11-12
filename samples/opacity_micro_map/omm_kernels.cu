#include "opacity_micro_map_shared.h"
#include "optix_micromap.h"

using namespace Shared;

static constexpr uint32_t WarpSize = 32;

CUDA_DEVICE_FUNCTION CUDA_INLINE float fetchAlpha(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    int2 pix) {
    union Alias {
        float4 f4;
        float2 f2;
        float f;
        float a[4];
        CUDA_DEVICE_FUNCTION Alias() {}
    } alias;

    float2 texCoord = make_float2(
        (pix.x + 0.5f) / texSize.x,
        (pix.y + 0.5f) / texSize.y);

    float alpha;
    if (numChannels == 4) {
        alias.f4 = tex2DLod<float4>(texture, texCoord.x, texCoord.y, 0.0f);
        alpha = alias.a[alphaChannelIdx];
    }
    else if (numChannels == 2) {
        alias.f2 = tex2DLod<float2>(texture, texCoord.x, texCoord.y, 0.0f);
        alpha = alias.a[alphaChannelIdx];
    }
    else {
        alpha = tex2DLod<float>(texture, texCoord.x, texCoord.y, 0.0f);
    }
    return alpha;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE bool isTransparent(float alpha) {
    return alpha < 0.5f;
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void rasterizeBottomFlatTriangle(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const float2 ps[3],
    uint32_t* transparentCount, uint32_t* numPixels) {
    const float invSlope01 = (ps[1].x - ps[0].x) / (ps[1].y - ps[0].y);
    const float invSlope02 = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);

    float curXFBegin = ps[0].x;
    float curXFEnd = ps[0].x;
    int32_t curX = static_cast<int32_t>(curXFBegin);
    int32_t curXEnd = static_cast<int32_t>(curXFEnd);
    int32_t curY = static_cast<int32_t>(ps[0].y);
    const int32_t yEnd = static_cast<int32_t>(ps[1].y);
    uint32_t curNumItemsPerWarp = 0;
    int2 item = make_int2(INT32_MAX, INT32_MAX);
    while (curY <= yEnd) {
        const uint32_t numItemsToFill =
            min(curXEnd - curX + 1, static_cast<int32_t>(WarpSize - curNumItemsPerWarp));
        if (threadIdx.x >= curNumItemsPerWarp &&
            threadIdx.x < (curNumItemsPerWarp + numItemsToFill))
            item = make_int2(curX + threadIdx.x - curNumItemsPerWarp, curY);
        curNumItemsPerWarp += numItemsToFill;
        *numPixels += numItemsToFill;
        if (curNumItemsPerWarp == WarpSize) {
            const float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
            *transparentCount += numTrsInWarp;
            curNumItemsPerWarp = 0;
        }
        curX += numItemsToFill;
        if (curX > curXEnd) {
            curXFBegin += invSlope01;
            curXFEnd += invSlope02;
            curX = static_cast<int32_t>(curXFBegin);
            curXEnd = static_cast<int32_t>(curXFEnd);
            ++curY;
        }
    }
    if (curNumItemsPerWarp > 0) {
        float alpha = 1.0f;
        if (threadIdx.x < curNumItemsPerWarp)
            alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
        const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
        *transparentCount += numTrsInWarp;
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void rasterizeTopFlatTriangle(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const float2 ps[3], bool ignoreFirstLine,
    uint32_t* transparentCount, uint32_t* numPixels) {
    const float invSlope02 = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);
    const float invSlope12 = (ps[2].x - ps[1].x) / (ps[2].y - ps[1].y);

    float curXFBegin = ps[2].x;
    float curXFEnd = ps[2].x;
    int32_t curX = static_cast<int32_t>(curXFBegin);
    int32_t curXEnd = static_cast<int32_t>(curXFEnd);
    int32_t curY = static_cast<int32_t>(ps[2].y);
    const int32_t yEnd = static_cast<int32_t>(ps[0].y) + ignoreFirstLine;
    uint32_t curNumItemsPerWarp = 0;
    int2 item = make_int2(INT32_MAX, INT32_MAX);
    while (curY >= yEnd) {
        const uint32_t numItemsToFill =
            min(curXEnd - curX + 1, static_cast<int32_t>(WarpSize - curNumItemsPerWarp));
        if (threadIdx.x >= curNumItemsPerWarp &&
            threadIdx.x < (curNumItemsPerWarp + numItemsToFill))
            item = make_int2(curX + threadIdx.x - curNumItemsPerWarp, curY);
        curNumItemsPerWarp += numItemsToFill;
        *numPixels += numItemsToFill;
        if (curNumItemsPerWarp == WarpSize) {
            const float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
            *transparentCount += numTrsInWarp;
            curNumItemsPerWarp = 0;
        }
        curX += numItemsToFill;
        if (curX > curXEnd) {
            curXFBegin -= invSlope12;
            curXFEnd -= invSlope02;
            curX = static_cast<int32_t>(curXFBegin);
            curXEnd = static_cast<int32_t>(curXFEnd);
            --curY;
        }
    }
    if (curNumItemsPerWarp > 0) {
        float alpha = 1.0f;
        if (threadIdx.x < curNumItemsPerWarp)
            alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
        const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
        *transparentCount += numTrsInWarp;
    }
}



CUDA_DEVICE_FUNCTION CUDA_INLINE void evaluateSingleTriangleTransparency(
    uint32_t triIdx,
    const Vertex* vertices, const Triangle* triangles, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    uint32_t* numTransparentPixels, uint32_t* numPixels) {
    const Triangle &tri = triangles[triIdx];
    const float2 texCoords[] = {
        vertices[tri.index0].texCoord,
        vertices[tri.index1].texCoord,
        vertices[tri.index2].texCoord
    };
    float2 fPixs[3] = {
        make_float2(texSize.x * texCoords[0].x, texSize.y * texCoords[0].y),
        make_float2(texSize.x * texCoords[1].x, texSize.y * texCoords[1].y),
        make_float2(texSize.x * texCoords[2].x, texSize.y * texCoords[2].y)
    };

    const auto swap = [](float2 &a, float2 &b) {
        float2 temp = a;
        a = b;
        b = temp;
    };

    // Sort vertices to be Y-ascending
    if (fPixs[0].y > fPixs[1].y)
        swap(fPixs[0], fPixs[1]);
    if (fPixs[1].y > fPixs[2].y)
        swap(fPixs[1], fPixs[2]);
    if (fPixs[0].y > fPixs[1].y)
        swap(fPixs[0], fPixs[1]);
    
    *numTransparentPixels = 0;
    *numPixels = 0;

    // Top-Flat
    if (fPixs[0].y == fPixs[1].y) {
        // Make the triangle CCW
        if (fPixs[0].x < fPixs[1].x)
            swap(fPixs[0], fPixs[1]);

        rasterizeTopFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            fPixs, false,
            numTransparentPixels, numPixels);
    }
    // Bottom-Flat
    else if (fPixs[1].y == fPixs[2].y) {
        // Make the triangle CCW
        if (fPixs[1].x >= fPixs[2].x)
            swap(fPixs[1], fPixs[2]);

        rasterizeBottomFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            fPixs,
            numTransparentPixels, numPixels);
    }
    // General
    else {
        const float t = (fPixs[1].y - fPixs[0].y) / (fPixs[2].y - fPixs[0].y);
        const float2 newP = make_float2(
            fPixs[0].x + t * (fPixs[2].x - fPixs[0].x),
            fPixs[1].y);

        float2 ps[3];
        ps[0] = fPixs[0];
        ps[1] = fPixs[1];
        ps[2] = newP;
        // Make the triangle CCW
        if (ps[1].x >= ps[2].x)
            swap(ps[1], ps[2]);

        rasterizeBottomFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            numTransparentPixels, numPixels);

        ps[0] = newP;
        ps[1] = fPixs[1];
        ps[2] = fPixs[2];
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeTopFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            ps, true,
            numTransparentPixels, numPixels);
    }
}

CUDA_DEVICE_KERNEL void evaluateTriangleTransparencies(
    const Vertex* vertices, const Triangle* triangles, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    volatile uint32_t* numFetchedTriangles,
    uint32_t* ommFormatCounts, uint64_t* ommSizes) {
    while (true) {
        uint32_t curNumFetches;
        if (threadIdx.x == 0)
            curNumFetches = *numFetchedTriangles;
        curNumFetches = __shfl_sync(0xFFFFFFFF, curNumFetches, 0);
        if (curNumFetches >= numTriangles)
            return;

        uint32_t baseTriIdx;
        if (threadIdx.x == 0)
            baseTriIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), WarpSize);
        baseTriIdx = __shfl_sync(0xFFFFFFFF, baseTriIdx, 0);

        for (uint32_t triSubIdx = 0; triSubIdx < WarpSize; ++triSubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t triIdx = baseTriIdx + triSubIdx;
            if (triIdx >= numTriangles)
                return;

            uint32_t numTransparentPixels;
            uint32_t numPixels;
            evaluateSingleTriangleTransparency(
                triIdx,
                vertices, triangles, numTriangles,
                texture, texSize, numChannels, alphaChannelIdx,
                &numTransparentPixels, &numPixels);

            if (threadIdx.x == 0) {
                const bool singleState = numTransparentPixels == 0 || numTransparentPixels == numPixels;
                constexpr int32_t minLevel = OMMFormat_None;
                constexpr int32_t maxLevel = OMMFormat_Level4;
                const int32_t level = singleState ? 0 :
                    min(max(static_cast<int32_t>(
                        std::log(static_cast<float>(numPixels)) / std::log(4.0f)
                        ) - 4, minLevel), maxLevel); // -4: ad-hoc offset
                atomicAdd(&ommFormatCounts[level], 1u);

                const uint32_t ommSizeInBits = level == 0 ? 0 : 2 * (1 << (2 * level));
                const uint32_t ommSizeInBytes = (ommSizeInBits + 7) / 8;
                ommSizes[triIdx] = ommSizeInBytes;
            }
        }
    }
}



CUDA_DEVICE_KERNEL void createOMMDescriptors(
    const uint64_t* ommOffsets, uint32_t numTriangles,
    uint32_t* descCounter,
    OptixOpacityMicromapDesc* ommDescs, void* ommIndices, uint32_t ommIndexSize) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const auto ommIndices8 = reinterpret_cast<uint8_t*>(ommIndices);
    const auto ommIndices16 = reinterpret_cast<uint16_t*>(ommIndices);
    const auto ommIndices32 = reinterpret_cast<uint32_t*>(ommIndices);

    const uint64_t ommOffset = ommOffsets[triIdx];
    const uint32_t ommSize = static_cast<uint32_t>(ommOffsets[triIdx + 1] - ommOffset);
    const uint32_t numMicroTris = 4 * ommSize;
    const uint32_t ommLevel = tzcnt(numMicroTris) >> 1;
    if (ommLevel == 0) {
        const uint32_t ommIndex = OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX_FULLY_OPAQUE;
        if (ommIndexSize == 1)
            ommIndices8[triIdx] = ommIndex;
        else if (ommIndexSize == 2)
            ommIndices16[triIdx] = ommIndex;
        else/* if (ommIndexSize == 4)*/
            ommIndices32[triIdx] = ommIndex;
        return;
    }

    const uint32_t descIdx = atomicAdd(descCounter, 1u);
    OptixOpacityMicromapDesc &ommDesc = ommDescs[descIdx];
    ommDesc.byteOffset = ommOffset;
    ommDesc.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
    ommDesc.subdivisionLevel = ommLevel;

    if (ommIndexSize == 1)
        ommIndices8[triIdx] = descIdx;
    else if (ommIndexSize == 2)
        ommIndices16[triIdx] = descIdx;
    else/* if (ommIndexSize == 4)*/
        ommIndices32[triIdx] = descIdx;
}



CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t evaluateSingleMicroTriangle(
    float2 fPixA, float2 fPixB, float2 fPixC,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx) {
    const auto rasterizeBottomFlatTriangle = [&]
    (const float2 ps[3], uint32_t* numPixels, uint32_t* numTransparents) {
        const float invSlope01 = (ps[1].x - ps[0].x) / (ps[1].y - ps[0].y);
        const float invSlope02 = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);

        float curXFBegin = ps[0].x;
        float curXFEnd = ps[0].x;
        int32_t curY = static_cast<int32_t>(ps[0].y);
        const int32_t yEnd = static_cast<int32_t>(ps[1].y);
        while (curY <= yEnd) {
            int32_t curXBegin = static_cast<int32_t>(curXFBegin);
            int32_t curXEnd = static_cast<int32_t>(curXFEnd);
            for (int32_t curX = curXBegin; curX <= curXEnd; ++curX) {
                int2 item = make_int2(curX, curY);
                float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
                if (isTransparent(alpha))
                    ++*numTransparents;
            }
            curXFBegin += invSlope01;
            curXFEnd += invSlope02;
            ++curY;

            *numPixels += curXEnd - curXBegin + 1;
        }
    };

    const auto rasterizeTopFlatTriangle = [&]
    (const float2 ps[3], bool ignoreFirstLine, uint32_t* numPixels, uint32_t* numTransparents) {
        const float invSlope02 = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);
        const float invSlope12 = (ps[2].x - ps[1].x) / (ps[2].y - ps[1].y);

        float curXFBegin = ps[2].x;
        float curXFEnd = ps[2].x;
        int32_t curY = static_cast<int32_t>(ps[2].y);
        const int32_t yEnd = static_cast<int32_t>(ps[0].y) + ignoreFirstLine;
        while (curY >= yEnd) {
            int32_t curXBegin = static_cast<int32_t>(curXFBegin);
            int32_t curXEnd = static_cast<int32_t>(curXFEnd);
            for (int32_t curX = curXBegin; curX <= curXEnd; ++curX) {
                int2 item = make_int2(curX, curY);
                float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
                if (isTransparent(alpha))
                    ++*numTransparents;
            }
            curXFBegin -= invSlope12;
            curXFEnd -= invSlope02;
            --curY;

            *numPixels += curXEnd - curXBegin + 1;
        }
    };

    const auto swap = [](float2 &a, float2 &b) {
        float2 temp = a;
        a = b;
        b = temp;
    };

    // Sort vertices to be Y-ascending
    if (fPixA.y > fPixB.y)
        swap(fPixA, fPixB);
    if (fPixB.y > fPixC.y)
        swap(fPixB, fPixC);
    if (fPixA.y > fPixB.y)
        swap(fPixA, fPixB);

    float2 ps[3];
    ps[0] = fPixA;
    ps[1] = fPixB;
    ps[2] = fPixC;

    uint32_t numPixels = 0;
    uint32_t numTransparents = 0;

    // Top-Flat
    if (ps[0].y == ps[1].y) {
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeTopFlatTriangle(ps, false, &numPixels, &numTransparents);
    }
    // Bottom-Flat
    else if (ps[1].y == ps[2].y) {
        // Make the triangle CCW
        if (ps[1].x >= ps[2].x)
            swap(ps[1], ps[2]);

        rasterizeBottomFlatTriangle(ps, &numPixels, &numTransparents);
    }
    // General
    else {
        const float t = (fPixB.y - fPixA.y) / (fPixC.y - fPixA.y);
        const float2 newP = make_float2(
            fPixA.x + t * (fPixC.x - fPixA.x),
            fPixB.y);

        ps[0] = fPixA;
        ps[1] = fPixB;
        ps[2] = newP;
        // Make the triangle CCW
        if (ps[1].x >= ps[2].x)
            swap(ps[1], ps[2]);

        rasterizeBottomFlatTriangle(ps, &numPixels, &numTransparents);

        ps[0] = newP;
        ps[1] = fPixB;
        ps[2] = fPixC;
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeTopFlatTriangle(ps, true, &numPixels, &numTransparents);
    }

    uint32_t state;
    if (numTransparents == 0)
        state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
    else if (numTransparents == numPixels)
        state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
    else if (2 * numTransparents < numPixels)
        state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
    else
        state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;

    return state;
}

CUDA_DEVICE_KERNEL void evaluateMicroTriangleTransparencies(
    const Vertex* vertices, const Triangle* triangles, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const uint64_t* ommOffsets,
    volatile uint32_t* numFetchedTriangles,
    uint8_t* opacityMicroMaps) {
    while (true) {
        if (*numFetchedTriangles >= numTriangles)
            return;

        uint32_t baseTriIdx;
        if (threadIdx.x == 0)
            baseTriIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), WarpSize);
        baseTriIdx = __shfl_sync(0xFFFFFFFF, baseTriIdx, 0);

        for (uint32_t triSubIdx = 0; triSubIdx < WarpSize; ++triSubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t triIdx = baseTriIdx + triSubIdx;
            if (triIdx >= numTriangles)
                return;

            const Triangle &tri = triangles[triIdx];
            const float2 tcA = vertices[tri.index0].texCoord;
            const float2 tcB = vertices[tri.index1].texCoord;
            const float2 tcC = vertices[tri.index2].texCoord;
            const float2 fTexSize = make_float2(texSize.x, texSize.y);

            const uint64_t ommOffset = ommOffsets[triIdx];
            const uint32_t ommSize = static_cast<uint32_t>(ommOffsets[triIdx + 1] - ommOffset);
            uint8_t* const opacityMicroMap = opacityMicroMaps + ommOffset;

            // JP: Opacity Micro-Mapのクリア。
            // EN: Clear the opacity micro-map.
            const uint32_t numDwords = ommSize / 4;
            for (uint32_t dwBaseIdx = 0; dwBaseIdx < numDwords; dwBaseIdx += WarpSize) {
                const uint32_t dwIdx = dwBaseIdx + threadIdx.x;
                if (dwIdx < numDwords)
                    reinterpret_cast<uint32_t*>(opacityMicroMap)[dwIdx] = 0;
            }

            const uint32_t numMicroTris = 4 * ommSize;
            const uint32_t ommLevel = tzcnt(numMicroTris) >> 1;
            for (uint32_t microTriBaseIdx = 0; microTriBaseIdx < numMicroTris; microTriBaseIdx += WarpSize) {
                // JP: 各スレッドがマイクロ三角形のステートを計算する。
                // EN: Each thread computes the state of a micro triangle.
                // TODO: Upper/Lower Micro-Triangleを適切に振り分けてDivergenceを抑える。
                const uint32_t microTriIdx = microTriBaseIdx + threadIdx.x;
                if (microTriIdx >= numMicroTris)
                    break;

                float2 bc0, bc1, bc2;
                optixMicromapIndexToBaseBarycentrics(microTriIdx, ommLevel, bc0, bc1, bc2);

                const float2 fPix0 =
                    fTexSize * ((1.0f - (bc0.x + bc0.y)) * tcA + bc0.x * tcB + bc0.y * tcC);
                const float2 fPix1 =
                    fTexSize * ((1.0f - (bc1.x + bc1.y)) * tcA + bc1.x * tcB + bc1.y * tcC);
                const float2 fPix2 =
                    fTexSize * ((1.0f - (bc2.x + bc2.y)) * tcA + bc2.x * tcB + bc2.y * tcC);

                const uint32_t state = evaluateSingleMicroTriangle(
                    fPix0, fPix1, fPix2,
                    texture, texSize, numChannels, alphaChannelIdx);

                const uint32_t binIdx = microTriIdx / 16;
                const uint32_t offsetInBin = 2 * (microTriIdx % 16);
                atomicOr(
                    reinterpret_cast<uint32_t*>(opacityMicroMap) + binIdx,
                    state << offsetInBin);
            }
        }
    }
}