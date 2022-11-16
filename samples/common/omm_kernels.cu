#include "omm_generator.h"
#include "optix_micromap.h"

using namespace shared;

static constexpr uint32_t WarpSize = 32;

struct Triangle {
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
};

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



// TODO: Conservativeなラスタライザーの実装。

enum class FlatTriangleType {
    BottomFlat,
    TopFlat,
};

template <FlatTriangleType triType, bool ignoreFirstLine>
CUDA_DEVICE_FUNCTION CUDA_INLINE void rasterizeFlatTriangle(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const float2 ps[2],
    uint32_t* numTransparentPixels, uint32_t* numPixels) {
    float invBeginSlope;
    float invEndSlope;
    float curXFBegin;
    float curXFEnd;
    int32_t curY;
    int32_t yEnd;
    if constexpr (triType == FlatTriangleType::BottomFlat) {
        invBeginSlope = (ps[1].x - ps[0].x) / (ps[1].y - ps[0].y);
        invEndSlope = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);

        curXFBegin = ps[0].x;
        curXFEnd = ps[0].x;
        curY = static_cast<int32_t>(ps[0].y);
        yEnd = static_cast<int32_t>(ps[1].y);
    }
    else /*if constexpr (triType == FlatTriangleType::TopFlat)*/ {
        invEndSlope = -(ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);
        invBeginSlope = -(ps[2].x - ps[1].x) / (ps[2].y - ps[1].y);

        curXFBegin = ps[2].x;
        curXFEnd = ps[2].x;
        curY = static_cast<int32_t>(ps[2].y);
        yEnd = static_cast<int32_t>(ps[0].y) + ignoreFirstLine;
    }

    /*
    JP: 三角形をラスタライズして各スレッドでテクスチャーフェッチ、総ピクセル数と透明ピクセル数をカウントする。
    EN: Rasterize the triangle and fetch the texture by each thread, then count the total number of pixels
        and the number of transparent pixels.
    */
    int32_t curX = static_cast<int32_t>(curXFBegin);
    int32_t curXEnd = static_cast<int32_t>(curXFEnd);
    uint32_t curNumItemsPerWarp = 0;
    int2 item = make_int2(INT32_MAX, INT32_MAX);
    while (triType == FlatTriangleType::BottomFlat ? curY <= yEnd : curY >= yEnd) {
        // JP: Warpの空いているスレッドにタスクを充填する。
        // EN: Assign tasks to available threads in the warp.
        const uint32_t numItemsToFill =
            min(curXEnd - curX + 1, static_cast<int32_t>(WarpSize - curNumItemsPerWarp));
        if (threadIdx.x >= curNumItemsPerWarp &&
            threadIdx.x < (curNumItemsPerWarp + numItemsToFill))
            item = make_int2(curX + threadIdx.x - curNumItemsPerWarp, curY);
        curNumItemsPerWarp += numItemsToFill;
        *numPixels += numItemsToFill;

        // JP: Warpがいっぱいになったらテクスチャーフェッチを実行する。
        // EN: Once the warp becomes full, perform texture fetch.
        if (curNumItemsPerWarp == WarpSize) {
            const float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
            *numTransparentPixels += numTrsInWarp;
            curNumItemsPerWarp = 0;
        }

        curX += numItemsToFill;
        if (curX > curXEnd) {
            curXFBegin += invBeginSlope;
            curXFEnd += invEndSlope;
            curX = static_cast<int32_t>(curXFBegin);
            curXEnd = static_cast<int32_t>(curXFEnd);
            if constexpr (triType == FlatTriangleType::BottomFlat)
                ++curY;
            else /*if constexpr (triType == FlatTriangleType::TopFlat)*/
                --curY;
        }
    }
    // JP: 最後に余ったテクスチャーフェッチタスクを実行する。
    // EN: Finally, perform the remaining texture fetch tasks.
    if (curNumItemsPerWarp > 0) {
        float alpha = 1.0f;
        if (threadIdx.x < curNumItemsPerWarp)
            alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
        const uint32_t numTrsInWarp = __popc(__ballot_sync(0xFFFFFFFF, isTransparent(alpha)));
        *numTransparentPixels += numTrsInWarp;
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void evaluateSingleTriangleTransparency(
    uint32_t triIdx,
    const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    uint32_t* numTransparentPixels, uint32_t* numPixels) {
    auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);
    const float2 tcs[] = {
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index0]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index1]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index2]),
    };
    float2 fPixs[3] = {
        make_float2(texSize.x * tcs[0].x, texSize.y * tcs[0].y),
        make_float2(texSize.x * tcs[1].x, texSize.y * tcs[1].y),
        make_float2(texSize.x * tcs[2].x, texSize.y * tcs[2].y)
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

        rasterizeFlatTriangle<FlatTriangleType::TopFlat, false>(
            texture, texSize, numChannels, alphaChannelIdx,
            fPixs,
            numTransparentPixels, numPixels);
    }
    // Bottom-Flat
    else if (fPixs[1].y == fPixs[2].y) {
        // Make the triangle CCW
        if (fPixs[1].x >= fPixs[2].x)
            swap(fPixs[1], fPixs[2]);

        rasterizeFlatTriangle<FlatTriangleType::BottomFlat, false>(
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

        rasterizeFlatTriangle<FlatTriangleType::BottomFlat, false>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            numTransparentPixels, numPixels);

        ps[0] = newP;
        ps[1] = fPixs[1];
        ps[2] = fPixs[2];
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeFlatTriangle<FlatTriangleType::TopFlat, true>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            numTransparentPixels, numPixels);
    }
}

CUDA_DEVICE_KERNEL void countOMMFormats(
    const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    OMMFormat minSubdivLevel, OMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    bool useIndexBuffer, volatile uint32_t* numFetchedTriangles,
    uint32_t* ommFormatCounts, uint32_t* perTriInfos, uint32_t* hasOmmFlags, uint64_t* ommSizes) {
    while (true) {
        uint32_t curNumFetches;
        if (threadIdx.x == 0)
            curNumFetches = *numFetchedTriangles;
        curNumFetches = __shfl_sync(0xFFFFFFFF, curNumFetches, 0);
        if (curNumFetches >= numTriangles)
            return;

        constexpr uint32_t numTrisPerFetch = 8;
        uint32_t baseTriIdx;
        if (threadIdx.x == 0)
            baseTriIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), numTrisPerFetch);
        baseTriIdx = __shfl_sync(0xFFFFFFFF, baseTriIdx, 0);

        for (uint32_t triSubIdx = 0; triSubIdx < numTrisPerFetch; ++triSubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t triIdx = baseTriIdx + triSubIdx;
            if (triIdx >= numTriangles)
                return;

            uint32_t numTransparentPixels;
            uint32_t numPixels;
            evaluateSingleTriangleTransparency(
                triIdx,
                texCoords, vertexStride, triangles, triangleStride, numTriangles,
                texture, texSize, numChannels, alphaChannelIdx,
                &numTransparentPixels, &numPixels);

            if (threadIdx.x == 0) {
                uint32_t state;
                if (numTransparentPixels == 0)
                    state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
                else if (numTransparentPixels == numPixels)
                    state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
                else if (2 * numTransparentPixels < numPixels)
                    state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
                else
                    state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;

                // JP: ラスタライズされた結果のテクセル数から分割レベルを計算する。
                //     他にも良い指標があるかもしれない。
                // EN: Determine the subdivision level from the number of texels computed from the rasterization.
                //     There may be other good parameters.
                const bool isSingleState =
                    state == OPTIX_OPACITY_MICROMAP_STATE_OPAQUE ||
                    state == OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
                const int32_t minLevel = static_cast<int32_t>(minSubdivLevel);
                const int32_t maxLevel = static_cast<int32_t>(maxSubdivLevel);
                const int32_t level = isSingleState ? 0 :
                    min(max(static_cast<int32_t>(
                        std::log(static_cast<float>(numPixels)) / std::log(4.0f)
                        ) - 4 + subdivLevelBias, minLevel), maxLevel); // -4: ad-hoc offset
                const OMMFormat singleStateFormat = useIndexBuffer ? OMMFormat_None : OMMFormat_Level0;
                atomicAdd(&ommFormatCounts[isSingleState ? singleStateFormat : level], 1u);

                // JP: Dword単位に切り上げた三角形のOMMサイズを記録する。
                //     インデックスバッファーを使わない場合、すべての三角形がOMMを持つ。
                // EN: Record the OMM size of the triangle with rounding up to in Dwords.
                //     Every triangle has an OMM when not using an index buffer.
                // TODO: バイト単位にする？
                const uint32_t ommSizeInBits = isSingleState ?
                    (useIndexBuffer ? 0 : 2) :
                    2 * (1 << (2 * level));
                const uint32_t ommSizeInDwords = (ommSizeInBits + 31) / 32;
                hasOmmFlags[triIdx] = ommSizeInDwords > 0;
                ommSizes[triIdx] = 4 * ommSizeInDwords;

                // JP: 三角形の状態を記録する。
                // EN: Record the triangle state.
                PerTriInfo triInfo = {};
                triInfo.state = state;
                triInfo.level = level;
                const uint32_t triInfoBinIdx = triIdx / 4;
                const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
                atomicOr(&perTriInfos[triInfoBinIdx], triInfo.asUInt << offsetInTriInfoBin);
            }
        }
    }
}



CUDA_DEVICE_KERNEL void createOMMDescriptors(
    const uint32_t* perTriInfos, const uint32_t* triToOmmMap, const uint64_t* ommOffsets, uint32_t numTriangles,
    bool useIndexBuffer,
    OptixOpacityMicromapDesc* ommDescs, void* ommIndices, uint32_t ommIndexSize) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const auto ommIndices8 = reinterpret_cast<int8_t*>(ommIndices);
    const auto ommIndices16 = reinterpret_cast<int16_t*>(ommIndices);
    const auto ommIndices32 = reinterpret_cast<int32_t*>(ommIndices);

    const uint32_t triInfoBinIdx = triIdx / 4;
    const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
    PerTriInfo triInfo;
    triInfo.asUInt = (perTriInfos[triInfoBinIdx] >> offsetInTriInfoBin) & 0xFF;

    if (useIndexBuffer &&
        (triInfo.state == OPTIX_OPACITY_MICROMAP_STATE_OPAQUE ||
         triInfo.state == OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT)) {
        // OPTIX_OPACITY_MICROMAP_PREDEFINED_INDEX
        const int32_t ommIndex = -1 - static_cast<int32_t>(triInfo.state);
        if (ommIndexSize == 1)
            ommIndices8[triIdx] = ommIndex;
        else if (ommIndexSize == 2)
            ommIndices16[triIdx] = ommIndex;
        else/* if (ommIndexSize == 4)*/
            ommIndices32[triIdx] = ommIndex;
        return;
    }

    const int32_t ommIdx = static_cast<int32_t>(triToOmmMap[triIdx]);
    OptixOpacityMicromapDesc &ommDesc = ommDescs[ommIdx];
    ommDesc.byteOffset = ommOffsets[triIdx];
    ommDesc.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
    ommDesc.subdivisionLevel = triInfo.level;

    if (useIndexBuffer) {
        if (ommIndexSize == 1)
            ommIndices8[triIdx] = ommIdx;
        else if (ommIndexSize == 2)
            ommIndices16[triIdx] = ommIdx;
        else/* if (ommIndexSize == 4)*/
            ommIndices32[triIdx] = ommIdx;
    }
}



template <FlatTriangleType triType, bool ignoreFirstLine>
CUDA_DEVICE_FUNCTION CUDA_INLINE void rasterizeFlatMicroTriangle(
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const float2 ps[2],
    uint32_t* numTransparentPixels, uint32_t* numPixels) {
    float invBeginSlope;
    float invEndSlope;
    float curXFBegin;
    float curXFEnd;
    int32_t curY;
    int32_t yEnd;
    if constexpr (triType == FlatTriangleType::BottomFlat) {
        invBeginSlope = (ps[1].x - ps[0].x) / (ps[1].y - ps[0].y);
        invEndSlope = (ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);

        curXFBegin = ps[0].x;
        curXFEnd = ps[0].x;
        curY = static_cast<int32_t>(ps[0].y);
        yEnd = static_cast<int32_t>(ps[1].y);
    }
    else /*if constexpr (triType == FlatTriangleType::TopFlat)*/ {
        invEndSlope = -(ps[2].x - ps[0].x) / (ps[2].y - ps[0].y);
        invBeginSlope = -(ps[2].x - ps[1].x) / (ps[2].y - ps[1].y);

        curXFBegin = ps[2].x;
        curXFEnd = ps[2].x;
        curY = static_cast<int32_t>(ps[2].y);
        yEnd = static_cast<int32_t>(ps[0].y) + ignoreFirstLine;
    }

    while (triType == FlatTriangleType::BottomFlat ? curY <= yEnd : curY >= yEnd) {
        int32_t curXBegin = static_cast<int32_t>(curXFBegin);
        int32_t curXEnd = static_cast<int32_t>(curXFEnd);
        for (int32_t curX = curXBegin; curX <= curXEnd; ++curX) {
            int2 item = make_int2(curX, curY);
            float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            if (isTransparent(alpha))
                ++*numTransparentPixels;
        }
        curXFBegin += invBeginSlope;
        curXFEnd += invEndSlope;
        if constexpr (triType == FlatTriangleType::BottomFlat)
            ++curY;
        else /*if constexpr (triType == FlatTriangleType::TopFlat)*/
            --curY;

        *numPixels += curXEnd - curXBegin + 1;
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE uint32_t evaluateSingleMicroTriangle(
    float2 fPixA, float2 fPixB, float2 fPixC,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx) {
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

    uint32_t numTrPixels = 0;
    uint32_t numPixels = 0;

    // Top-Flat
    if (ps[0].y == ps[1].y) {
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeFlatMicroTriangle<FlatTriangleType::TopFlat, false>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            &numTrPixels, &numPixels);
    }
    // Bottom-Flat
    else if (ps[1].y == ps[2].y) {
        // Make the triangle CCW
        if (ps[1].x >= ps[2].x)
            swap(ps[1], ps[2]);

        rasterizeFlatMicroTriangle<FlatTriangleType::BottomFlat, false>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            &numTrPixels, &numPixels);
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

        rasterizeFlatMicroTriangle<FlatTriangleType::BottomFlat, false>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            &numTrPixels, &numPixels);

        ps[0] = newP;
        ps[1] = fPixB;
        ps[2] = fPixC;
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeFlatMicroTriangle<FlatTriangleType::TopFlat, true>(
            texture, texSize, numChannels, alphaChannelIdx,
            ps,
            &numTrPixels, &numPixels);
    }

    uint32_t state;
    if (numTrPixels == 0)
        state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
    else if (numTrPixels == numPixels)
        state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
    else if (2 * numTrPixels < numPixels)
        state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
    else
        state = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;

    return state;
}

CUDA_DEVICE_KERNEL void evaluateMicroTriangleTransparencies(
    const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    const uint32_t* perTriInfos, const uint64_t* ommOffsets,
    volatile uint32_t* numFetchedTriangles,
    uint8_t* opacityMicroMaps) {
    while (true) {
        if (*numFetchedTriangles >= numTriangles)
            return;

        constexpr uint32_t numTrisPerFetch = 8;
        uint32_t baseTriIdx;
        if (threadIdx.x == 0)
            baseTriIdx = atomicAdd(const_cast<uint32_t*>(numFetchedTriangles), numTrisPerFetch);
        baseTriIdx = __shfl_sync(0xFFFFFFFF, baseTriIdx, 0);

        for (uint32_t triSubIdx = 0; triSubIdx < numTrisPerFetch; ++triSubIdx) {
            // JP: Warp中の全スレッドが同じ三角形を処理する。
            // EN: All the threads in a warp process the same triangle.
            const uint32_t triIdx = baseTriIdx + triSubIdx;
            if (triIdx >= numTriangles)
                return;

            const uint32_t triInfoBinIdx = triIdx / 4;
            const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
            PerTriInfo triInfo;
            triInfo.asUInt = (perTriInfos[triInfoBinIdx] >> offsetInTriInfoBin) & 0xFF;

            auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);
            const float2 tcA = reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index0]);
            const float2 tcB = reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index1]);
            const float2 tcC = reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index2]);
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

            const uint32_t numMicroTris = 1 << (2 * triInfo.level);
            for (uint32_t microTriBaseIdx = 0; microTriBaseIdx < numMicroTris; microTriBaseIdx += WarpSize) {
                // JP: 各スレッドがMicro Triangleのステートを計算する。
                // EN: Each thread computes the state of a micro triangle.
                // TODO: Upper/Lower Micro-Triangleを適切に振り分けてDivergenceを抑える。
                const uint32_t microTriIdx = microTriBaseIdx + threadIdx.x;
                if (microTriIdx >= numMicroTris)
                    break;

                float2 bc0, bc1, bc2;
                optixMicromapIndexToBaseBarycentrics(microTriIdx, triInfo.level, bc0, bc1, bc2);

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