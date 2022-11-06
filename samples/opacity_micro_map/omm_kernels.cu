#include "opacity_micro_map_shared.h"

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
            float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            if (isTransparent(alpha))
                atomicAdd(transparentCount, 1u);
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
    if (threadIdx.x < curNumItemsPerWarp) {
        float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
        if (isTransparent(alpha))
            atomicAdd(transparentCount, 1u);
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
    const int32_t yEnd = static_cast<int32_t>(ps[0].y) - ignoreFirstLine;
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
            float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
            if (isTransparent(alpha))
                atomicAdd(transparentCount, 1u);
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
    if (threadIdx.x < curNumItemsPerWarp) {
        float alpha = fetchAlpha(texture, texSize, numChannels, alphaChannelIdx, item);
        if (isTransparent(alpha))
            atomicAdd(transparentCount, 1u);
    }
}

CUDA_DEVICE_FUNCTION CUDA_INLINE void evaluateSingleTriangleTransparency(
    uint32_t triIdx,
    const Vertex* vertices, const Triangle* triangles, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    uint32_t* numFetchedTriangles,
    uint32_t* transparentCounts, uint32_t* numPixelsValues) {
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
    
    uint32_t* const transparentCount = &transparentCounts[triIdx];
    uint32_t numPixels = 0;

    // Top-Flat
    if (fPixs[0].y == fPixs[1].y) {
        // Make the triangle CCW
        if (fPixs[0].x < fPixs[1].x)
            swap(fPixs[0], fPixs[1]);

        rasterizeTopFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            fPixs, false,
            transparentCount, &numPixels);
    }
    // Bottom-Flat
    else if (fPixs[1].y == fPixs[2].y) {
        // Make the triangle CCW
        if (fPixs[1].x >= fPixs[2].x)
            swap(fPixs[1], fPixs[2]);

        rasterizeBottomFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            fPixs,
            transparentCount, &numPixels);
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
            transparentCount, &numPixels);

        ps[0] = newP;
        ps[1] = fPixs[1];
        ps[2] = fPixs[2];
        // Make the triangle CCW
        if (ps[0].x < ps[1].x)
            swap(ps[0], ps[1]);

        rasterizeTopFlatTriangle(
            texture, texSize, numChannels, alphaChannelIdx,
            ps, true,
            transparentCount, &numPixels);
    }

    if (threadIdx.x == 0)
        numPixelsValues[triIdx] = numPixels;
}

CUDA_DEVICE_KERNEL void evaluateTriangleTransparencies(
    const Vertex* vertices, const Triangle* triangles, uint32_t numTriangles,
    CUtexObject texture, uint2 texSize, uint32_t numChannels, uint32_t alphaChannelIdx,
    uint32_t* numFetchedTriangles,
    uint32_t* transparentCounts, uint32_t* numPixelsValues) {
    while (true) {
        if (*numFetchedTriangles >= numTriangles)
            return;

        uint32_t baseTriIdx;
        if (threadIdx.x == 0)
            baseTriIdx = atomicAdd(numFetchedTriangles, WarpSize);
        baseTriIdx = __shfl_sync(0xFFFFFFFF, baseTriIdx, 0);

        for (uint32_t subTriIdx = 0; subTriIdx < WarpSize; ++subTriIdx) {
            const uint32_t triIdx = baseTriIdx + subTriIdx;
            if (triIdx >= numTriangles)
                return;
            evaluateSingleTriangleTransparency(
                triIdx,
                vertices, triangles, numTriangles,
                texture, texSize, numChannels, alphaChannelIdx,
                numFetchedTriangles,
                transparentCounts, numPixelsValues);
        }
    }
}



/*
*/
CUDA_DEVICE_KERNEL void countOMMFormats(
    const uint32_t* transparentCounts, const uint32_t* numPixelsValues, uint32_t numTriangles,
    uint32_t* ommFormatCounts, uint64_t* ommSizes) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;
    const uint32_t trCount = transparentCounts[triIdx];
    const uint32_t numPixels = numPixelsValues[triIdx];
    const bool singleState = trCount == 0 || trCount == numPixels;
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
