#include "dmm_generator_private.h"
#include "optix_micromap.h"

using namespace shared;

static constexpr uint32_t WarpSize = 32;

CUDA_DEVICE_KERNEL void computeMeshAABB(
    const uint8_t* positions, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    AABBAsOrderedInt* meshAabbAsOrderedInt) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const bool isValidThread = triIdx < numTriangles;

    CUDA_SHARED_MEM uint32_t b_memForBlockAabb[sizeof(AABBAsOrderedInt) / sizeof(uint32_t)];
    auto &blockAabb = reinterpret_cast<AABBAsOrderedInt &>(b_memForBlockAabb);
    if (threadIdx.x == 0)
        blockAabb = AABBAsOrderedInt();
    __syncthreads();

    if (isValidThread) {
        auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);

        const float3 triPositions[] = {
            reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[0]]),
            reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[1]]),
            reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[2]]),
        };

        AABB triAabb;
        triAabb.unify(triPositions[0]).unify(triPositions[1]).unify(triPositions[2]);
        AABBAsOrderedInt triAabbAsOrderedInt = triAabb;
        // Divergent branchの中にあって良い？
        atomicMin_block(&blockAabb.minP.x, triAabbAsOrderedInt.minP.x);
        atomicMin_block(&blockAabb.minP.y, triAabbAsOrderedInt.minP.y);
        atomicMin_block(&blockAabb.minP.z, triAabbAsOrderedInt.minP.z);
        atomicMax_block(&blockAabb.maxP.x, triAabbAsOrderedInt.maxP.x);
        atomicMax_block(&blockAabb.maxP.y, triAabbAsOrderedInt.maxP.y);
        atomicMax_block(&blockAabb.maxP.z, triAabbAsOrderedInt.maxP.z);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicMin(&meshAabbAsOrderedInt->minP.x, blockAabb.minP.x);
        atomicMin(&meshAabbAsOrderedInt->minP.y, blockAabb.minP.y);
        atomicMin(&meshAabbAsOrderedInt->minP.z, blockAabb.minP.z);
        atomicMax(&meshAabbAsOrderedInt->maxP.x, blockAabb.maxP.x);
        atomicMax(&meshAabbAsOrderedInt->maxP.y, blockAabb.maxP.y);
        atomicMax(&meshAabbAsOrderedInt->maxP.z, blockAabb.maxP.z);
    }
}



CUDA_DEVICE_KERNEL void finalizeMeshAABB(
    AABBAsOrderedInt* meshAabbAsOrderedInt,
    AABB* meshAabb, float* meshAabbArea) {
    if (threadIdx.x > 0)
        return;

    *meshAabb = static_cast<AABB>(*meshAabbAsOrderedInt);
    *meshAabbArea = 2 * meshAabb->calcHalfSurfaceArea();
}



CUDA_DEVICE_KERNEL void determineTargetSubdivLevels(
    const float* meshAabbArea,
    const uint8_t* positions, const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    uint2 texSize,
    DMMFormat minSubdivLevel, DMMFormat maxSubdivLevel, int32_t subdivLevelBias,
    uint32_t* perTriInfos) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);

    const float3 triPositions[] = {
        reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[0]]),
        reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[1]]),
        reinterpret_cast<const float3 &>(positions[vertexStride * tri.indices[2]]),
    };
    const float normTriArea = 0.5f * length(cross(
        triPositions[2] - triPositions[0], triPositions[1] - triPositions[0])) / *meshAabbArea;

    const float2 triTexCoords[] = {
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[0]]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[1]]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[2]]),
    };
    const float2 texSizeF = make_float2(texSize.x, texSize.y);
    const float numTexelsF = 0.5f * fabsf(cross(
        texSizeF * (triTexCoords[2] - triTexCoords[0]),
        texSizeF * (triTexCoords[1] - triTexCoords[0])));

    const float targetSubdivLevelF = std::log(numTexelsF) / std::log(4.0f)
        + std::log(normTriArea) / std::log(4.0f) + 4; // +4: ad-hoc offset
    //printf("Tri %u: tgt level: %g\n", triIdx, targetSubdivLevelF);
    const int32_t minLevel = static_cast<int32_t>(minSubdivLevel);
    const int32_t maxLevel = static_cast<int32_t>(maxSubdivLevel);
    const int32_t targetSubdivLevel =
        min(max(static_cast<int32_t>(targetSubdivLevelF) + subdivLevelBias, minLevel), maxLevel);

    PerTriInfo triInfo = {};
    triInfo.level = targetSubdivLevel;
    const uint32_t triInfoBinIdx = triIdx / 4;
    const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
    atomicOr(&perTriInfos[triInfoBinIdx], triInfo.asUInt << offsetInTriInfoBin);
}

CUDA_DEVICE_KERNEL void adjustSubdivLevels(
    const TriNeighborList* triNeighborLists, uint32_t numTriangles,
    const uint32_t* srcPerTriInfos, uint32_t* dstPerTriInfos) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const uint32_t triInfoBinIdx = triIdx / 4;
    const uint32_t offsetInTriInfoBin = 8 * (triIdx % 4);
    PerTriInfo triInfo = {};
    triInfo.asUInt = (srcPerTriInfos[triInfoBinIdx] >> offsetInTriInfoBin) & 0xFF;

    uint32_t maxLevelInNeighbors = 0;
    const TriNeighborList &triNeighborList = triNeighborLists[triIdx];
    for (uint32_t i = 0; i < 3; ++i) {
        uint32_t nbTriIdx = triNeighborList.neighbors[i];
        if (nbTriIdx == 0xFFFFFFFF)
            continue;

        const uint32_t nbTriInfoBinIdx = nbTriIdx / 4;
        const uint32_t nbOffsetInTriInfoBin = 8 * (nbTriIdx % 4);
        PerTriInfo nbTriInfo = {};
        nbTriInfo.asUInt = (srcPerTriInfos[nbTriInfoBinIdx] >> nbOffsetInTriInfoBin) & 0xFF;

        maxLevelInNeighbors = max(nbTriInfo.level, maxLevelInNeighbors);
    }
    triInfo.level = max(triInfo.level, max(maxLevelInNeighbors, 1u) - 1);

    atomicOr(&dstPerTriInfos[triInfoBinIdx], triInfo.asUInt << offsetInTriInfoBin);
}