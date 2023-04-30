#pragma once

#include "common.h"

namespace shared {
    struct DirectedEdge {
        uint32_t vertexIndexA;
        uint32_t vertexIndexB;

        CUDA_DEVICE_FUNCTION bool operator==(const DirectedEdge &r) const {
            return vertexIndexA == r.vertexIndexA && vertexIndexB == r.vertexIndexB;
        }
        CUDA_DEVICE_FUNCTION bool operator!=(const DirectedEdge &r) const {
            return vertexIndexA != r.vertexIndexA || vertexIndexB != r.vertexIndexB;
        }

        CUDA_DEVICE_FUNCTION bool operator<(const DirectedEdge &r) const {
            if (vertexIndexA < r.vertexIndexA)
                return true;
            if (vertexIndexA > r.vertexIndexA)
                return false;

            // vertexIndexA == r.vertexIndexA
            if (vertexIndexB < r.vertexIndexB)
                return true;
            if (vertexIndexB > r.vertexIndexB)
                return false;

            // *this == r
            return false;
        }
        CUDA_DEVICE_FUNCTION bool operator<=(const DirectedEdge &r) const {
            if (vertexIndexA < r.vertexIndexA)
                return true;
            if (vertexIndexA > r.vertexIndexA)
                return false;

            // vertexIndexA == r.vertexIndexA
            if (vertexIndexB < r.vertexIndexB)
                return true;
            if (vertexIndexB > r.vertexIndexB)
                return false;

            // *this == r
            return true;
        }
    };

    struct HalfEdge {
        uint32_t twinHalfEdgeIndex;
        uint32_t orgVertexIndex;
        // 三角形メッシュ限定なので暗黙的に求まる。
        //uint32_t triangleIndex;
        //uint32_t prevHalfEdgeIndex;
        //uint32_t nextHalfEdgeIndex;
    };

    struct TriNeighborList {
        uint32_t neighbors[3];
    };

    struct TriTexCoordTuple {
        float2 tcA;
        float2 tcB;
        float2 tcC;

        CUDA_DEVICE_FUNCTION bool operator==(const TriTexCoordTuple &r) const {
            return tcA == r.tcA && tcB == r.tcB && tcC == r.tcC;
        }
        CUDA_DEVICE_FUNCTION bool operator!=(const TriTexCoordTuple &r) const {
            return tcA != r.tcA || tcB != r.tcB || tcC != r.tcC;
        }

        CUDA_DEVICE_FUNCTION bool operator<(const TriTexCoordTuple &r) const {
            if (tcA.x < r.tcA.x)
                return true;
            if (tcA.x > r.tcA.x)
                return false;
            if (tcA.y < r.tcA.y)
                return true;
            if (tcA.y > r.tcA.y)
                return false;

            // tcA == r.tcA
            if (tcB.x < r.tcB.x)
                return true;
            if (tcB.x > r.tcB.x)
                return false;
            if (tcB.y < r.tcB.y)
                return true;
            if (tcB.y > r.tcB.y)
                return false;

            // tcA == r.tcA && tcB == r.tcB
            if (tcC.x < r.tcC.x)
                return true;
            if (tcC.x > r.tcC.x)
                return false;
            if (tcC.y < r.tcC.y)
                return true;
            if (tcC.y > r.tcC.y)
                return false;

            // *this == r
            return false;
        }
    };
}

#if !defined(__CUDA_ARCH__)

template <typename T = void>
inline auto allocate(uintptr_t &curOffset, size_t numElems = 1, size_t alignment = alignof(T))
-> std::conditional_t<std::is_same_v<T, void>, uintptr_t, T*> {
    uint64_t mask = alignment - 1;
    uintptr_t ret = (curOffset + mask) & ~mask;
    if constexpr (std::is_same_v<T, void>) {
        curOffset = ret + numElems;
        return ret;
    }
    else {
        curOffset = ret + sizeof(T) * numElems;
        return reinterpret_cast<T*>(ret);
    }
}

template <typename T>
static void read(std::vector<T> &dataOnHost, T* dataOnDevice) {
    CUDADRV_CHECK(cuMemcpyDtoH(
        dataOnHost.data(), reinterpret_cast<CUdeviceptr>(dataOnDevice),
        sizeof(T) * dataOnHost.size()));
}



void initializeMicroMapGeneratorKernels(const std::filesystem::path &ptxDirPath);



size_t __getScratchMemSizeForSortDirectedEdges(uint32_t numEdges);

void __sortDirectedEdges(
    shared::DirectedEdge* edges, uint32_t* edgeIndices, uint32_t numEdges,
    void* scratchMem, size_t scratchMemSize);



size_t __getScratchMemSizeForSortTriTexCoordTuples(uint32_t numTriangles);

void __sortTriTexCoordTuples(
    shared::TriTexCoordTuple* tuples, uint32_t* triIndices, uint32_t numTriangles,
    void* scratchMem, size_t scratchMemSize);

#endif
