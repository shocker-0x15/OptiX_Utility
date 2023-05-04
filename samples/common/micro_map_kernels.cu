#include "micro_map_generator_private.h"

using namespace shared;

CUDA_DEVICE_KERNEL void computeMeshAABB(
    StridedBuffer<float3> positions, StridedBuffer<Triangle> triangles,
    AABBAsOrderedInt* meshAabbAsOrderedInt) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const bool isValidThread = triIdx < triangles.numElements;

    CUDA_SHARED_MEM uint32_t b_memForBlockAabb[sizeof(AABBAsOrderedInt) / sizeof(uint32_t)];
    auto &blockAabb = reinterpret_cast<AABBAsOrderedInt &>(b_memForBlockAabb);
    if (threadIdx.x == 0)
        blockAabb = AABBAsOrderedInt();
    __syncthreads();

    if (isValidThread) {
        const Triangle &tri = triangles[triIdx];

        const float3 triPositions[] = {
            positions[tri.indices[0]],
            positions[tri.indices[1]],
            positions[tri.indices[2]],
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



CUDA_DEVICE_KERNEL void initializeHalfEdges(
    StridedBuffer<Triangle> triangles,
    DirectedEdge* edges, uint32_t* halfEdgeIndices, HalfEdge* halfEdges) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= triangles.numElements)
        return;

    const Triangle &tri = triangles[triIdx];
    constexpr uint32_t numFaceVertices = 3;
    const uint32_t edgeBaseIdx = numFaceVertices * triIdx;
    for (uint32_t i = 0; i < numFaceVertices; ++i) {
        uint32_t vIdx = tri.indices[i];
        uint32_t edgeIdx = edgeBaseIdx + i;

        DirectedEdge &edge = edges[edgeIdx];
        edge.vertexIndexA = vIdx;
        edge.vertexIndexB = tri.indices[(i + 1) % numFaceVertices];

        halfEdgeIndices[edgeIdx] = edgeIdx;

        HalfEdge &halfEdge = halfEdges[edgeIdx];
        halfEdge.twinHalfEdgeIndex = 0xFFFFFFFF;
        halfEdge.orgVertexIndex = vIdx;
        //halfEdge.triangleIndex = triIdx;
        //halfEdge.prevHalfEdgeIndex = edgeBaseIdx + (i + (numFaceVertices - 1)) % numFaceVertices;
        //halfEdge.nextHalfEdgeIndex = edgeBaseIdx + (i + 1) % numFaceVertices;
    }
}

CUDA_DEVICE_KERNEL void findTwinHalfEdges(
    const DirectedEdge* sortedEdges, const uint32_t* sortedHalfEdgeIndices,
    HalfEdge* halfEdges, uint32_t numHalfEdges) {
    const uint32_t halfEdgeIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (halfEdgeIdx >= numHalfEdges)
        return;

    HalfEdge &halfEdge = halfEdges[halfEdgeIdx];
    const HalfEdge &nextHalfEdge = halfEdges[halfEdgeIdx + ((halfEdgeIdx + 1) % 3 == 0 ? -2 : 1)];

    DirectedEdge twinEdge;
    twinEdge.vertexIndexA = nextHalfEdge.orgVertexIndex;
    twinEdge.vertexIndexB = halfEdge.orgVertexIndex;
    uint32_t idx = 0;
    bool found = false;
    for (uint32_t d = nextPowerOf2(numHalfEdges) >> 1; d >= 1; d >>= 1) {
        if (idx + d >= numHalfEdges)
            continue;
        if (sortedEdges[idx + d] <= twinEdge) {
            idx += d;
            found = sortedEdges[idx] == twinEdge;
            if (found)
                break;
        }
    }

    if (found)
        halfEdge.twinHalfEdgeIndex = sortedHalfEdgeIndices[idx];
}

CUDA_DEVICE_KERNEL void findTriangleNeighbors(
    const HalfEdge* halfEdges, TriNeighborList* neighborLists, uint32_t numTriangles) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    TriNeighborList &neighborList = neighborLists[triIdx];
    constexpr uint32_t numFaceVertices = 3;
    const uint32_t edgeBaseIdx = numFaceVertices * triIdx;
    for (uint32_t i = 0; i < numFaceVertices; ++i) {
        uint32_t edgeIdx = edgeBaseIdx + i;
        const HalfEdge &halfEdge = halfEdges[edgeIdx];
        uint32_t neighborTriIdx = 0xFFFFFFFF;
        if (halfEdge.twinHalfEdgeIndex != 0xFFFFFFFF)
            neighborTriIdx = halfEdge.twinHalfEdgeIndex / 3;
        neighborList.neighbors[i] = neighborTriIdx;
    }
}



CUDA_DEVICE_KERNEL void extractTexCoords(
    StridedBuffer<float2> texCoords, StridedBuffer<Triangle> triangles,
    TriTexCoordTuple* triTcTuples, uint32_t* triIndices) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= triangles.numElements)
        return;

    const Triangle &tri = triangles[triIdx];
    TriTexCoordTuple tuple{
        texCoords[tri.indices[0]],
        texCoords[tri.indices[1]],
        texCoords[tri.indices[2]],
    };
    triTcTuples[triIdx] = tuple;
    triIndices[triIdx] = triIdx;
}

CUDA_DEVICE_KERNEL void testIfTCTupleIsUnique(
    const TriTexCoordTuple* triTcTuples, uint32_t* refTupleIndices, uint32_t numTriangles) {
    const uint32_t tupleIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tupleIdx >= numTriangles)
        return;

    const TriTexCoordTuple &triTcTuple = triTcTuples[tupleIdx];
    uint32_t refTupleIdx = tupleIdx;
    if (tupleIdx > 0) {
        const TriTexCoordTuple &prevTriTcTuple = triTcTuples[tupleIdx - 1];
        if (triTcTuple == prevTriTcTuple)
            refTupleIdx = 0;
    }

    refTupleIndices[tupleIdx] = refTupleIdx;
}



CUDA_DEVICE_KERNEL void testIfMicroMapKeyIsUnique(
    MicroMapKey* triMicroMapKeys, uint32_t* refKeyIndices, uint32_t numTriangles) {
    const uint32_t keyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (keyIdx >= numTriangles)
        return;

    const MicroMapKey &mmKey = triMicroMapKeys[keyIdx];
    uint32_t refTupleIdx = keyIdx;
    if (keyIdx > 0) {
        const MicroMapKey &prevMmKey = triMicroMapKeys[keyIdx - 1];
        if (mmKey == prevMmKey)
            refTupleIdx = 0;
    }

    refKeyIndices[keyIdx] = refTupleIdx;
}
