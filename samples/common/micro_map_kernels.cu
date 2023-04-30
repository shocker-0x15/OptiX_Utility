#include "micro_map_generator_private.h"

using namespace shared;

CUDA_DEVICE_KERNEL void initializeHalfEdges(
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    DirectedEdge* edges, uint32_t* halfEdgeIndices, HalfEdge* halfEdges) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);
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
    const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    TriTexCoordTuple* triTcTuples, uint32_t* triIndices) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);
    TriTexCoordTuple tuple{
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[0]]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[1]]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.indices[2]]),
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
