#include "micro_map_generator_private.h"

using namespace shared;

struct Triangle {
    uint32_t index0;
    uint32_t index1;
    uint32_t index2;
};

CUDA_DEVICE_KERNEL void extractTexCoords(
    const uint8_t* texCoords, uint64_t vertexStride,
    const uint8_t* triangles, uint64_t triangleStride, uint32_t numTriangles,
    TriTexCoordTuple* triTcTuples, uint32_t* triIndices) {
    const uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    auto &tri = reinterpret_cast<const Triangle &>(triangles[triangleStride * triIdx]);
    TriTexCoordTuple tuple{
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index0]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index1]),
        reinterpret_cast<const float2 &>(texCoords[vertexStride * tri.index2]),
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
