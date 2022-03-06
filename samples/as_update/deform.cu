#pragma once

#include "as_update_shared.h"

using namespace Shared;

CUDA_DEVICE_KERNEL void deform(
    const Vertex* originalVertices, Vertex* vertices, uint32_t numVertices,
    float amplitude, float t) {
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    float3 orgPos = originalVertices[vIdx].position;

    // JP: ノイズによって頂点に適当な変異を加える。
    // EN: Displace vertices by random amount by noise.
    PerlinNoise3D noiseX(0);
    PerlinNoise3D noiseY(0);
    PerlinNoise3D noiseZ(0);
    float3 epX = orgPos + 100 * make_float3(0.21f, -0.34f, 0.72f);
    float3 epY = orgPos + 100 * make_float3(-0.33f, -0.31f, -0.48f);
    float3 epZ = orgPos + 100 * make_float3(-0.23f, -0.66f, 0.12f);
    float3 displace = make_float3(noiseX.evaluate(epX, 0.025f),
                                  noiseY.evaluate(epY, 0.025f),
                                  noiseZ.evaluate(epZ, 0.025f));
    vertices[vIdx].position = orgPos + amplitude * t * displace;
    vertices[vIdx].normal = make_float3(0, 0, 0);
}

CUDA_DEVICE_KERNEL void accumulateVertexNormals(
    Vertex* vertices, Triangle* triangles, uint32_t numTriangles) {
    uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const Triangle &tri = triangles[triIdx];
    Vertex &v0 = vertices[tri.index0];
    Vertex &v1 = vertices[tri.index1];
    Vertex &v2 = vertices[tri.index2];

    const auto atomicAddNormalAsInt32 = [](float3* dstN, const int3 &vn) {
        atomicAdd(reinterpret_cast<int32_t*>(&dstN->x), vn.x);
        atomicAdd(reinterpret_cast<int32_t*>(&dstN->y), vn.y);
        atomicAdd(reinterpret_cast<int32_t*>(&dstN->z), vn.z);
    };

    float3 vn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
    constexpr int32_t coeffFloatToFixed = 1 << 24;
    int32_t vnx = static_cast<int32_t>(vn.x * coeffFloatToFixed);
    int32_t vny = static_cast<int32_t>(vn.y * coeffFloatToFixed);
    int32_t vnz = static_cast<int32_t>(vn.z * coeffFloatToFixed);
    int3 vnInt32 = make_int3(vnx, vny, vnz);

    atomicAddNormalAsInt32(&v0.normal, vnInt32);
    atomicAddNormalAsInt32(&v1.normal, vnInt32);
    atomicAddNormalAsInt32(&v2.normal, vnInt32);
}

CUDA_DEVICE_KERNEL void normalizeVertexNormals(
    Vertex* vertices, uint32_t numVertices) {
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    float3 vn = vertices[vIdx].normal;
    int32_t vnx = *reinterpret_cast<int32_t*>(&vn.x);
    int32_t vny = *reinterpret_cast<int32_t*>(&vn.y);
    int32_t vnz = *reinterpret_cast<int32_t*>(&vn.z);
    constexpr float coeffFixedToFloat = 1.0f / (1 << 24);
    vn = make_float3(vnx * coeffFixedToFloat,
                     vny * coeffFixedToFloat,
                     vnz * coeffFixedToFloat);
    
    vertices[vIdx].normal = normalize(vn);
}
