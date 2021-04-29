#pragma once

#include "uber_shared.h"

CUDA_DEVICE_KERNEL void deform(
    const Shared::Vertex* originalVertices, Shared::Vertex* vertices, uint32_t numVertices,
    float t) {
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;

    float3 spherePos = 3 * normalize(originalVertices[vIdx].position);
    vertices[vIdx].position = (1 - t) * originalVertices[vIdx].position + t * spherePos;
    vertices[vIdx].normal = make_float3(0, 0, 0);
}

CUDA_DEVICE_KERNEL void accumulateVertexNormals(
    Shared::Vertex* vertices,
    Shared::Triangle* triangles, uint32_t numTriangles) {
    uint32_t triIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (triIdx >= numTriangles)
        return;

    const Shared::Triangle &tri = triangles[triIdx];
    Shared::Vertex &v0 = vertices[tri.index0];
    Shared::Vertex &v1 = vertices[tri.index1];
    Shared::Vertex &v2 = vertices[tri.index2];

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

CUDA_DEVICE_KERNEL void normalizeVertexNormals(Shared::Vertex* vertices, uint32_t numVertices) {
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
