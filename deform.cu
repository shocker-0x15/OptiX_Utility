#pragma once

#include "shared.h"

extern "C" __global__ void deform(const Shared::Vertex* originalVertices, Shared::Vertex* vertices, uint32_t numVertices,
                                  float t) {
    uint32_t vIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (vIdx >= numVertices)
        return;
    
    float3 spherePos = normalize(originalVertices[vIdx].position);
    vertices[vIdx].position = (1 - t) * originalVertices[vIdx].position + t * spherePos;
}
