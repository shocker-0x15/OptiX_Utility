#pragma once

#include "uber_shared.h"

using namespace Shared;

CUDA_DEVICE_KERNEL void calculateBoundingBoxesForSpheres(const SphereParameter * sphereParameters, AABB * aabbs, uint32_t numSpheres) {
    uint32_t sphereIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (sphereIdx >= numSpheres)
        return;

    const SphereParameter &param = sphereParameters[sphereIdx];
    AABB &aabb = aabbs[sphereIdx];

    float3 hd = make_float3(param.radius, param.radius, param.radius);
    aabb.minP = param.center - hd;
    aabb.maxP = param.center + hd;
}