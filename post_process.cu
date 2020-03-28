#pragma once

#include "shared.h"

extern "C" __global__ void postProcess(const float4* accumBuffer, uint32_t imageSizeX, uint32_t imageSizeY, uint32_t numAccumFrames,
                                       float4* outputBuffer) {
    uint32_t ipx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ipy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ipx >= imageSizeX || ipy >= imageSizeY)
        return;
    uint32_t idx = ipy * imageSizeX + ipx;
    float3 pix = getXYZ(accumBuffer[idx]) / (float)numAccumFrames;
    pix.x = 1 - std::exp(-pix.x);
    pix.y = 1 - std::exp(-pix.y);
    pix.z = 1 - std::exp(-pix.z);
    outputBuffer[idx] = make_float4(pix, 1.0f);
}
