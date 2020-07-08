#pragma once

#include "uber_shared.h"

extern "C" __global__ void postProcess(
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
    optixu::NativeBlockBuffer2D<float4> accumBuffer,
#else
    optixu::BlockBuffer2D<float4, 1> accumBuffer,
#endif
    uint32_t imageSizeX, uint32_t imageSizeY, uint32_t numAccumFrames,
    CUsurfObject outputBuffer) {
    uint32_t ipx = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t ipy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ipx >= imageSizeX || ipy >= imageSizeY)
        return;
    float3 pix = getXYZ(accumBuffer.read(make_uint2(ipx, ipy))) / (float)numAccumFrames;
    pix.x = 1 - std::exp(-pix.x);
    pix.y = 1 - std::exp(-pix.y);
    pix.z = 1 - std::exp(-pix.z);
    surf2Dwrite(make_float4(pix, 1.0f), outputBuffer, ipx * sizeof(float4), ipy);
}
