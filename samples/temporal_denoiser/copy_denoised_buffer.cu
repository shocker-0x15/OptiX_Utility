#include "temporal_denoiser_shared.h"

CUDA_DEVICE_KERNEL void copyDenoisedBuffer(float4* linearBuffer,
                                           optixu::NativeBlockBuffer2D<float4> outputBuffer,
                                           uint2 imageSize) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    outputBuffer.write(launchIndex, linearBuffer[linearIndex]);
}
