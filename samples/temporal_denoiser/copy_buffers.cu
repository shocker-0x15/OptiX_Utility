#include "temporal_denoiser_shared.h"

CUDA_DEVICE_KERNEL void copyBuffers(optixu::NativeBlockBuffer2D<float4> colorAccumBuffer,
                                    optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer,
                                    optixu::NativeBlockBuffer2D<float4> normalAccumBuffer,
                                    float4* linearColorBuffer,
                                    float4* linearAlbedoBuffer,
                                    float4* linearNormalBuffer,
                                    uint2 imageSize) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    linearColorBuffer[linearIndex] = colorAccumBuffer.read(launchIndex);
    linearAlbedoBuffer[linearIndex] = albedoAccumBuffer.read(launchIndex);
    float3 normal = getXYZ(normalAccumBuffer.read(launchIndex));
    if (normal.x != 0 || normal.y != 0 || normal.z != 0)
        normal = normalize(normal);
    linearNormalBuffer[linearIndex] = make_float4(normal, 1.0f);
}
