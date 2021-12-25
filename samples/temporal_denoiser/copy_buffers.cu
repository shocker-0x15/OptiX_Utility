#include "temporal_denoiser_shared.h"

using namespace Shared;

CUDA_DEVICE_KERNEL void copyToLinearBuffers(
    optixu::NativeBlockBuffer2D<float4> colorAccumBuffer,
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

CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    void* linearBuffer,
    BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer,
    uint2 imageSize) {
    uint2 launchIndex = make_uint2(blockDim.x * blockIdx.x + threadIdx.x,
                                   blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= imageSize.x ||
        launchIndex.y >= imageSize.y)
        return;

    uint32_t linearIndex = launchIndex.y * imageSize.x + launchIndex.x;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (bufferTypeToDisplay) {
    case BufferToDisplay::NoisyBeauty:
    case BufferToDisplay::DenoisedBeauty: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        // simple tone-map
        value.x = 1 - std::exp(-value.x);
        value.y = 1 - std::exp(-value.y);
        value.z = 1 - std::exp(-value.z);
        break;
    }
    case BufferToDisplay::Albedo: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        break;
    }
    case BufferToDisplay::Normal: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        value.x = 0.5f + 0.5f * value.x;
        value.y = 0.5f + 0.5f * value.y;
        value.z = 0.5f + 0.5f * value.z;
        break;
    }
    case BufferToDisplay::Flow: {
        auto typedLinearBuffer = reinterpret_cast<float2*>(linearBuffer);
        float2 f2Value = typedLinearBuffer[linearIndex];
        value = make_float4(fminf(fmaxf(motionVectorScale * f2Value.x + motionVectorOffset, 0.0f), 1.0f),
                            fminf(fmaxf(motionVectorScale * f2Value.y + motionVectorOffset, 0.0f), 1.0f),
                            motionVectorOffset, 1.0f);
        break;
    }
    default:
        break;
    }
    outputBuffer.write(launchIndex, value);
}
