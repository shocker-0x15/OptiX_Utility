#include "temporal_denoiser_shared.h"

CUDA_DEVICE_KERNEL void copyToOutputBuffer(void* linearBuffer,
                                           Shared::BufferToDisplay bufferTypeToDisplay,
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
    case Shared::BufferToDisplay::NoisyBeauty:
    case Shared::BufferToDisplay::DenoisedBeauty: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        // simple tone-map
        value.x = 1 - std::exp(-value.x);
        value.y = 1 - std::exp(-value.y);
        value.z = 1 - std::exp(-value.z);
        break;
    }
    case Shared::BufferToDisplay::Albedo: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        break;
    }
    case Shared::BufferToDisplay::Normal: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        value = typedLinearBuffer[linearIndex];
        value.x = 0.5f + 0.5f * value.x;
        value.y = 0.5f + 0.5f * value.y;
        value.z = 0.5f + 0.5f * value.z;
        break;
    }
    case Shared::BufferToDisplay::Flow: {
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
