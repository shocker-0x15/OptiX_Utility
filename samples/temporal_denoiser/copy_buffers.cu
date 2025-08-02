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
    float3 normal = normalize(getXYZ(normalAccumBuffer.read(launchIndex)));
    if (!allFinite(normal))
        normal = make_float3(0, 0, 0);
    linearNormalBuffer[linearIndex] = make_float4(normal, 1.0f);
}

CUDA_DEVICE_KERNEL void visualizeToOutputBuffer(
    void* linearBuffer,
    BufferToDisplay bufferTypeToDisplay,
    float motionVectorOffset, float motionVectorScale,
    optixu::NativeBlockBuffer2D<float4> outputBuffer,
    int2 outputBufferSize, int2 srcImageSize, bool performUpscale, bool useLowResRendering) {
    int2 launchIndex = make_int2(blockDim.x * blockIdx.x + threadIdx.x,
                                 blockDim.y * blockIdx.y + threadIdx.y);
    if (launchIndex.x >= outputBufferSize.x ||
        launchIndex.y >= outputBufferSize.y)
        return;

    int2 pixel = launchIndex;
    if (useLowResRendering && (!performUpscale || bufferTypeToDisplay != BufferToDisplay::DenoisedBeauty)) {
        pixel.x /= 2;
        pixel.y /= 2;
    }

    uint32_t linearIndex = pixel.y * srcImageSize.x + pixel.x;
    float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    switch (bufferTypeToDisplay) {
    case BufferToDisplay::NoisyBeauty:
    case BufferToDisplay::DenoisedBeauty: {
        auto typedLinearBuffer = reinterpret_cast<float4*>(linearBuffer);
        if (bufferTypeToDisplay == BufferToDisplay::DenoisedBeauty && !performUpscale && useLowResRendering) {
            float xInP = (launchIndex.x % 2 + 0.5f) / 2;
            float yInP = (launchIndex.y % 2 + 0.5f) / 2;
            int32_t npx = stc::clamp(pixel.x + ((xInP - 0.5f) > 0 ? 1 : -1), 0, srcImageSize.x - 1);
            int32_t npy = stc::clamp(pixel.y + ((yInP - 0.5f) > 0 ? 1 : -1), 0, srcImageSize.y - 1);
            float s = std::fabs(xInP - 0.5f);
            float t = std::fabs(yInP - 0.5f);
            value =
                (1.0f - s) * (1.0f - t) * typedLinearBuffer[pixel.y * srcImageSize.x + pixel.x] +
                s * (1.0f - t) * typedLinearBuffer[pixel.y * srcImageSize.x + npx] +
                (1.0f - s) * t * typedLinearBuffer[npy * srcImageSize.x + pixel.x] +
                s * t * typedLinearBuffer[npy * srcImageSize.x + npx];
        }
        else {
            value = typedLinearBuffer[linearIndex];
        }
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
