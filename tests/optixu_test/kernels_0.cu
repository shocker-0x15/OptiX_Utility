#include "shared.h"

using namespace shared;

RT_PIPELINE_LAUNCH_PARAMETERS PipelineLaunchParameters0 plp;

CUDA_DEVICE_KERNEL void RT_RG_NAME(rg0)() {
    uint32_t value = 0;
    optixu::trace<Pipeline0Payload0Signature>(
        plp.travHandle,
        make_float3(0, 0, 0), make_float3(0, 0, 1), 0.0f, INFINITY, 0.0f,
        0xFF, OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        value);
}

CUDA_DEVICE_KERNEL void RT_EX_NAME(ex0)() {
}

CUDA_DEVICE_KERNEL void RT_MS_NAME(ms0)() {
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(ch0)() {
    uint32_t value;
    optixu::getPayloads<Pipeline0Payload0Signature>(&value);
    value += 1;
    optixu::setPayloads<Pipeline0Payload0Signature>(&value);
}

CUDA_DEVICE_KERNEL void RT_CH_NAME(ch1)() {
    uint32_t value;
    optixu::getPayloads<Pipeline0Payload0Signature>(&value);
    value += 2;
    optixu::setPayloads<Pipeline0Payload0Signature>(&value);
}

CUDA_DEVICE_KERNEL void RT_AH_NAME(ah0)() {
}

CUDA_DEVICE_KERNEL void RT_IS_NAME(is0)() {
}

CUDA_DEVICE_KERNEL void RT_DC_NAME(dc0)() {
}

CUDA_DEVICE_KERNEL void RT_CC_NAME(cc0)() {
}