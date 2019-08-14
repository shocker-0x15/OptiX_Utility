#pragma once

#include "shared.h"

using namespace Shared;

extern "C" __constant__ InterfaceVariables iv;

extern "C" __global__
void __raygen__fill() {
    uint3 launchIndex = optixGetLaunchIndex();
    RayGenData* rgData = (RayGenData*)optixGetSbtDataPointer();
    int32_t index = iv.imageSize.x * launchIndex.y + launchIndex.x;
    float x = (float)launchIndex.x / iv.imageSize.x;
    float y = (float)launchIndex.y / iv.imageSize.y;
    iv.outputBuffer[index] = make_float4(rgData->r * x, rgData->g * y, rgData->b, 1.0f);
}
