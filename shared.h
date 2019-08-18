#pragma once

#include <optix.h>
#include <cstdint>

namespace Shared {
    struct PipelineLaunchParameters {
        int2 imageSize;
        float4* outputBuffer;
    };

    struct RayGenData {
        float r, g, b;
    };
}
