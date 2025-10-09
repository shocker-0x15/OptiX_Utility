#pragma once

#include "../../samples/common/common.h"

namespace shared {
    struct PipelineLaunchParameters0 {
        OptixTraversableHandle travHandle;
    };

    struct PipelineLaunchParameters1 {
        OptixTraversableHandle travHandle;
    };

    using Pipeline0Payload0Signature = optixu::PayloadSignature<uint32_t>;
    using Pipeline1Payload0Signature = optixu::PayloadSignature<float3>;
}
