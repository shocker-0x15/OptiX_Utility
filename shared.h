#pragma once

#include <optix.h>
#include <cstdint>

#if defined(__CUDA_ARCH__)
#   define RT_FUNCTION __forceinline__ __device__
#   define RT_PROGRAM extern "C" __global__
#else
#   define RT_FUNCTION
#endif

namespace Shared {
    enum RayType {
        RayType_Search = 0,
        RayType_Visibility,
        NumRayTypes
    };



    struct Vertex {
        float3 position;
        float3 normal;
        float2 texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };



    class PCG32RNG {
        uint64_t state;

    public:
        RT_FUNCTION PCG32RNG() {}

        RT_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
        }

        RT_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
    };



    struct GeometryData {
        Vertex* vertexBuffer;
        Triangle* triangleBuffer;
    };

    struct MaterialData {
        float3 albedo;
    };

    struct HitGroupData {
        GeometryData geom;
        MaterialData mat;
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle topGroup;
        int2 imageSize;
        uint32_t numAccumFrames;
        PCG32RNG* rngBuffer;
        float4* outputBuffer;
        PerspectiveCamera camera;
    };
}
