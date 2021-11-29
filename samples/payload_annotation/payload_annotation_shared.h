#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;

    static constexpr bool usePayloadAnnotation = true;
#define USE_RAW_INTRINSICS 0



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
        CUDA_DEVICE_FUNCTION PCG32RNG() {}

        void setState(uint32_t _state) { state = _state; }

        CUDA_DEVICE_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_DEVICE_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;
    };


    
    struct GeometryData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
    };

    struct MaterialData {
        CUtexObject texture;
        float3 albedo;
        bool isEmitter;

        MaterialData() :
            texture(0),
            albedo(make_float3(0.0f, 0.0f, 0.5f)),
            isEmitter(false) {}
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        uint32_t numAccumFrames;
        optixu::NativeBlockBuffer2D<PCG32RNG> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
    };



    struct PathFlags {
        unsigned int pathLength : 31;
        unsigned int terminate : 1;
    };
}

#define SearchRayPayloadSignature Shared::PCG32RNG, float3, float3, float3, float3, Shared::PathFlags
#define VisibilityRayPayloadSignature float
