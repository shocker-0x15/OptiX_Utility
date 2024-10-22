#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;



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
        CUDA_COMMON_FUNCTION CUDA_INLINE PCG32RNG() {}

        CUDA_COMMON_FUNCTION CUDA_INLINE void setState(uint32_t _state) { state = _state; }

        CUDA_COMMON_FUNCTION CUDA_INLINE uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_COMMON_FUNCTION CUDA_INLINE float getFloat0cTo1o() {
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


    
    struct GeometryInstanceData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
        uint32_t matIndex;
    };

    struct MaterialData;

    struct BSDFData {
        union {
            struct LambertBRDFData {
                float3 reflectance;
            } asLambertBRDF;
            struct MirrorBRDFData {
                float3 f0Reflectance;
            } asMirrorBRDF;
            struct GlassBSDFData {
                float ior;
            } asGlassBSDF;
        };

        CUDA_DEVICE_FUNCTION CUDA_INLINE BSDFData() {}
    };

    // JP: 各種Callable Programのシグネチャーを定義する。
    // EN: Define the signatures of callable programs.
    using SetUpBSDF = optixu::DirectCallableProgramID<
        void(const MaterialData &matData, const float2 &texCoord, BSDFData* bsdfData)>;
    using BSDF_sampleF = optixu::DirectCallableProgramID<float3(
        const BSDFData &bsdfData, const float3 &givenLocalDir, const float uDir[2],
        float3* sampledDir, float* probDens, bool* deltaSampled)>;
    using BSDF_evaluateF = optixu::DirectCallableProgramID<float3(
        const BSDFData &bsdfData, const float3 &givenLocalDir,
        const float3 &sampledDir)>;

    struct MaterialData {
        union {
            struct Matte {
                CUtexObject texture;
                float3 reflectance;
            } asMatte;
            struct Mirror {
                CUtexObject texture;
                float3 f0Reflectance;
            } asMirror;
            struct Glass {
                float ior;
            } asGlass;
        };

        // JP: これらCallable Programはメモリ上ではただのSBT中のインデックスである。
        // EN: These callable programs are just indices of the SBT on the memory.
        SetUpBSDF setUpBSDF;
        BSDF_sampleF sampleF;
        BSDF_evaluateF evaluateF;

        bool isEmitter;

        MaterialData() {}
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        ROBuffer<MaterialData> materialBuffer;
        optixu::BlockBuffer2D<PCG32RNG, 1> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> accumBuffer;
        uint32_t numAccumFrames;
        PerspectiveCamera camera;
    };



    struct SearchRayPayload {
        float3 alpha;
        float3 contribution;
        float3 origin;
        float3 direction;
        uint32_t pathLength : 30;
        uint32_t terminate : 1;
        uint32_t deltaSampled : 1;
    };

    using SearchRayPayloadSignature = optixu::PayloadSignature<PCG32RNG, SearchRayPayload*>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;
}

