#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;

    static constexpr bool usePayloadAnnotation = true;



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

        CUDA_COMMON_FUNCTION CUDA_INLINE void setState(uint64_t _state) { state = _state; }

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


    
    struct GeometryData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
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
        uint32_t pathLength : 31;
        uint32_t terminate : 1;
    };

    // JP: 通常のPayloadSignature型の代わりにアクセス情報を記述したAnnotatedPayloadSignature型を使用する。
    // EN: Use AnnotatedPayloadSignature type which describes access information
    //     instead of the ordinary PayloadSignature type.
    using SearchRayPayloadSignature =
        //optixu::PayloadSignature<PCG32RNG, float3, float3, float3, float3, PathFlags>;
        optixu::AnnotatedPayloadSignature<
            optixu::AnnotatedPayload<
                PCG32RNG, // rng
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_NONE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>,
            optixu::AnnotatedPayload<
                float3, // alpha
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
                OPTIX_PAYLOAD_SEMANTICS_CH_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_NONE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>,
            optixu::AnnotatedPayload<
                float3, // contribution
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
                OPTIX_PAYLOAD_SEMANTICS_CH_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>,
            optixu::AnnotatedPayload<
                float3, // origin
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
                OPTIX_PAYLOAD_SEMANTICS_CH_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_NONE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>,
            optixu::AnnotatedPayload<
                float3, // direction
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ |
                OPTIX_PAYLOAD_SEMANTICS_CH_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_NONE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>,
            // JP: Missプログラムではterminateに書込みしか行っていないように見えるが、
            //     flagsはビットフィールドなのでRead-Writeとして取り扱う必要がある。
            // EN: The miss program seems to only write to "terminate" but
            //     since flags is a bit field, it needs to be handled as read-write.
            optixu::AnnotatedPayload<
                PathFlags, // flags
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_CH_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_MS_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_AH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>
        >;
    using VisibilityRayPayloadSignature =
        //optixu::PayloadSignature<float>;
        optixu::AnnotatedPayloadSignature<
            optixu::AnnotatedPayload<
                float, // visibility
                OPTIX_PAYLOAD_SEMANTICS_TRACE_CALLER_READ_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_CH_NONE |
                OPTIX_PAYLOAD_SEMANTICS_MS_NONE |
                OPTIX_PAYLOAD_SEMANTICS_AH_WRITE |
                OPTIX_PAYLOAD_SEMANTICS_IS_NONE>
        >;
}
