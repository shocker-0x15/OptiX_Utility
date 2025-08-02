#pragma once

#include "../common/common.h"

#define USE_NATIVE_BLOCK_BUFFER2D

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

    struct CurveVertex {
        float3 position;
        float width;
        float2 texCoord;
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



    struct SphereParameter {
        float3 center;
        float radius;
        float texCoordMultiplier;
    };



    struct HitPointParameter {
        float b0, b1;
        int32_t primIndex;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE static HitPointParameter get();
#endif
    };


    
    struct GeometryData;
    
    using ProgDecodeHitPoint = optixu::DirectCallableProgramID<void(const HitPointParameter &, const GeometryData &, float3*, float3*, float2*)>;
    
    struct GeometryData {
        union {
            struct {
                ROBuffer<Vertex> vertexBuffer;
                ROBuffer<Triangle> triangleBuffer;
            };
            struct {
                ROBuffer<CurveVertex> curveVertexBuffer;
                ROBuffer<uint32_t> segmentIndexBuffer;
            };
            struct {
                ROBuffer<AABB> aabbBuffer;
                ROBuffer<SphereParameter> paramBuffer;
            };
        };
        ProgDecodeHitPoint decodeHitPointFunc;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION CUDA_INLINE void decodeHitPoint(
            const HitPointParameter &hitPointParam,
            float3* p, float3* sn, float2* texCoord) const {
            if (isnan(hitPointParam.b1)) { // curves
                *p = optixGetWorldRayOrigin() + optixGetRayTmax() * optixGetWorldRayDirection();
                *p = optixTransformPointFromWorldToObjectSpace(*p);
            }
            decodeHitPointFunc(hitPointParam, *this, p, sn, texCoord);
            *p = optixTransformPointFromObjectToWorldSpace(*p);
            *sn = normalize(optixTransformNormalFromObjectToWorldSpace(*sn));
        }
#endif
    };

    struct MaterialData {
        float3 albedo;
        union {
            struct {
                uint32_t program : 16;
                uint32_t texID : 16;
            };
            uint32_t misc;
        };

        MaterialData() :
            albedo(make_float3(0.0f, 0.0f, 0.5f)),
            misc(0xFFFFFFFF) {}
    };



    struct PipelineLaunchParameters {
        const OptixTraversableHandle* travHandles;
        ROBuffer<MaterialData> materialData;
        ROBuffer<GeometryData> geomInstData;
        uint32_t travIndex;
        int2 imageSize;
        uint32_t numAccumFrames;
        optixu::BlockBuffer2D<PCG32RNG, 1> rngBuffer;
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
        optixu::NativeBlockBuffer2D<float4> accumBuffer;
#else
        optixu::BlockBuffer2D<float4, 1> accumBuffer;
#endif
        PerspectiveCamera camera;
        uint32_t matLightIndex;
        CUtexObject* textures;
    };



    struct SearchRayPayload {
        float3 alpha;
        float3 contribution;
        float3 origin;
        float3 direction;
        uint32_t pathLength : 30;
        uint32_t specularBounce : 1;
        uint32_t terminate : 1;
    };

    using SphereAttributeSignature = optixu::AttributeSignature<float, float>;
    using SearchRayPayloadSignature = optixu::PayloadSignature<OptixTraversableHandle, PCG32RNG, SearchRayPayload*>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;
}
