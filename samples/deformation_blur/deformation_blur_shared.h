﻿#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;

    static constexpr bool useEmbeddedVertexData = false;



    enum RayType {
        RayType_Primary = 0,
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
    };

    struct Sphere {
        float3 center;
        float radius;
    };

    struct PartialSphereParameter {
        AABB aabb;
        float3 center;
        float radius;
        float minPhi;
        float maxPhi;
        float minTheta;
        float maxTheta;
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
        union {
            struct {
                ROBuffer<ROBuffer<Vertex>> vertexBuffers;
                ROBuffer<Triangle> triangleBuffer;
            };
            struct {
                ROBuffer<ROBuffer<CurveVertex>> curveVertexBuffers;
                ROBuffer<uint32_t> segmentIndexBuffer;
            };
            struct {
                ROBuffer<ROBuffer<Sphere>> sphereBuffers;
            };
            struct {
                ROBuffer<ROBuffer<PartialSphereParameter>> partialSphereParamBuffers;
            };
        };
        uint32_t numMotionSteps;
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize; // Note that CUDA/OptiX built-in vector types with width 2 require 8-byte alignment.
        optixu::BlockBuffer2D<PCG32RNG, 4> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        float timeBegin;
        float timeEnd;
        uint32_t numAccumFrames;
        PerspectiveCamera camera;
        PCG32RNG globalRNG;
        uint32_t usePerPixelRNGs : 1;
    };

    using PartialSphereAttributeSignature = optixu::AttributeSignature<float, float>;
    using MyPayloadSignature = optixu::PayloadSignature<float3>;
}
