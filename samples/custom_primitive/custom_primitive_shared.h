#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;



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

    struct SphereParameter {
        float3 center;
        float radius;
        float texCoordMultiplier;
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
                const Vertex* vertexBuffer;
                const Triangle* triangleBuffer;
            };
            struct {
                const AABB* aabbBuffer;
                const SphereParameter* paramBuffer;
            };
        };
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        const GeometryData* geomInstData;
        int2 imageSize; // Note that CUDA/OptiX built-in vector types with width 2 require 8-byte alignment.
        optixu::BlockBuffer2D<float4, 1> resultBuffer;
        PerspectiveCamera camera;
    };

    using SphereAttributeSignature = optixu::AttributeSignature<float, float>;
    using PayloadSignature = optixu::PayloadSignature<float3>;
}
