#pragma once

#include "../common/common.h"

namespace Shared {
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



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;
    };


    
    struct GeometryData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
        Matrix3x3 matSR_N; // pre-transform normal matrix

        CUDA_COMMON_FUNCTION CUDA_INLINE float3 transformNormal(const float3 &n) const {
            return matSR_N * n;
        }
    };

    struct alignas(OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT) GeometryPreTransform {
        float matSRT[12]; // row major (used by OptiX internal)

        GeometryPreTransform() {}
        GeometryPreTransform(const Matrix3x3 &matSR, const float3 &translate) {
            matSRT[0] = matSR.m00; matSRT[1] = matSR.m01; matSRT[ 2] = matSR.m02; matSRT[ 3] = translate.x;
            matSRT[4] = matSR.m10; matSRT[5] = matSR.m11; matSRT[ 6] = matSR.m12; matSRT[ 7] = translate.y;
            matSRT[8] = matSR.m20; matSRT[9] = matSR.m21; matSRT[10] = matSR.m22; matSRT[11] = translate.z;
        }
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize; // Note that CUDA/OptiX built-in vector types with width 2 require 8-byte alignment.
        optixu::BlockBuffer2D<float4, 1> resultBuffer;
        PerspectiveCamera camera;
    };

    using MyPayloadSignature = optixu::PayloadSignature<float3>;
}
