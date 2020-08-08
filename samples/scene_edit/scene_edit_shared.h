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
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
    };



    struct alignas(OPTIX_GEOMETRY_TRANSFORM_BYTE_ALIGNMENT) GeometryInstancePreTransform {
        float raw[12];
        float3 scale;
        Quaternion orientation;
        float3 translation;

        GeometryInstancePreTransform() :
            raw{ 1.0f, 0.0f, 0.0f, 0.0f,
                 0.0f, 1.0f, 0.0f, 0.0f,
                 0.0f, 0.0f, 1.0f, 0.0f },
            scale(make_float3(1.0f, 1.0f, 1.0f)) {}
#if defined(__CUDA_ARCH__) || defined(OPTIX_CODE_COMPLETION)
        CUDA_DEVICE_FUNCTION float3 transformNormalFromObjectToWorld(const float3 &n) const {
            float3 sn = n / scale;
            return orientation.toMatrix3x3() * sn;
        }
#endif
    };



    struct GASData {
        const GeometryInstancePreTransform* preTransforms;
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        const GeometryData* geomInstData;
        const GASData* gasData;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> resultBuffer;
        PerspectiveCamera camera;
    };
}
