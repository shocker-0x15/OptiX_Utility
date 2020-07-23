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



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        const GeometryData* geomInstData;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> resultBuffer;
        PerspectiveCamera camera;
    };
}
