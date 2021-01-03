#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;



    enum PickRayType {
        PickRayType_Primary = 0,
        NumPickRayTypes
    };
    
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
    };

    struct EquirectangularCamera {
        float horizentalExtent;
        float verticalExtent;
    };


    
    struct MaterialData {
        uint32_t matID;
        float3 color;
    };
    
    struct GeometryData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
        uint32_t geomID;
    };

    struct GASData {
        uint32_t gasID;
    };



    struct PickInfo {
        uint32_t instanceIndex;
        uint32_t matIndex;
        uint32_t primIndex;
        uint32_t instanceID;
        uint32_t gasID;
        uint32_t geomID;
        uint32_t matID;
        unsigned int hit : 1;
    };



    struct PickPipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        PerspectiveCamera perspCamera;
        EquirectangularCamera equirecCamera;
        float3 position;
        Matrix3x3 orientation;
        int2 mousePosition;
        PickInfo* pickInfo;
    };
    
    struct RenderPipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        PerspectiveCamera perspCamera;
        EquirectangularCamera equirecCamera;
        float3 position;
        Matrix3x3 orientation;
        float colorInterp;
        const PickInfo* pickInfo;
        optixu::NativeBlockBuffer2D<float4> resultBuffer;
    };
}
