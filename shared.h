#pragma once

#include <optix.h>
#include <cstdint>

namespace Shared {
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



    struct PerspectiveCamera {
        float aspect;
        float fovY;
    };

    struct RayGenData {
        PerspectiveCamera camera;
    };

    struct MissData {
        float3 bgRadiance;
    };

    struct GeometryData {
        Vertex* vertexBuffer;
        Triangle* triangleBuffer;
    };

    struct MaterialData {
        float3 albedo;
    };

    struct HitGroupData {
        GeometryData geom;
        MaterialData mat;
    };

    struct PipelineLaunchParameters {
        OptixTraversableHandle topGroup;
        int2 imageSize;
        float4* outputBuffer;
    };
}
