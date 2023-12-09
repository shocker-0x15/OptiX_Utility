#pragma once

#include "../common/common.h"
#include "../common/micro_map/dmm_generator.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;



    enum RayType {
        RayType_Primary = 0,
        RayType_Visibility,
        NumRayTypes
    };



    struct Vertex {
        float3 position;
        float3 normal;
        float3 tc0Direction;
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


    
    struct GeometryInstanceData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;

        // Just for debug visualization
        ROBuffer<OptixDisplacementMicromapDesc> dmmDescBuffer;
        CUdeviceptr dmmIndexBuffer;

        CUtexObject albedoTexture;
        float3 albedo;

        CUtexObject normalTexture;

        uint32_t dmmIndexSize : 3;
    };



    enum VisualizationMode : uint32_t {
        VisualizationMode_Final = 0,
        VisualizationMode_Barycentric,
        VisualizationMode_MicroBarycentric,
        VisualizationMode_SubdivLevel,
        VisualizationMode_Normal,
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
        float3 lightDirection;
        float3 lightRadiance;
        float3 envRadiance;
        float2 subPixelOffset;
        uint32_t visualizationMode : 3;
        uint32_t sampleIndex : 8;
        uint32_t drawBaseEdges : 1;
        uint32_t enableNormalMap : 1;
    };



    struct HitPointFlags {
        uint32_t nearBaseTriEdge : 1;
    };

    using PrimaryRayPayloadSignature = optixu::PayloadSignature<float3, HitPointFlags>;
    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;
}
