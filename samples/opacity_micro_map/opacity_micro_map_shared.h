#pragma once

#include "../common/common.h"

static constexpr bool useSimpleScene = false;

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
        float2 texCoord;
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };



    enum OMMFormat : uint32_t {
        OMMFormat_None = 0, // TODO: Level 0は無視？
        OMMFormat_Level1, // 4 micro-tris,
        OMMFormat_Level2, // 16 micro-tris
        OMMFormat_Level3, // 64 micro-tris
        OMMFormat_Level4, // 256 micro-tris
        NumOMMFormats
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;
    };


    
    struct GeometryInstanceData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
        CUtexObject texture;
        float3 albedo;
    };



    enum VisualizationMode : uint32_t {
        VisualizationMode_Final = 0,
        VisualizationMode_NumPrimaryAnyHits,
        VisualizationMode_NumShadowAnyHits,
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
        float3 lightDirection;
        float3 lightRadiance;
        float3 envRadiance;
        unsigned int visualizationMode : 2;
        unsigned int sampleIndex : 8;
        unsigned int superSampleSizeMinus1 : 4;
    };



    using PrimaryRayPayloadSignature =
        optixu::PayloadSignature<float3, uint32_t>;
    using VisibilityRayPayloadSignature =
        optixu::PayloadSignature<float, uint32_t>;
}
