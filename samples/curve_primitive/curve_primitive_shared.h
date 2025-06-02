#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;

    static constexpr bool useEmbeddedVertexData = true;



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

    struct RibbonVertex {
        float3 position;
        float3 normal;
        float width;
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;
    };



    union GeometryData {
        struct {
            ROBuffer<Vertex> vertexBuffer;
            ROBuffer<Triangle> triangleBuffer;
        };
        struct {
            ROBuffer<CurveVertex> curveVertexBuffer;
            ROBuffer<uint32_t> segmentIndexBuffer;
        };
        struct {
            ROBuffer<RibbonVertex> ribbonVertexBuffer;
            ROBuffer<uint32_t> segmentIndexBuffer;
        };
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize; // Note that CUDA/OptiX built-in vector types with width 2 require 8-byte alignment.
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
        float2 subPixelOffset;
        uint32_t sampleIndex : 8;
        uint32_t enableRocapsRefinement : 1;
    };

    using MyPayloadSignature = optixu::PayloadSignature<float3>;
}
