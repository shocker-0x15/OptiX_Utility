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
    };

    struct Triangle {
        uint32_t index0, index1, index2;
    };

    struct LocalTriangle {
        uint8_t index0, index1, index2;
    };

    struct Sphere {
        float3 center;
        float radius;
    };

    struct Cluster {
        Sphere bounds;
        float error;
        Sphere parentBounds;
        float parentError;
        uint32_t vertPoolStartIndex;
        uint32_t triPoolStartIndex;
        uint32_t childIndexPoolStartIndex;
        uint32_t parentStartClusterIndex;
        uint32_t vertexCount : 12;
        uint32_t triangleCount : 12;
        uint32_t childCount : 4;
        uint32_t parentCount : 4;
        uint32_t padding0;
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;

        CUDA_COMMON_FUNCTION CUDA_INLINE float2 calcScreenPosition(const float3 &posInWorld) const {
            Matrix3x3 invOri = inverse(orientation);
            float3 posInView = invOri * (posInWorld - position);
            float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan(fovY / 2);
            float w = aspect * h;
            return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                               1 - (posAtZ1.y + 0.5f * h) / h);
        }
    };



    struct GeometryData {
        ROBuffer<Vertex> vertexBuffer;
        ROBuffer<Triangle> triangleBuffer;
    };



    enum LoDMode : uint32_t {
        LoDMode_ViewAdaptive = 0,
        LoDMode_ManualUniform,
    };

    enum VisualizationMode : uint32_t {
        VisualizationMode_GeometricNormal = 0,
        VisualizationMode_Cluster,
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
        float2 subPixelOffset;
        uint32_t sampleIndex : 8;
        uint32_t visMode : 3;
    };



    using MyPayloadSignature = optixu::PayloadSignature<uint32_t, float3>;
}
