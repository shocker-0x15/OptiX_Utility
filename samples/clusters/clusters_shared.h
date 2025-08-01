﻿#pragma once

#include "../common/common.h"

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
        uint32_t level : 6;
        uint32_t padding0 : 26;
    };

    struct ClusterSetInfo {
        OptixClusterAccelBuildInputTrianglesArgs* argsArray; // args array for CLAS set build
        CUdeviceptr* clasHandles; // CLAS handles' destination for CLAS set build
        uint32_t* usedFlags; // Each bit indicates whether the corresponding cluster is used in the frame.
        uint32_t* indexMapClusterToClasBuild; // cluster index => CLAS build index
        uint32_t argsCountToBuild;
    };

    struct InstanceTransform {
        float scale;
        Quaternion orientation;
        float3 position;
    };

    struct InstanceStaticInfo {
        struct {
            uint32_t* indexMapClasHandleToCluster; // CLAS handle index for GAS build => cluster index
            CUdeviceptr* clasHandles; // handles for Cluster GAS build
        } cgas;
    };

    struct InstanceDynamicInfo {
        struct {
            uint32_t clasHandleCount;
        } cgas;
        InstanceTransform transform;
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



    struct NormalMeshData {
        ROBuffer<Vertex> vertices;
        ROBuffer<Triangle> triangles;
        AABB bbox;
    };

    struct ClusteredMeshData {
        ROBuffer<Vertex> vertexPool;
        ROBuffer<LocalTriangle> trianglePool;
        ROBuffer<Cluster> clusters;
        AABB bbox;
    };



    enum LoDMode : uint32_t {
        LoDMode_ViewAdaptive = 0,
        LoDMode_ManualUniform,
    };

    enum VisualizationMode : uint32_t {
        VisualizationMode_Final = 0,
        VisualizationMode_ShadingNormal,
        VisualizationMode_GeometricNormal,
        VisualizationMode_Cluster,
        VisualizationMode_Level,
        VisualizationMode_Triangle,
    };



    struct GeometryConfig {
        LoDMode lodMode;
        uint32_t manualUniformLevel;
        uint32_t positionTruncateBitWidth;
    };



    struct HitInfo {
        const ClusteredMeshData* cMeshData;
        uint32_t instIndex;
        uint32_t clusterId;
        uint32_t primIndex;
        float2 barycentrics;
        float3 shadingNormal;
        float3 geomNormal;
    };

    struct PickInfo {
        uint32_t instanceIndex;
        uint32_t clusterId;
        uint32_t primitiveIndex;
        float2 barycentrics;
        Cluster cluster;
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> colorAccumBuffer;
        PerspectiveCamera camera;
        ROBuffer<InstanceStaticInfo> instStaticInfoBuffer;
        ROBuffer<InstanceDynamicInfo> instDynamicInfoBuffer;
        PickInfo* pickInfo;
        uint2 mousePosition;
        float3 lightDirection;
        float3 lightRadiance;
        float3 envRadiance;
        float2 subPixelOffset;
        uint32_t sampleIndex : 8;
        uint32_t visMode : 3;
        uint32_t posTruncateBitWidth : 5;
    };



    using MyPayloadSignature = optixu::PayloadSignature<HitInfo, float3>;

    using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;
}
