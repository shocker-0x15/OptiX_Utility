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

    struct GASChildData {
        float3 scale;
        Quaternion orientation;
        float3 translation;

        CUDA_COMMON_FUNCTION void setPreTransform(
            const float3 &_scale, const float _rollPitchYaw[3], const float3 &_trans) {
            scale = _scale;
            orientation = qFromEulerAngles(_rollPitchYaw[0],
                                           _rollPitchYaw[1],
                                           _rollPitchYaw[2]);
            translation = _trans;
        }

        CUDA_COMMON_FUNCTION float3 transformNormalFromObjectToWorld(const float3 &n) const {
            float3 sn = n / scale;
            return orientation.toMatrix3x3() * sn;
        }
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> resultBuffer;
        PerspectiveCamera camera;
    };

    using PayloadSignature = optixu::PayloadSignature<float3>;
}

