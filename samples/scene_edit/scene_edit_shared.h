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
            scale(make_float3(1.0f, 1.0f, 1.0f)), translation(make_float3(1.0f, 1.0f, 1.0f)) {}

        void setSRT(const float3 &_scale, const float _rollPitchYaw[3], const float3 &_trans) {
            scale = _scale;
            orientation = qFromEulerAngles(_rollPitchYaw[0],
                                           _rollPitchYaw[1],
                                           _rollPitchYaw[2]);
            translation = _trans;

            Matrix3x3 matSR = orientation.toMatrix3x3() * scale3x3(scale);
            raw[0] = matSR.m00; raw[1] = matSR.m01; raw[ 2] = matSR.m02; raw[ 3] = translation.x;
            raw[4] = matSR.m10; raw[5] = matSR.m11; raw[ 6] = matSR.m12; raw[ 7] = translation.y;
            raw[8] = matSR.m20; raw[9] = matSR.m21; raw[10] = matSR.m22; raw[11] = translation.z;
        }

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
        int2 imageSize;
        optixu::NativeBlockBuffer2D<float4> resultBuffer;
        PerspectiveCamera camera;
    };
}
