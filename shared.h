#pragma once

#include "optix_util.h"

//#define USE_BUFFER2D

RT_FUNCTION float3 getXYZ(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}

RT_FUNCTION float4 make_float4(const float3 &v, float w) {
    return make_float4(v.x, v.y, v.z, w);
}

RT_FUNCTION float2 operator-(const float2 &v) {
    return make_float2(-v.x, -v.y);
}
RT_FUNCTION float2 operator+(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x + v1.x, v0.y + v1.y);
}
RT_FUNCTION float2 operator-(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x - v1.x, v0.y - v1.y);
}
RT_FUNCTION float2 operator*(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
RT_FUNCTION float2 operator*(float s, const float2 &v) {
    return make_float2(s * v.x, s * v.y);
}
RT_FUNCTION float2 operator*(const float2 &v, float s) {
    return make_float2(s * v.x, s * v.y);
}
RT_FUNCTION float2 operator/(const float2 &v, float s) {
    float r = 1 / s;
    return r * v;
}

RT_FUNCTION float3 operator-(const float3 &v) {
    return make_float3(-v.x, -v.y, -v.z);
}
RT_FUNCTION float3 operator+(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
RT_FUNCTION float3 operator-(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
RT_FUNCTION float3 operator*(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
RT_FUNCTION float3 operator*(float s, const float3 &v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
RT_FUNCTION float3 operator*(const float3 &v, float s) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
RT_FUNCTION float3 operator/(const float3 &v, float s) {
    float r = 1 / s;
    return r * v;
}

RT_FUNCTION float dot(const float3 &v0, const float3 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
RT_FUNCTION float3 cross(const float3 &v0, const float3 &v1) {
    return make_float3(v0.y * v1.z - v0.z * v1.y,
                       v0.z * v1.x - v0.x * v1.z,
                       v0.x * v1.y - v0.y * v1.x);
}
RT_FUNCTION float length(const float3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
RT_FUNCTION float3 normalize(const float3 &v) {
    return v / length(v);
}



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



    class PCG32RNG {
        uint64_t state;

    public:
        RT_FUNCTION PCG32RNG() {}

        RT_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-(int32_t)rot) & 31));
        }

        RT_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
    };



    struct GeometryData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
    };

    struct MaterialData {
        float3 albedo;
        union {
            struct {
                unsigned int program : 16;
                unsigned int texID : 16;
            };
            uint32_t misc;
        };

        MaterialData() :
            albedo(make_float3(0.0f, 0.0f, 0.5f)),
            misc(0xFFFFFFFF) {}
    };



    struct PipelineLaunchParameters {
        const OptixTraversableHandle* travHandles;
        const MaterialData* materialData;
        const GeometryData* geomInstData;
        uint32_t travIndex;
        int2 imageSize;
        uint32_t numAccumFrames;
        PCG32RNG* rngBuffer;
#if defined(USE_BUFFER2D)
        optix::WritableBuffer2D<float4> accumBuffer;
#else
        float4* accumBuffer;
#endif
        PerspectiveCamera camera;
        uint32_t matLightIndex;
        CUtexObject* textures;
    };
}
