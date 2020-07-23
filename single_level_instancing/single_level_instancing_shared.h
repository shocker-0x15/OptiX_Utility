#pragma once

#include "../optix_util.h"

CUDA_DEVICE_FUNCTION float3 getXYZ(const float4 &v) {
    return make_float3(v.x, v.y, v.z);
}

CUDA_DEVICE_FUNCTION float3 make_float3(float v) {
    return make_float3(v, v, v);
}
CUDA_DEVICE_FUNCTION float4 make_float4(const float3 &v, float w) {
    return make_float4(v.x, v.y, v.z, w);
}

CUDA_DEVICE_FUNCTION float2 operator-(const float2 &v) {
    return make_float2(-v.x, -v.y);
}
CUDA_DEVICE_FUNCTION float2 operator+(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x + v1.x, v0.y + v1.y);
}
CUDA_DEVICE_FUNCTION float2 operator-(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x - v1.x, v0.y - v1.y);
}
CUDA_DEVICE_FUNCTION float2 operator*(const float2 &v0, const float2 &v1) {
    return make_float2(v0.x * v1.x, v0.y * v1.y);
}
CUDA_DEVICE_FUNCTION float2 operator*(float s, const float2 &v) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_DEVICE_FUNCTION float2 operator*(const float2 &v, float s) {
    return make_float2(s * v.x, s * v.y);
}
CUDA_DEVICE_FUNCTION float2 operator/(const float2 &v, float s) {
    float r = 1 / s;
    return r * v;
}

CUDA_DEVICE_FUNCTION float3 operator-(const float3 &v) {
    return make_float3(-v.x, -v.y, -v.z);
}
CUDA_DEVICE_FUNCTION float3 operator+(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x + v1.x, v0.y + v1.y, v0.z + v1.z);
}
CUDA_DEVICE_FUNCTION float3 &operator+=(float3 &v0, const float3 &v1) {
    v0.x += v1.x;
    v0.y += v1.y;
    v0.z += v1.z;
    return v0;
}
CUDA_DEVICE_FUNCTION float3 operator-(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x - v1.x, v0.y - v1.y, v0.z - v1.z);
}
CUDA_DEVICE_FUNCTION float3 operator*(const float3 &v0, const float3 &v1) {
    return make_float3(v0.x * v1.x, v0.y * v1.y, v0.z * v1.z);
}
CUDA_DEVICE_FUNCTION float3 operator*(float s, const float3 &v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_DEVICE_FUNCTION float3 operator*(const float3 &v, float s) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
CUDA_DEVICE_FUNCTION float3 &operator*=(float3 &v, float s) {
    v.x *= s;
    v.y *= s;
    v.z *= s;
    return v;
}
CUDA_DEVICE_FUNCTION float3 operator/(const float3 &v, float s) {
    float r = 1 / s;
    return r * v;
}
CUDA_DEVICE_FUNCTION float3 &operator/=(float3 &v, float s) {
    float r = 1 / s;
    return v *= r;
}

CUDA_DEVICE_FUNCTION float dot(const float3 &v0, const float3 &v1) {
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}
CUDA_DEVICE_FUNCTION float3 cross(const float3 &v0, const float3 &v1) {
    return make_float3(v0.y * v1.z - v0.z * v1.y,
                       v0.z * v1.x - v0.x * v1.z,
                       v0.x * v1.y - v0.y * v1.x);
}
CUDA_DEVICE_FUNCTION float length(const float3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}
CUDA_DEVICE_FUNCTION float sqLength(const float3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}
CUDA_DEVICE_FUNCTION float3 normalize(const float3 &v) {
    return v / length(v);
}



struct Matrix3x3 {
    union {
        struct { float m00, m10, m20; };
        float3 c0;
    };
    union {
        struct { float m01, m11, m21; };
        float3 c1;
    };
    union {
        struct { float m02, m12, m22; };
        float3 c2;
    };

    CUDA_DEVICE_FUNCTION Matrix3x3() :
        c0(make_float3(1, 0, 0)),
        c1(make_float3(0, 1, 0)),
        c2(make_float3(0, 0, 1)) { }
    CUDA_DEVICE_FUNCTION Matrix3x3(const float array[9]) :
        m00(array[0]), m10(array[1]), m20(array[2]),
        m01(array[3]), m11(array[4]), m21(array[5]),
        m02(array[6]), m12(array[7]), m22(array[8]) { }
    CUDA_DEVICE_FUNCTION Matrix3x3(const float3 &col0, const float3 &col1, const float3 &col2) :
        c0(col0), c1(col1), c2(col2)
    { }

    CUDA_DEVICE_FUNCTION Matrix3x3 operator+() const { return *this; }
    CUDA_DEVICE_FUNCTION Matrix3x3 operator-() const { return Matrix3x3(-c0, -c1, -c2); }

    CUDA_DEVICE_FUNCTION Matrix3x3 operator+(const Matrix3x3 &mat) const { return Matrix3x3(c0 + mat.c0, c1 + mat.c1, c2 + mat.c2); }
    CUDA_DEVICE_FUNCTION Matrix3x3 operator-(const Matrix3x3 &mat) const { return Matrix3x3(c0 - mat.c0, c1 - mat.c1, c2 - mat.c2); }
    CUDA_DEVICE_FUNCTION Matrix3x3 operator*(const Matrix3x3 &mat) const {
        const float3 r[] = { row(0), row(1), row(2) };
        return Matrix3x3(make_float3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0)),
                         make_float3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1)),
                         make_float3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2)));
    }
    CUDA_DEVICE_FUNCTION float3 operator*(const float3 &v) const {
        const float3 r[] = { row(0), row(1), row(2) };
        return make_float3(dot(r[0], v),
                           dot(r[1], v),
                           dot(r[2], v));
    }

    CUDA_DEVICE_FUNCTION Matrix3x3 &operator*=(const Matrix3x3 &mat) {
        const float3 r[] = { row(0), row(1), row(2) };
        c0 = make_float3(dot(r[0], mat.c0), dot(r[1], mat.c0), dot(r[2], mat.c0));
        c1 = make_float3(dot(r[0], mat.c1), dot(r[1], mat.c1), dot(r[2], mat.c1));
        c2 = make_float3(dot(r[0], mat.c2), dot(r[1], mat.c2), dot(r[2], mat.c2));
        return *this;
    }

    CUDA_DEVICE_FUNCTION float3 row(unsigned int r) const {
        //Assert(r < 3, "\"r\" is out of range [0, 2].");
        switch (r) {
        case 0:
            return make_float3(m00, m01, m02);
        case 1:
            return make_float3(m10, m11, m12);
        case 2:
            return make_float3(m20, m21, m22);
        default:
            return make_float3(0, 0, 0);
        }
    }

    CUDA_DEVICE_FUNCTION Matrix3x3 &transpose() {
        std::swap(m10, m01); std::swap(m20, m02);
        std::swap(m21, m12);
        return *this;
    }
    CUDA_DEVICE_FUNCTION Matrix3x3 &inverse() {
        float det = 1.0f / (m00 * m11 * m22 + m01 * m12 * m20 + m02 * m10 * m21 -
                            m02 * m11 * m20 - m01 * m10 * m22 - m00 * m12 * m21);
        Matrix3x3 m;
        m.m00 =  det * (m11 * m22 - m12 * m21); m.m01 = -det * (m01 * m22 - m02 * m21); m.m02 =  det * (m01 * m12 - m02 * m11);
        m.m10 = -det * (m10 * m22 - m12 * m20); m.m11 =  det * (m00 * m22 - m02 * m20); m.m12 = -det * (m00 * m12 - m02 * m10);
        m.m20 =  det * (m10 * m21 - m11 * m20); m.m21 = -det * (m00 * m21 - m01 * m20); m.m22 =  det * (m00 * m11 - m01 * m10);
        *this = m;

        return *this;
    }
};

CUDA_DEVICE_FUNCTION Matrix3x3 transpose(const Matrix3x3 &mat) {
    Matrix3x3 ret = mat;
    return ret.transpose();
}
CUDA_DEVICE_FUNCTION Matrix3x3 inverse(const Matrix3x3 &mat) {
    Matrix3x3 ret = mat;
    return ret.inverse();
}

CUDA_DEVICE_FUNCTION Matrix3x3 scale3x3(const float3 &s) {
    return Matrix3x3(s.x * make_float3(1, 0, 0),
                     s.y * make_float3(0, 1, 0),
                     s.z * make_float3(0, 0, 1));
}
CUDA_DEVICE_FUNCTION Matrix3x3 scale3x3(float sx, float sy, float sz) {
    return scale3x3(make_float3(sx, sy, sz));
}
CUDA_DEVICE_FUNCTION Matrix3x3 scale3x3(float s) {
    return scale3x3(make_float3(s, s, s));
}

CUDA_DEVICE_FUNCTION Matrix3x3 rotate3x3(float angle, const float3 &axis) {
    Matrix3x3 matrix;
    float3 nAxis = normalize(axis);
    float s = std::sin(angle);
    float c = std::cos(angle);
    float oneMinusC = 1 - c;

    matrix.m00 = nAxis.x * nAxis.x * oneMinusC + c;
    matrix.m10 = nAxis.x * nAxis.y * oneMinusC + nAxis.z * s;
    matrix.m20 = nAxis.z * nAxis.x * oneMinusC - nAxis.y * s;
    matrix.m01 = nAxis.x * nAxis.y * oneMinusC - nAxis.z * s;
    matrix.m11 = nAxis.y * nAxis.y * oneMinusC + c;
    matrix.m21 = nAxis.y * nAxis.z * oneMinusC + nAxis.x * s;
    matrix.m02 = nAxis.z * nAxis.x * oneMinusC + nAxis.y * s;
    matrix.m12 = nAxis.y * nAxis.z * oneMinusC - nAxis.x * s;
    matrix.m22 = nAxis.z * nAxis.z * oneMinusC + c;

    return matrix;
}
CUDA_DEVICE_FUNCTION Matrix3x3 rotate3x3(float angle, float ax, float ay, float az) {
    return rotate3x3(angle, make_float3(ax, ay, az));
}
CUDA_DEVICE_FUNCTION Matrix3x3 rotateX3x3(float angle) { return rotate3x3(angle, make_float3(1, 0, 0)); }
CUDA_DEVICE_FUNCTION Matrix3x3 rotateY3x3(float angle) { return rotate3x3(angle, make_float3(0, 1, 0)); }
CUDA_DEVICE_FUNCTION Matrix3x3 rotateZ3x3(float angle) { return rotate3x3(angle, make_float3(0, 0, 1)); }



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



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        const GeometryData* geomInstData;
        int2 imageSize; // Note that CUDA/OptiX built-in vector types with width 2 require 8-byte alignment.
        optixu::BlockBuffer2D<float4, 1> resultBuffer;
        PerspectiveCamera camera;
    };
}
