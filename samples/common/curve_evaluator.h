#pragma once

#include "common.h"

// Based on OptiX SDK/cuda/curve.h

namespace curve {
    CUDA_DEVICE_FUNCTION float clamp(float x, float minx, float maxx) {
        return fminf(fmaxf(x, minx), maxx);
    }



    template <OptixPrimitiveType curveType>
    CUDA_DEVICE_FUNCTION constexpr uint32_t getNumControlPoints() {
        static_assert(false, "Invalid curve type.");
        return 0;
    }
    template <>
    CUDA_DEVICE_FUNCTION constexpr uint32_t getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR>() {
        return 2;
    }
    template <>
    CUDA_DEVICE_FUNCTION constexpr uint32_t getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE>() {
        return 3;
    }
    template <>
    CUDA_DEVICE_FUNCTION constexpr uint32_t getNumControlPoints<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>() {
        return 4;
    }



    template <OptixPrimitiveType curveType>
    struct Interpolator;

    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR> {
        float4 m_p[2];

        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[2]) {
            m_p[0] = cps[0];
            m_p[1] = cps[1] - cps[0];
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            return m_p[0] + u * m_p[1];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            return make_float3(m_p[0]) + u * make_float3(m_p[1]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            return m_p[0].w + u * m_p[1].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            return fminf(radius(uA), radius(uB));
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            return fmaxf(radius(uA), radius(uB));
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            return m_p[1];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            return make_float3(m_p[1]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            return m_p[1].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
    };

    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE> {
        float4 m_p[3];

        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[3]) {
            m_p[0] = cps[1] / 2.0f + cps[0] / 2.0f;
            m_p[1] = cps[1] - cps[0];
            m_p[2] = cps[0] / 2.0f - cps[1] + cps[2] / 2.0f;
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            return m_p[0] + u * m_p[1] + u * u * m_p[2];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            return make_float3(m_p[0]) + u * make_float3(m_p[1]) + u * u * make_float3(m_p[2]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            return m_p[0].w + u * m_p[1].w + u * u * m_p[2].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            float root1 = clamp(-0.5f * m_p[1].w / m_p[2].w, uA, uB);
            return fminf(fminf(radius(uA), radius(uB)), radius(root1));
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            if (m_p[1].w == 0.0f && m_p[2].w == 0.0f)
                return m_p[0].w; // constant width
            float root1 = clamp(-0.5f * m_p[1].w / m_p[2].w, uA, uB);
            return fmaxf(fmaxf(radius(uA), radius(uB)), radius(root1));
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            return m_p[1] + 2 * u * m_p[2];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            return make_float3(m_p[1]) + 2 * u * make_float3(m_p[2]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            return m_p[1].w + 2 * u * m_p[2].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return 2 * m_p[2];
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return 2 * make_float3(m_p[2]);
        }
    };

    template <>
    struct Interpolator<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE> {
        float4 m_p[4];

        CUDA_DEVICE_FUNCTION static float3 coeffs(float u) {
            float uu = u * u;
            float u3 = (1 / 6.0f) * uu * u;
            return make_float3(u3 + 0.5f * (u - uu), uu - 4 * u3, u3);
        }

        CUDA_DEVICE_FUNCTION Interpolator(const float4 cps[3]) {
            m_p[0] = (cps[2] + cps[0]) / 6 + (4 / 6.0f) * cps[1];
            m_p[1] = cps[2] - cps[0];
            m_p[2] = cps[2] - cps[1];
            m_p[3] = cps[3] - cps[1];
        }

        CUDA_DEVICE_FUNCTION float4 position_radius(float u) const {
            float3 c = coeffs(u);
            return m_p[0] + c.x * m_p[1] + c.y * m_p[2] + c.z * m_p[3];
        }
        CUDA_DEVICE_FUNCTION float3 position(float u) const {
            float3 c = coeffs(u);
            return make_float3(m_p[0]) + c.x * make_float3(m_p[1]) + c.y * make_float3(m_p[2]) + c.z * make_float3(m_p[3]);
        }
        CUDA_DEVICE_FUNCTION float radius(float u) const {
            float3 c = coeffs(u);
            return m_p[0].w + c.x * m_p[1].w + c.y * m_p[2].w + c.z * m_p[3].w;
        }

        CUDA_DEVICE_FUNCTION float minRadius(float uA, float uB) const {
            // a + 2 b u - c u^2
            float a = m_p[1].w;
            float b = 2 * m_p[2].w - m_p[1].w;
            float c = 4 * m_p[2].w - m_p[1].w - m_p[3].w;
            float rmin = fminf(radius(uA), radius(uB));
            if (fabsf(c) < 1e-5f) {
                float root1 = clamp(-0.5f * a / b, uA, uB);
                return fminf(rmin, radius(root1));
            }
            else {
                float det = b * b + a * c;
                det = det <= 0.0f ? 0.0f : sqrt(det);
                float root1 = clamp((b + det) / c, uA, uB);
                float root2 = clamp((b - det) / c, uA, uB);
                return fminf(rmin, fminf(radius(root1), radius(root2)));
            }
        }
        CUDA_DEVICE_FUNCTION float maxRadius(float uA, float uB) const {
            if (m_p[1].w == 0 && m_p[2].w == 0 && m_p[3].w == 0)
                return m_p[0].w; // constant width
            // a + 2 b u - c u^2
            float a = m_p[1].w;
            float b = 2 * m_p[2].w - m_p[1].w;
            float c = 4 * m_p[2].w - m_p[1].w - m_p[3].w;
            float rmax = fmaxf(radius(uA), radius(uB));
            if (fabsf(c) < 1e-5f) {
                float root1 = clamp(-0.5f * a / b, uA, uB);
                return fmaxf(rmax, radius(root1));
            }
            else {
                float det = b * b + a * c;
                det = det <= 0.0f ? 0.0f : sqrt(det);
                float root1 = clamp((b + det) / c, uA, uB);
                float root2 = clamp((b - det) / c, uA, uB);
                return fmaxf(rmax, fmaxf(radius(root1), radius(root2)));
            }
        }

        CUDA_DEVICE_FUNCTION float4 dPosition_dRadius(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            float v = 1 - u;
            return 0.5f * v * v * m_p[1] + 2 * v * u * m_p[2] + 0.5f * u * u * m_p[3];
        }
        CUDA_DEVICE_FUNCTION float3 dPosition(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            float v = 1 - u;
            return 0.5f * v * v * make_float3(m_p[1]) + 2 * v * u * make_float3(m_p[2]) + 0.5f * u * u * make_float3(m_p[3]);
        }
        CUDA_DEVICE_FUNCTION float dRadius(float u) const {
            // adjust u to avoid problems with tripple knots.
            if (u == 0)
                u = 0.000001f;
            if (u == 1)
                u = 0.999999f;
            float v = 1 - u;
            return 0.5f * v * v * m_p[1].w + 2 * v * u * m_p[2].w + 0.5f * u * u * m_p[3].w;
        }

        CUDA_DEVICE_FUNCTION float4 ddPosition_ddRadius(float u) const {
            return 2 * m_p[2] - m_p[1] + (m_p[1] - 4 * m_p[2] + m_p[3]) * u;
        }
        CUDA_DEVICE_FUNCTION float3 ddPosition(float u) const {
            return 2 * make_float3(m_p[2]) - make_float3(m_p[1]) + (make_float3(m_p[1]) - 4 * make_float3(m_p[2]) + make_float3(m_p[3])) * u;
        }
    };



    template <OptixPrimitiveType curveType>
    class Evaluator {
        Interpolator<curveType> m_interpolator;

    public:
        CUDA_DEVICE_FUNCTION Evaluator(const float4 controlPoints[getNumControlPoints<curveType>()]) :
            m_interpolator(controlPoints) {}

        // type - 0     ~ cylindrical approximation (correct if radius' == 0)
        //        1     ~ conic       approximation (correct if curve'' == 0)
        //        other ~ the bona fide surface normal
        template <uint32_t type = 2>
        CUDA_DEVICE_FUNCTION float3 calcNormal(float u, const float3 &hitPointInObject) const {
            float3 hp = hitPointInObject;

            float3 normal;
            if (u == 0.0f) {
                if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
                    normal = hp - make_float3(m_interpolator.m_p[0]); // special handling for round endcaps
                else
                    normal = -m_interpolator.dPosition(0); // special handling for flat endcaps
            }
            else if (u >= 1.0f) {
                if constexpr (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR) {
                    // reconstruct second control point (Note: the interpolator pre-transforms
                    // the control-points to speed up repeated evaluation.
                    float3 p1 = make_float3(m_interpolator.m_p[1]) + make_float3(m_interpolator.m_p[0]);
                    normal = hp - p1; // special handling for round endcaps
                }
                else {
                    normal = m_interpolator.dPosition(1); // special handling for flat endcaps
                }
            }
            else {
                // hp is a point that is near the curve's offset surface,
                // usually ray.origin + ray.direction * rayt.
                // We will push it exactly to the surface by projecting it to the plane(p,d).
                // The function derivation:
                // we (implicitly) transform the curve into coordinate system
                // {p, o1 = normalize(hp - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
                // curve'(t) = (0, length(d), 0); hp = (r, 0, 0);
                float4 p4 = m_interpolator.position_radius(u);
                float3 p = make_float3(p4);
                float  r = p4.w; // == length(hp - p) if hp is already on the surface
                float4 d4 = m_interpolator.dPosition_dRadius(u);
                float3 d = make_float3(d4);
                float  dr = d4.w;
                float  dd = dot(d, d);

                float3 o1 = hp - p;          // dot(modified_o1, d) == 0 by design:
                o1 -= (dot(o1, d) / dd) * d; // first, project hp to the plane(p,d)
                o1 *= r / length(o1);        // and then drop it to the surface
                hp = p + o1;                 // fine-tuning the hit point
                if constexpr (type == 0) {
                    normal = o1; // cylindrical approximation
                }
                else {
                    if constexpr (curveType != OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR && type != 1)
                        dd -= dot(m_interpolator.ddPosition(u), o1);
                    normal = dd * o1 - (dr * r) * d;
                }
            }

            return normal; // non-normalized
        }

        CUDA_DEVICE_FUNCTION float3 calcTangent(float u) const {
            float3 tangent = m_interpolator.dPosition(u);
            return tangent; // non-normalized;
        }
    };
}
