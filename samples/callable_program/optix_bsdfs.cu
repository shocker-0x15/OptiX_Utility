#include "callable_program_shared.h"

using namespace Shared;

CUDA_DEVICE_FUNCTION CUDA_INLINE float3 schlickFresnel(const float3 &f0Reflectance, float cos) {
    return f0Reflectance + (make_float3(1.0f) - f0Reflectance) * pow5(1 - cos);
}

CUDA_DEVICE_FUNCTION CUDA_INLINE float fresnel(float etaEnter, float etaExit, float cosEnter) {
    float sinExit = etaEnter / etaExit * std::sqrt(std::fmax(0.0f, 1.0f - cosEnter * cosEnter));
    if (sinExit >= 1.0f) {
        return 1.0f;
    }
    else {
        float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit * sinExit));
        float Rparl = ((etaExit * cosEnter) - (etaEnter * cosExit)) / ((etaExit * cosEnter) + (etaEnter * cosExit));
        float Rperp = ((etaEnter * cosEnter) - (etaExit * cosExit)) / ((etaEnter * cosEnter) + (etaExit * cosExit));
        return (Rparl * Rparl + Rperp * Rperp) / 2.0f;
    }
}



RT_CALLABLE_PROGRAM void RT_DC_NAME(setUpLambertBRDF)(
    const MaterialData &matData, const float2 &texCoord, BSDFData* bsdfData) {
    float3 reflectance;
    if (matData.asMatte.texture) {
        reflectance = getXYZ(tex2DLod<float4>(
            matData.asMatte.texture, texCoord.x, texCoord.y, 0.0f));
    }
    else {
        reflectance = matData.asMatte.reflectance;
    }
    bsdfData->asLambertBRDF.reflectance = reflectance;
}

class LambertBRDF {
    float3 m_reflectance;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE LambertBRDF(const float3 &reflectance) :
        m_reflectance(reflectance) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 sampleF(
        const float3 &givenLocalDir, const float uDir[2],
        float3* sampledDir, float* probDens, bool* deltaSampled) const
    {
        *sampledDir = cosineSampleHemisphere(uDir[0], uDir[1]);
        const float oneOverPi = 1.0f / pi_v<float>;
        *probDens = sampledDir->z * oneOverPi;
        *deltaSampled = false;
        sampledDir->z *= givenLocalDir.z >= 0 ? 1 : -1;
        const float3 fsValue = m_reflectance * oneOverPi;
        return fsValue;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 evaluateF(
        const float3 &givenLocalDir,
        const float3 &sampledDir) const
    {
        if (givenLocalDir.z * sampledDir.z <= 0.0f) {
            const float3 fs = make_float3(0.0f, 0.0f, 0.0f);
            return fs;
        }
        const float oneOverPi = 1.0f / pi_v<float>;
        const float3 fsValue = m_reflectance * oneOverPi;
        return fsValue;
    }
};

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(LambertBRDF_sampleF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float uDir[2],
    float3* sampledDir, float* probDens, bool* deltaSampled) {
    LambertBRDF bsdf(bsdfData.asLambertBRDF.reflectance);
    return bsdf.sampleF(givenLocalDir, uDir, sampledDir, probDens, deltaSampled);
}

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(LambertBRDF_evaluateF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float3 &sampledDir) {
    LambertBRDF bsdf(bsdfData.asLambertBRDF.reflectance);
    return bsdf.evaluateF(givenLocalDir, sampledDir);
}



RT_CALLABLE_PROGRAM void RT_DC_NAME(setUpMirrorBRDF)(
    const MaterialData &matData, const float2 &texCoord, BSDFData* bsdfData) {
    float3 f0Reflectance;
    if (matData.asMatte.texture) {
        f0Reflectance = getXYZ(tex2DLod<float4>(
            matData.asMirror.texture, texCoord.x, texCoord.y, 0.0f));
    }
    else {
        f0Reflectance = matData.asMirror.f0Reflectance;
    }
    bsdfData->asMirrorBRDF.f0Reflectance = f0Reflectance;
}

class MirrorBRDF {
    float3 m_f0Reflectance;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE MirrorBRDF(const float3 &f0Reflectance) :
        m_f0Reflectance(f0Reflectance) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 sampleF(
        const float3 &givenLocalDir, const float uDir[2],
        float3* sampledDir, float* probDens, bool* deltaSampled) const
    {
        *sampledDir = make_float3(-givenLocalDir.x, -givenLocalDir.y, givenLocalDir.z);
        *probDens = 1.0f;
        *deltaSampled = true;
        float3 ret = schlickFresnel(m_f0Reflectance, givenLocalDir.z) *
            (1.0f / std::fabs(givenLocalDir.z));
        return ret;
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 evaluateF(
        const float3 &givenLocalDir,
        const float3 &sampledDir) const
    {
        const float3 fsValue = make_float3(0.0f, 0.0f, 0.0f);
        return fsValue;
    }
};

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(MirrorBRDF_sampleF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float uDir[2],
    float3* sampledDir, float* probDens, bool* deltaSampled) {
    MirrorBRDF bsdf(bsdfData.asMirrorBRDF.f0Reflectance);
    return bsdf.sampleF(givenLocalDir, uDir, sampledDir, probDens, deltaSampled);
}

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(MirrorBRDF_evaluateF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float3 &sampledDir) {
    MirrorBRDF bsdf(bsdfData.asMirrorBRDF.f0Reflectance);
    return bsdf.evaluateF(givenLocalDir, sampledDir);
}



RT_CALLABLE_PROGRAM void RT_DC_NAME(setUpGlassBSDF)(
    const MaterialData &matData, const float2 &texCoord, BSDFData* bsdfData) {
    bsdfData->asGlassBSDF.ior = matData.asGlass.ior;
}

class GlassBSDF {
    float m_ior;

public:
    CUDA_DEVICE_FUNCTION CUDA_INLINE GlassBSDF(float ior) :
        m_ior(ior) {}

    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 sampleF(
        const float3 &givenLocalDir, const float uDir[2],
        float3* sampledDir, float* probDens, bool* deltaSampled) const
    {
        bool entering = givenLocalDir.z >= 0.0f;

        const float eEnter = entering ? 1.0f : m_ior;
        const float eExit = entering ? m_ior : 1.0f;

        float3 dirV = entering ? givenLocalDir : -givenLocalDir;

        *deltaSampled = true;
        float F = fresnel(eEnter, eExit, dirV.z);
        float uComp = uDir[1];
        if (uComp < F) {
            if (dirV.z == 0.0f) {
                *probDens = 0.0f;
                return make_float3(0.0f);
            }
            float3 dirL = make_float3(-dirV.x, -dirV.y, dirV.z);
            *sampledDir = entering ? dirL : -dirL;
            *probDens = F;
            float3 ret = make_float3(F * (1.0f / std::fabs(dirV.z)));

            return ret;
        }
        else {
            float sinEnter2 = 1.0f - dirV.z * dirV.z;
            float recRelIOR = eEnter / eExit;// reciprocal of relative IOR.
            float sinExit2 = recRelIOR * recRelIOR * sinEnter2;

            if (sinExit2 >= 1.0f) {
                *probDens = 0.0f;
                return make_float3(0.0f);
            }
            float cosExit = std::sqrt(std::fmax(0.0f, 1.0f - sinExit2));
            float3 dirL = make_float3(recRelIOR * -dirV.x, recRelIOR * -dirV.y, -cosExit);
            *sampledDir = entering ? dirL : -dirL;
            *probDens = (1.0f - F);

            float3 ret = make_float3(1.0f - F);

            float squeezeFactor = pow2(eEnter / eExit);
            ret *= squeezeFactor / std::fabs(cosExit);
            *probDens *= squeezeFactor;

            return ret;
        }
    }
    CUDA_DEVICE_FUNCTION CUDA_INLINE float3 evaluateF(
        const float3 &givenLocalDir,
        const float3 &sampledDir) const
    {
        const float3 fsValue = make_float3(0.0f, 0.0f, 0.0f);
        return fsValue;
    }
};

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(GlassBSDF_sampleF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float uDir[2],
    float3* sampledDir, float* probDens, bool* deltaSampled) {
    GlassBSDF bsdf(bsdfData.asGlassBSDF.ior);
    return bsdf.sampleF(givenLocalDir, uDir, sampledDir, probDens, deltaSampled);
}

RT_CALLABLE_PROGRAM float3 RT_DC_NAME(GlassBSDF_evaluateF)(
    const BSDFData &bsdfData, const float3 &givenLocalDir, const float3 &sampledDir) {
    GlassBSDF bsdf(bsdfData.asGlassBSDF.ior);
    return bsdf.evaluateF(givenLocalDir, sampledDir);
}
