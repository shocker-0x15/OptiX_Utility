#pragma once

#include "../common/common.h"

namespace Shared {
    static constexpr float Pi = 3.14159265358979323846f;



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
        CUDA_DEVICE_FUNCTION PCG32RNG() {}

        void setState(uint32_t _state) { state = _state; }

        CUDA_DEVICE_FUNCTION uint32_t operator()() {
            uint64_t oldstate = state;
            // Advance internal state
            state = oldstate * 6364136223846793005ULL + 1;
            // Calculate output function (XSH RR), uses old state for max ILP
            uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            uint32_t rot = oldstate >> 59u;
            return (xorshifted >> rot) | (xorshifted << ((-static_cast<int32_t>(rot)) & 31));
        }

        CUDA_DEVICE_FUNCTION float getFloat0cTo1o() {
            uint32_t fractionBits = ((*this)() >> 9) | 0x3f800000;
            return *(float*)&fractionBits - 1.0f;
        }
    };



    struct PerspectiveCamera {
        float aspect;
        float fovY;
        float3 position;
        Matrix3x3 orientation;

#if defined(__CUDA_ARCH__) || defined(OPTIXU_Platform_CodeCompletion)
        CUDA_DEVICE_FUNCTION float2 calcScreenPosition(const float3 &posInWorld) const {
            Matrix3x3 invOri = inverse(orientation);
            float3 posInView = invOri * (posInWorld - position);
            float2 posAtZ1 = make_float2(posInView.x / posInView.z, posInView.y / posInView.z);
            float h = 2 * std::tan(fovY / 2);
            float w = aspect * h;
            return make_float2(1 - (posAtZ1.x + 0.5f * w) / w,
                               1 - (posAtZ1.y + 0.5f * h) / h);
        }
#endif
    };


    
    struct GeometryData {
        const Vertex* vertexBuffer;
        const Triangle* triangleBuffer;
    };

    struct MaterialData {
        CUtexObject texture;
        float3 albedo;
        bool isEmitter;

        MaterialData() :
            texture(0),
            albedo(make_float3(0.0f, 0.0f, 0.5f)),
            isEmitter(false) {}
    };

    struct InstanceData {
        float scale;
        Matrix3x3 rotation;
        float3 translation;
        float prevScale;
        Matrix3x3 prevRotation;
        float3 prevTranslation;

        InstanceData() :
            scale(1.0f), translation(make_float3(0.0f, 0.0f, 0.0f)),
            prevScale(1.0f), prevTranslation(make_float3(0.0f, 0.0f, 0.0f)) {}
        InstanceData(float _scale, const Matrix3x3 &_rotation, const float3 &_translation) :
            scale(_scale), rotation(_rotation), translation(_translation),
            prevScale(_scale), prevRotation(_rotation), prevTranslation(_translation) {}
    };



    struct PipelineLaunchParameters {
        OptixTraversableHandle travHandle;
        int2 imageSize;
        uint32_t numAccumFrames;
        optixu::BlockBuffer2D<PCG32RNG, 1> rngBuffer;
        optixu::NativeBlockBuffer2D<float4> beautyAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> albedoAccumBuffer;
        optixu::NativeBlockBuffer2D<float4> normalAccumBuffer;
        float2* linearFlowBuffer;
        PerspectiveCamera camera;
        PerspectiveCamera prevCamera;
        const InstanceData* instances;
        unsigned int enableJittering : 1;
        unsigned int resetFlowBuffer : 1;
    };



    struct SearchRayPayload {
        float3 alpha;
        float3 contribution;
        float3 origin;
        float3 direction;
        struct {
            unsigned int pathLength : 30;
            unsigned int terminate : 1;
        };
    };

    struct DenoiserData {
        float3 firstHitAlbedo;
        float3 firstHitNormal;
        float3 firstHitPrevPositionInWorld;
    };



    enum class BufferToDisplay {
        NoisyBeauty = 0,
        Albedo,
        Normal,
        Flow,
        DenoisedBeauty,
    };
}

#define SearchRayPayloadSignature Shared::PCG32RNG, Shared::SearchRayPayload*, Shared::DenoiserData*
#define VisibilityRayPayloadSignature float
