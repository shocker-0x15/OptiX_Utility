#pragma once

#include "optix_util.h"

#include <optix_function_table_definition.h>

#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include <stdexcept>

#define OPTIX_CHECK(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK_LOG(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n" \
               << "Log: " << log << (logSize > sizeof(log) ? "<TRUNCATED>" : "") \
               << "\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

namespace optix {
#define OPTIX_ALIAS_PIMPL(Name) using _ ## Name = Name::Priv

    OPTIX_ALIAS_PIMPL(Context);
    OPTIX_ALIAS_PIMPL(Material);
    OPTIX_ALIAS_PIMPL(Scene);
    OPTIX_ALIAS_PIMPL(GeometryInstance);
    OPTIX_ALIAS_PIMPL(GeometryAccelerationStructure);
    OPTIX_ALIAS_PIMPL(InstanceAccelerationStructure);
    OPTIX_ALIAS_PIMPL(Pipeline);
    OPTIX_ALIAS_PIMPL(Module);
    OPTIX_ALIAS_PIMPL(ProgramGroup);



#define OPTIX_OPAQUE_BRIDGE(BaseName) \
    friend class BaseName; \
\
    BaseName getPublicType() { \
        BaseName ret; \
        ret.m = this; \
        return ret; \
    } \
\
    static BaseName::Priv* extract(BaseName publicType) { \
        return publicType.m; \
    }

    template <typename PublicType>
    static typename PublicType::Priv* extract(const PublicType &obj) {
        return PublicType::Priv::extract(obj);
    }



    struct SizeAlign {
        uint32_t size;
        uint32_t alignment;

        constexpr SizeAlign() : size(0), alignment(0) {}
        constexpr SizeAlign(uint32_t s, uint32_t a) : size(s), alignment(a) {}

        SizeAlign &add(const SizeAlign &sa, uint32_t* offset) {
            uint32_t mask = sa.alignment - 1;
            alignment = std::max(alignment, sa.alignment);
            size = (size + mask) & ~mask;
            if (offset)
                *offset = size;
            size += sa.size;
            return *this;
        }
        SizeAlign &operator+=(const SizeAlign &sa) {
            return add(sa, nullptr);
        }
        SizeAlign &alignUp() {
            uint32_t mask = alignment - 1;
            size = (size + mask) & ~mask;
            return *this;
        }
    };

    SizeAlign max(const SizeAlign &sa0, const SizeAlign &sa1) {
        return SizeAlign{ std::max(sa0.size, sa1.size), std::max(sa0.alignment, sa1.alignment) };
    }



    class Context::Priv {
        OptixDeviceContext rawContext;

    public:
        Priv() :
            rawContext(nullptr) {}

        OPTIX_OPAQUE_BRIDGE(Context);

        OptixDeviceContext getRawContext() const {
            return rawContext;
        }
    };



    class Material::Priv {
    public:
        struct Key {
            const _Pipeline* pipeline;
            uint32_t rayType;

            bool operator<(const Key &rKey) const {
                if (pipeline < rKey.pipeline) {
                    return true;
                }
                else if (pipeline == rKey.pipeline) {
                    if (rayType < rKey.rayType)
                        return true;
                }
                return false;
            }
        };

        struct Info {
            const _ProgramGroup* program;
            uint8_t recordData[128];
            SizeAlign sizeAlign;

            Info() : program(nullptr), sizeAlign(0, 0) {}
            Info(const _ProgramGroup* _program, const void* data, size_t size, size_t align) {
                program = _program;
                std::memcpy(recordData, data, size);
                sizeAlign.size = size;
                sizeAlign.alignment = align;
            }
        };

    private:
        const _Context* context;

        std::map<Key, Info> infos;

    public:
        Priv(const _Context* ctxt) : context(ctxt) {}

        OPTIX_OPAQUE_BRIDGE(Material);

        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        const Info &getSBTRecord(const _Pipeline* pipeline, uint32_t rayTypeIdx) const {
            Key key{ pipeline, rayTypeIdx };
            return infos.at(key);
        }
    };



    class Scene::Priv {
        struct SBTOffsetKey {
            const _GeometryAccelerationStructure* gas;
            uint32_t matSetIndex;

            bool operator<(const SBTOffsetKey &rKey) const {
                if (gas < rKey.gas) {
                    return true;
                }
                else if (gas == rKey.gas) {
                    if (matSetIndex < rKey.matSetIndex)
                        return true;
                }
                return false;
            }
        };

        const _Context* context;
        std::set<const _GeometryAccelerationStructure*> geomASs;
        std::map<SBTOffsetKey, uint32_t> sbtOffsets;
        uint32_t numSBTRecords;
        bool sbtOffsetsAreDirty;
        std::set<const _InstanceAccelerationStructure*> instASs;

    public:
        Priv(const _Context* ctxt) : context(ctxt), sbtOffsetsAreDirty(true) {}

        OPTIX_OPAQUE_BRIDGE(Scene);

        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        void removeGAS(const _GeometryAccelerationStructure* gas) {
            geomASs.erase(gas);
        }

        void removeIAS(const _InstanceAccelerationStructure* ias) {
            instASs.erase(ias);
        }

        bool sbtOffsetsGenerationIsDone() const {
            return !sbtOffsetsAreDirty;
        }

        uint32_t getSBTOffset(_GeometryAccelerationStructure* gas, uint32_t matSetIdx) {
            return sbtOffsets.at(SBTOffsetKey{ gas, matSetIdx });
        }

        SizeAlign calcHitGroupRecordStride(const _Pipeline* pipeline) const;

        uint32_t getNumSBTRecords() const {
            return numSBTRecords;
        }

        void fillSBTRecords(const _Pipeline* pipeline, uint8_t* records, uint32_t stride) const;
    };



    class GeometryInstance::Priv {
    public:
        struct MaterialKey {
            uint32_t matSetIndex;
            uint32_t matIndex;

            bool operator<(const MaterialKey &rKey) const {
                if (matSetIndex < rKey.matSetIndex) {
                    return true;
                }
                else if (matSetIndex == rKey.matSetIndex) {
                    if (matIndex < rKey.matIndex)
                        return true;
                }
                return false;
            }
        };

    private:
        _Scene* scene;

        // TODO: support deformation blur (multiple vertex buffers)
        CUdeviceptr vertexBufferArray[1];
        Buffer* vertexBuffer;
        Buffer* triangleBuffer;
        Buffer* materialIndexOffsetBuffer;
        std::vector<uint32_t> buildInputFlags; // per SBT record

        uint8_t recordData[128];
        SizeAlign sizeAlign;

        std::map<MaterialKey, const _Material*> materials;

    public:
        Priv(_Scene* _scene) :
            scene(_scene),
            vertexBuffer(nullptr), triangleBuffer(nullptr), materialIndexOffsetBuffer(nullptr) {
        }

        OPTIX_OPAQUE_BRIDGE(GeometryInstance);

        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        void fillBuildInput(OptixBuildInput* input) const;

        SizeAlign calcHitGroupRecordStride(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes) const;
        uint32_t getNumSBTRecords() const;
        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                uint8_t* records, size_t stride) const;
    };



    class GeometryAccelerationStructure::Priv {
        _Scene* scene;

        std::vector<uint32_t> numRayTypesValues;

        std::vector<_GeometryInstance*> children;
        std::vector<OptixBuildInput> buildInputs;

        OptixAccelBuildOptions buildOptions;

        size_t accelBufferSize;
        Buffer accelBuffer;
        Buffer accelTempBuffer;

        Buffer compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        void fillBuildInputs();

    public:
        Priv(_Scene* _scene) : scene(_scene) {
            compactedSizeOnDevice.initialize(BufferType::Device, 1, sizeof(size_t), 0);

            propertyCompactedSize = OptixAccelEmitDesc{};
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

            available = false;
            compactedAvailable = false;
        }
        ~Priv() {
            compactedSizeOnDevice.finalize();
        }

        OPTIX_OPAQUE_BRIDGE(GeometryAccelerationStructure);

        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        uint32_t getNumMaterialSets() const {
            return static_cast<uint32_t>(numRayTypesValues.size());
        }

        uint32_t getNumRayTypes(uint32_t matSetIdx) const {
            return numRayTypesValues[matSetIdx];
        }

        SizeAlign calcHitGroupRecordStride(const _Pipeline* pipeline) const;

        uint32_t calcNumSBTRecords(uint32_t matSetIdx) const;

        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records, uint32_t stride) const;

        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            optixAssert(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            return 0;
        }
    };



    enum class InstanceType {
        GAS = 0,
    };

    struct _Instance {
        OptixInstance rawInstance;
        uint32_t matSetIndex;

        InstanceType type;
        union {
            _GeometryAccelerationStructure* gas;
        };

        _Instance(_GeometryAccelerationStructure* _gas, uint32_t matSetIdx, const float transform[12]) :
            matSetIndex(matSetIdx), type(InstanceType::GAS), gas(_gas) {
            rawInstance = OptixInstance{};
            rawInstance.instanceId = 0;
            rawInstance.visibilityMask = 0xFF;
            if (transform) {
                std::copy_n(transform, 12, rawInstance.transform);
            }
            else {
                float identity[] = {
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0
                };
                std::copy_n(identity, 12, rawInstance.transform);
            }
            rawInstance.flags = OPTIX_INSTANCE_FLAG_NONE;
        }
    };



    class InstanceAccelerationStructure::Priv {
        _Scene* scene;

        std::vector<_Instance> children;
        OptixBuildInput buildInput;
        Buffer instanceBuffer;

        OptixAccelBuildOptions buildOptions;

        size_t accelBufferSize;
        Buffer accelBuffer;
        Buffer accelTempBuffer;

        Buffer compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        void setupInstances();

        void fillBuildInput();

    public:
        Priv(_Scene* _scene) : scene(_scene) {
            compactedSizeOnDevice.initialize(BufferType::Device, 1, sizeof(size_t), 0);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

            available = false;
            compactedAvailable = false;
        }
        ~Priv() {
            instanceBuffer.finalize();

            compactedSizeOnDevice.finalize();
        }

        OPTIX_OPAQUE_BRIDGE(InstanceAccelerationStructure);

        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        bool isReady() const {
            return available || compactedAvailable;
        }

        bool isReadyRecursive() const {
            if (!isReady())
                return false;

            for (int i = 0; i < children.size(); ++i) {
                const _Instance &inst = children[i];

                uint32_t cNumSBTRecords = 0;
                size_t cMaxSize = 0;
                if (inst.type == InstanceType::GAS) {
                    if (!inst.gas->isReady())
                        return false;
                }
                else {
                    optixAssert_NotImplemented();
                }
            }

            return true;
        }

        OptixTraversableHandle getHandle() const {
            optixAssert(isReady(), "Traversable handle is not ready.");
            if (compactedAvailable)
                return compactedHandle;
            if (available)
                return handle;
            optixAssert_ShouldNotBeCalled();
            return 0;
        }
    };



    class Pipeline::Priv {
        const _Context* context;
        OptixPipeline rawPipeline;

        uint32_t maxTraceDepth;
        OptixPipelineCompileOptions pipelineCompileOptions;
        size_t sizeOfPipelineLaunchParams;
        std::set<OptixProgramGroup> programGroups;

        const _Scene* scene;
        uint32_t numMissRayTypes;

        _ProgramGroup* rayGenProgram;
        _ProgramGroup* exceptionProgram;
        std::vector<_ProgramGroup*> missPrograms;
        Buffer rayGenRecord;
        Buffer missRecords;
        Buffer hitGroupRecords;

        OptixShaderBindingTable sbt;

        struct {
            unsigned int pipelineLinked : 1;
            unsigned int sbtSetup : 1;
        };

        void createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group);

        void setupShaderBindingTable();

    public:
        Priv(const _Context* ctxt) : context(ctxt),
            maxTraceDepth(0),
            scene(nullptr), numMissRayTypes(0),
            rayGenProgram(nullptr), exceptionProgram(nullptr),
            pipelineLinked(false), sbtSetup(false) {
        }

        OPTIX_OPAQUE_BRIDGE(Pipeline);

        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        void destroyProgram(OptixProgramGroup group);
    };



    class Module::Priv {
        const _Pipeline* pipeline;
        OptixModule rawModule;

    public:
        Priv(const _Pipeline* pl, OptixModule module) : pipeline(pl), rawModule(module) {}

        OPTIX_OPAQUE_BRIDGE(Module);



        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixModule getRawModule() const {
            return rawModule;
        }
    };



    class ProgramGroup::Priv {
        _Pipeline* pipeline;
        OptixProgramGroup rawGroup;

    public:
        Priv(_Pipeline* pl, OptixProgramGroup group) : pipeline(pl), rawGroup(group) {}

        OPTIX_OPAQUE_BRIDGE(ProgramGroup);



        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixProgramGroup getRawProgramGroup() const {
            return rawGroup;
        }

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };
}
