#pragma once

#include "optix_util.h"

#include <optix_function_table_definition.h>

#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include <intrin.h>

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
    static std::runtime_error make_runtime_error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);

        return std::runtime_error(str);
    }

#define THROW_RUNTIME_ERROR(expr, fmt, ...) do { if (!(expr)) throw make_runtime_error(fmt, ##__VA_ARGS__); } while (0)

    static void logCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
        optixPrintf("[%2u][%12s]: %s\n", level, tag, message);
    }



    class SlotFinder {
        uint32_t m_numLayers;
        uint32_t m_numLowestFlagBins;
        uint32_t m_numTotalCompiledFlagBins;
        uint32_t* m_flagBins;
        uint32_t* m_offsetsToOR_AND;
        uint32_t* m_numUsedFlagsUnderBinList;
        uint32_t* m_offsetsToNumUsedFlags;
        uint32_t* m_numFlagsInLayerList;

        SlotFinder(const SlotFinder &) = delete;
        SlotFinder &operator=(const SlotFinder &) = delete;

        void aggregate();
    public:
        static constexpr uint32_t InvalidSlotIndex = 0xFFFFFFFF;

        SlotFinder() :
            m_numLayers(0), m_numLowestFlagBins(0), m_numTotalCompiledFlagBins(0),
            m_flagBins(nullptr), m_offsetsToOR_AND(nullptr),
            m_numUsedFlagsUnderBinList(nullptr), m_offsetsToNumUsedFlags(nullptr),
            m_numFlagsInLayerList(nullptr) {
        }
        ~SlotFinder() {
        }

        void initialize(uint32_t numSlots);

        void finalize();

        SlotFinder &operator=(SlotFinder &&inst) {
            finalize();

            m_numLayers = inst.m_numLayers;
            m_numLowestFlagBins = inst.m_numLowestFlagBins;
            m_numTotalCompiledFlagBins = inst.m_numTotalCompiledFlagBins;
            m_flagBins = inst.m_flagBins;
            m_offsetsToOR_AND = inst.m_offsetsToOR_AND;
            m_numUsedFlagsUnderBinList = inst.m_numUsedFlagsUnderBinList;
            m_offsetsToNumUsedFlags = inst.m_offsetsToNumUsedFlags;
            m_numFlagsInLayerList = inst.m_numFlagsInLayerList;
            inst.m_flagBins = nullptr;
            inst.m_offsetsToOR_AND = nullptr;
            inst.m_numUsedFlagsUnderBinList = nullptr;
            inst.m_offsetsToNumUsedFlags = nullptr;
            inst.m_numFlagsInLayerList = nullptr;

            return *this;
        }
        SlotFinder(SlotFinder &&inst) {
            *this = std::move(inst);
        }

        void resize(uint32_t numSlots);

        void reset() {
            std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
            std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
        }

        uint32_t getNumLayers() const {
            return m_numLayers;
        }

        const uint32_t* getOffsetsToOR_AND() const {
            return m_offsetsToOR_AND;
        }

        const uint32_t* getOffsetsToNumUsedFlags() const {
            return m_offsetsToNumUsedFlags;
        }

        const uint32_t* getNumFlagsInLayerList() const {
            return m_numFlagsInLayerList;
        }



        void setInUse(uint32_t slotIdx);

        void setNotInUse(uint32_t slotIdx);

        bool getUsage(uint32_t slotIdx) const {
            uint32_t binIdx = slotIdx / 32;
            uint32_t flagIdxInBin = slotIdx % 32;
            uint32_t flagBin = m_flagBins[binIdx];

            return (bool)((flagBin >> flagIdxInBin) & 0x1);
        }

        uint32_t getFirstAvailableSlot() const;

        uint32_t getFirstUsedSlot() const;

        uint32_t find_nthUsedSlot(uint32_t n) const;

        uint32_t getNumSlots() const {
            return m_numFlagsInLayerList[0];
        }

        uint32_t getNumUsed() const {
            return m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[m_numLayers - 1]];
        }

        void debugPrint() const;
    };



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



    struct HitGroupSBTRecord {
        uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
        uint32_t matSlotIndex;
        uint32_t geomInstSlotIndex;
    };

    struct HitGroupSBT {
        TypedBuffer<HitGroupSBTRecord> records;
    };



    class Context::Priv {
        CUcontext cudaContext;
        OptixDeviceContext rawContext;
        Buffer materialDataBuffer;
        SlotFinder materialDataSlotFinder;

    public:
        OPTIX_OPAQUE_BRIDGE(Context);

        Priv() :
            rawContext(nullptr) {
            constexpr uint32_t InitialMaterialDataStride = 16;
            constexpr uint32_t NumInitialMaterialData = 2;

            materialDataBuffer.initialize(cudaContext, BufferType::Device, NumInitialMaterialData, InitialMaterialDataStride, 0);
            materialDataSlotFinder.initialize(NumInitialMaterialData);
        }

        CUcontext getCUDAContext() const {
            return cudaContext;
        }
        OptixDeviceContext getRawContext() const {
            return rawContext;
        }

        uint32_t requestMaterialDataSlot();
        void releaseMaterialDataSlot(uint32_t index);
        void setMaterialData(uint32_t index, const void* data, size_t size, size_t alignment);
    };



    class Material::Priv {
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

        _Context* context;
        uint32_t slotIndex;

        std::map<Key, const _ProgramGroup*> programs;

    public:
        OPTIX_OPAQUE_BRIDGE(Material);

        Priv(_Context* ctxt) :
            context(ctxt), slotIndex(context->requestMaterialDataSlot()) {}
        ~Priv() {
            context->releaseMaterialDataSlot(slotIndex);
        }

        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }

        void setRecordData(const _Pipeline* pipeline, uint32_t rayType, HitGroupSBTRecord* record) const;
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
        Buffer geomInstDataBuffer;
        SlotFinder geomInstDataSlotFinder;
        std::set<_GeometryAccelerationStructure*> geomASs;
        std::map<SBTOffsetKey, uint32_t> sbtOffsets;
        uint32_t numSBTRecords;
        std::set<_InstanceAccelerationStructure*> instASs;
        struct {
            unsigned int sbtLayoutIsUpToDate : 1;
        };

        std::map<const _Pipeline*, HitGroupSBT*> hitGroupSBTs;

    public:
        OPTIX_OPAQUE_BRIDGE(Scene);

        Priv(const _Context* ctxt) : context(ctxt), sbtLayoutIsUpToDate(false) {
            constexpr uint32_t InitialGeomInstDataStride = 16;
            constexpr uint32_t NumInitialGeomInstData = 16;

            CUcontext cudaContext = context->getCUDAContext();
            geomInstDataBuffer.initialize(cudaContext, BufferType::Device, NumInitialGeomInstData, InitialGeomInstDataStride, 0);
            geomInstDataSlotFinder.initialize(NumInitialGeomInstData);
        }

        CUcontext getCUDAContext() const {
            return context->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        uint32_t requestGeometryInstanceDataSlot();
        void releaseGeometryInstanceDataSlot(uint32_t index);
        void setGeometryInstanceData(uint32_t index, const void* data, size_t size, size_t alignment);



        void addGAS(_GeometryAccelerationStructure* gas) {
            geomASs.insert(gas);
        }
        void removeGAS(_GeometryAccelerationStructure* gas) {
            geomASs.erase(gas);
        }
        void addIAS(_InstanceAccelerationStructure* ias) {
            instASs.insert(ias);
        }
        void removeIAS(_InstanceAccelerationStructure* ias) {
            instASs.erase(ias);
        }

        void registerPipeline(const _Pipeline* pipeline);
        void generateSBTLayout(const _Pipeline* pipeline);
        bool sbtLayoutGenerationDone() const {
            return sbtLayoutIsUpToDate;
        }
        uint32_t getSBTOffset(_GeometryAccelerationStructure* gas, uint32_t matSetIdx) {
            return sbtOffsets.at(SBTOffsetKey{ gas, matSetIdx });
        }

        void setupHitGroupSBT(const _Pipeline* pipeline);
        const HitGroupSBT* getHitGroupSBT(const _Pipeline* pipeline);
    };



    class GeometryInstance::Priv {
        _Scene* scene;
        uint32_t slotIndex;

        // TODO: support deformation blur (multiple vertex buffers)
        CUdeviceptr vertexBufferArray[1];
        Buffer* vertexBuffer;
        Buffer* triangleBuffer;
        TypedBuffer<uint32_t>* materialIndexOffsetBuffer;
        std::vector<uint32_t> buildInputFlags; // per SBT record

        std::vector<std::vector<const _Material*>> materials;

    public:
        OPTIX_OPAQUE_BRIDGE(GeometryInstance);

        Priv(_Scene* _scene) :
            scene(_scene),
            slotIndex(scene->requestGeometryInstanceDataSlot()),
            vertexBuffer(nullptr), triangleBuffer(nullptr), materialIndexOffsetBuffer(nullptr) {
        }
        ~Priv() {
            scene->releaseGeometryInstanceDataSlot(slotIndex);
        }

        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        void fillBuildInput(OptixBuildInput* input) const;

        uint32_t getNumSBTRecords() const;
        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                HitGroupSBTRecord* records) const;
    };



    class GeometryAccelerationStructure::Priv {
        _Scene* scene;

        std::vector<uint32_t> numRayTypesPerMaterialSet;

        std::vector<_GeometryInstance*> children;
        std::vector<OptixBuildInput> buildInputs;

        OptixAccelBuildOptions buildOptions;

        size_t accelBufferSize;
        Buffer accelBuffer;
        Buffer accelTempBuffer;

        TypedBuffer<size_t> compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int preferFastTrace : 1;
            unsigned int allowUpdate : 1;
            unsigned int allowCompaction : 1;
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        void fillBuildInputs();

    public:
        OPTIX_OPAQUE_BRIDGE(GeometryAccelerationStructure);

        Priv(_Scene* _scene) : scene(_scene) {
            scene->addGAS(this);

            compactedSizeOnDevice.initialize(scene->getCUDAContext(), BufferType::Device, 1);

            propertyCompactedSize = OptixAccelEmitDesc{};
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = reinterpret_cast<CUdeviceptr>(compactedSizeOnDevice.getDevicePointer());

            preferFastTrace = true;
            allowUpdate = false;
            allowCompaction = false;

            available = false;
            compactedAvailable = false;
        }
        ~Priv() {
            compactedSizeOnDevice.finalize();

            scene->removeGAS(this);
        }

        CUcontext getCUDAContext() const {
            return scene->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        uint32_t getNumMaterialSets() const {
            return static_cast<uint32_t>(numRayTypesPerMaterialSet.size());
        }

        uint32_t getNumRayTypes(uint32_t matSetIdx) const {
            return numRayTypesPerMaterialSet[matSetIdx];
        }

        uint32_t calcNumSBTRecords(uint32_t matSetIdx) const;

        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, HitGroupSBTRecord* records) const;

        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            THROW_RUNTIME_ERROR(isReady(), "Traversable handle is not ready.");
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

        TypedBuffer<size_t> compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int preferFastTrace : 1;
            unsigned int allowUpdate : 1;
            unsigned int allowCompaction : 1;
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        void setupInstances();

        void fillBuildInput();

    public:
        OPTIX_OPAQUE_BRIDGE(InstanceAccelerationStructure);

        Priv(_Scene* _scene) : scene(_scene) {
            scene->addIAS(this);

            compactedSizeOnDevice.initialize(scene->getCUDAContext(), BufferType::Device, 1);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = reinterpret_cast<CUdeviceptr>(compactedSizeOnDevice.getDevicePointer());

            preferFastTrace = true;
            allowUpdate = false;
            allowCompaction = false;

            available = false;
            compactedAvailable = false;
        }
        ~Priv() {
            instanceBuffer.finalize();

            compactedSizeOnDevice.finalize();

            scene->removeIAS(this);
        }

        CUcontext getCUDAContext() const {
            return scene->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return scene->getRawContext();
        }



        bool isReady() const {
            return available || compactedAvailable;
        }

        OptixTraversableHandle getHandle() const {
            THROW_RUNTIME_ERROR(isReady(), "Traversable handle is not ready.");
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

        _Scene* scene;
        uint32_t numMissRayTypes;

        _ProgramGroup* rayGenProgram;
        _ProgramGroup* exceptionProgram;
        std::vector<_ProgramGroup*> missPrograms;
        Buffer rayGenRecord;
        Buffer missRecords;

        OptixShaderBindingTable sbt;

        struct {
            unsigned int pipelineLinked : 1;
            unsigned int sbtAllocDone : 1;
            unsigned int sbtIsUpToDate : 1;
        };

        void setupShaderBindingTable();

    public:
        OPTIX_OPAQUE_BRIDGE(Pipeline);

        Priv(const _Context* ctxt) : context(ctxt),
            maxTraceDepth(0),
            scene(nullptr), numMissRayTypes(0),
            rayGenProgram(nullptr), exceptionProgram(nullptr),
            pipelineLinked(false), sbtAllocDone(false), sbtIsUpToDate(false) {
        }

        CUcontext getCUDAContext() const {
            return context->getCUDAContext();
        }
        OptixDeviceContext getRawContext() const {
            return context->getRawContext();
        }



        void createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group);
        void destroyProgram(OptixProgramGroup group);
    };



    class Module::Priv {
        const _Pipeline* pipeline;
        OptixModule rawModule;

    public:
        OPTIX_OPAQUE_BRIDGE(Module);

        Priv(const _Pipeline* pl, OptixModule _rawModule) : pipeline(pl), rawModule(_rawModule) {}



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
        OPTIX_OPAQUE_BRIDGE(ProgramGroup);

        Priv(_Pipeline* pl, OptixProgramGroup _rawGroup) : pipeline(pl), rawGroup(_rawGroup) {}



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
