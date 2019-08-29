#include "optix_wrapper.h"
#include <optix_function_table_definition.h>

#include <vector>
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
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }



    static void logCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
        optixPrintf("[%2u][%12s]: %s\n", level, tag, message);
    }


    
#define OPTIX_ALIAS_PIMPL(Name) using _ ## Name = Name::Impl

   OPTIX_ALIAS_PIMPL(Context);
   OPTIX_ALIAS_PIMPL(ProgramGroup);
   OPTIX_ALIAS_PIMPL(GeometryInstance);
   OPTIX_ALIAS_PIMPL(GeometryAccelerationStructure);
   OPTIX_ALIAS_PIMPL(InstanceAccelerationStructure);



#define OPTIX_OPAQUE_BRIDGE(BaseName) \
    BaseName getPublicType() { \
        BaseName ret; \
        ret.m_opaque = this; \
        return ret; \
    } \
 \
    static BaseName::Impl* extract(BaseName publicType) { \
        return publicType.m_opaque; \
    }



    struct Context::Impl {
        OptixDeviceContext rawContext;

        uint32_t numRayTypes;
        uint32_t maxTraceDepth;
        OptixPipelineCompileOptions pipelineCompileOptions;
        size_t sizeOfPipelineLaunchParams;
        std::vector<OptixModule> modules;
        std::vector<OptixProgramGroup> programGroups;
        OptixPipeline pipeline;

        _ProgramGroup* rayGenProgram;
        _ProgramGroup* exceptionProgram;
        std::vector<_ProgramGroup*> missPrograms;

        struct {
            unsigned int pipelineLinked : 1;
        };

        Impl() : 
            rawContext(nullptr), rayGenProgram(nullptr), exceptionProgram(nullptr), pipelineLinked(false) {}

        OPTIX_OPAQUE_BRIDGE(Context);

        OptixDeviceContext getRawContext() const {
            return rawContext;
        }
    };



    struct ProgramGroup::Impl {
        OptixProgramGroup rawGroup;

        Impl(OptixProgramGroup group) : rawGroup(group) {
        }

        OPTIX_OPAQUE_BRIDGE(ProgramGroup);

        void packHeader(uint8_t* record) const {
            OPTIX_CHECK(optixSbtRecordPackHeader(rawGroup, record));
        }
    };


    
    struct HitGroupSet {
        const _ProgramGroup* hitGroup;
        uint8_t recordData[128];
        size_t dataSize;

        HitGroupSet() : hitGroup(nullptr) {}
        HitGroupSet(const _ProgramGroup* group, const void* data, size_t size) {
            hitGroup = group;
            std::memcpy(recordData, data, size);
            size_t alignMask = OPTIX_SBT_RECORD_ALIGNMENT - 1;
            dataSize = (size + alignMask) & ~alignMask;
        }
    };

    struct GeometryInstance::Impl {
        _Context* context;

        CUdeviceptr vertexBufferArray[1];
        Buffer* vertexBuffer;
        Buffer* triangleBuffer;
        Buffer* hitGroupIndexBuffer;
        std::vector<std::vector<HitGroupSet>> hitGroupSets;
        std::vector<uint32_t> buildInputFlags;

        Impl(_Context* ctxt) :
            context(ctxt),
            vertexBuffer(nullptr), triangleBuffer(nullptr), hitGroupIndexBuffer(nullptr) {
        }

        OPTIX_OPAQUE_BRIDGE(GeometryInstance);

        void fillBuildInput(OptixBuildInput* input) const {
            optixAssert(hitGroupSets.size() == buildInputFlags.size(), "These sizes should match.");

            std::memset(input, 0, sizeof(*input));

            input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            OptixBuildInputTriangleArray &triArray = input->triangleArray;

            triArray.vertexBuffers = vertexBufferArray;
            triArray.numVertices = vertexBuffer->numElements();
            triArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triArray.vertexStrideInBytes = vertexBuffer->stride();

            triArray.indexBuffer = triangleBuffer->getDevicePointer();
            triArray.numIndexTriplets = triangleBuffer->numElements();
            triArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triArray.indexStrideInBytes = triangleBuffer->stride();
            triArray.primitiveIndexOffset = 0;

            triArray.numSbtRecords = hitGroupSets.size();
            if (triArray.numSbtRecords > 1) {
                optixAssert_NotImplemented();
                triArray.sbtIndexOffsetBuffer = hitGroupIndexBuffer->getDevicePointer();
                triArray.sbtIndexOffsetSizeInBytes = 4;
                triArray.sbtIndexOffsetStrideInBytes = hitGroupIndexBuffer->stride();
            }
            else {
                triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
                triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
                triArray.sbtIndexOffsetStrideInBytes = 0; // No effect
            }

            triArray.preTransform = 0;

            triArray.flags = buildInputFlags.data();
        }

        // without #RayTypes multiplier
        void calcNumSBTRecordsAndMaxSize(uint32_t* numSBTRecords, size_t* maxSize) const {
            *numSBTRecords = hitGroupSets.size();
            if (maxSize == nullptr)
                return;
            *maxSize = 0;
            for (int i = 0; i < hitGroupSets.size(); ++i)
                for (int j = 0; j < hitGroupSets[i].size(); ++j)
                    *maxSize = std::max(*maxSize, OPTIX_SBT_RECORD_HEADER_SIZE + hitGroupSets[i][j].dataSize);
        }

        uintptr_t fillSBTRecords(uint8_t* sbtRecords, size_t stride) const {
            uintptr_t offset = 0;
            for (int i = 0; i < hitGroupSets.size(); ++i) {
                for (int j = 0; j < hitGroupSets[i].size(); ++j) {
                    const auto &hitGroupSet = hitGroupSets[i][j];
                    hitGroupSet.hitGroup->packHeader(sbtRecords + offset);
                    offset += OPTIX_SBT_RECORD_HEADER_SIZE;
                    std::copy_n(hitGroupSet.recordData, hitGroupSet.dataSize, sbtRecords + offset);
                    offset += stride;
                }
            }
            return offset;
        }
    };



    struct GeometryAccelerationStructure::Impl {
        _Context* context;

        std::vector<_GeometryInstance*> children;
        std::vector<OptixBuildInput> buildInputs;

        OptixAccelBuildOptions buildOptions;

        size_t accelBufferSize;
        CUDAHelper::Buffer accelBuffer;
        CUDAHelper::Buffer accelTempBuffer;

        CUDAHelper::Buffer compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        CUDAHelper::Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        Impl(_Context* ctxt) : context(ctxt) {
            compactedSizeOnDevice.initialize(CUDAHelper::BufferType::Device, 1, sizeof(size_t), 0);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

            available = false;
            compactedAvailable = false;
        }
        ~Impl() {
            compactedSizeOnDevice.finalize();
        }

        OPTIX_OPAQUE_BRIDGE(GeometryAccelerationStructure);

        bool isReady() const {
            return available || compactedAvailable;
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

        void fillBuildInputs() {
            for (int i = 0; i < children.size(); ++i)
                children[i]->fillBuildInput(&buildInputs[i]);
        }

        void calcNumSBTRecordsAndMaxSize(uint32_t* numSBTRecords, size_t* maxSize) const {
            *numSBTRecords = 0;
            if (maxSize)
                *maxSize = 0;
            for (int i = 0; i < children.size(); ++i) {
                uint32_t cNumSBTRecords = 0;
                size_t cMaxSize = 0;
                children[i]->calcNumSBTRecordsAndMaxSize(&cNumSBTRecords, maxSize ? &cMaxSize : nullptr);
                *numSBTRecords += cNumSBTRecords;
                if (maxSize)
                    *maxSize = std::max(*maxSize, cMaxSize);
            }
        }

        uintptr_t fillSBTRecords(uint8_t* sbtRecords, size_t stride) const {
            uintptr_t offset = 0;
            for (int i = 0; i < children.size(); ++i)
                offset += children[i]->fillSBTRecords(sbtRecords + offset, stride);
            return offset;
        }
    };



    enum class InstanceType {
        GAS = 0,
    };
    
    struct _Instance {
        OptixInstance rawInstance;

        InstanceType type;
        union {
            _GeometryAccelerationStructure* gas;
        };

        _Instance(_GeometryAccelerationStructure* _gas, const float transform[12]) :
            type(InstanceType::GAS), gas(_gas) {
            std::memset(&rawInstance, 0, sizeof(rawInstance));
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

    
    
    struct InstanceAccelerationStructure::Impl {
        _Context* context;

        std::vector<_Instance> children;
        OptixBuildInput buildInput;
        CUDAHelper::Buffer instanceBuffer;

        OptixAccelBuildOptions buildOptions;

        size_t accelBufferSize;
        CUDAHelper::Buffer accelBuffer;
        CUDAHelper::Buffer accelTempBuffer;

        CUDAHelper::Buffer compactedSizeOnDevice;
        size_t compactedSize;
        OptixAccelEmitDesc propertyCompactedSize;
        CUDAHelper::Buffer compactedAccelBuffer;

        OptixTraversableHandle handle;
        OptixTraversableHandle compactedHandle;
        struct {
            unsigned int available : 1;
            unsigned int compactedAvailable : 1;
        };

        Impl(_Context* ctxt) : context(ctxt) {
            compactedSizeOnDevice.initialize(CUDAHelper::BufferType::Device, 1, sizeof(size_t), 0);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

            available = false;
            compactedAvailable = false;
        }
        ~Impl() {
            compactedSizeOnDevice.finalize();
        }

        OPTIX_OPAQUE_BRIDGE(InstanceAccelerationStructure);

        bool isReady() const {
            return available || compactedAvailable;
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

        void setupInstances() {
            instanceBuffer.initialize(BufferType::Device, children.size(), sizeof(OptixInstance), 0);
            auto instancesD = (OptixInstance*)instanceBuffer.map();

            // TODO: 同じGASを複数回参照している場合の対応。
            //       同じレコードで良いのならレコード数を削れる。
            size_t sbtOffset = 0;
            for (int i = 0; i < children.size(); ++i) {
                _Instance &inst = children[i];

                if (inst.type == InstanceType::GAS) {
                    if (!inst.gas->isReady())
                        throw std::runtime_error("A GAS is not ready.");
                    inst.rawInstance.traversableHandle = inst.gas->getHandle();
                    inst.rawInstance.sbtOffset = sbtOffset * context->numRayTypes;
                    uint32_t numSBTRecords = 0;
                    inst.gas->calcNumSBTRecordsAndMaxSize(&numSBTRecords, nullptr);
                    sbtOffset += numSBTRecords;
                }
                else {
                    optixAssert_NotImplemented();
                }

                instancesD[i] = inst.rawInstance;
            }

            instanceBuffer.unmap();
        }

        void fillBuildInput() {
            std::memset(&buildInput, 0, sizeof(buildInput));
            buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            OptixBuildInputInstanceArray &instArray = buildInput.instanceArray;
            instArray.instances = instanceBuffer.getDevicePointer();
            instArray.numInstances = children.size();
        }

        void calcNumSBTRecordsAndMaxSize(uint32_t* numSBTRecords, size_t* maxSize) const {
            *numSBTRecords = 0;
            *maxSize = 0;
            for (int i = 0; i < children.size(); ++i) {
                const _Instance &inst = children[i];

                uint32_t cNumSBTRecords = 0;
                size_t cMaxSize = 0;
                if (inst.type == InstanceType::GAS) {
                    inst.gas->calcNumSBTRecordsAndMaxSize(&cNumSBTRecords, &cMaxSize);
                }
                else {
                    optixAssert_NotImplemented();
                }

                *numSBTRecords += cNumSBTRecords;
                *maxSize = std::max(*maxSize, cMaxSize);
            }
        }

        void fillSBTRecords() {
            optixAssert_NotImplemented();
        }
    };

    
    
    Context Context::create() {
        // JP: CUDAの初期化。
        //     ゼロは現在のCUDAコンテキストを意味する。
        // EN: initialize CUDA.
        //     Zero means taking the current CUDA ontext.
        CUDA_CHECK(cudaFree(0));
        CUcontext cudaContext = 0;

        OPTIX_CHECK(optixInit());

        Context ret;
        ret.m_opaque = new _Context();
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &logCallBack;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &ret.m_opaque->rawContext));

        return ret;
    }

    void Context::destroy() {
        delete m_opaque;
    }



    void Context::setNumRayTypes(uint32_t numRayTypes) const {
        m_opaque->numRayTypes = numRayTypes;
        m_opaque->missPrograms.resize(numRayTypes, nullptr);
    }

    void Context::setMaxTraceDepth(uint32_t maxTraceDepth) const {
        m_opaque->maxTraceDepth = maxTraceDepth;
    }

    void Context::setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                     bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) const {
        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set a pipeline compile options common among modules in the pipeline and the pipeline itself.
        std::memset(&m_opaque->pipelineCompileOptions, 0, sizeof(m_opaque->pipelineCompileOptions));
        m_opaque->pipelineCompileOptions.numPayloadValues = numPayloadValues;
        m_opaque->pipelineCompileOptions.numAttributeValues = numAttributeValues;
        m_opaque->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m_opaque->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m_opaque->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m_opaque->pipelineCompileOptions.exceptionFlags = exceptionFlags;

        m_opaque->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }

    int32_t Context::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const {
        int32_t moduleID = m_opaque->modules.size();

        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = maxRegisterCount;
        moduleCompileOptions.optLevel = optLevel;
        moduleCompileOptions.debugLevel = debugLevel;

        OptixModule module;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_opaque->rawContext,
                                                 &moduleCompileOptions,
                                                 &m_opaque->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &module));

        if (module)
            m_opaque->modules.push_back(module);
        else
            moduleID = -1;

        return moduleID;
    }

    ProgramGroup Context::createRayGenProgram(int32_t moduleID, const char* entryFunctionName) const {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = m_opaque->modules[moduleID];
        desc.raygen.entryFunctionName = entryFunctionName;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_opaque->rawContext,
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                &group));

        m_opaque->programGroups.push_back(group);

        return (new _ProgramGroup(group))->getPublicType();
    }

    ProgramGroup Context::createExceptionProgram(int32_t moduleID, const char* entryFunctionName) const {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.exception.module = m_opaque->modules[moduleID];
        desc.exception.entryFunctionName = entryFunctionName;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_opaque->rawContext,
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                &group));

        m_opaque->programGroups.push_back(group);

        return (new _ProgramGroup(group))->getPublicType();
    }

    ProgramGroup Context::createMissProgram(int32_t moduleID, const char* entryFunctionName) const {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = m_opaque->modules[moduleID];
        desc.miss.entryFunctionName = entryFunctionName;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_opaque->rawContext,
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                &group));

        m_opaque->programGroups.push_back(group);

        return (new _ProgramGroup(group))->getPublicType();
    }

    ProgramGroup Context::createHitProgramGroup(int32_t moduleID_CH, const char* entryFunctionNameCH,
                                                int32_t moduleID_AH, const char* entryFunctionNameAH,
                                                int32_t moduleID_IS, const char* entryFunctionNameIS) const {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH)
            desc.hitgroup.moduleCH = m_opaque->modules[moduleID_CH];
        desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        if (entryFunctionNameAH)
            desc.hitgroup.moduleAH = m_opaque->modules[moduleID_AH];
        desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        if (entryFunctionNameIS)
            desc.hitgroup.moduleIS = m_opaque->modules[moduleID_IS];
        desc.hitgroup.entryFunctionNameIS = entryFunctionNameIS;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_opaque->rawContext,
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                &group));

        m_opaque->programGroups.push_back(group);

        return (new _ProgramGroup(group))->getPublicType();
    }

    ProgramGroup Context::createCallableGroup(int32_t moduleID_DC, const char* entryFunctionNameDC,
                                              int32_t moduleID_CC, const char* entryFunctionNameCC) const {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        desc.callables.moduleDC = m_opaque->modules[moduleID_DC];
        desc.callables.entryFunctionNameDC = entryFunctionNameDC;
        desc.callables.moduleCC = m_opaque->modules[moduleID_CC];
        desc.callables.entryFunctionNameCC = entryFunctionNameCC;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(m_opaque->rawContext,
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                &group));

        m_opaque->programGroups.push_back(group);

        return (new _ProgramGroup(group))->getPublicType();
    }

    void Context::setRayGenerationProgram(ProgramGroup rayGen) const {
        m_opaque->rayGenProgram = _ProgramGroup::extract(rayGen);
    }

    void Context::setExceptionProgram(ProgramGroup exception) const {
        m_opaque->exceptionProgram = _ProgramGroup::extract(exception);

    }

    void Context::setMissProgram(uint32_t rayType, ProgramGroup miss) const {
        optixAssert(rayType < m_opaque->numRayTypes, "Invalid ray type.");
        m_opaque->missPrograms[rayType] = _ProgramGroup::extract(miss);
    }

    void Context::linkPipeline(OptixCompileDebugLevel debugLevel, bool overrideUseMotionBlur) const {
        if (!m_opaque->pipelineLinked) {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = m_opaque->maxTraceDepth;
            pipelineLinkOptions.debugLevel = debugLevel;
            pipelineLinkOptions.overrideUsesMotionBlur = overrideUseMotionBlur;

            char log[4096];
            size_t logSize = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(m_opaque->rawContext,
                                                &m_opaque->pipelineCompileOptions,
                                                &pipelineLinkOptions,
                                                m_opaque->programGroups.data(), m_opaque->programGroups.size(),
                                                log, &logSize,
                                                &m_opaque->pipeline));

            m_opaque->pipelineLinked = true;
        }
    }

    GeometryInstance Context::createGeometryInstance() const {
        return (new _GeometryInstance(m_opaque))->getPublicType();
    }

    GeometryAccelerationStructure Context::createGeometryAccelerationStructure() const {
        return (new _GeometryAccelerationStructure(m_opaque))->getPublicType();
    }

    InstanceAccelerationStructure Context::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m_opaque))->getPublicType();
    }

    void Context::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) {

    }



    void ProgramGroup::destroy() {
        optixProgramGroupDestroy(m_opaque->rawGroup);
        delete m_opaque;
    }



    void GeometryInstance::destroy() {
        delete m_opaque;
    }
    
    void GeometryInstance::setVertexBuffer(Buffer* vertexBuffer) const {
        m_opaque->vertexBuffer = vertexBuffer;
        m_opaque->vertexBufferArray[0] = vertexBuffer->getDevicePointer();
    }

    void GeometryInstance::setTriangleBuffer(Buffer* triangleBuffer) const {
        m_opaque->triangleBuffer = triangleBuffer;
    }

    void GeometryInstance::setNumHitGroups(uint32_t num) const {
        m_opaque->hitGroupSets.resize(num);
        for (int i = 0; i < num; ++i)
            m_opaque->hitGroupSets[i].resize(m_opaque->context->numRayTypes);
        m_opaque->buildInputFlags.resize(num, OPTIX_GEOMETRY_FLAG_NONE);
    }

    void GeometryInstance::setGeometryFlags(uint32_t idx, OptixGeometryFlags flags) const {
        optixAssert(idx < m_opaque->buildInputFlags.size(), "Out of bounds.");
        m_opaque->buildInputFlags[idx] = flags;
    }

    void GeometryInstance::setHitGroup(uint32_t idx, uint32_t rayType, const ProgramGroup &hitGroup,
                                       const void* sbtRecordData, size_t size) const {
        optixAssert(idx < m_opaque->hitGroupSets.size(), "Out of bounds.");
        optixAssert(rayType < m_opaque->context->numRayTypes, "Invalid ray type.");
        m_opaque->hitGroupSets[idx][rayType] = HitGroupSet(_ProgramGroup::extract(hitGroup), sbtRecordData, size);
    }



    void GeometryAccelerationStructure::destroy() {
        delete m_opaque;
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) const {
        auto _geomInst = _GeometryInstance::extract(geomInst);
        m_opaque->children.push_back(_geomInst);
        // JP: この段階では値を設定しないでおく。
        m_opaque->buildInputs.push_back(OptixBuildInput{});
    }
    
    void GeometryAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, CUstream stream) const {
        m_opaque->fillBuildInputs();

        if (!m_opaque->available) {
            std::memset(&m_opaque->buildOptions, 0, sizeof(m_opaque->buildOptions));
            m_opaque->buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                                 (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                                 (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
            //buildOptions.motionOptions

            OptixAccelBufferSizes bufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m_opaque->context->getRawContext(), &m_opaque->buildOptions,
                                                     m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                                     &bufferSizes));

            m_opaque->accelBufferSize = bufferSizes.outputSizeInBytes;
            m_opaque->accelTempBuffer.initialize(CUDAHelper::BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m_opaque->accelBuffer.initialize(CUDAHelper::BufferType::Device, m_opaque->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream,
                                    &m_opaque->buildOptions, m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    m_opaque->accelBuffer.getDevicePointer(), m_opaque->accelBuffer.size(),
                                    &m_opaque->handle,
                                    compactionEnabled ? &m_opaque->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m_opaque->available = true;
        m_opaque->compactedHandle = 0;
        m_opaque->compactedAvailable = false;
    }

    void GeometryAccelerationStructure::compaction(CUstream rebuildOrUpdateStream, CUstream stream) const {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_opaque->available || m_opaque->compactedAvailable || !compactionEnabled)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDA_CHECK(cudaStreamSynchronize(rebuildOrUpdateStream));
        CUDA_CHECK(cudaMemcpy(&m_opaque->compactedSize, (void*)m_opaque->propertyCompactedSize.result, sizeof(m_opaque->compactedSize), cudaMemcpyDeviceToHost));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m_opaque->compactedSize, (void*)m_opaque->propertyCompactedSize.result, sizeof(m_opaque->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        if (m_opaque->compactedSize < m_opaque->accelBuffer.size()) {
            m_opaque->compactedAccelBuffer.initialize(CUDAHelper::BufferType::Device, m_opaque->compactedSize, 1, 0);

            OPTIX_CHECK(optixAccelCompact(m_opaque->context->getRawContext(), stream,
                                          m_opaque->handle, m_opaque->compactedAccelBuffer.getDevicePointer(), m_opaque->compactedAccelBuffer.size(),
                                          &m_opaque->compactedHandle));

            m_opaque->compactedAvailable = true;
        }
    }

    void GeometryAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_opaque->compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m_opaque->accelBuffer.finalize();

        m_opaque->handle = 0;
        m_opaque->available = false;
    }

    void GeometryAccelerationStructure::update(CUstream stream) const {
        bool updateEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m_opaque->available && !m_opaque->compactedAvailable) || !updateEnabled)
            return;

        const CUDAHelper::Buffer &accelBuffer = m_opaque->compactedAvailable ? m_opaque->compactedAccelBuffer : m_opaque->accelBuffer;
        OptixTraversableHandle &handle = m_opaque->compactedAvailable ? m_opaque->compactedHandle : m_opaque->handle;

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream,
                                    &m_opaque->buildOptions, m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m_opaque->isReady();
    }

    OptixTraversableHandle GeometryAccelerationStructure::getHandle() const {
        return m_opaque->getHandle();
    }



    void InstanceAccelerationStructure::destroy() {
        delete m_opaque;
    }

    void InstanceAccelerationStructure::addChild(const GeometryAccelerationStructure &gas, const float instantTransform[12]) const {
        auto _gas = _GeometryAccelerationStructure::extract(gas);

        _Instance inst = _Instance(_GeometryAccelerationStructure::extract(gas), instantTransform);

        m_opaque->children.push_back(inst);
    }

    void InstanceAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, CUstream stream) const {
        m_opaque->setupInstances();
        m_opaque->fillBuildInput();

        if (!m_opaque->available) {
            std::memset(&m_opaque->buildOptions, 0, sizeof(m_opaque->buildOptions));
            m_opaque->buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                                 (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                                 (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
            //buildOptions.motionOptions

            OptixAccelBufferSizes bufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m_opaque->context->getRawContext(), &m_opaque->buildOptions,
                                                     &m_opaque->buildInput, 1,
                                                     &bufferSizes));

            m_opaque->accelBufferSize = bufferSizes.outputSizeInBytes;
            m_opaque->accelTempBuffer.initialize(CUDAHelper::BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m_opaque->accelBuffer.initialize(CUDAHelper::BufferType::Device, m_opaque->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream, &m_opaque->buildOptions, &m_opaque->buildInput, 1,
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    m_opaque->accelBuffer.getDevicePointer(), m_opaque->accelBuffer.size(),
                                    &m_opaque->handle,
                                    compactionEnabled ? &m_opaque->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m_opaque->available = true;
        m_opaque->compactedHandle = 0;
        m_opaque->compactedAvailable = false;
    }

    void InstanceAccelerationStructure::compaction(CUstream rebuildOrUpdateStream, CUstream stream) const {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_opaque->available || m_opaque->compactedAvailable || !compactionEnabled)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDA_CHECK(cudaStreamSynchronize(rebuildOrUpdateStream));
        CUDA_CHECK(cudaMemcpy(&m_opaque->compactedSize, (void*)m_opaque->propertyCompactedSize.result, sizeof(m_opaque->compactedSize), cudaMemcpyDeviceToHost));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m_opaque->compactedSize, (void*)m_opaque->propertyCompactedSize.result, sizeof(m_opaque->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        if (m_opaque->compactedSize < m_opaque->accelBuffer.size()) {
            m_opaque->compactedAccelBuffer.initialize(CUDAHelper::BufferType::Device, m_opaque->compactedSize, 1, 0);

            OPTIX_CHECK(optixAccelCompact(m_opaque->context->getRawContext(), stream,
                                          m_opaque->handle, m_opaque->compactedAccelBuffer.getDevicePointer(), m_opaque->compactedAccelBuffer.size(),
                                          &m_opaque->compactedHandle));

            m_opaque->compactedAvailable = true;
        }
    }

    void InstanceAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_opaque->compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m_opaque->accelBuffer.finalize();

        m_opaque->handle = 0;
        m_opaque->available = false;
    }

    void InstanceAccelerationStructure::update(CUstream stream) const {
        bool updateEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m_opaque->available && !m_opaque->compactedAvailable) || !updateEnabled)
            return;

        const CUDAHelper::Buffer &accelBuffer = m_opaque->compactedAvailable ? m_opaque->compactedAccelBuffer : m_opaque->accelBuffer;
        OptixTraversableHandle &handle = m_opaque->compactedAvailable ? m_opaque->compactedHandle : m_opaque->handle;

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream,
                                    &m_opaque->buildOptions, &m_opaque->buildInput, 1,
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m_opaque->isReady();
    }

    OptixTraversableHandle InstanceAccelerationStructure::getHandle() const {
        return m_opaque->getHandle();
    }
}
