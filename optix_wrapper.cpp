#include "optix_wrapper.h"
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
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }

    std::runtime_error make_runtime_error(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[4096];
        vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
        va_end(args);

        return std::runtime_error(str);
    }



    static void logCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
        optixPrintf("[%2u][%12s]: %s\n", level, tag, message);
    }


    
#define OPTIX_ALIAS_PIMPL(Name) using _ ## Name = Name::Impl

   OPTIX_ALIAS_PIMPL(Context);
   OPTIX_ALIAS_PIMPL(Pipeline);
   OPTIX_ALIAS_PIMPL(Module);
   OPTIX_ALIAS_PIMPL(ProgramGroup);
   OPTIX_ALIAS_PIMPL(GeometryInstance);
   OPTIX_ALIAS_PIMPL(GeometryAccelerationStructure);
   OPTIX_ALIAS_PIMPL(InstanceAccelerationStructure);



#define OPTIX_OPAQUE_BRIDGE(BaseName) \
    friend class BaseName; \
 \
    BaseName getPublicType() { \
        BaseName ret; \
        ret.m = this; \
        return ret; \
    } \
 \
    static BaseName::Impl* extract(BaseName publicType) { \
        return publicType.m; \
    }



    class Context::Impl {
        OptixDeviceContext rawContext;

    public:
        Impl() : 
            rawContext(nullptr) {}

        OPTIX_OPAQUE_BRIDGE(Context);

        OptixDeviceContext getRawContext() const {
            return rawContext;
        }
    };



    class Pipeline::Impl {
        const _Context* context;
        OptixPipeline rawPipeline;

        uint32_t numRayTypes;
        uint32_t maxTraceDepth;
        OptixPipelineCompileOptions pipelineCompileOptions;
        size_t sizeOfPipelineLaunchParams;
        std::set<OptixProgramGroup> programGroups;

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

        void createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group) {
            char log[4096];
            size_t logSize = sizeof(log);
            OPTIX_CHECK_LOG(optixProgramGroupCreate(context->getRawContext(),
                                                    &desc, 1, // num program groups
                                                    &options,
                                                    log, &logSize,
                                                    group));
            programGroups.insert(*group);
        }
        void destroyProgram(OptixProgramGroup group) {
            optixAssert(programGroups.count(group) > 0, "This program group has not been registered.");
            programGroups.erase(group);
            OPTIX_CHECK(optixProgramGroupDestroy(group));
        }

        void setupShaderBindingTable() {
            if (!sbtSetup) {
                if (rayGenProgram == nullptr)
                    throw make_runtime_error("Ray generation program is not set.");

                for (int i = 0; i < numRayTypes; ++i)
                    if (missPrograms[i] == nullptr)
                        throw make_runtime_error("Miss program is not set for some ray types.");

                std::memset(&sbt, 0, sizeof(sbt));
                {
                    sbt.raygenRecord = rayGenRecord.getDevicePointer();

                    sbt.exceptionRecord = 0;

                    for (int i = 0; i < numRayTypes; ++i) {
                        void* dst = (void*)(missRecords.getDevicePointer() + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                    }
                }

                sbtSetup = true;
            }
        }

    public:
        Impl(const _Context* ctxt) : context(ctxt), 
            numRayTypes(0), maxTraceDepth(0),
            rayGenProgram(nullptr), exceptionProgram(nullptr), 
            pipelineLinked(false), sbtSetup(false) {
        }

        OPTIX_OPAQUE_BRIDGE(Pipeline);

        uint32_t getNumRayTypes() const {
            if (numRayTypes == 0)
                throw make_runtime_error("Num ray types is not set.");

            return numRayTypes;
        }
    };



    class Module::Impl {
        const _Pipeline* pipeline;
        OptixModule rawModule;

    public:
        Impl(const _Pipeline* pl, OptixModule module) : pipeline(pl), rawModule(module) {
        }

        OPTIX_OPAQUE_BRIDGE(Module);

        const _Pipeline* getPipeline() const {
            return pipeline;
        }

        OptixModule getRawModule() const {
            return rawModule;
        }
    };



    class ProgramGroup::Impl {
        const _Pipeline* pipeline;
        OptixProgramGroup rawGroup;

    public:
        Impl(const _Pipeline* pl, OptixProgramGroup group) : pipeline(pl), rawGroup(group) {
        }

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


    
    class GeometryInstance::Impl {
    public:
        struct HitGroupData {
            const _ProgramGroup* program;
            uint8_t recordData[128];
            size_t dataSize;

            HitGroupData() : program(nullptr), dataSize(0) {}
            HitGroupData(const _ProgramGroup* _program, const void* data, size_t size) {
                program = _program;
                std::memcpy(recordData, data, size);
                size_t alignMask = OPTIX_SBT_RECORD_ALIGNMENT - 1;
                dataSize = (size + alignMask) & ~alignMask;
            }
        };

        struct PerPipelineInfo {
            HitGroupData* hitGroupData; // per ray type, per SBT record

            PerPipelineInfo() : hitGroupData(nullptr) {}
        };

    private:
        _Context* context;

        CUdeviceptr vertexBufferArray[1];
        Buffer* vertexBuffer;
        Buffer* triangleBuffer;
        Buffer* hitGroupIndexBuffer;
        std::vector<uint32_t> buildInputFlags; // per SBT record

        std::map<const _Pipeline*, PerPipelineInfo> perPipelineInfos;

    public:
        Impl(_Context* ctxt) :
            context(ctxt),
            vertexBuffer(nullptr), triangleBuffer(nullptr) {
        }

        OPTIX_OPAQUE_BRIDGE(GeometryInstance);

        void fillBuildInput(OptixBuildInput* input) const {
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

            triArray.numSbtRecords = buildInputFlags.size();
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
        void getNumSBTRecords(uint32_t* numSBTRecords) const {
            *numSBTRecords = buildInputFlags.size();
        }

        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint8_t* sbtRecords, size_t stride) const {
            if (perPipelineInfos.count(pipeline) == 0)
                throw make_runtime_error("The pipeline %p has not been registered.", pipeline);

            const PerPipelineInfo &info = perPipelineInfos.at(pipeline);

            uint32_t numHitGroups = buildInputFlags.size();
            uint32_t numRayTypes = pipeline->getNumRayTypes();
            for (int i = 0; i < numHitGroups * numRayTypes; ++i) {
                uintptr_t offset = i * stride;
                const auto &hitGroupData = info.hitGroupData[i];

                hitGroupData.program->packHeader(sbtRecords + offset);
                offset += OPTIX_SBT_RECORD_HEADER_SIZE;

                std::copy_n(hitGroupData.recordData, hitGroupData.dataSize, sbtRecords + offset);
            }

            return numHitGroups * numRayTypes;
        }
    };



    class GeometryAccelerationStructure::Impl {
    public:
        struct PerPipelineInfo {

        };

    private:
        _Context* context;

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

        void fillBuildInputs() {
            buildInputs.resize(children.size(), OptixBuildInput{});
            for (int i = 0; i < children.size(); ++i)
                children[i]->fillBuildInput(&buildInputs[i]);
        }

    public:
        Impl(_Context* ctxt) : context(ctxt) {
            compactedSizeOnDevice.initialize(BufferType::Device, 1, sizeof(size_t), 0);

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

        void calcNumSBTRecords(uint32_t* numSBTRecords) const {
            *numSBTRecords = 0;
            for (int i = 0; i < children.size(); ++i) {
                uint32_t cNumSBTRecords = 0;
                children[i]->getNumSBTRecords(&cNumSBTRecords);
                *numSBTRecords += cNumSBTRecords;
            }
        }

        uint32_t fillSBTRecords(const _Pipeline* pipeline, uint8_t* sbtRecords, size_t stride) const {
            uint32_t numSBTRecords = 0;
            for (int i = 0; i < children.size(); ++i)
                numSBTRecords += children[i]->fillSBTRecords(pipeline, sbtRecords + stride * numSBTRecords, stride);
            return numSBTRecords;
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

    
    
    class InstanceAccelerationStructure::Impl {
        _Context* context;

        std::vector<_Instance> children;
        OptixBuildInput buildInput;
        Buffer instanceBuffer;

        OptixAccelBuildOptions buildOptions;
        uint32_t maxNumRayTypes;

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

        void setupInstances() {
            instanceBuffer.finalize();

            instanceBuffer.initialize(BufferType::Device, children.size(), sizeof(OptixInstance), 0);
            auto instancesD = (OptixInstance*)instanceBuffer.map();

            // TODO: 同じGASを複数回参照している場合の対応。
            //       同じレコードで良いのならレコード数を削れる。
            size_t sbtOffset = 0;
            for (int i = 0; i < children.size(); ++i) {
                _Instance &inst = children[i];

                if (inst.type == InstanceType::GAS) {
                    if (!inst.gas->isReady())
                        throw make_runtime_error("GAS %p is not ready.", inst.gas);

                    inst.rawInstance.traversableHandle = inst.gas->getHandle();
                    inst.rawInstance.sbtOffset = sbtOffset * maxNumRayTypes;
                    uint32_t cNumSBTRecords = 0;
                    inst.gas->calcNumSBTRecords(&cNumSBTRecords);
                    sbtOffset += cNumSBTRecords;
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

    public:
        Impl(_Context* ctxt) : context(ctxt) {
            compactedSizeOnDevice.initialize(BufferType::Device, 1, sizeof(size_t), 0);

            std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
            propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
            propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

            available = false;
            compactedAvailable = false;
        }
        ~Impl() {
            instanceBuffer.finalize();

            compactedSizeOnDevice.finalize();
        }

        OPTIX_OPAQUE_BRIDGE(InstanceAccelerationStructure);

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

        void calcNumSBTRecords(uint32_t* numSBTRecords) const {
            *numSBTRecords = 0;
            for (int i = 0; i < children.size(); ++i) {
                const _Instance &inst = children[i];

                uint32_t cNumSBTRecords = 0;
                if (inst.type == InstanceType::GAS) {
                    inst.gas->calcNumSBTRecords(&cNumSBTRecords);
                }
                else {
                    optixAssert_NotImplemented();
                }

                *numSBTRecords += cNumSBTRecords;
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
        ret.m = new _Context();
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &logCallBack;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &ret.m->rawContext));

        return ret;
    }

    void Context::destroy() {
        OPTIX_CHECK(optixDeviceContextDestroy(m->rawContext));
        delete m;
        m = nullptr;
    }



    Pipeline Context::createPipeline() const {
        return (new _Pipeline(m))->getPublicType();
    }

    GeometryInstance Context::createGeometryInstance() const {
        return (new _GeometryInstance(m))->getPublicType();
    }

    GeometryAccelerationStructure Context::createGeometryAccelerationStructure() const {
        return (new _GeometryAccelerationStructure(m))->getPublicType();
    }

    InstanceAccelerationStructure Context::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m))->getPublicType();
    }



    void Pipeline::destroy() {
        if (m->pipelineLinked)
            OPTIX_CHECK(optixPipelineDestroy(m->rawPipeline));
        delete m;
        m = nullptr;
    }



    void Pipeline::setNumRayTypes(uint32_t numRayTypes) const {
        m->numRayTypes = numRayTypes;
        m->missPrograms.resize(numRayTypes, nullptr);
    }

    void Pipeline::setMaxTraceDepth(uint32_t maxTraceDepth) const {
        m->maxTraceDepth = maxTraceDepth;
    }

    void Pipeline::setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                      bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) const {
        // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
        // EN: Set a pipeline compile options common among modules in the pipeline and the pipeline itself.
        std::memset(&m->pipelineCompileOptions, 0, sizeof(m->pipelineCompileOptions));
        m->pipelineCompileOptions.numPayloadValues = numPayloadValues;
        m->pipelineCompileOptions.numAttributeValues = numAttributeValues;
        m->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m->pipelineCompileOptions.exceptionFlags = exceptionFlags;

        m->sizeOfPipelineLaunchParams = sizeOfLaunchParams;
    }



    Module Pipeline::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const {
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = maxRegisterCount;
        moduleCompileOptions.optLevel = optLevel;
        moduleCompileOptions.debugLevel = debugLevel;

        OptixModule module;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m->context->getRawContext(),
                                                 &moduleCompileOptions,
                                                 &m->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &module));

        return (new _Module(m, module))->getPublicType();
    }

    void Pipeline::destroyModule(Module module) const {
        auto _module = _Module::extract(module);
        OPTIX_CHECK(optixModuleDestroy(_module->getRawModule()));
        delete _module;
    }



    ProgramGroup Pipeline::createRayGenProgram(Module module, const char* entryFunctionName) const {
        auto _module = _Module::extract(module);
        if (_module->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = _module->getRawModule();
        desc.raygen.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createExceptionProgram(Module module, const char* entryFunctionName) const {
        auto _module = _Module::extract(module);
        if (_module->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
        desc.exception.module = _module->getRawModule();
        desc.exception.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createMissProgram(Module module, const char* entryFunctionName) const {
        auto _module = _Module::extract(module);
        if (_module && _module->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given module.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        if (_module)
            desc.miss.module = _module->getRawModule();
        desc.miss.entryFunctionName = entryFunctionName;

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createHitProgramGroup(Module module_CH, const char* entryFunctionNameCH,
                                                 Module module_AH, const char* entryFunctionNameAH,
                                                 Module module_IS, const char* entryFunctionNameIS) const {
        auto _module_CH = _Module::extract(module_CH);
        auto _module_AH = _Module::extract(module_AH);
        auto _module_IS = _Module::extract(module_IS);
        if (entryFunctionNameCH && _module_CH && _module_CH->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given CH module.");
        if (entryFunctionNameAH && _module_AH && _module_AH->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given AH module.");
        if (entryFunctionNameIS && _module_IS && _module_IS->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given IS module.");

        if (entryFunctionNameCH == nullptr &&
            entryFunctionNameAH == nullptr &&
            entryFunctionNameIS == nullptr)
            throw make_runtime_error("Either of CH/AH/IS entry function name should be provided.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        if (entryFunctionNameCH && _module_CH) {
            desc.hitgroup.moduleCH = _module_CH->getRawModule();
            desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        }
        if (entryFunctionNameAH && _module_AH) {
            desc.hitgroup.moduleAH = _module_AH->getRawModule();
            desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
        }
        if (entryFunctionNameIS && _module_IS) {
            desc.hitgroup.moduleIS = _module_IS->getRawModule();
            desc.hitgroup.entryFunctionNameIS = entryFunctionNameIS;
        }

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    ProgramGroup Pipeline::createCallableGroup(Module module_DC, const char* entryFunctionNameDC,
                                               Module module_CC, const char* entryFunctionNameCC) const {
        auto _module_DC = _Module::extract(module_DC);
        auto _module_CC = _Module::extract(module_CC);
        if (entryFunctionNameDC && _module_DC && _module_DC->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given DC module.");
        if (entryFunctionNameCC && _module_CC && _module_CC->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given CC module.");

        if (entryFunctionNameDC == nullptr && entryFunctionNameCC == nullptr)
            throw make_runtime_error("Either of CC/DC entry function name should be provided.");

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
        if (entryFunctionNameDC && _module_DC) {
            desc.callables.moduleDC = _module_DC->getRawModule();
            desc.callables.entryFunctionNameDC = entryFunctionNameDC;
        }
        if (entryFunctionNameCC && _module_CC) {
            desc.callables.moduleCC = _module_CC->getRawModule();
            desc.callables.entryFunctionNameCC = entryFunctionNameCC;
        }

        OptixProgramGroupOptions options = {};

        OptixProgramGroup group;
        m->createProgram(desc, options, &group);

        return (new _ProgramGroup(m, group))->getPublicType();
    }

    void Pipeline::destroyProgramGroup(ProgramGroup program) const {
        auto _program = _ProgramGroup::extract(program);
        m->destroyProgram(_program->getRawProgramGroup());
        delete _program;
    }



    void Pipeline::link(OptixCompileDebugLevel debugLevel, bool overrideUseMotionBlur) const {
        if (m->pipelineLinked)
            throw make_runtime_error("This pipeline has been already linked.");

        if (!m->pipelineLinked) {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = m->maxTraceDepth;
            pipelineLinkOptions.debugLevel = debugLevel;
            pipelineLinkOptions.overrideUsesMotionBlur = overrideUseMotionBlur;

            std::vector<OptixProgramGroup> groups;
            groups.resize(m->programGroups.size());
            std::copy(m->programGroups.cbegin(), m->programGroups.cend(), groups.begin());

            char log[4096];
            size_t logSize = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(m->context->getRawContext(),
                                                &m->pipelineCompileOptions,
                                                &pipelineLinkOptions,
                                                groups.data(), groups.size(),
                                                log, &logSize,
                                                &m->rawPipeline));

            m->pipelineLinked = true;



            m->rayGenRecord.initialize(BufferType::Device, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
            m->missRecords.initialize(BufferType::Device, m->numRayTypes, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
        }
    }



    void Pipeline::setRayGenerationProgram(ProgramGroup program) const {
        auto _program = _ProgramGroup::extract(program);
        if (_program == nullptr)
            throw make_runtime_error("Invalid program %p.", _program);
        if (_program->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given program (group).");

        m->rayGenProgram = _program;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        auto _program = _ProgramGroup::extract(program);
        if (_program == nullptr)
            throw make_runtime_error("Invalid program %p.", _program);
        if (_program->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given program (group).");

        m->exceptionProgram = _program;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        auto _program = _ProgramGroup::extract(program);
        if (rayType >= m->numRayTypes)
            throw make_runtime_error("Invalid ray type.");
        if (_program == nullptr)
            throw make_runtime_error("Invalid program %p.", _program);
        if (_program->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given program (group).");

        m->missPrograms[rayType] = _program;
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) {
        m->setupShaderBindingTable();

        OPTIX_CHECK(optixLaunch(m->rawPipeline, stream, plpOnDevice, m->sizeOfPipelineLaunchParams,
                                &m->sbt, dimX, dimY, dimZ));
    }



    void GeometryInstance::destroy() {
        for (auto &it : m->perPipelineInfos) {
            if (it.second.hitGroupData)
                delete[] it.second.hitGroupData;
        }
        delete m;
        m = nullptr;
    }
    
    void GeometryInstance::setVertexBuffer(Buffer* vertexBuffer) const {
        m->vertexBuffer = vertexBuffer;
        m->vertexBufferArray[0] = vertexBuffer->getDevicePointer();
    }

    void GeometryInstance::setTriangleBuffer(Buffer* triangleBuffer) const {
        m->triangleBuffer = triangleBuffer;
    }

    void GeometryInstance::setNumHitGroups(uint32_t numHitGroups) const {
        if (m->buildInputFlags.size() != 0)
            throw make_runtime_error("Number of hit groups has been already set.");

        m->buildInputFlags.resize(numHitGroups, OPTIX_GEOMETRY_FLAG_NONE);
    }

    void GeometryInstance::setGeometryFlags(uint32_t hitGroupIdx, OptixGeometryFlags flags) const {
        size_t numHitGroups = m->buildInputFlags.size();
        if (hitGroupIdx >= numHitGroups)
            throw make_runtime_error("Out of hit group bounds [0, %u).", (uint32_t)numHitGroups);

        m->buildInputFlags[hitGroupIdx] = flags;
    }

    void GeometryInstance::setHitGroup(Pipeline pipeline, uint32_t hitGroupIdx, uint32_t rayType, const ProgramGroup &hitGroup,
                                       const void* sbtRecordData, size_t size) const {
        const auto _pipeline = _Pipeline::extract(pipeline);
        if (_pipeline == nullptr)
            throw make_runtime_error("Invalid pipeline.");

        bool isNewPipeline = m->perPipelineInfos.count(_pipeline) == 0;

        _GeometryInstance::PerPipelineInfo &info = m->perPipelineInfos[_pipeline];
        size_t numHitGroups = m->buildInputFlags.size();
        if (hitGroupIdx >= numHitGroups)
            throw make_runtime_error("Out of hit group bounds [0, %u).", (uint32_t)numHitGroups);
        uint32_t numRayTypes = _pipeline->getNumRayTypes();
        if (rayType >= numRayTypes)
            throw make_runtime_error("Invalid ray type.");

        if (isNewPipeline)
            info.hitGroupData = new _GeometryInstance::HitGroupData[numHitGroups * numRayTypes];
        info.hitGroupData[hitGroupIdx * numRayTypes + rayType] = _GeometryInstance::HitGroupData(_ProgramGroup::extract(hitGroup), sbtRecordData, size);
    }



    void GeometryAccelerationStructure::destroy() {
        m->accelBuffer.finalize();
        m->accelTempBuffer.finalize();

        delete m;
        m = nullptr;
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) const {
        auto _geomInst = _GeometryInstance::extract(geomInst);
        if (_geomInst == nullptr)
            throw make_runtime_error("Invalid geometry instance %p.", _geomInst);

        m->children.push_back(_geomInst);
    }
    
    void GeometryAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, CUstream stream) const {
        m->fillBuildInputs();

        {
            m->accelBuffer.finalize();
            m->accelTempBuffer.finalize();

            std::memset(&m->buildOptions, 0, sizeof(m->buildOptions));
            m->buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                          (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                          (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
            //buildOptions.motionOptions

            OptixAccelBufferSizes bufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m->context->getRawContext(), &m->buildOptions,
                                                     m->buildInputs.data(), m->buildInputs.size(),
                                                     &bufferSizes));

            m->accelBufferSize = bufferSizes.outputSizeInBytes;
            m->accelTempBuffer.initialize(BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m->accelBuffer.initialize(BufferType::Device, m->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->context->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    m->accelTempBuffer.getDevicePointer(), m->accelTempBuffer.size(),
                                    m->accelBuffer.getDevicePointer(), m->accelBuffer.size(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m->available = true;
        m->compactedHandle = 0;
        m->compactedAvailable = false;
    }

    void GeometryAccelerationStructure::compaction(CUstream rebuildOrUpdateStream, CUstream stream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->available || m->compactedAvailable || !compactionEnabled)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDA_CHECK(cudaStreamSynchronize(rebuildOrUpdateStream));
        CUDA_CHECK(cudaMemcpy(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        if (m->compactedSize < m->accelBuffer.size()) {
            m->compactedAccelBuffer.initialize(BufferType::Device, m->compactedSize, 1, 0);

            OPTIX_CHECK(optixAccelCompact(m->context->getRawContext(), stream,
                                          m->handle, m->compactedAccelBuffer.getDevicePointer(), m->compactedAccelBuffer.size(),
                                          &m->compactedHandle));

            m->compactedAvailable = true;
        }
    }

    void GeometryAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m->accelBuffer.finalize();

        m->handle = 0;
        m->available = false;
    }

    void GeometryAccelerationStructure::update(CUstream stream) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m->available && !m->compactedAvailable) || !updateEnabled)
            return;

        const Buffer &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->context->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    m->accelTempBuffer.getDevicePointer(), m->accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle GeometryAccelerationStructure::getHandle() const {
        return m->getHandle();
    }



    void InstanceAccelerationStructure::destroy() {
        m->accelBuffer.finalize();
        m->accelTempBuffer.finalize();

        delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::addChild(const GeometryAccelerationStructure &gas, const float instantTransform[12]) const {
        auto _gas = _GeometryAccelerationStructure::extract(gas);
        if (_gas == nullptr)
            throw make_runtime_error("Invalid GAS %p.", _gas);

        _Instance inst = _Instance(_gas, instantTransform);

        m->children.push_back(inst);
    }

    void InstanceAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, uint32_t maxNumRayTypes, CUstream stream) const {
        m->maxNumRayTypes = maxNumRayTypes;
        m->setupInstances();
        m->fillBuildInput();

        {
            m->accelBuffer.finalize();
            m->accelTempBuffer.finalize();

            std::memset(&m->buildOptions, 0, sizeof(m->buildOptions));
            m->buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                          (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                          (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
            //buildOptions.motionOptions

            OptixAccelBufferSizes bufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m->context->getRawContext(), &m->buildOptions,
                                                     &m->buildInput, 1,
                                                     &bufferSizes));

            m->accelBufferSize = bufferSizes.outputSizeInBytes;
            m->accelTempBuffer.initialize(BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m->accelBuffer.initialize(BufferType::Device, m->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->context->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
                                    m->accelTempBuffer.getDevicePointer(), m->accelTempBuffer.size(),
                                    m->accelBuffer.getDevicePointer(), m->accelBuffer.size(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m->available = true;
        m->compactedHandle = 0;
        m->compactedAvailable = false;
    }

    void InstanceAccelerationStructure::compaction(CUstream rebuildOrUpdateStream, CUstream stream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->available || m->compactedAvailable || !compactionEnabled)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDA_CHECK(cudaStreamSynchronize(rebuildOrUpdateStream));
        CUDA_CHECK(cudaMemcpy(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        if (m->compactedSize < m->accelBuffer.size()) {
            m->compactedAccelBuffer.initialize(BufferType::Device, m->compactedSize, 1, 0);

            OPTIX_CHECK(optixAccelCompact(m->context->getRawContext(), stream,
                                          m->handle, m->compactedAccelBuffer.getDevicePointer(), m->compactedAccelBuffer.size(),
                                          &m->compactedHandle));

            m->compactedAvailable = true;
        }
    }

    void InstanceAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m->accelBuffer.finalize();

        m->handle = 0;
        m->available = false;
    }

    void InstanceAccelerationStructure::update(CUstream stream) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m->available && !m->compactedAvailable) || !updateEnabled)
            return;

        const Buffer &accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->context->getRawContext(), stream,
                                    &m->buildOptions, &m->buildInput, 1,
                                    m->accelTempBuffer.getDevicePointer(), m->accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
    }

    OptixTraversableHandle InstanceAccelerationStructure::getHandle() const {
        return m->getHandle();
    }
}
