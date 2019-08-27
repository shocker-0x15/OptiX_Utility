#include "optix_wrapper.h"

#include <vector>
#include <algorithm>

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
   //OPTIX_ALIAS_PIMPL(InstanceAccelerationStructure);



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
        std::vector<OptixModule> modules;
        std::vector<OptixProgramGroup> programGroups;
        OptixPipeline pipeline;

        Impl() : rawContext(nullptr) {}

        OPTIX_OPAQUE_BRIDGE(Context);

        void linkPipeline() {
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = maxTraceDepth;
            pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
            pipelineLinkOptions.overrideUsesMotionBlur = false;

            char log[4096];
            size_t logSize = sizeof(log);
            OPTIX_CHECK_LOG(optixPipelineCreate(rawContext,
                                                &pipelineCompileOptions,
                                                &pipelineLinkOptions,
                                                programGroups.data(), programGroups.size(),
                                                log, &logSize,
                                                &pipeline));
        }

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
        HitGroupSet(const _ProgramGroup* group, void* data, size_t size) {
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
        uint32_t getNumSBTRecords() const {
            return hitGroupSets.size();
        }

        size_t calcSBTRecordMaxSize() const {
            size_t ret = 0;
            for (int i = 0; i < hitGroupSets.size(); ++i)
                for (int j = 0; j < hitGroupSets[i].size(); ++j)
                    ret = std::max(ret, OPTIX_SBT_RECORD_HEADER_SIZE + hitGroupSets[i][j].dataSize);
            return ret;
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

        uint32_t calcNumSBTRecords() const {
            uint32_t numSBTRecords = 0;
            for (int i = 0; i < children.size(); ++i)
                numSBTRecords += children[i]->getNumSBTRecords();
            return numSBTRecords;
        }

        size_t calcSBTRecordMaxSize() const {
            size_t ret = 0;
            for (int i = 0; i < children.size(); ++i)
                ret = std::max(ret, children[i]->calcSBTRecordMaxSize());
            return ret;
        }

        uintptr_t fillSBTRecords(uint8_t* sbtRecords, size_t stride) const {
            uintptr_t offset = 0;
            for (int i = 0; i < children.size(); ++i)
                offset += children[i]->fillSBTRecords(sbtRecords + offset, stride);
            return offset;
        }
    };



    //struct InstanceAccelerationStructure::Impl {
    //    _Context &context;

    //    std::vector<_Instance> children;
    //    OptixBuildInput buildInput;
    //    CUDAHelper::Buffer instanceBuffer;

    //    OptixAccelBuildOptions buildOptions;

    //    size_t accelBufferSize;
    //    CUDAHelper::Buffer accelBuffer;
    //    CUDAHelper::Buffer accelTempBuffer;

    //    CUDAHelper::Buffer compactedSizeOnDevice;
    //    size_t compactedSize;
    //    OptixAccelEmitDesc propertyCompactedSize;
    //    CUDAHelper::Buffer compactedAccelBuffer;

    //    OptixTraversableHandle handle;
    //    OptixTraversableHandle compactedHandle;
    //    struct {
    //        unsigned int available : 1;
    //        unsigned int compactedAvailable : 1;
    //    };

    //    _InstanceAccelerationStructure(_Context &ctxt) : context(ctxt) {
    //        compactedSizeOnDevice.initialize(CUDAHelper::BufferType::Device, 1, sizeof(size_t), 0);

    //        std::memset(&propertyCompactedSize, 0, sizeof(propertyCompactedSize));
    //        propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    //        propertyCompactedSize.result = compactedSizeOnDevice.getDevicePointer();

    //        available = false;
    //        compactedAvailable = false;
    //    }
    //    ~_InstanceAccelerationStructure() {
    //        compactedSizeOnDevice.finalize();
    //    }

    //    void fillSBTRecords() {
    //        optixAssert_NotImplemented();
    //    }
    //};

    
    
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



    void Context::setNumRayTypes(uint32_t numRayTypes) {
        m_opaque->numRayTypes = numRayTypes;
    }

    void Context::setMaxTraceDepth(uint32_t maxTraceDepth) {
        m_opaque->maxTraceDepth = maxTraceDepth;
    }

    void Context::setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName,
                                     bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) {
        std::memset(&m_opaque->pipelineCompileOptions, 0, sizeof(m_opaque->pipelineCompileOptions));
        m_opaque->pipelineCompileOptions.numPayloadValues = numPayloadValues;
        m_opaque->pipelineCompileOptions.numAttributeValues = numAttributeValues;
        m_opaque->pipelineCompileOptions.pipelineLaunchParamsVariableName = launchParamsVariableName;
        m_opaque->pipelineCompileOptions.usesMotionBlur = useMotionBlur;
        m_opaque->pipelineCompileOptions.traversableGraphFlags = traversableGraphFlags;
        m_opaque->pipelineCompileOptions.exceptionFlags = exceptionFlags;
    }

    int32_t Context::createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) {
        int32_t moduleID = m_opaque->modules.size();
        m_opaque->modules.push_back(OptixModule());
        OptixModule &module = m_opaque->modules.back();

        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = maxRegisterCount;
        moduleCompileOptions.optLevel = optLevel;
        moduleCompileOptions.debugLevel = debugLevel;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m_opaque->rawContext,
                                                 &moduleCompileOptions,
                                                 &m_opaque->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &module));

        return moduleID;
    }

    ProgramGroup Context::createRayGenProgram(int32_t moduleID, const char* entryFunctionName) {
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

    ProgramGroup Context::createExceptionProgram(int32_t moduleID, const char* entryFunctionName) {
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

    ProgramGroup Context::createMissProgram(int32_t moduleID, const char* entryFunctionName) {
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
                                                int32_t moduleID_IS, const char* entryFunctionNameIS) {
        OptixProgramGroup group;

        OptixProgramGroupOptions options = {};

        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = m_opaque->modules[moduleID_CH];
        desc.hitgroup.entryFunctionNameCH = entryFunctionNameCH;
        desc.hitgroup.moduleAH = m_opaque->modules[moduleID_AH];
        desc.hitgroup.entryFunctionNameAH = entryFunctionNameAH;
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
                                              int32_t moduleID_CC, const char* entryFunctionNameCC) {
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

    GeometryInstance Context::createGeometryInstance() {
        return (new _GeometryInstance(m_opaque))->getPublicType();
    }

    GeometryAccelerationStructure Context::createGeometryAccelerationStructure() {
        return (new _GeometryAccelerationStructure(m_opaque))->getPublicType();
    }



    void ProgramGroup::destroy() {
        optixProgramGroupDestroy(m_opaque->rawGroup);
        delete m_opaque;
    }



    void GeometryInstance::destroy() {
        delete m_opaque;
    }
    
    void GeometryInstance::setVertexBuffer(Buffer &vertexBuffer) {
        m_opaque->vertexBuffer = &vertexBuffer;
        m_opaque->vertexBufferArray[0] = vertexBuffer.getDevicePointer();
    }

    void GeometryInstance::setTriangleBuffer(Buffer &triangleBuffer) {
        m_opaque->triangleBuffer = &triangleBuffer;
    }

    void GeometryInstance::setNumHitGroups(uint32_t num) {
        m_opaque->hitGroupSets.resize(num);
        for (int i = 0; i < num; ++i)
            m_opaque->hitGroupSets[i].resize(m_opaque->context->numRayTypes);
        m_opaque->buildInputFlags.resize(num, OPTIX_GEOMETRY_FLAG_NONE);
    }

    void GeometryInstance::setGeometryFlags(uint32_t idx, OptixGeometryFlags flags) {
        optixAssert(idx < m_opaque->buildInputFlags.size(), "Out of bounds.");
        m_opaque->buildInputFlags[idx] = flags;
    }

    void GeometryInstance::setHitGroup(uint32_t idx, uint32_t rayType, const ProgramGroup &hitGroup,
                                       void* sbtRecordData, size_t size) {
        optixAssert(idx < m_opaque->hitGroupSets.size(), "Out of bounds.");
        optixAssert(rayType < m_opaque->context->numRayTypes, "Invalid ray type.");
        m_opaque->hitGroupSets[idx][rayType] = HitGroupSet(_ProgramGroup::extract(hitGroup), sbtRecordData, size);
    }



    void GeometryAccelerationStructure::destroy() {
        delete m_opaque;
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) {
        auto _geomInst = _GeometryInstance::extract(geomInst);
        m_opaque->children.push_back(_geomInst);
        // JP: この段階では値を設定しないでおく。
        m_opaque->buildInputs.push_back(OptixBuildInput{});
    }
    
    void GeometryAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, CUstream stream) {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        for (int i = 0; i < m_opaque->children.size(); ++i)
            m_opaque->children[i]->fillBuildInput(&m_opaque->buildInputs[i]);

        if (!m_opaque->available) {
            std::memset(&m_opaque->buildOptions, 0, sizeof(m_opaque->buildOptions));
            m_opaque->buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                                 (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                                 (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
            //buildOptions.motionOptions

            OptixAccelBufferSizes bufferSizes;
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m_opaque->context->getRawContext(), &m_opaque->buildOptions, m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                                     &bufferSizes));

            m_opaque->accelBufferSize = bufferSizes.outputSizeInBytes;
            m_opaque->accelTempBuffer.initialize(CUDAHelper::BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m_opaque->accelBuffer.initialize(CUDAHelper::BufferType::Device, m_opaque->accelBufferSize, 1, 0);
        }

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream, &m_opaque->buildOptions, m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    m_opaque->accelBuffer.getDevicePointer(), m_opaque->accelBuffer.size(),
                                    &m_opaque->handle,
                                    compactionEnabled ? &m_opaque->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m_opaque->available = true;
        m_opaque->compactedHandle = 0;
        m_opaque->compactedAvailable = false;
    }

    void GeometryAccelerationStructure::compaction(CUstream rebuildOrUpdateStream, CUstream stream) {
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

            OPTIX_CHECK(optixAccelCompact(m_opaque->context->getRawContext(), stream, m_opaque->handle, m_opaque->compactedAccelBuffer.getDevicePointer(), m_opaque->compactedAccelBuffer.size(),
                                          &m_opaque->compactedHandle));

            m_opaque->compactedAvailable = true;
        }
    }

    void GeometryAccelerationStructure::removeUncompacted(CUstream compactionStream) {
        bool compactionEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_opaque->compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m_opaque->accelBuffer.finalize();

        m_opaque->handle = 0;
        m_opaque->available = false;
    }

    void GeometryAccelerationStructure::update(CUstream stream) {
        bool updateEnabled = (m_opaque->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m_opaque->available && !m_opaque->compactedAvailable) || !updateEnabled)
            return;

        const CUDAHelper::Buffer &accelBuffer = m_opaque->compactedAvailable ? m_opaque->compactedAccelBuffer : m_opaque->accelBuffer;
        OptixTraversableHandle &handle = m_opaque->compactedAvailable ? m_opaque->compactedHandle : m_opaque->handle;

        m_opaque->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m_opaque->context->getRawContext(), stream, &m_opaque->buildOptions, m_opaque->buildInputs.data(), m_opaque->buildInputs.size(),
                                    m_opaque->accelTempBuffer.getDevicePointer(), m_opaque->accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }
}
