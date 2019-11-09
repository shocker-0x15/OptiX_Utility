#include "optix_util_private.h"

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



    Material Context::createMaterial() const {
        return (new _Material(m))->getPublicType();
    }
    
    Scene Context::createScene() const {
        return (new _Scene(m))->getPublicType();
    }

    Pipeline Context::createPipeline() const {
        return (new _Pipeline(m))->getPublicType();
    }



    void Material::destroy() {
        delete m;
        m = nullptr;
    }
    
    void Material::setData(uint32_t rayType, const ProgramGroup &hitGroup,
                           const void* sbtRecordData, size_t size, size_t alignment) const {
        auto _pipeline = extract(hitGroup)->getPipeline();
        if (_pipeline == nullptr)
            throw make_runtime_error("Invalid pipeline.");

        _Material::Key key{ _pipeline, rayType };
        m->infos[key] = _Material::Info(extract(hitGroup), sbtRecordData, size, alignment);
    }



    SizeAlign Scene::Priv::calcHitGroupRecordStride(const _Pipeline* pipeline) const {
        SizeAlign maxSizeAlign;
        for (auto gas : geomASs) {
            SizeAlign sizeAlign = gas->calcHitGroupRecordStride(pipeline);
            maxSizeAlign = max(maxSizeAlign, sizeAlign);
        }
        return maxSizeAlign;
    }

    void Scene::Priv::fillSBTRecords(const _Pipeline* pipeline, uint8_t* records, uint32_t stride) const {
        for (auto gas : geomASs) {
            for (int j = 0; j < gas->getNumMaterialSets(); ++j) {
                records += gas->fillSBTRecords(pipeline, j, records, stride);
            }
        }
    }
    
    void Scene::destroy() {
        delete m;
        m = nullptr;
    }
    
    GeometryInstance Scene::createGeometryInstance() const {
        auto _geomInst = new _GeometryInstance(m);
        return _geomInst->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure() const {
        auto _geomAS = new _GeometryAccelerationStructure(m);
        m->geomASs.insert(_geomAS);
        return _geomAS->getPublicType();
    }

    InstanceAccelerationStructure Scene::createInstanceAccelerationStructure() const {
        auto _instAS = new _InstanceAccelerationStructure(m);
        m->instASs.insert(_instAS);
        return _instAS->getPublicType();
    }

    void Scene::generateSBTLayout() const {
        uint32_t sbtOffset = 0;
        m->sbtOffsets.clear();
        for (auto gas : m->geomASs) {
            for (int matSetIdx = 0; matSetIdx < gas->getNumMaterialSets(); ++matSetIdx) {
                uint32_t gasNumSBTRecords = gas->calcNumSBTRecords(matSetIdx);
                _Scene::SBTOffsetKey key = { gas, matSetIdx };
                m->sbtOffsets[key] = sbtOffset;
                sbtOffset += gasNumSBTRecords;
            }
        }
        m->numSBTRecords = sbtOffset;
        m->sbtOffsetsAreDirty = false;
    }



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input) const {
        *input = OptixBuildInput{};

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
            triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer->getDevicePointer();
            triArray.sbtIndexOffsetSizeInBytes = 4;
            triArray.sbtIndexOffsetStrideInBytes = materialIndexOffsetBuffer->stride();
        }
        else {
            triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
            triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
            triArray.sbtIndexOffsetStrideInBytes = 0; // No effect
        }

        triArray.preTransform = 0;

        triArray.flags = buildInputFlags.data();
    }

    SizeAlign GeometryInstance::Priv::calcHitGroupRecordStride(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes) const {
        SizeAlign maxSizeAlign;
        MaterialKey key{ matSetIdx, -1 };
        for (int matIdx = 0; matIdx < buildInputFlags.size(); ++matIdx) {
            key.matIndex = matIdx;
            if (materials.count(key) == 0)
                throw make_runtime_error("No material registered for Material set: %u, Material: %u", matSetIdx, matIdx);

            const _Material* mat = materials.at(key);
            for (int rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                // Header Region
                SizeAlign stride(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
                // GeometryInstance Region
                stride += sizeAlign;
                // Material Region
                stride += mat->getSBTRecord(pipeline, rIdx).sizeAlign;
                stride.alignUp();

                maxSizeAlign = max(maxSizeAlign, stride);
            }
        }

        return maxSizeAlign;
    }

    uint32_t GeometryInstance::Priv::getNumSBTRecords() const {
        return static_cast<uint32_t>(buildInputFlags.size());
    }

    uint32_t GeometryInstance::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                                    uint8_t* records, size_t stride) const {
        const uint8_t* orgHead = records;

        MaterialKey key{ matSetIdx, -1 };
        for (int matIdx = 0; matIdx < buildInputFlags.size(); ++matIdx) {
            key.matIndex = matIdx;
            if (materials.count(key) == 0)
                throw make_runtime_error("No material registered for Material set: %u, Material: %u", matSetIdx, matIdx);

            const _Material* mat = materials.at(key);
            for (int rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                const Material::Priv::Info &matInfo = mat->getSBTRecord(pipeline, rIdx);

                SizeAlign sa;
                uint32_t offset;
                constexpr SizeAlign saHeader(OPTIX_SBT_RECORD_HEADER_SIZE, OPTIX_SBT_RECORD_ALIGNMENT);
                // Header Region
                sa.add(saHeader, &offset);
                matInfo.program->packHeader(records + offset);
                // GeometryInstance Region
                sa.add(sizeAlign, &offset);
                std::memcpy(records + offset, recordData, sizeAlign.size);
                // Material Region
                sa.add(matInfo.sizeAlign, &offset);
                std::memcpy(records + offset, matInfo.recordData, matInfo.sizeAlign.size);

                records += stride;
            }
        }

        return static_cast<uint32_t>(records - orgHead);
    }

    void GeometryInstance::destroy() {
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

    void GeometryInstance::setMaterialIndexOffsetBuffer(Buffer* matIdxOffsetBufferX) const {
        m->materialIndexOffsetBuffer = matIdxOffsetBufferX;
    }

    void GeometryInstance::setData(const void* sbtRecordData, size_t size, size_t alignment) const {
        std::memcpy(m->recordData, sbtRecordData, size);
        m->sizeAlign = SizeAlign(size, alignment);
    }

    void GeometryInstance::setNumMaterials(uint32_t numMaterials) const {
        if (m->buildInputFlags.size() != 0)
            throw make_runtime_error("Number of hit groups has been already set.");

        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
    }

    void GeometryInstance::setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const {
        size_t numMaterials = m->buildInputFlags.size();
        if (matIdx >= numMaterials)
            throw make_runtime_error("Out of material bounds [0, %u).", (uint32_t)numMaterials);

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->buildInputFlags.size();
        if (matIdx >= numMaterials)
            throw make_runtime_error("Out of material bounds [0, %u).", (uint32_t)numMaterials);

        _GeometryInstance::MaterialKey key{ matSetIdx, matIdx };
        m->materials[key] = extract(mat);
    }



    void GeometryAccelerationStructure::Priv::fillBuildInputs() {
        buildInputs.resize(children.size(), OptixBuildInput{});
        for (int i = 0; i < children.size(); ++i)
            children[i]->fillBuildInput(&buildInputs[i]);
    }
    
    SizeAlign GeometryAccelerationStructure::Priv::calcHitGroupRecordStride(const _Pipeline* pipeline) const {
        SizeAlign maxSizeAlign;
        for (int matSetIdx = 0; matSetIdx < numRayTypesValues.size(); ++matSetIdx) {
            uint32_t numRayTypes = numRayTypesValues[matSetIdx];
            for (int i = 0; i < children.size(); ++i) {
                SizeAlign sizeAlign = children[i]->calcHitGroupRecordStride(pipeline, matSetIdx, numRayTypes);
                maxSizeAlign = max(maxSizeAlign, sizeAlign);
            }
        }
        return maxSizeAlign;
    }

    uint32_t GeometryAccelerationStructure::Priv::calcNumSBTRecords(uint32_t matSetIdx) const {
        uint32_t numSBTRecords = 0;
        for (int i = 0; i < children.size(); ++i)
            numSBTRecords += children[i]->getNumSBTRecords();
        numSBTRecords *= numRayTypesValues[matSetIdx];

        return numSBTRecords;
    }

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint8_t* records, uint32_t stride) const {
        if (matSetIdx >= numRayTypesValues.size())
            throw make_runtime_error("Material set index %u is out of bound [0, %u).",
                                     matSetIdx, static_cast<uint32_t>(numRayTypesValues.size()));

        const uint8_t* orgHead = records;
        for (int i = 0; i < children.size(); ++i)
            records += children[i]->fillSBTRecords(pipeline, matSetIdx, numRayTypesValues[matSetIdx], records, stride);

        return static_cast<uint32_t>(records - orgHead);
    }
    
    void GeometryAccelerationStructure::destroy() {
        m->scene->removeGAS(m);

        m->accelBuffer.finalize();
        m->accelTempBuffer.finalize();

        delete m;
        m = nullptr;
    }

    void GeometryAccelerationStructure::setNumMaterialSets(uint32_t numMatSets) const {
        m->numRayTypesValues.resize(numMatSets, 0);
    }

    void GeometryAccelerationStructure::setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const {
        if (matSetIdx >= m->numRayTypesValues.size())
            throw make_runtime_error("Material set index %u is out of bounds [0, %u).",
                                     matSetIdx, static_cast<uint32_t>(m->numRayTypesValues.size()));
        m->numRayTypesValues[matSetIdx] = numRayTypes;
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) const {
        auto _geomInst = extract(geomInst);
        if (_geomInst == nullptr)
            throw make_runtime_error("Invalid geometry instance %p.", _geomInst);

        m->children.push_back(_geomInst);

        m->available = false;
        m->compactedAvailable = false;
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
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                     m->buildInputs.data(), m->buildInputs.size(),
                                                     &bufferSizes));

            m->accelBufferSize = bufferSizes.outputSizeInBytes;
            m->accelTempBuffer.initialize(BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m->accelBuffer.initialize(BufferType::Device, m->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
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

            OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
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
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
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



    void InstanceAccelerationStructure::Priv::setupInstances() {
        if (!scene->sbtOffsetsGenerationIsDone())
            throw make_runtime_error("SBT layout generation should be done before.");

        instanceBuffer.finalize();

        instanceBuffer.initialize(BufferType::Device, children.size(), sizeof(OptixInstance), 0);
        auto instancesD = (OptixInstance*)instanceBuffer.map();

        for (int i = 0; i < children.size(); ++i) {
            _Instance &inst = children[i];

            if (inst.type == InstanceType::GAS) {
                if (!inst.gas->isReady())
                    throw make_runtime_error("GAS %p is not ready.", inst.gas);

                inst.rawInstance.traversableHandle = inst.gas->getHandle();
                inst.rawInstance.sbtOffset = scene->getSBTOffset(inst.gas, inst.matSetIndex);
            }
            else {
                optixAssert_NotImplemented();
            }

            instancesD[i] = inst.rawInstance;
        }

        instanceBuffer.unmap();
    }

    void InstanceAccelerationStructure::Priv::fillBuildInput() {
        buildInput = OptixBuildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        OptixBuildInputInstanceArray &instArray = buildInput.instanceArray;
        instArray.instances = instanceBuffer.getDevicePointer();
        instArray.numInstances = static_cast<uint32_t>(children.size());
    }
    
    void InstanceAccelerationStructure::destroy() {
        m->scene->removeIAS(m);

        m->accelBuffer.finalize();
        m->accelTempBuffer.finalize();

        delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::addChild(const GeometryAccelerationStructure &gas, uint32_t matSetIdx, const float instantTransform[12]) const {
        auto _gas = extract(gas);
        if (_gas == nullptr)
            throw make_runtime_error("Invalid GAS %p.", _gas);
        if (matSetIdx >= _gas->getNumMaterialSets())
            throw make_runtime_error("Material set index %u is out of bound [0, %u).", matSetIdx, _gas->getNumMaterialSets());

        _Instance inst = _Instance(_gas, matSetIdx, instantTransform);

        m->children.push_back(inst);

        m->available = false;
        m->compactedAvailable = false;
    }

    void InstanceAccelerationStructure::rebuild(bool preferFastTrace, bool allowUpdate, bool enableCompaction, CUstream stream) const {
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
            OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                     &m->buildInput, 1,
                                                     &bufferSizes));

            m->accelBufferSize = bufferSizes.outputSizeInBytes;
            m->accelTempBuffer.initialize(BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

            m->accelBuffer.initialize(BufferType::Device, m->accelBufferSize, 1, 0);
        }

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
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

            OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
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
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
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



    void Pipeline::Priv::createProgram(const OptixProgramGroupDesc &desc, const OptixProgramGroupOptions &options, OptixProgramGroup* group) {
        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(context->getRawContext(),
                                                &desc, 1, // num program groups
                                                &options,
                                                log, &logSize,
                                                group));
        programGroups.insert(*group);
    }

    void Pipeline::Priv::destroyProgram(OptixProgramGroup group) {
        optixAssert(programGroups.count(group) > 0, "This program group has not been registered.");
        programGroups.erase(group);
        OPTIX_CHECK(optixProgramGroupDestroy(group));
    }
    
    void Pipeline::Priv::setupShaderBindingTable() {
        if (!sbtSetup) {
            if (scene == nullptr)
                throw make_runtime_error("Scene is not set.");

            if (rayGenProgram == nullptr)
                throw make_runtime_error("Ray generation program is not set.");

            for (int i = 0; i < numMissRayTypes; ++i)
                if (missPrograms[i] == nullptr)
                    throw make_runtime_error("Miss program is not set for some ray types.");

            sbt = OptixShaderBindingTable{};
            {
                rayGenRecord.initialize(BufferType::Device, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
                auto rayGenRecordOnHost = reinterpret_cast<uint8_t*>(rayGenRecord.map());
                rayGenProgram->packHeader(rayGenRecordOnHost);
                rayGenRecord.unmap();

                missRecords.initialize(BufferType::Device, numMissRayTypes, OPTIX_SBT_RECORD_HEADER_SIZE, 0);
                auto missRecordsOnHost = reinterpret_cast<uint8_t*>(missRecords.map());
                for (int i = 0; i < numMissRayTypes; ++i) {
                    missPrograms[i]->packHeader(missRecordsOnHost + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                }
                missRecords.unmap();

                SizeAlign hitGroupRecordStride = scene->calcHitGroupRecordStride(this);
                uint32_t numHitGroupRecords = scene->getNumSBTRecords();
                hitGroupRecords.initialize(BufferType::Device, numHitGroupRecords, hitGroupRecordStride.size, 0);

                auto hitGroupRecordsOnHost = reinterpret_cast<uint8_t*>(hitGroupRecords.map());
                scene->fillSBTRecords(this, hitGroupRecordsOnHost, hitGroupRecordStride.size);
                hitGroupRecords.unmap();



                sbt.raygenRecord = rayGenRecord.getDevicePointer();

                sbt.exceptionRecord = 0;

                sbt.missRecordBase = missRecords.getDevicePointer();
                sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
                sbt.missRecordCount = numMissRayTypes;

                sbt.hitgroupRecordBase = hitGroupRecords.getDevicePointer();
                sbt.hitgroupRecordStrideInBytes = hitGroupRecordStride.size;
                sbt.hitgroupRecordCount = numHitGroupRecords;

                sbt.callablesRecordBase = 0;
                sbt.callablesRecordCount = 0;
            }

            sbtSetup = true;
        }
    }

    void Pipeline::destroy() {
        if (m->pipelineLinked)
            OPTIX_CHECK(optixPipelineDestroy(m->rawPipeline));
        delete m;
        m = nullptr;
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

        OptixModule rawModule;

        char log[4096];
        size_t logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(m->context->getRawContext(),
                                                 &moduleCompileOptions,
                                                 &m->pipelineCompileOptions,
                                                 ptxString.c_str(), ptxString.size(),
                                                 log, &logSize,
                                                 &rawModule));

        return (new _Module(m, rawModule))->getPublicType();
    }



    ProgramGroup Pipeline::createRayGenProgram(Module module, const char* entryFunctionName) const {
        _Module* _module = extract(module);
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
        _Module* _module = extract(module);
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
        _Module* _module = extract(module);
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
        _Module* _module_CH = extract(module_CH);
        _Module* _module_AH = extract(module_AH);
        _Module* _module_IS = extract(module_IS);
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
        _Module* _module_DC = extract(module_DC);
        _Module* _module_CC = extract(module_CC);
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
                                                groups.data(), static_cast<uint32_t>(groups.size()),
                                                log, &logSize,
                                                &m->rawPipeline));

            m->pipelineLinked = true;
        }
    }



    void Pipeline::setScene(Scene scene) const {
        m->scene = extract(scene);
    }

    void Pipeline::setNumMissRayTypes(uint32_t numMissRayTypes) const {
        m->numMissRayTypes = numMissRayTypes;
        m->missPrograms.resize(m->numMissRayTypes);
    }
    
    void Pipeline::setRayGenerationProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        if (_program == nullptr)
            throw make_runtime_error("Invalid program %p.", _program);
        if (_program->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given program (group).");

        m->rayGenProgram = _program;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        if (_program == nullptr)
            throw make_runtime_error("Invalid program %p.", _program);
        if (_program->getPipeline() != m)
            throw make_runtime_error("Pipeline mismatch for the given program (group).");

        m->exceptionProgram = _program;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        if (rayType >= m->numMissRayTypes)
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



    void Module::destroy() {
        OPTIX_CHECK(optixModuleDestroy(m->rawModule));
        delete m;
        m = nullptr;
    }



    void ProgramGroup::destroy() {
        m->pipeline->destroyProgram(m->rawGroup);
        delete m;
        m = nullptr;
    }
}
