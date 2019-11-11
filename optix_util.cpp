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
        THROW_RUNTIME_ERROR(_pipeline, "Invalid pipeline.");

        _Material::Key key{ _pipeline, rayType };
        m->infos[key].update(extract(hitGroup), sbtRecordData, size, alignment);
    }



    SizeAlign Scene::Priv::calcHitGroupRecordStride(const _Pipeline* pipeline) const {
        SizeAlign maxSizeAlign;
        for (_GeometryAccelerationStructure* gas : geomASs) {
            SizeAlign sizeAlign = gas->calcHitGroupRecordStride(pipeline);
            maxSizeAlign = max(maxSizeAlign, sizeAlign);
        }
        return maxSizeAlign;
    }

    void Scene::Priv::registerPipeline(const _Pipeline* pipeline) {
        THROW_RUNTIME_ERROR(hitGroupSBTs.count(pipeline) == 0, "This pipeline %p has been already registered.", pipeline);
        auto hitGroupSBT = new HitGroupSBT();
        if (sbtLayoutIsUpToDate) {
            SizeAlign stride = calcHitGroupRecordStride(pipeline);
            hitGroupSBT->records.initialize(BufferType::Device, numSBTRecords, stride.size, 0);
        }

        hitGroupSBTs[pipeline] = hitGroupSBT;
    }

    void Scene::Priv::setupHitGroupSBT(const _Pipeline* pipeline) {
        HitGroupSBT* hitGroupSBT = hitGroupSBTs.at(pipeline);
        auto records = reinterpret_cast<uint8_t*>(hitGroupSBT->records.map());
        uint32_t stride = hitGroupSBT->records.stride();

        for (_GeometryAccelerationStructure* gas : geomASs) {
            for (int j = 0; j < gas->getNumMaterialSets(); ++j) {
                uint32_t numRecords = gas->fillSBTRecords(pipeline, j, records, stride);
                records += numRecords * stride;
            }
        }

        hitGroupSBT->records.unmap();
    }

    const HitGroupSBT* Scene::Priv::getHitGroupSBT(const _Pipeline* pipeline) {
        return hitGroupSBTs.at(pipeline);
    }

    void Scene::destroy() {
        for (auto it = m->hitGroupSBTs.crbegin(); it != m->hitGroupSBTs.crend(); ++it)
            delete it->second;
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
        if (m->sbtLayoutIsUpToDate)
            return;

        uint32_t sbtOffset = 0;
        m->sbtOffsets.clear();
        for (_GeometryAccelerationStructure* gas : m->geomASs) {
            for (int matSetIdx = 0; matSetIdx < gas->getNumMaterialSets(); ++matSetIdx) {
                uint32_t gasNumSBTRecords = gas->calcNumSBTRecords(matSetIdx);
                _Scene::SBTOffsetKey key = { gas, matSetIdx };
                m->sbtOffsets[key] = sbtOffset;
                sbtOffset += gasNumSBTRecords;
            }
        }
        m->numSBTRecords = sbtOffset;
        m->sbtLayoutIsUpToDate = true;

        for (auto it = m->hitGroupSBTs.begin(); it != m->hitGroupSBTs.end(); ++it) {
            HitGroupSBT* hitGroupSBT = it->second;
            hitGroupSBT->records.finalize();
            SizeAlign stride = m->calcHitGroupRecordStride(it->first);
            hitGroupSBT->records.initialize(BufferType::Device, m->numSBTRecords, stride.size, 0);
        }
    }

    void Scene::setupASsAndSBTLayout(CUstream stream) const {
        for (_GeometryAccelerationStructure* _gas : m->geomASs) {
            if (!_gas->isReady()) {
                GeometryAccelerationStructure gas = _gas->getPublicType();
                gas.rebuild(stream);
                gas.compaction(stream, stream);
                gas.removeUncompacted(stream);
            }
        }

        generateSBTLayout();

        for (_InstanceAccelerationStructure* _ias : m->instASs) {
            if (!_ias->isReady()) {
                InstanceAccelerationStructure ias = _ias->getPublicType();
                ias.rebuild(stream);
                ias.compaction(stream, stream);
                ias.removeUncompacted(stream);
            }
        }
    }

    void Scene::markSBTLayoutDirty() const {
        m->sbtLayoutIsUpToDate = false;

        for (_InstanceAccelerationStructure* _ias : m->instASs) {
            InstanceAccelerationStructure ias = _ias->getPublicType();
            ias.markDirty();
        }
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
            THROW_RUNTIME_ERROR(materials.count(key) > 0,
                                "No material registered for Material set: %u, Material: %u", matSetIdx, matIdx);

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
        uint32_t numRecords = 0;

        MaterialKey key{ matSetIdx, -1 };
        for (int matIdx = 0; matIdx < buildInputFlags.size(); ++matIdx) {
            key.matIndex = matIdx;
            THROW_RUNTIME_ERROR(materials.count(key) > 0,
                                "No material registered for Material set: %u, Material: %u", matSetIdx, matIdx);

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
                ++numRecords;
            }
        }

        return numRecords;
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

    void GeometryInstance::setNumMaterials(uint32_t numMaterials, Buffer* matIdxOffsetBuffer) const {
        THROW_RUNTIME_ERROR(numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        THROW_RUNTIME_ERROR((numMaterials == 1) != (matIdxOffsetBuffer != nullptr),
                            "Material index offset buffer must be provided when multiple materials are used, otherwise, must not be provided.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        m->materialIndexOffsetBuffer = matIdxOffsetBuffer;
    }

    void GeometryInstance::setData(const void* sbtRecordData, size_t size, size_t alignment) const {
        std::memcpy(m->recordData, sbtRecordData, size);
        m->sizeAlign = SizeAlign(size, alignment);
    }

    void GeometryInstance::setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const {
        size_t numMaterials = m->buildInputFlags.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", (uint32_t)numMaterials);

        m->buildInputFlags[matIdx] = flags;
    }

    void GeometryInstance::setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const {
        size_t numMaterials = m->buildInputFlags.size();
        THROW_RUNTIME_ERROR(matIdx < numMaterials,
                            "Out of material bounds [0, %u).", (uint32_t)numMaterials);

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
        THROW_RUNTIME_ERROR(matSetIdx < numRayTypesValues.size(),
                            "Material set index %u is out of bound [0, %u).",
                            matSetIdx, static_cast<uint32_t>(numRayTypesValues.size()));

        uint32_t sumRecords = 0;
        for (int i = 0; i < children.size(); ++i) {
            uint32_t numRecords = children[i]->fillSBTRecords(pipeline, matSetIdx, numRayTypesValues[matSetIdx],
                                                              records, stride);
            records += numRecords * stride;
            sumRecords += numRecords;
        }

        return sumRecords;
    }
    
    void GeometryAccelerationStructure::destroy() {
        m->scene->removeGAS(m);

        m->accelBuffer.finalize();
        m->accelTempBuffer.finalize();

        delete m;
        m = nullptr;
    }

    void GeometryAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace == preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate == allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction == allowCompaction;

        if (changed)
            markDirty();
    }

    void GeometryAccelerationStructure::setNumMaterialSets(uint32_t numMatSets) const {
        m->numRayTypesValues.resize(numMatSets, 0);

        m->scene->getPublicType().markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const {
        THROW_RUNTIME_ERROR(matSetIdx < m->numRayTypesValues.size(),
                            "Material set index %u is out of bounds [0, %u).",
                            matSetIdx, static_cast<uint32_t>(m->numRayTypesValues.size()));
        m->numRayTypesValues[matSetIdx] = numRayTypes;

        m->scene->getPublicType().markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);

        m->children.push_back(_geomInst);

        markDirty();
    }

    void GeometryAccelerationStructure::rebuild(CUstream stream) const {
        m->fillBuildInputs();

        {
            m->accelBuffer.finalize();
            m->accelTempBuffer.finalize();

            std::memset(&m->buildOptions, 0, sizeof(m->buildOptions));
            m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                          (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                          (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
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

    void GeometryAccelerationStructure::markDirty() const {
        m->available = false;
        m->compactedAvailable = false;

        m->scene->getPublicType().markSBTLayoutDirty();
    }



    void InstanceAccelerationStructure::Priv::setupInstances() {
        THROW_RUNTIME_ERROR(scene->sbtLayoutGenerationDone(),
                            "SBT layout generation should be done before.");

        instanceBuffer.finalize();

        instanceBuffer.initialize(BufferType::Device, children.size(), sizeof(OptixInstance), 0);
        auto instancesD = (OptixInstance*)instanceBuffer.map();

        for (int i = 0; i < children.size(); ++i) {
            _Instance &inst = children[i];

            if (inst.type == InstanceType::GAS) {
                THROW_RUNTIME_ERROR(inst.gas->isReady(), "GAS %p is not ready.", inst.gas);

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

    void InstanceAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace == preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate == allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction == allowCompaction;

        if (changed)
            markDirty();
    }

    void InstanceAccelerationStructure::addChild(const GeometryAccelerationStructure &gas, uint32_t matSetIdx, const float instantTransform[12]) const {
        auto _gas = extract(gas);
        THROW_RUNTIME_ERROR(_gas, "Invalid GAS %p.", _gas);
        THROW_RUNTIME_ERROR(matSetIdx < _gas->getNumMaterialSets(),
                            "Material set index %u is out of bound [0, %u).", matSetIdx, _gas->getNumMaterialSets());

        _Instance inst = _Instance(_gas, matSetIdx, instantTransform);

        m->children.push_back(inst);

        markDirty();
    }

    void InstanceAccelerationStructure::rebuild(CUstream stream) const {
        m->setupInstances();
        m->fillBuildInput();

        {
            m->accelBuffer.finalize();
            m->accelTempBuffer.finalize();

            std::memset(&m->buildOptions, 0, sizeof(m->buildOptions));
            m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                          (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                          (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
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

    void InstanceAccelerationStructure::markDirty() const {
        m->available = false;
        m->compactedAvailable = false;
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
        if (!sbtAllocDone) {
            rayGenRecord.finalize();
            rayGenRecord.initialize(BufferType::Device, 1, OPTIX_SBT_RECORD_HEADER_SIZE, 0);

            missRecords.finalize();
            missRecords.initialize(BufferType::Device, numMissRayTypes, OPTIX_SBT_RECORD_HEADER_SIZE, 0);

            sbtAllocDone = true;
        }

        if (!sbtIsUpToDate) {
            THROW_RUNTIME_ERROR(rayGenProgram, "Ray generation program is not set.");

            for (int i = 0; i < numMissRayTypes; ++i)
                THROW_RUNTIME_ERROR(missPrograms[i], "Miss program is not set for ray type %d.", i);

            sbt = OptixShaderBindingTable{};
            {
                auto rayGenRecordOnHost = reinterpret_cast<uint8_t*>(rayGenRecord.map());
                rayGenProgram->packHeader(rayGenRecordOnHost);
                rayGenRecord.unmap();

                auto missRecordsOnHost = reinterpret_cast<uint8_t*>(missRecords.map());
                for (int i = 0; i < numMissRayTypes; ++i)
                    missPrograms[i]->packHeader(missRecordsOnHost + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                missRecords.unmap();

                scene->setupHitGroupSBT(this);
                const HitGroupSBT* hitGroupSBT = scene->getHitGroupSBT(this);



                sbt.raygenRecord = rayGenRecord.getDevicePointer();

                sbt.exceptionRecord = 0;

                sbt.missRecordBase = missRecords.getDevicePointer();
                sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
                sbt.missRecordCount = numMissRayTypes;

                sbt.hitgroupRecordBase = hitGroupSBT->records.getDevicePointer();
                sbt.hitgroupRecordStrideInBytes = hitGroupSBT->records.stride();
                sbt.hitgroupRecordCount = hitGroupSBT->records.numElements();

                sbt.callablesRecordBase = 0;
                sbt.callablesRecordCount = 0;
            }

            sbtIsUpToDate = true;
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
        THROW_RUNTIME_ERROR(_module->getPipeline() == m, "Pipeline mismatch for the given module.");

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
        THROW_RUNTIME_ERROR(_module->getPipeline() == m, "Pipeline mismatch for the given module.");

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
        THROW_RUNTIME_ERROR(_module == nullptr || _module->getPipeline() == m,
                            "Pipeline mismatch for the given module.");

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
        THROW_RUNTIME_ERROR(entryFunctionNameCH == nullptr || _module_CH == nullptr ||
                            _module_CH->getPipeline() == m,
                            "Pipeline mismatch for the given CH module.");
        THROW_RUNTIME_ERROR(entryFunctionNameAH == nullptr || _module_AH == nullptr ||
                            _module_AH->getPipeline() == m,
                            "Pipeline mismatch for the given AH module.");
        THROW_RUNTIME_ERROR(entryFunctionNameIS == nullptr || _module_IS == nullptr ||
                            _module_IS->getPipeline() == m,
                            "Pipeline mismatch for the given IS module.");

        THROW_RUNTIME_ERROR(entryFunctionNameCH || entryFunctionNameAH || entryFunctionNameIS,
                            "Either of CH/AH/IS entry function name should be provided.");

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
        THROW_RUNTIME_ERROR(entryFunctionNameDC == nullptr || _module_DC == nullptr ||
                            _module_DC->getPipeline() == m,
                            "Pipeline mismatch for the given DC module.");
        THROW_RUNTIME_ERROR(entryFunctionNameCC == nullptr || _module_CC == nullptr ||
                            _module_CC->getPipeline() == m,
                            "Pipeline mismatch for the given CC module.");

        THROW_RUNTIME_ERROR(entryFunctionNameDC || entryFunctionNameCC,
                            "Either of CC/DC entry function name should be provided.");

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
        THROW_RUNTIME_ERROR(!m->pipelineLinked, "This pipeline has been already linked.");

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
        m->scene->registerPipeline(m);
    }

    void Pipeline::setNumMissRayTypes(uint32_t numMissRayTypes) const {
        m->numMissRayTypes = numMissRayTypes;
        m->missPrograms.resize(m->numMissRayTypes);
        m->sbtAllocDone = false;
    }
    
    void Pipeline::setRayGenerationProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->rayGenProgram = _program;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->exceptionProgram = _program;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(rayType < m->numMissRayTypes, "Invalid ray type.");
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->missPrograms[rayType] = _program;
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) {
        THROW_RUNTIME_ERROR(m->scene, "Scene is not set.");

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
