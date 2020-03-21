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



    static inline uint32_t tzcnt(uint32_t x) {
        return _tzcnt_u32(x);
    }

    static inline uint32_t lzcnt(uint32_t x) {
        return _lzcnt_u32(x);
    }

    static inline int32_t popcnt(uint32_t x) {
        return _mm_popcnt_u32(x);
    }

    static inline uint32_t nextPowOf2Exponent(uint32_t x) {
        if (x == 0)
            return 0;
        return 32 - lzcnt(x - 1);
    }

    static inline uint32_t nextPowOf2(uint32_t x) {
        return 1 << nextPowOf2Exponent(x);
    }

    static inline uint32_t nextMultiplesForPowOf2(uint32_t x, uint32_t exponent) {
        uint32_t mask = (1 << exponent) - 1;
        return (x + mask) & ~mask;
    }

    static inline uint32_t nextMultiplierForPowOf2(uint32_t x, uint32_t exponent) {
        return nextMultiplesForPowOf2(x, exponent) >> exponent;
    }

    static inline uint32_t nthSetBit(uint32_t value, uint32_t n) {
        uint32_t idx = 0;
        int32_t count;
        if (n >= popcnt(value))
            return 0xFFFFFFFF;

        for (uint32_t width = 16; width >= 1; width >>= 1) {
            if (value == 0)
                return 0xFFFFFFFF;

            uint32_t mask = (1 << width) - 1;
            count = popcnt(value & mask);
            if (n >= count) {
                value >>= width;
                n -= count;
                idx += width;
            }
        }

        return idx;
    }



    void SlotFinder::initialize(uint32_t numSlots) {
        m_numLayers = 1;
        m_numLowestFlagBins = nextMultiplierForPowOf2(numSlots, 5);

        // e.g. factor 4
        // 0 | 1101 | 0011 | 1001 | 1011 | 0010 | 1010 | 0000 | 1011 | 1110 | 0101 | 111* | **** | **** | **** | **** | **** | 43 flags
        // OR bins:
        // 1 | 1      1      1      1    | 1      1      0      1    | 1      1      1      *    | *      *      *      *    | 11
        // 2 | 1                           1                           1                           *                         | 3
        // AND bins
        // 1 | 0      0      0      0    | 0      0      0      0    | 0      0      1      *    | *      *      *      *    | 11
        // 2 | 0                           0                           0                           *                         | 3
        //
        // numSlots: 43
        // numLowestFlagBins: 11
        // numLayers: 3
        //
        // Memory Order
        // LowestFlagBins (layer 0) | OR, AND Bins (layer 1) | ... | OR, AND Bins (layer n-1)
        // Offset Pair to OR, AND (layer 0) | ... | Offset Pair to OR, AND (layer n-1)
        // NumUsedFlags (layer 0) | ... | NumUsedFlags (layer n-1)
        // Offset to NumUsedFlags (layer 0) | ... | Offset to NumUsedFlags (layer n-1)
        // NumFlags (layer 0) | ... | NumFlags (layer n-1)

        uint32_t numFlagBinsInLayer = m_numLowestFlagBins;
        m_numTotalCompiledFlagBins = 0;
        while (numFlagBinsInLayer > 1) {
            ++m_numLayers;
            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);
            m_numTotalCompiledFlagBins += 2 * numFlagBinsInLayer; // OR bins and AND bins
        }

        size_t memSize = sizeof(uint32_t) *
            ((m_numLowestFlagBins + m_numTotalCompiledFlagBins) +
             m_numLayers * 2 +
             (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2) +
             m_numLayers +
             m_numLayers);
        void* mem = malloc(memSize);

        uintptr_t memHead = (uintptr_t)mem;
        m_flagBins = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins);

        m_offsetsToOR_AND = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * m_numLayers * 2;

        m_numUsedFlagsUnderBinList = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * (m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2);

        m_offsetsToNumUsedFlags = (uint32_t*)memHead;
        memHead += sizeof(uint32_t) * m_numLayers;

        m_numFlagsInLayerList = (uint32_t*)memHead;

        uint32_t layer = 0;
        uint32_t offsetToOR_AND = 0;
        uint32_t offsetToNumUsedFlags = 0;
        {
            m_numFlagsInLayerList[layer] = numSlots;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numSlots, 5);

            m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
            m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND;
            m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

            offsetToOR_AND += numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }
        while (numFlagBinsInLayer > 1) {
            ++layer;
            m_numFlagsInLayerList[layer] = numFlagBinsInLayer;

            numFlagBinsInLayer = nextMultiplierForPowOf2(numFlagBinsInLayer, 5);

            m_offsetsToOR_AND[2 * layer + 0] = offsetToOR_AND;
            m_offsetsToOR_AND[2 * layer + 1] = offsetToOR_AND + numFlagBinsInLayer;
            m_offsetsToNumUsedFlags[layer] = offsetToNumUsedFlags;

            offsetToOR_AND += 2 * numFlagBinsInLayer;
            offsetToNumUsedFlags += numFlagBinsInLayer;
        }

        std::fill_n(m_flagBins, m_numLowestFlagBins + m_numTotalCompiledFlagBins, 0);
        std::fill_n(m_numUsedFlagsUnderBinList, m_numLowestFlagBins + m_numTotalCompiledFlagBins / 2, 0);
    }

    void SlotFinder::finalize() {
        if (m_flagBins)
            free(m_flagBins);
        m_flagBins = nullptr;
    }

    void SlotFinder::aggregate() {
        uint32_t offsetToOR_last = m_offsetsToOR_AND[2 * 0 + 0];
        uint32_t offsetToAND_last = m_offsetsToOR_AND[2 * 0 + 1];
        uint32_t offsetToNumUsedFlags_last = m_offsetsToNumUsedFlags[0];
        for (int layer = 1; layer < m_numLayers; ++layer) {
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t offsetToOR = m_offsetsToOR_AND[2 * layer + 0];
            uint32_t offsetToAND = m_offsetsToOR_AND[2 * layer + 1];
            uint32_t offsetToNumUsedFlags = m_offsetsToNumUsedFlags[layer];
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t &ORFlagBin = m_flagBins[offsetToOR + binIdx];
                uint32_t &ANDFlagBin = m_flagBins[offsetToAND + binIdx];
                uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[offsetToNumUsedFlags + binIdx];

                uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
                for (int bit = 0; bit < numFlagsInBin; ++bit) {
                    uint32_t lBinIdx = 32 * binIdx + bit;
                    uint32_t lORFlagBin = m_flagBins[offsetToOR_last + lBinIdx];
                    uint32_t lANDFlagBin = m_flagBins[offsetToAND_last + lBinIdx];
                    uint32_t lNumFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer - 1] - 32 * lBinIdx);
                    if (lORFlagBin != 0)
                        ORFlagBin |= 1 << bit;
                    if (popcnt(lANDFlagBin) == lNumFlagsInBin)
                        ANDFlagBin |= 1 << bit;
                    numUsedFlagsUnderBin += m_numUsedFlagsUnderBinList[offsetToNumUsedFlags_last + lBinIdx];
                }
            }

            offsetToOR_last = offsetToOR;
            offsetToAND_last = offsetToAND;
            offsetToNumUsedFlags_last = offsetToNumUsedFlags;
        }
    }

    void SlotFinder::resize(uint32_t numSlots) {
        if (numSlots == m_numFlagsInLayerList[0])
            return;

        SlotFinder newFinder;
        newFinder.initialize(numSlots);

        uint32_t numLowestFlagBins = std::min(m_numLowestFlagBins, newFinder.m_numLowestFlagBins);
        for (int binIdx = 0; binIdx < numLowestFlagBins; ++binIdx) {
            uint32_t numFlagsInBin = std::min(32u, numSlots - 32 * binIdx);
            uint32_t mask = numFlagsInBin >= 32 ? 0xFFFFFFFF : ((1 << numFlagsInBin) - 1);
            uint32_t value = m_flagBins[0 + binIdx] & mask;
            newFinder.m_flagBins[0 + binIdx] = value;
            newFinder.m_numUsedFlagsUnderBinList[0 + binIdx] = popcnt(value);
        }

        newFinder.aggregate();

        *this = std::move(newFinder);
    }

    void SlotFinder::setInUse(uint32_t slotIdx) {
        if (getUsage(slotIdx))
            return;

        bool setANDFlag = false;
        uint32_t flagIdxInLayer = slotIdx;
        for (int layer = 0; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            // JP: 最下層ではOR/ANDは同じ実体だがsetANDFlagが初期値falseであるので設定は1回きり。
            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            ORFlagBin |= (1 << flagIdxInBin);
            if (setANDFlag)
                ANDFlagBin |= (1 << flagIdxInBin);
            ++numUsedFlagsUnderBin;

            // JP: このビンに利用可能なスロットが無くなった場合は次のANDレイヤーもフラグを立てる。
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            setANDFlag = popcnt(ANDFlagBin) == numFlagsInBin;

            flagIdxInLayer = binIdx;
        }
    }

    void SlotFinder::setNotInUse(uint32_t slotIdx) {
        if (!getUsage(slotIdx))
            return;

        bool resetORFlag = false;
        uint32_t flagIdxInLayer = slotIdx;
        for (int layer = 0; layer < m_numLayers; ++layer) {
            uint32_t binIdx = flagIdxInLayer / 32;
            uint32_t flagIdxInBin = flagIdxInLayer % 32;

            // JP: 最下層ではOR/ANDは同じ実体だがresetORFlagが初期値falseであるので設定は1回きり。
            uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            if (resetORFlag)
                ORFlagBin &= ~(1 << flagIdxInBin);
            ANDFlagBin &= ~(1 << flagIdxInBin);
            --numUsedFlagsUnderBin;

            // JP: このビンに使用中スロットが無くなった場合は次のORレイヤーのフラグを下げる。
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            resetORFlag = ORFlagBin == 0;

            flagIdxInLayer = binIdx;
        }
    }

    uint32_t SlotFinder::getFirstAvailableSlot() const {
        uint32_t binIdx = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t ANDFlagBinOffset = m_offsetsToOR_AND[2 * layer + 1];
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t ANDFlagBin = m_flagBins[ANDFlagBinOffset + binIdx];

            if (popcnt(ANDFlagBin) != numFlagsInBin) {
                // JP: このビンに利用可能なスロットを発見。
                binIdx = tzcnt(~ANDFlagBin) + 32 * binIdx;
            }
            else {
                // JP: 利用可能なスロットが見つからなかった。
                return 0xFFFFFFFF;
            }
        }

        optixAssert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return binIdx;
    }

    uint32_t SlotFinder::getFirstUsedSlot() const {
        uint32_t binIdx = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t ORFlagBinOffset = m_offsetsToOR_AND[2 * layer + 0];
            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            uint32_t ORFlagBin = m_flagBins[ORFlagBinOffset + binIdx];

            if (ORFlagBin != 0) {
                // JP: このビンに使用中のスロットを発見。
                binIdx = tzcnt(ORFlagBin) + 32 * binIdx;
            }
            else {
                // JP: 使用中スロットが見つからなかった。
                return 0xFFFFFFFF;
            }
        }

        optixAssert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return binIdx;
    }

    uint32_t SlotFinder::find_nthUsedSlot(uint32_t n) const {
        if (n >= getNumUsed())
            return 0xFFFFFFFF;

        uint32_t startBinIdx = 0;
        uint32_t accNumUsed = 0;
        for (int layer = m_numLayers - 1; layer >= 0; --layer) {
            uint32_t numUsedFlagsOffset = m_offsetsToNumUsedFlags[layer];
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            for (int binIdx = startBinIdx; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

                // JP: 現在のビンの配下にインデックスnの使用中スロットがある。
                if (accNumUsed + numUsedFlagsUnderBin > n) {
                    startBinIdx = 32 * binIdx;
                    if (layer == 0) {
                        uint32_t flagBin = m_flagBins[binIdx];
                        startBinIdx += nthSetBit(flagBin, n - accNumUsed);
                    }
                    break;
                }

                accNumUsed += numUsedFlagsUnderBin;
            }
        }

        optixAssert(startBinIdx < m_numFlagsInLayerList[0], "Invalid value.");
        return startBinIdx;
    }

    void SlotFinder::debugPrint() const {
        uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
        devPrintf("----");
        for (int binIdx = 0; binIdx < numLowestFlagBins; ++binIdx) {
            devPrintf("------------------------------------");
        }
        devPrintf("\n");
        for (int layer = m_numLayers - 1; layer > 0; --layer) {
            devPrintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
            devPrintf(" OR:");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        devPrintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[layer];
                    if (!valid)
                        continue;

                    bool b = (ORFlagBin >> i) & 0x1;
                    devPrintf("%c", b ? '|' : '_');
                }
            }
            devPrintf("\n");
            devPrintf("AND:");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        devPrintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[layer];
                    if (!valid)
                        continue;

                    bool b = (ANDFlagBin >> i) & 0x1;
                    devPrintf("%c", b ? '|' : '_');
                }
            }
            devPrintf("\n");
            devPrintf("    ");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
                devPrintf("                            %8u", numUsedFlagsUnderBin);
            }
            devPrintf("\n");
        }
        {
            devPrintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
            uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
            devPrintf("   :");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t ORFlagBin = m_flagBins[binIdx];
                for (int i = 0; i < 32; ++i) {
                    if (i % 8 == 0)
                        devPrintf(" ");

                    bool valid = binIdx * 32 + i < m_numFlagsInLayerList[0];
                    if (!valid)
                        continue;

                    bool b = (ORFlagBin >> i) & 0x1;
                    devPrintf("%c", b ? '|' : '_');
                }
            }
            devPrintf("\n");
            devPrintf("    ");
            for (int binIdx = 0; binIdx < numFlagBinsInLayer; ++binIdx) {
                uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
                devPrintf("                            %8u", numUsedFlagsUnderBin);
            }
            devPrintf("\n");
        }
    }



    Context Context::create(CUcontext cudaContext) {
        return (new _Context(cudaContext))->getPublicType();
    }

    void Context::destroy() {
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



    void Material::Priv::setRecordData(const _Pipeline* pipeline, uint32_t rayType, HitGroupSBTRecord* record) const {
        Key key{ pipeline, rayType };
        const _ProgramGroup* hitGroup = programs.at(key);
        hitGroup->packHeader(record->header);
        record->data.materialData = userData;
    }
    
    void Material::destroy() {
        delete m;
        m = nullptr;
    }

    void Material::setHitGroup(uint32_t rayType, const ProgramGroup &hitGroup) {
        auto _pipeline = extract(hitGroup)->getPipeline();
        THROW_RUNTIME_ERROR(_pipeline, "Invalid pipeline.");

        _Material::Key key{ _pipeline, rayType };
        m->programs[key] = extract(hitGroup);
    }
    
    void Material::setUserData(uint32_t data) const {
        m->userData = data;
    }



    
    void Scene::Priv::registerPipeline(const _Pipeline* pipeline) {
        THROW_RUNTIME_ERROR(hitGroupSBTs.count(pipeline) == 0, "This pipeline %p has been already registered.", pipeline);
        auto hitGroupSBT = new HitGroupSBT();
        hitGroupSBT->records.initialize(getCUDAContext(), s_BufferType, 1);
        hitGroupSBTs[pipeline] = hitGroupSBT;
    }

    void Scene::Priv::generateSBTLayout() {
        if (sbtLayoutIsUpToDate)
            return;

        uint32_t sbtOffset = 0;
        sbtOffsets.clear();
        for (_GeometryAccelerationStructure* gas : geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            for (int matSetIdx = 0; matSetIdx < numMatSets; ++matSetIdx) {
                uint32_t gasNumSBTRecords = gas->calcNumSBTRecords(matSetIdx);
                _Scene::SBTOffsetKey key = { gas, matSetIdx };
                sbtOffsets[key] = sbtOffset;
                sbtOffset += gasNumSBTRecords;
            }
        }
        numSBTRecords = sbtOffset;
        sbtLayoutIsUpToDate = true;
    }

    void Scene::Priv::markSBTLayoutDirty() {
        sbtLayoutIsUpToDate = false;

        for (_InstanceAccelerationStructure* _ias : instASs) {
            InstanceAccelerationStructure ias = _ias->getPublicType();
            ias.markDirty();
        }
    }

    const HitGroupSBT* Scene::Priv::setupHitGroupSBT(const _Pipeline* pipeline) {
        HitGroupSBT* hitGroupSBT = hitGroupSBTs.at(pipeline);
        hitGroupSBT->records.resize(numSBTRecords);

        HitGroupSBTRecord* records = hitGroupSBT->records.map();

        for (_GeometryAccelerationStructure* gas : geomASs) {
            uint32_t numMatSets = gas->getNumMaterialSets();
            for (int j = 0; j < numMatSets; ++j) {
                uint32_t numRecords = gas->fillSBTRecords(pipeline, j, records);
                records += numRecords;
            }
        }

        hitGroupSBT->records.unmap();

        return hitGroupSBT;
    }

    bool Scene::Priv::isReady() {
        for (_GeometryAccelerationStructure* _gas : geomASs) {
            if (!_gas->isReady()) {
                return false;
            }
        }

        for (_InstanceAccelerationStructure* _ias : instASs) {
            if (!_ias->isReady()) {
                return false;
            }
        }

        return true;
    }

    void Scene::destroy() {
        for (auto it = m->hitGroupSBTs.crbegin(); it != m->hitGroupSBTs.crend(); ++it)
            delete it->second;
        delete m;
        m = nullptr;
    }
    
    GeometryInstance Scene::createGeometryInstance() const {
        return (new _GeometryInstance(m))->getPublicType();
    }

    GeometryAccelerationStructure Scene::createGeometryAccelerationStructure() const {
        return (new _GeometryAccelerationStructure(m))->getPublicType();
    }

    InstanceAccelerationStructure Scene::createInstanceAccelerationStructure() const {
        return (new _InstanceAccelerationStructure(m))->getPublicType();
    }



    void GeometryInstance::Priv::fillBuildInput(OptixBuildInput* input) const {
        *input = OptixBuildInput{};

        input->type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        OptixBuildInputTriangleArray &triArray = input->triangleArray;

        triArray.vertexBuffers = vertexBufferArray;
        triArray.numVertices = vertexBuffer->numElements();
        triArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triArray.vertexStrideInBytes = vertexBuffer->stride();

        triArray.indexBuffer = triangleBuffer->getCUdeviceptr();
        triArray.numIndexTriplets = triangleBuffer->numElements();
        triArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triArray.indexStrideInBytes = triangleBuffer->stride();
        triArray.primitiveIndexOffset = 0;

        triArray.numSbtRecords = buildInputFlags.size();
        if (triArray.numSbtRecords > 1) {
            optixAssert_NotImplemented();
            triArray.sbtIndexOffsetBuffer = materialIndexOffsetBuffer->getCUdeviceptr();
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

    uint32_t GeometryInstance::Priv::getNumSBTRecords() const {
        return static_cast<uint32_t>(buildInputFlags.size());
    }

    uint32_t GeometryInstance::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, uint32_t numRayTypes,
                                                    HitGroupSBTRecord* records) const {
        THROW_RUNTIME_ERROR(matSetIdx < materialSets.size(),
                            "Out of material set bound: [0, %u)", static_cast<uint32_t>(materialSets.size()));

        const std::vector<const _Material*> &materialSet = materialSets[matSetIdx];
        HitGroupSBTRecord* recordPtr = records;
        uint32_t numMaterials = buildInputFlags.size();
        for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
            const _Material* mat = materialSet[matIdx];
            THROW_RUNTIME_ERROR(mat, "No material set for %u-%u.", matSetIdx, matIdx);
            for (int rIdx = 0; rIdx < numRayTypes; ++rIdx) {
                mat->setRecordData(pipeline, rIdx, recordPtr);
                recordPtr->data.geomInstData = userData;
                ++recordPtr;
            }
        }

        return numMaterials * numRayTypes;
    }

    void GeometryInstance::destroy() {
        delete m;
        m = nullptr;
    }

    void GeometryInstance::setVertexBuffer(Buffer* vertexBuffer) const {
        m->vertexBuffer = vertexBuffer;
        m->vertexBufferArray[0] = vertexBuffer->getCUdeviceptr();
    }

    void GeometryInstance::setTriangleBuffer(Buffer* triangleBuffer) const {
        m->triangleBuffer = triangleBuffer;
    }

    void GeometryInstance::setNumMaterials(uint32_t numMaterials, TypedBuffer<uint32_t>* matIdxOffsetBuffer) const {
        THROW_RUNTIME_ERROR(numMaterials > 0, "Invalid number of materials %u.", numMaterials);
        THROW_RUNTIME_ERROR((numMaterials == 1) != (matIdxOffsetBuffer != nullptr),
                            "Material index offset buffer must be provided when multiple materials are used, otherwise, must not be provided.");
        m->buildInputFlags.resize(numMaterials, OPTIX_GEOMETRY_FLAG_NONE);
        m->materialIndexOffsetBuffer = matIdxOffsetBuffer;
    }

    void GeometryInstance::setUserData(uint32_t data) const {
        m->userData = data;
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

        uint32_t prevNumMatSets = m->materialSets.size();
        if (matSetIdx >= prevNumMatSets) {
            m->materialSets.resize(matSetIdx + 1);
            for (int i = prevNumMatSets; i < m->materialSets.size(); ++i)
                m->materialSets[i].resize(numMaterials, nullptr);
        }
        m->materialSets[matSetIdx][matIdx] = extract(mat);
    }



    uint32_t GeometryAccelerationStructure::Priv::calcNumSBTRecords(uint32_t matSetIdx) const {
        uint32_t numSBTRecords = 0;
        for (int i = 0; i < children.size(); ++i)
            numSBTRecords += children[i]->getNumSBTRecords();
        numSBTRecords *= numRayTypesPerMaterialSet[matSetIdx];

        return numSBTRecords;
    }

    uint32_t GeometryAccelerationStructure::Priv::fillSBTRecords(const _Pipeline* pipeline, uint32_t matSetIdx, HitGroupSBTRecord* records) const {
        THROW_RUNTIME_ERROR(matSetIdx < numRayTypesPerMaterialSet.size(),
                            "Material set index %u is out of bound [0, %u).",
                            matSetIdx, static_cast<uint32_t>(numRayTypesPerMaterialSet.size()));

        uint32_t numRayTypes = numRayTypesPerMaterialSet[matSetIdx];
        uint32_t sumRecords = 0;
        for (int i = 0; i < children.size(); ++i) {
            uint32_t numRecords = children[i]->fillSBTRecords(pipeline, matSetIdx, numRayTypes, records);
            records += numRecords;
            sumRecords += numRecords;
        }

        return sumRecords;
    }
    
    void GeometryAccelerationStructure::destroy() {
        delete m;
        m = nullptr;
    }

    void GeometryAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace = preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;

        if (changed)
            markDirty();
    }

    void GeometryAccelerationStructure::setNumMaterialSets(uint32_t numMatSets) const {
        m->numRayTypesPerMaterialSet.resize(numMatSets, 0);

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const {
        THROW_RUNTIME_ERROR(matSetIdx < m->numRayTypesPerMaterialSet.size(),
                            "Material set index %u is out of bounds [0, %u).",
                            matSetIdx, static_cast<uint32_t>(m->numRayTypesPerMaterialSet.size()));
        m->numRayTypesPerMaterialSet[matSetIdx] = numRayTypes;

        m->scene->markSBTLayoutDirty();
    }

    void GeometryAccelerationStructure::addChild(const GeometryInstance &geomInst) const {
        auto _geomInst = extract(geomInst);
        THROW_RUNTIME_ERROR(_geomInst, "Invalid geometry instance %p.", _geomInst);

        m->children.push_back(_geomInst);

        markDirty();
    }

    void GeometryAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        // Fill build inputs.
        m->buildInputs.resize(m->children.size(), OptixBuildInput{});
        for (int i = 0; i < m->children.size(); ++i)
            m->children[i]->fillBuildInput(&m->buildInputs[i]);

        m->buildOptions = {};
        m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                      (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                      (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        //buildOptions.motionOptions

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 m->buildInputs.data(), m->buildInputs.size(),
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;
    }

    OptixTraversableHandle GeometryAccelerationStructure::rebuild(CUstream stream, const Buffer &accelBuffer, const Buffer &scratchBuffer) const {
        THROW_RUNTIME_ERROR(accelBuffer.sizeInBytes() >= m->memoryRequirement.outputSizeInBytes,
                            "Size of the given buffer is not enough.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m->accelBuffer = &accelBuffer;
        m->available = true;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void GeometryAccelerationStructure::prepareForCompact(CUstream rebuildOrUpdateStream, size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDADRV_CHECK(cuStreamSynchronize(rebuildOrUpdateStream));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        *compactedAccelBufferSize = m->compactedSize;
    }

    OptixTraversableHandle GeometryAccelerationStructure::compact(CUstream stream, const Buffer &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));

        m->compactedAccelBuffer = &compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void GeometryAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuStreamSynchronize(compactionStream));

        m->handle = 0;
        m->available = false;
    }

    OptixTraversableHandle GeometryAccelerationStructure::update(CUstream stream, const Buffer &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        const Buffer* accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, m->buildInputs.data(), m->buildInputs.size(),
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer->getCUdeviceptr(), accelBuffer->sizeInBytes(),
                                    &handle,
                                    nullptr, 0));

        return handle;
    }

    bool GeometryAccelerationStructure::isReady() const {
        return m->isReady();
    }

    void GeometryAccelerationStructure::markDirty() const {
        m->available = false;
        m->compactedAvailable = false;

        m->scene->markSBTLayoutDirty();
    }



    void InstanceAccelerationStructure::destroy() {
        delete m;
        m = nullptr;
    }

    void InstanceAccelerationStructure::setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction) {
        bool changed = false;
        changed |= m->preferFastTrace != preferFastTrace;
        m->preferFastTrace = preferFastTrace;
        changed |= m->allowUpdate != allowUpdate;
        m->allowUpdate = allowUpdate;
        changed |= m->allowCompaction != allowCompaction;
        m->allowCompaction = allowCompaction;

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

    void InstanceAccelerationStructure::prepareForBuild(OptixAccelBufferSizes* memoryRequirement) const {
        // Setup instances.
        {
            m->scene->generateSBTLayout();

            m->instanceBuffer.finalize();
            m->instanceBuffer.initialize(m->getCUDAContext(), s_BufferType, m->children.size());
            auto instances = m->instanceBuffer.map();

            for (int i = 0; i < m->children.size(); ++i) {
                _Instance &inst = m->children[i];

                if (inst.type == InstanceType::GAS) {
                    THROW_RUNTIME_ERROR(inst.gas->isReady(), "GAS %p is not ready.", inst.gas);

                    inst.rawInstance.traversableHandle = inst.gas->getHandle();
                    inst.rawInstance.sbtOffset = m->scene->getSBTOffset(inst.gas, inst.matSetIndex);
                }
                else {
                    optixAssert_NotImplemented();
                }

                instances[i] = inst.rawInstance;
            }

            m->instanceBuffer.unmap();
        }

        // Fill the build input.
        {
            m->buildInput = OptixBuildInput{};
            m->buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            OptixBuildInputInstanceArray &instArray = m->buildInput.instanceArray;
            instArray.instances = m->instanceBuffer.getCUdeviceptr();
            instArray.numInstances = static_cast<uint32_t>(m->children.size());
        }

        m->buildOptions = {};
        m->buildOptions.buildFlags = ((m->preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : OPTIX_BUILD_FLAG_PREFER_FAST_BUILD) |
                                      (m->allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                      (m->allowCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        //buildOptions.motionOptions

        OPTIX_CHECK(optixAccelComputeMemoryUsage(m->getRawContext(), &m->buildOptions,
                                                 &m->buildInput, 1,
                                                 &m->memoryRequirement));

        *memoryRequirement = m->memoryRequirement;
    }

    OptixTraversableHandle InstanceAccelerationStructure::rebuild(CUstream stream, const Buffer &accelBuffer, const Buffer &scratchBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream, &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer.getCUdeviceptr(), accelBuffer.sizeInBytes(),
                                    &m->handle,
                                    compactionEnabled ? &m->propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m->accelBuffer = &accelBuffer;
        m->available = true;
        m->compactedHandle = 0;
        m->compactedAvailable = false;

        return m->handle;
    }

    void InstanceAccelerationStructure::prepareForCompact(CUstream rebuildOrUpdateStream, size_t* compactedAccelBufferSize) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");

        if (m->compactedAvailable)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDADRV_CHECK(cuStreamSynchronize(rebuildOrUpdateStream));
        CUDADRV_CHECK(cuMemcpyDtoH(&m->compactedSize, m->propertyCompactedSize.result, sizeof(m->compactedSize)));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m->compactedSize, (void*)m->propertyCompactedSize.result, sizeof(m->compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        *compactedAccelBufferSize = m->compactedSize;
    }

    OptixTraversableHandle InstanceAccelerationStructure::compact(CUstream stream, const Buffer &compactedAccelBuffer) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;
        THROW_RUNTIME_ERROR(compactionEnabled, "This AS does not allow compaction.");
        THROW_RUNTIME_ERROR(m->available, "Uncompacted AS has not been built yet.");
        THROW_RUNTIME_ERROR(compactedAccelBuffer.sizeInBytes() >= m->compactedSize,
                            "Size of the given buffer is not enough.");

        OPTIX_CHECK(optixAccelCompact(m->getRawContext(), stream,
                                      m->handle, compactedAccelBuffer.getCUdeviceptr(), compactedAccelBuffer.sizeInBytes(),
                                      &m->compactedHandle));

        m->compactedAccelBuffer = &compactedAccelBuffer;
        m->compactedAvailable = true;

        return m->compactedHandle;
    }

    void InstanceAccelerationStructure::removeUncompacted(CUstream compactionStream) const {
        bool compactionEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m->compactedAvailable || !compactionEnabled)
            return;

        CUDADRV_CHECK(cuStreamSynchronize(compactionStream));

        m->handle = 0;
        m->available = false;
    }

    OptixTraversableHandle InstanceAccelerationStructure::update(CUstream stream, const Buffer &scratchBuffer) const {
        bool updateEnabled = (m->buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;
        THROW_RUNTIME_ERROR(updateEnabled, "This AS does not allow update.");
        THROW_RUNTIME_ERROR(m->available || m->compactedAvailable, "AS has not been built yet.");
        THROW_RUNTIME_ERROR(scratchBuffer.sizeInBytes() >= m->memoryRequirement.tempUpdateSizeInBytes,
                            "Size of the given scratch buffer is not enough.");

        const Buffer* accelBuffer = m->compactedAvailable ? m->compactedAccelBuffer : m->accelBuffer;
        OptixTraversableHandle &handle = m->compactedAvailable ? m->compactedHandle : m->handle;

        m->buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m->getRawContext(), stream,
                                    &m->buildOptions, &m->buildInput, 1,
                                    scratchBuffer.getCUdeviceptr(), scratchBuffer.sizeInBytes(),
                                    accelBuffer->getCUdeviceptr(), accelBuffer->sizeInBytes(),
                                    &handle,
                                    nullptr, 0));

        return handle;
    }

    bool InstanceAccelerationStructure::isReady() const {
        return m->isReady();
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
            missRecords.resize(numMissRayTypes, missRecords.stride());

            sbtAllocDone = true;
        }

        if (!sbtIsUpToDate) {
            THROW_RUNTIME_ERROR(rayGenProgram, "Ray generation program is not set.");

            for (int i = 0; i < numMissRayTypes; ++i)
                THROW_RUNTIME_ERROR(missPrograms[i], "Miss program is not set for ray type %d.", i);

            sbt = {};
            {
                auto rayGenRecordOnHost = rayGenRecord.map<uint8_t>();
                rayGenProgram->packHeader(rayGenRecordOnHost);
                rayGenRecord.unmap();

                if (exceptionProgram) {
                    auto exceptionRecordOnHost = exceptionRecord.map<uint8_t>();
                    exceptionProgram->packHeader(exceptionRecordOnHost);
                    exceptionRecord.unmap();
                }

                auto missRecordsOnHost = missRecords.map<uint8_t>();
                for (int i = 0; i < numMissRayTypes; ++i)
                    missPrograms[i]->packHeader(missRecordsOnHost + OPTIX_SBT_RECORD_HEADER_SIZE * i);
                missRecords.unmap();

                const HitGroupSBT* hitGroupSBT = scene->setupHitGroupSBT(this);



                sbt.raygenRecord = rayGenRecord.getCUdeviceptr();

                sbt.exceptionRecord = exceptionProgram ? exceptionRecord.getCUdeviceptr() : 0;

                sbt.missRecordBase = missRecords.getCUdeviceptr();
                sbt.missRecordStrideInBytes = OPTIX_SBT_RECORD_HEADER_SIZE;
                sbt.missRecordCount = numMissRayTypes;

                sbt.hitgroupRecordBase = hitGroupSBT->records.getCUdeviceptr();
                sbt.hitgroupRecordStrideInBytes = hitGroupSBT->records.stride();
                sbt.hitgroupRecordCount = hitGroupSBT->records.numElements();

                sbt.callablesRecordBase = 0;
                sbt.callablesRecordCount = 0;
            }

            sbtIsUpToDate = true;
        }
    }

    void Pipeline::destroy() {
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
        m->pipelineCompileOptions = {};
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
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setExceptionProgram(ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->exceptionProgram = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::setMissProgram(uint32_t rayType, ProgramGroup program) const {
        _ProgramGroup* _program = extract(program);
        THROW_RUNTIME_ERROR(rayType < m->numMissRayTypes, "Invalid ray type.");
        THROW_RUNTIME_ERROR(_program, "Invalid program %p.", _program);
        THROW_RUNTIME_ERROR(_program->getPipeline() == m, "Pipeline mismatch for the given program (group).");

        m->missPrograms[rayType] = _program;
        m->sbtIsUpToDate = false;
    }

    void Pipeline::launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const {
        THROW_RUNTIME_ERROR(m->scene, "Scene is not set.");
        THROW_RUNTIME_ERROR(m->scene->isReady(), "Scene is not ready.");

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
