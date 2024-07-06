#include "common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"

void devPrintf(const char* fmt, ...) {
#ifdef HP_Platform_Windows_MSVC
    va_list args;
    va_start(args, fmt);
    const int32_t reqStrSize = _vscprintf(fmt, args) + 1;
    va_end(args);

    static std::vector<char> str;
    if (reqStrSize > str.size())
        str.resize(reqStrSize);

    va_start(args, fmt);
    vsnprintf_s(str.data(), str.size(), _TRUNCATE, fmt, args);
    va_end(args);

#   if defined(UNICODE)
    const int32_t reqWstrSize = MultiByteToWideChar(CP_ACP, 0, str.data(), -1, nullptr, 0);
    static std::vector<wchar_t> wstr;
    if (reqWstrSize > wstr.size())
        wstr.resize(reqWstrSize);

    MultiByteToWideChar(
        CP_ACP, 0, str.data(), static_cast<int32_t>(str.size()),
        wstr.data(), static_cast<int32_t>(wstr.size()));
    OutputDebugString(wstr.data());
#   else
    OutputDebugString(str.data());
#   endif
#else
    va_list args;
    va_start(args, fmt);
    vprintf_s(fmt, args);
    va_end(args);
#endif
}



std::filesystem::path getExecutableDirectory() {
    static std::filesystem::path ret;

    static bool done = false;
    if (!done) {
#if defined(HP_Platform_Windows_MSVC)
        TCHAR filepath[1024];
        auto length = GetModuleFileName(NULL, filepath, 1024);
        Assert(length > 0, "Failed to query the executable path.");

        ret = filepath;
#else
        static_assert(false, "Not implemented");
#endif
        ret = ret.remove_filename();

        done = true;
    }

    return ret;
}



std::string readTxtFile(const std::filesystem::path &filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
}

std::vector<char> readBinaryFile(const std::filesystem::path &filepath) {
    std::vector<char> ret;

    std::ifstream ifs;
    ifs.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
    if (ifs.fail())
        return std::move(ret);

    std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ret.resize(fileSize);
    ifs.read(ret.data(), fileSize);

    return std::move(ret);
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
    for (int layer = 1; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        uint32_t offsetToOR = m_offsetsToOR_AND[2 * layer + 0];
        uint32_t offsetToAND = m_offsetsToOR_AND[2 * layer + 1];
        uint32_t offsetToNumUsedFlags = m_offsetsToNumUsedFlags[layer];
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t &ORFlagBin = m_flagBins[offsetToOR + binIdx];
            uint32_t &ANDFlagBin = m_flagBins[offsetToAND + binIdx];
            uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[offsetToNumUsedFlags + binIdx];

            uint32_t numFlagsInBin = std::min(32u, m_numFlagsInLayerList[layer] - 32 * binIdx);
            for (int bit = 0; bit < static_cast<int32_t>(numFlagsInBin); ++bit) {
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
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
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
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: �ŉ��w�ł�OR/AND�͓������̂���setANDFlag�������lfalse�ł���̂Őݒ��1�񂫂�B
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        ORFlagBin |= (1 << flagIdxInBin);
        if (setANDFlag)
            ANDFlagBin |= (1 << flagIdxInBin);
        ++numUsedFlagsUnderBin;

        // JP: ���̃r���ɗ��p�\�ȃX���b�g�������Ȃ����ꍇ�͎���AND���C���[���t���O�𗧂Ă�B
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
    for (int layer = 0; layer < static_cast<int32_t>(m_numLayers); ++layer) {
        uint32_t binIdx = flagIdxInLayer / 32;
        uint32_t flagIdxInBin = flagIdxInLayer % 32;

        // JP: �ŉ��w�ł�OR/AND�͓������̂���resetORFlag�������lfalse�ł���̂Őݒ��1�񂫂�B
        uint32_t &ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
        uint32_t &ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
        uint32_t &numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
        if (resetORFlag)
            ORFlagBin &= ~(1 << flagIdxInBin);
        ANDFlagBin &= ~(1 << flagIdxInBin);
        --numUsedFlagsUnderBin;

        // JP: ���̃r���Ɏg�p���X���b�g�������Ȃ����ꍇ�͎���OR���C���[�̃t���O��������B
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
            // JP: ���̃r���ɗ��p�\�ȃX���b�g�𔭌��B
            binIdx = tzcnt(~ANDFlagBin) + 32 * binIdx;
        }
        else {
            // JP: ���p�\�ȃX���b�g��������Ȃ������B
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
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
            // JP: ���̃r���Ɏg�p���̃X���b�g�𔭌��B
            binIdx = tzcnt(ORFlagBin) + 32 * binIdx;
        }
        else {
            // JP: �g�p���X���b�g��������Ȃ������B
            return 0xFFFFFFFF;
        }
    }

    Assert(binIdx < m_numFlagsInLayerList[0], "Invalid value.");
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
        for (int binIdx = startBinIdx; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[numUsedFlagsOffset + binIdx];

            // JP: ���݂̃r���̔z���ɃC���f�b�N�Xn�̎g�p���X���b�g������B
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

    Assert(startBinIdx < m_numFlagsInLayerList[0], "Invalid value.");
    return startBinIdx;
}

void SlotFinder::debugPrint() const {
    uint32_t numLowestFlagBins = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
    hpprintf("----");
    for (int binIdx = 0; binIdx < static_cast<int32_t>(numLowestFlagBins); ++binIdx) {
        hpprintf("------------------------------------");
    }
    hpprintf("\n");
    for (int layer = m_numLayers - 1; layer > 0; --layer) {
        hpprintf("layer %u (%u):\n", layer, m_numFlagsInLayerList[layer]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[layer], 5);
        hpprintf(" OR:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 0] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("AND:");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ANDFlagBin = m_flagBins[m_offsetsToOR_AND[2 * layer + 1] + binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[layer]);
                if (!valid)
                    continue;

                bool b = (ANDFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[m_offsetsToNumUsedFlags[layer] + binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
    {
        hpprintf("layer 0 (%u):\n", m_numFlagsInLayerList[0]);
        uint32_t numFlagBinsInLayer = nextMultiplierForPowOf2(m_numFlagsInLayerList[0], 5);
        hpprintf("   :");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t ORFlagBin = m_flagBins[binIdx];
            for (int i = 0; i < 32; ++i) {
                if (i % 8 == 0)
                    hpprintf(" ");

                bool valid = binIdx * 32 + i < static_cast<int32_t>(m_numFlagsInLayerList[0]);
                if (!valid)
                    continue;

                bool b = (ORFlagBin >> i) & 0x1;
                hpprintf("%c", b ? '|' : '_');
            }
        }
        hpprintf("\n");
        hpprintf("    ");
        for (int binIdx = 0; binIdx < static_cast<int32_t>(numFlagBinsInLayer); ++binIdx) {
            uint32_t numUsedFlagsUnderBin = m_numUsedFlagsUnderBinList[binIdx];
            hpprintf("                            %8u", numUsedFlagsUnderBin);
        }
        hpprintf("\n");
    }
}



void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const uint32_t* data) {
    if (filepath.extension() == ".png")
        stbi_write_png(filepath.string().c_str(), width, height, 4, data,
                       width * sizeof(uint32_t));
    else if (filepath.extension() == ".bmp")
        stbi_write_bmp(filepath.string().c_str(), width, height, 4, data);
    else
        Assert_ShouldNotBeCalled();
}

void saveImage(const std::filesystem::path &filepath, uint32_t width, uint32_t height, const float4* data,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto image = new uint32_t[width * height];
    for (int y = 0; y < static_cast<int32_t>(height); ++y) {
        for (int x = 0; x < static_cast<int32_t>(width); ++x) {
            float4 src = data[y * width + x];
            if (applyToneMap) {
                src.x = simpleToneMap_s(src.x);
                src.y = simpleToneMap_s(src.y);
                src.z = simpleToneMap_s(src.z);
            }
            if (apply_sRGB_gammaCorrection) {
                src.x = sRGB_gamma_s(src.x);
                src.y = sRGB_gamma_s(src.y);
                src.z = sRGB_gamma_s(src.z);
            }
            uint32_t &dst = image[y * width + x];
            dst = ((std::min<uint32_t>(static_cast<uint32_t>(src.x * 255), 255) << 0) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.y * 255), 255) << 8) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.z * 255), 255) << 16) |
                   (std::min<uint32_t>(static_cast<uint32_t>(src.w * 255), 255) << 24));
        }
    }

    saveImage(filepath, width, height, image);

    delete[] image;
}

void saveImage(const std::filesystem::path &filepath,
               uint32_t width, cudau::TypedBuffer<float4> &buffer,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    Assert(buffer.numElements() % width == 0, "Buffer's length is not divisible by the width.");
    uint32_t height = static_cast<uint32_t>(buffer.numElements()) / width;
    auto data = buffer.map();
    saveImage(filepath, width, height, data, applyToneMap, apply_sRGB_gammaCorrection);
    buffer.unmap();
}

void saveImage(const std::filesystem::path &filepath,
               cudau::Array &array,
               bool applyToneMap, bool apply_sRGB_gammaCorrection) {
    auto data = array.map<float4>();
    saveImage(
        filepath,
        static_cast<uint32_t>(array.getWidth()), static_cast<uint32_t>(array.getHeight()),
        data, applyToneMap, apply_sRGB_gammaCorrection);
    array.unmap();
}
