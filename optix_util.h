#pragma once

/*

現状ではあらゆるAPIに破壊的変更が入る可能性が非常に高い。

TODO:
- ASのダブルバッファリングの運用方法を考える。
- update()の運用方法を考える。
- SBTのダブルバッファリングの運用方法を考える。
- SlotFinderはこのまま必要ないなら削除する。

*/

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define OPTIX_Platform_Windows
#    if defined(_MSC_VER)
#        define OPTIX_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define OPTIX_Platform_macOS
#endif

#if defined(OPTIX_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#endif

#if !defined(__CUDA_ARCH__)
#include "cuda_helper.h"
#endif

#include <optix.h>
#if !defined(__CUDA_ARCH__)
#include <optix_stubs.h>
#endif
#include <cstdint>

#if defined(__CUDA_ARCH__)
#   define RT_FUNCTION __forceinline__ __device__
#   define RT_PROGRAM extern "C" __global__
#else
#   define RT_FUNCTION
#endif



namespace optix {

#ifdef _DEBUG
#   define OPTIX_ENABLE_ASSERT
#endif

#if defined(OPTIX_Platform_Windows_MSVC)
    void devPrintf(const char* fmt, ...);
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define optixPrintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define optixPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#if defined(OPTIX_ENABLE_ASSERT)
#   if defined(__CUDA_ARCH__)
#   define optixAssert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); } 0
#   else
#   define optixAssert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#   endif
#else
#   define optixAssert(expr, fmt, ...)
#endif

#define optixAssert_ShouldNotBeCalled() optixAssert(false, "Should not be called!")
#define optixAssert_NotImplemented() optixAssert(false, "Not implemented yet!")



    struct HitGroupSBTRecordData {
        uint32_t materialData;
        uint32_t geomInstData;
    };

#if defined(__CUDA_ARCH__)
    RT_FUNCTION HitGroupSBTRecordData getHitGroupSBTRecordData() {
        return *reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    }
#endif



#if !defined(__CUDA_ARCH__)
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



    using namespace CUDAHelper;

    /*

    Context --+-- Pipeline --+-- Module
              |              |
              |              +-- ProgramGroup
              |
              |
              +-- Material
              |
              |
              +-- Scene --+-- IAS
                          |
                          +-- GAS
                          |
                          +-- GeomInst

    JP: 
    EN: 

    */

    class Context;
    class Material;
    class Scene;
    class GeometryInstance;
    class GeometryAccelerationStructure;
    class InstanceAccelerationStructure;
    class Pipeline;
    class Module;
    class ProgramGroup;

#define OPTIX_PIMPL() \
public: \
    class Priv; \
private: \
    Priv* m = nullptr



    class Context {
        OPTIX_PIMPL();

    public:
        static Context create(CUcontext cudaContext);
        void destroy();

        Material createMaterial() const;
        Scene createScene() const;

        Pipeline createPipeline() const;
    };



    class Material {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setHitGroup(uint32_t rayType, const ProgramGroup &hitGroup);
        void setUserData(uint32_t data) const;
    };



    class Scene {
        OPTIX_PIMPL();

    public:
        void destroy();

        GeometryInstance createGeometryInstance() const;
        GeometryAccelerationStructure createGeometryAccelerationStructure() const;
        InstanceAccelerationStructure createInstanceAccelerationStructure() const;

        const OptixTraversableHandle* getTraversableHandles() const;
    };



    class GeometryInstance {
        OPTIX_PIMPL();

    public:
        void destroy();

        // After the program calls either of these methods or updates the contents of vertex/index buffers,
        // the program should call markDirty() of GASs to which the GeometryInstance is belonging.
        void setVertexBuffer(Buffer* vertexBuffer) const;
        void setTriangleBuffer(Buffer* triangleBuffer) const;
        void setNumMaterials(uint32_t numMaterials, TypedBuffer<uint32_t>* matIdxOffsetBuffer) const;

        void setUserData(uint32_t data) const;

        void setGeometryFlags(uint32_t matIdx, OptixGeometryFlags flags) const;
        void setMaterial(uint32_t matSetIdx, uint32_t matIdx, Material mat) const;
    };



    class GeometryAccelerationStructure {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction);
        void setNumMaterialSets(uint32_t numMatSets) const;
        void setNumRayTypes(uint32_t matSetIdx, uint32_t numRayTypes) const;

        void addChild(const GeometryInstance &geomInst) const;

        void rebuild(CUstream stream) const;
        void compact(CUstream rebuildOrUpdateStream, CUstream stream) const;
        void removeUncompacted(CUstream compactionStream) const;
        void update(CUstream stream) const;

        bool isReady() const;
        uint32_t getID() const;

        void markDirty() const;
    };



    class InstanceAccelerationStructure {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setConfiguration(bool preferFastTrace, bool allowUpdate, bool allowCompaction);

        void addChild(const GeometryAccelerationStructure &gas, uint32_t matSetIdx = 0, const float instantTransform[12] = nullptr) const;

        void rebuild(CUstream stream) const;
        void compact(CUstream rebuildOrUpdateStream, CUstream stream) const;
        void removeUncompacted(CUstream compactionStream) const;
        void update(CUstream stream) const;

        bool isReady() const;
        uint32_t getID() const;

        void markDirty() const;
    };



    class Pipeline {
        OPTIX_PIMPL();

    public:
        void destroy();

        void setMaxTraceDepth(uint32_t maxTraceDepth) const;
        void setPipelineOptions(uint32_t numPayloadValues, uint32_t numAttributeValues, const char* launchParamsVariableName, size_t sizeOfLaunchParams,
                                bool useMotionBlur, uint32_t traversableGraphFlags, uint32_t exceptionFlags) const;

        Module createModuleFromPTXString(const std::string &ptxString, int32_t maxRegisterCount, OptixCompileOptimizationLevel optLevel, OptixCompileDebugLevel debugLevel) const;

        ProgramGroup createRayGenProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createExceptionProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createMissProgram(Module module, const char* entryFunctionName) const;
        ProgramGroup createHitProgramGroup(Module module_CH, const char* entryFunctionNameCH,
                                           Module module_AH, const char* entryFunctionNameAH,
                                           Module module_IS, const char* entryFunctionNameIS) const;
        ProgramGroup createCallableGroup(Module module_DC, const char* entryFunctionNameDC,
                                         Module module_CC, const char* entryFunctionNameCC) const;

        void link(OptixCompileDebugLevel debugLevel, bool overrideUseMotionBlur) const;

        void setScene(Scene scene) const;
        void setNumMissRayTypes(uint32_t numMissRayTypes) const;

        void setRayGenerationProgram(ProgramGroup program) const;
        void setExceptionProgram(ProgramGroup program) const;
        void setMissProgram(uint32_t rayType, ProgramGroup program) const;

        void launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ) const;
    };



    class Module {
        OPTIX_PIMPL();

    public:
        void destroy();
    };



    class ProgramGroup {
        OPTIX_PIMPL();

    public:
        void destroy();
    };

#endif // #if !defined(__CUDA_ARCH__)
}
