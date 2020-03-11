#pragma once

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

#include "cuda_helper.h"

#include <optix.h>
#include <optix_stubs.h>



namespace optix {
#ifdef _DEBUG
#   define OPTIX_ENABLE_ASSERT
#endif

#ifdef OPTIX_Platform_Windows_MSVC
    void devPrintf(const char* fmt, ...);
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define optixPrintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define optixPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#ifdef OPTIX_ENABLE_ASSERT
#   define optixAssert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define optixAssert(expr, fmt, ...)
#endif

#define optixAssert_ShouldNotBeCalled() optixAssert(false, "Should not be called!")
#define optixAssert_NotImplemented() optixAssert(false, "Not implemented yet!")

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
        void setData(const void* data, size_t size, size_t alignment) const;
        template <typename RecordDataType>
        void setData(const RecordDataType &data) const {
            setData(&data, sizeof(RecordDataType), alignof(RecordDataType));
        }
    };



    class Scene {
        OPTIX_PIMPL();

    public:
        void destroy();

        GeometryInstance createGeometryInstance() const;
        GeometryAccelerationStructure createGeometryAccelerationStructure() const;
        InstanceAccelerationStructure createInstanceAccelerationStructure() const;

        void markSBTLayoutDirty() const;
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

        void setData(const void* data, size_t size, size_t alignment) const;
        template <typename RecordDataType>
        void setData(const RecordDataType &data) const {
            setData(&data, sizeof(RecordDataType), alignof(RecordDataType));
        }

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

        void launch(CUstream stream, CUdeviceptr plpOnDevice, uint32_t dimX, uint32_t dimY, uint32_t dimZ);
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
}
