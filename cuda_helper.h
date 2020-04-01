#pragma once

// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#   define CUDAHPlatform_Windows
#   if defined(__MINGW32__) // Defined for both 32 bit/64 bit MinGW
#       define CUDAHPlatform_Windows_MinGW
#   elif defined(_MSC_VER)
#       define CUDAHPlatform_Windows_MSVC
#   endif
#elif defined(__linux__)
#   define CUDAHPlatform_Linux
#elif defined(__APPLE__)
#   define CUDAHPlatform_macOS
#elif defined(__OpenBSD__)
#   define CUDAHPlatform_OpenBSD
#endif

#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <vector>
#include <sstream>

#include <GL/gl3w.h>

#include <cuda.h>
#include <cudaGL.h>
#include <vector_types.h>

#ifdef _DEBUG
#   define CUDAH_ENABLE_ASSERT
#endif

#ifdef CUDAH_ENABLE_ASSERT
#   define CUDAHAssert(expr, fmt, ...) \
    if (!(expr)) { \
        CUDAHelper::devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); \
        CUDAHelper::devPrintf(fmt"\n", ##__VA_ARGS__); \
        abort(); \
    } 0
#else
#   define CUDAHAssert(expr, fmt, ...)
#endif

#define CUDAHAssert_ShouldNotBeCalled() CUDAHAssert(false, "Should not be called!")
#define CUDAHAssert_NotImplemented() CUDAHAssert(false, "Not implemented yet!")

#define CUDADRV_CHECK(call) \
    do { \
        CUresult error = call; \
        if (error != CUDA_SUCCESS) { \
            std::stringstream ss; \
            const char* errMsg = "failed to get an error message."; \
            cuGetErrorString(error, &errMsg); \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << errMsg \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::stringstream ss; \
            ss << "CUDA call (" << #call << " ) failed with error: '" \
               << cudaGetErrorString(error) \
               << "' (" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



namespace CUDAHelper {
    void devPrintf(const char* fmt, ...);



    using ConstVoidPtr = const void*;
    
    inline void addArgPointer(ConstVoidPtr* pointer) {}

    template <typename HeadType, typename... TailTypes>
    void addArgPointer(ConstVoidPtr* pointer, HeadType &&head, TailTypes&&... tails) {
        *pointer = &head;
        addArgPointer(pointer + 1, std::forward<TailTypes>(tails)...);
    }

    template <typename... ArgTypes>
    void callKernel(CUstream stream, CUfunction kernel, const dim3 &gridDim, const dim3 &blockDim, uint32_t sharedMemSize,
                    ArgTypes&&... args) {
        ConstVoidPtr argPointers[30];
        addArgPointer(argPointers, std::forward<ArgTypes>(args)...);

        CUDADRV_CHECK(cuLaunchKernel(kernel,
                                     gridDim.x, gridDim.y, gridDim.z,
                                     blockDim.x, blockDim.y, blockDim.z,
                                     sharedMemSize, stream, const_cast<void**>(argPointers), nullptr));
    }



    class Timer {
        CUcontext m_context;
        CUevent m_startEvent;
        CUevent m_endEvent;

    public:
        void initialize(CUcontext context) {
            m_context = context;
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventCreate(&m_startEvent, CU_EVENT_BLOCKING_SYNC));
            CUDADRV_CHECK(cuEventCreate(&m_endEvent, CU_EVENT_BLOCKING_SYNC));
        }
        void finalize() {
            CUDADRV_CHECK(cuCtxSetCurrent(m_context));
            CUDADRV_CHECK(cuEventDestroy(m_endEvent));
            CUDADRV_CHECK(cuEventDestroy(m_startEvent));
            m_context = nullptr;
        }

        void start(CUstream stream) const {
            CUDADRV_CHECK(cuEventRecord(m_startEvent, stream));
        }
        void stop(CUstream stream) const {
            CUDADRV_CHECK(cuEventRecord(m_endEvent, stream));
        }

        float report() const {
            float ret = 0.0f;
            CUDADRV_CHECK(cuEventSynchronize(m_endEvent));
            CUDADRV_CHECK(cuEventElapsedTime(&ret, m_startEvent, m_endEvent));
            return ret;
        }
    };



    enum class BufferType {
        Device = 0,     // not preferred, typically slower than ZERO_COPY
        GL_Interop = 1, // single device only, preferred for single device
        ZeroCopy = 2,   // general case, preferred for multi-gpu if not fully nvlink connected
        P2P = 3         // fully connected only, preferred for fully nvlink connected
    };

    class Buffer {
        CUcontext m_cudaContext;
        BufferType m_type;

        uint32_t m_numElements;
        uint32_t m_stride;

        void* m_hostPointer;
        CUdeviceptr m_devicePointer;
        void* m_mappedPointer;

        uint32_t m_GLBufferID;
        CUgraphicsResource m_cudaGfxResource;

        struct {
            unsigned int m_initialized : 1;
            unsigned int m_mapped : 1;
        };

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

    public:
        Buffer();
        Buffer(CUcontext context, BufferType type,
               uint32_t numElements, uint32_t stride, uint32_t glBufferID = 0) : Buffer() {
            initialize(context, type, numElements, stride, glBufferID);
        }
        ~Buffer();

        Buffer(Buffer &&b);
        Buffer &operator=(Buffer &&b);

        void initialize(CUcontext context, BufferType type,
                        uint32_t numElements, uint32_t stride, uint32_t glBufferID);
        void finalize();

        void resize(uint32_t numElements, uint32_t stride);

        CUdeviceptr getCUdeviceptr() const {
            return m_devicePointer;
        }
        CUdeviceptr getCUdeviceptrAt(uint32_t idx) const {
            return m_devicePointer + m_stride * idx;
        }
        size_t sizeInBytes() const {
            return m_numElements * m_stride;
        }
        size_t stride() const {
            return m_stride;
        }
        size_t numElements() const {
            return m_numElements;
        }
        bool isInitialized() const {
            return m_initialized;
        }

        CUdeviceptr beginCUDAAccess(CUstream stream);
        void endCUDAAccess(CUstream stream);

        void* map();
        template <typename T>
        T* map() {
            return reinterpret_cast<T*>(map());
        }
        void unmap();

        Buffer copy() const;
    };



    template <typename T>
    class TypedBuffer : public Buffer {
    public:
        TypedBuffer() {}
        TypedBuffer(CUcontext context, BufferType type, int32_t numElements) {
            Buffer::initialize(context, type, numElements, sizeof(T), 0);
        }
        TypedBuffer(CUcontext context, BufferType type, int32_t numElements, const T &value) {
            Buffer::initialize(context, type, numElements, sizeof(T), 0);
            T* values = (T*)map();
            for (int i = 0; i < numElements; ++i)
                values[i] = value;
            unmap();
        }

        void initialize(CUcontext context, BufferType type, int32_t numElements) {
            Buffer::initialize(context, type, numElements, sizeof(T), 0);
        }
        void initialize(CUcontext context, BufferType type, int32_t numElements, const T &value) {
            Buffer::initialize(context, type, numElements, sizeof(T), 0);
            T* values = (T*)Buffer::map();
            for (int i = 0; i < numElements; ++i)
                values[i] = value;
            Buffer::unmap();
        }
        void initialize(CUcontext context, BufferType type, const T* v, uint32_t numElements) {
            initialize(context, type, numElements);
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v, numElements * sizeof(T)));
        }
        void initialize(CUcontext context, BufferType type, const std::vector<T> &v) {
            initialize(context, type, v.size());
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getCUdeviceptr(), v.data(), v.size() * sizeof(T)));
        }
        void finalize() {
            Buffer::finalize();
        }

        void resize(int32_t numElements) {
            Buffer::resize(numElements, sizeof(T));
        }

        T* getDevicePointer() const {
            return reinterpret_cast<T*>(getCUdeviceptr());
        }
        T* getDevicePointerAt(uint32_t idx) const {
            return reinterpret_cast<T*>(getCUdeviceptrAt(idx));
        }

        T* map() {
            return reinterpret_cast<T*>(Buffer::map());
        }

        T operator[](uint32_t idx) {
            const T* values = map();
            T ret = values[idx];
            unmap();
            return ret;
        }

        TypedBuffer<T> copy() const {
            TypedBuffer<T> ret;
            // safe ?
            *reinterpret_cast<Buffer*>(&ret) = Buffer::copy();
            return ret;
        }
    };

    template <typename T>
    class TypedHostBuffer {
        std::vector<T> m_values;

    public:
        TypedHostBuffer() {}
        TypedHostBuffer(TypedBuffer<T> &b) {
            m_values.resize(b.numElements());
            auto srcValues = b.map();
            std::copy_n(srcValues, b.numElements(), m_values.data());
            b.unmap();
        }

        T* getPointer() {
            return m_values.data();
        }
        size_t numElements() const {
            return m_values.size();
        }

        const T &operator[](uint32_t idx) const {
            return m_values[idx];
        }
        T &operator[](uint32_t idx) {
            return m_values[idx];
        }
    };



    enum class ArrayElementType {
        UInt8x4,
        UInt16x4,
        UInt32x4,
        Int8x4,
        Int16x4,
        Int32x4,
        Halfx4,
        Floatx4,
    };

    //enum class ArrayType {
    //    E_1D = 0,
    //    E_2D,
    //    E_3D,
    //    E_1DLayered,
    //    E_2DLayered,
    //    Cubemap,
    //    CubemapLayered,
    //};
    
    class Array {
        CUcontext m_cudaContext;

        //ArrayType m_arrayType;
        uint32_t m_width;
        uint32_t m_height;
        uint32_t m_depth;
        uint32_t m_stride;
        ArrayElementType m_elemType;

        CUarray m_array;
        void* m_mappedPointer;

        struct {
            unsigned int m_initialized : 1;
            unsigned int m_mapped : 1;
        };

        Array(const Array &) = delete;
        Array &operator=(const Array &) = delete;

        void initialize(CUcontext context, ArrayElementType elemType,
                        uint32_t width, uint32_t height, uint32_t depth,
                        bool cubemap, bool layered);

    public:
        Array();
        Array(CUcontext context, ArrayElementType elemType, uint32_t width, uint32_t height) {
            initialize(context, elemType, width, height);
        }
        ~Array();

        Array(Array &&b);
        Array &operator=(Array &&b);

        void initialize(CUcontext context, ArrayElementType elemType, uint32_t length) {
            initialize(context, elemType, length, 0, 0, false, false);
        }
        void initialize(CUcontext context, ArrayElementType elemType, uint32_t width, uint32_t height) {
            initialize(context, elemType, width, height, 0, false, false);
        }
        void initialize(CUcontext context, ArrayElementType elemType, uint32_t width, uint32_t height, uint32_t depth) {
            initialize(context, elemType, width, height, 0, false, false);
        }
        void finalize();

        CUarray getCUarray() const {
            return m_array;
        }

        uint32_t getWidth() const {
            return m_width;
        }
        uint32_t getHeight() const {
            return m_height;
        }
        uint32_t getDepth() const {
            return m_depth;
        }

        void* map();
        template <typename T>
        T* map() {
            return reinterpret_cast<T*>(map());
        }
        void unmap();
    };



    //enum class MipmappedArrayType {
    //    E_1DMipmapped = 0,
    //    E_2DMipmapped,
    //    E_3DMipmapped,
    //};



    enum class TextureWrapMode {
        Repeat = CU_TR_ADDRESS_MODE_WRAP,
        Clamp = CU_TR_ADDRESS_MODE_CLAMP,
        Mirror = CU_TR_ADDRESS_MODE_MIRROR,
        Border = CU_TR_ADDRESS_MODE_BORDER,
    };

    enum class TextureFilterMode {
        Point = CU_TR_FILTER_MODE_POINT,
        Linear = CU_TR_FILTER_MODE_LINEAR,
    };

    enum class TextureIndexingMode {
        NormalizedCoordinates = 0,
        ArrayIndex,
    };

    enum class TextureReadMode {
        ElementType = 0,
        NormalizedFloat,
        NormalizedFloat_sRGB
    };
    
    class TextureSampler {
        CUDA_RESOURCE_DESC m_resDesc;
        CUDA_TEXTURE_DESC m_texDesc;
        CUDA_RESOURCE_VIEW_DESC m_resViewDesc;
        CUtexObject m_texObject;
        struct {
            unsigned int m_texObjectCreated : 1;
            unsigned int m_texObjectIsUpToDate : 1;
        };

        TextureSampler(const TextureSampler &) = delete;
        TextureSampler &operator=(const TextureSampler &) = delete;

    public:
        TextureSampler() : m_texObjectCreated(false), m_texObjectIsUpToDate(false) {
            m_resDesc = {};
            m_texDesc = {};
            m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
            m_resViewDesc = {};
            // TODO: support block compression formats.
        }
        ~TextureSampler() {
            if (m_texObjectCreated)
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
        }

        void destroyTextureObject() {
            if (m_texObjectIsUpToDate) {
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
                m_texObjectIsUpToDate = false;
                m_texObjectCreated = false;
            }
        }

        TextureSampler(TextureSampler &&t) {
            m_resDesc = t.m_resDesc;
            m_texDesc = t.m_texDesc;
            m_texObject = t.m_texObject;
            m_texObjectCreated = t.m_texObjectCreated;
            m_texObjectIsUpToDate = t.m_texObjectIsUpToDate;

            t.m_texObjectCreated = false;
        }
        TextureSampler &operator=(TextureSampler &&t) {
            m_resDesc = t.m_resDesc;
            m_texDesc = t.m_texDesc;
            m_texObject = t.m_texObject;
            m_texObjectCreated = t.m_texObjectCreated;
            m_texObjectIsUpToDate = t.m_texObjectIsUpToDate;

            t.m_texObjectCreated = false;
        }

        void setArray(const Array &array) {
            m_resDesc.resType = CU_RESOURCE_TYPE_ARRAY;
            m_resDesc.res.array.hArray = array.getCUarray();
            m_texObjectIsUpToDate = false;
        }

        void setFilterMode(TextureFilterMode xy, TextureFilterMode mipmap) {
            m_texDesc.filterMode = static_cast<CUfilter_mode>(xy);
            m_texDesc.mipmapFilterMode = static_cast<CUfilter_mode>(mipmap);
            m_texObjectIsUpToDate = false;
        }
        void setWrapMode(uint32_t dim, TextureWrapMode mode) {
            if (dim >= 3)
                return;
            m_texDesc.addressMode[dim] = static_cast<CUaddress_mode>(mode);
            m_texObjectIsUpToDate = false;
        }
        void setBorderColor(float r, float g, float b, float a) {
            m_texDesc.borderColor[0] = r;
            m_texDesc.borderColor[1] = g;
            m_texDesc.borderColor[2] = b;
            m_texDesc.borderColor[3] = a;
            m_texObjectIsUpToDate = false;
        }
        void setIndexingMode(TextureIndexingMode mode) {
            if (mode == TextureIndexingMode::ArrayIndex)
                m_texDesc.flags &= ~CU_TRSF_NORMALIZED_COORDINATES;
            else
                m_texDesc.flags |= CU_TRSF_NORMALIZED_COORDINATES;
            m_texObjectIsUpToDate = false;
        }
        void setReadMode(TextureReadMode mode) {
            if (mode == TextureReadMode::ElementType)
                m_texDesc.flags |= CU_TRSF_READ_AS_INTEGER;
            else
                m_texDesc.flags &= ~CU_TRSF_READ_AS_INTEGER;

            if (mode == TextureReadMode::NormalizedFloat_sRGB)
                m_texDesc.flags |= CU_TRSF_SRGB;
            else
                m_texDesc.flags &= ~CU_TRSF_SRGB;
            m_texObjectIsUpToDate = false;
        }

        CUtexObject getTextureObject() {
            if (m_texObjectCreated && !m_texObjectIsUpToDate)
                CUDADRV_CHECK(cuTexObjectDestroy(m_texObject));
            if (!m_texObjectIsUpToDate) {
                CUDADRV_CHECK(cuTexObjectCreate(&m_texObject, &m_resDesc, &m_texDesc, nullptr));
                m_texObjectCreated = true;
                m_texObjectIsUpToDate = true;
            }
            return m_texObject;
        }
    };
}
