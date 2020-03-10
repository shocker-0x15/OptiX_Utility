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
        BufferType m_type;

        int32_t m_numElements;
        int32_t m_stride;

        void* m_hostPointer;
        CUdeviceptr m_devicePointer;
        void* m_mappedPointer;

        uint32_t m_GLBufferID;
        CUgraphicsResource m_cudaGfxResource;

        CUcontext m_cudaContext;

        struct {
            unsigned int m_initialized : 1;
            unsigned int m_mapped : 1;
        };

        Buffer(const Buffer &) = delete;
        Buffer &operator=(const Buffer &) = delete;

    public:
        Buffer();
        Buffer(CUcontext context, BufferType type, int32_t numElements, int32_t stride, uint32_t glBufferID = 0) : Buffer() {
            initialize(context, type, numElements, stride, glBufferID);
        }
        ~Buffer();

        Buffer(Buffer &&b);
        Buffer &operator=(Buffer &&b);

        void initialize(CUcontext context, BufferType type, int32_t numElements, int32_t stride, uint32_t glBufferID);
        void finalize();

        void resize(int32_t numElements, int32_t stride);

        CUdeviceptr getDevicePointer() const {
            return (CUdeviceptr)m_devicePointer;
        }
        size_t sizeInBytes() const {
            return (size_t)m_numElements * m_stride;
        }
        size_t stride() const {
            return m_stride;
        }
        size_t numElements() const {
            return (size_t)m_numElements;
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
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getDevicePointer(), v, numElements * sizeof(T)));
        }
        void initialize(CUcontext context, BufferType type, const std::vector<T> &v) {
            initialize(context, type, v.size());
            CUDADRV_CHECK(cuMemcpyHtoD(Buffer::getDevicePointer(), v.data(), v.size() * sizeof(T)));
        }
        void finalize() {
            Buffer::finalize();
        }

        void resize(int32_t numElements) {
            Buffer::resize(numElements, sizeof(T));
        }

        T* getDevicePointer() const {
            return reinterpret_cast<T*>(Buffer::getDevicePointer());
        }
        T* getDevicePointerAt(uint32_t idx) const {
            CUdeviceptr ptr = Buffer::getDevicePointer();
            return reinterpret_cast<T*>(ptr + sizeof(T) * idx);
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
}
