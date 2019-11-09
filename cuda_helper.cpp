#include "cuda_helper.h"

#ifdef CUDAHPlatform_Windows_MSVC
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#endif



namespace CUDAHelper {
#ifdef CUDAHPlatform_Windows_MSVC
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        char str[1024];
        vsprintf_s(str, fmt, args);
        va_end(args);
        OutputDebugString(str);
    }
#else
    void devPrintf(const char* fmt, ...) {
        va_list args;
        va_start(args, fmt);
        vprintf_s(fmt, args);
        va_end(args);
    }
#endif



    Buffer::Buffer() :
        m_deviceIndex(0),
        m_hostPointer(nullptr), m_devicePointer(nullptr), m_mappedPointer(nullptr),
        m_GLBufferID(0), m_cudaGfxResource(nullptr),
        m_initialized(false), m_mapped(false) {
    }

    Buffer::~Buffer() {
    }



    void Buffer::makeCurrent() {
        CUDA_CHECK(cudaSetDevice(m_deviceIndex));
    }


    
    void Buffer::initialize(BufferType type, int32_t numElements, int32_t stride, uint32_t glBufferID) {
        if (m_initialized)
            throw std::runtime_error("Buffer is already initialized.");

        m_type = type;

        // If using GL Interop, expect that the active device is also the display device.
        if (m_type == BufferType::GL_Interop) {
            int32_t currentDevice;
            int32_t isDisplayDevice;
            CUDA_CHECK(cudaGetDevice(&currentDevice));
            CUDA_CHECK(cudaDeviceGetAttribute(&isDisplayDevice, cudaDevAttrKernelExecTimeout, currentDevice));
            if (!isDisplayDevice)
                throw std::runtime_error("GL Interop is only available on the display device.");
        }

        m_numElements = numElements;
        m_stride = stride;

        m_GLBufferID = glBufferID;

        makeCurrent();

        if (m_type == BufferType::Device || m_type == BufferType::P2P)
            CUDA_CHECK(cudaMalloc(&m_devicePointer, m_numElements * m_stride));

        if (m_type == BufferType::GL_Interop)
            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaGfxResource, m_GLBufferID, cudaGraphicsRegisterFlagsNone)); // TODO

        if (m_type == BufferType::ZeroCopy) {
            CUDA_CHECK(cudaHostAlloc(&m_hostPointer, m_numElements * m_stride, cudaHostAllocPortable | cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&m_devicePointer, m_hostPointer, 0));
        }

        m_initialized = true;
    }

    void Buffer::finalize() {
        if (!m_initialized)
            return;

        makeCurrent();

        if (m_type == BufferType::ZeroCopy) {
            CUDA_CHECK(cudaFreeHost(m_hostPointer));
            m_devicePointer = nullptr;
            m_hostPointer = nullptr;
        }

        if (m_type == BufferType::GL_Interop) {
            CUDA_CHECK(cudaGraphicsUnregisterResource(m_cudaGfxResource));
            m_devicePointer = 0;
        }

        if (m_type == BufferType::Device || m_type == BufferType::P2P) {
            CUDA_CHECK(cudaFree(m_devicePointer));
            m_devicePointer = nullptr;
        }

        m_mappedPointer = nullptr;;
        m_stride = 0;
        m_numElements = 0;

        m_initialized = false;
    }



    CUdeviceptr Buffer::beginCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        if (m_type == BufferType::GL_Interop) {
            makeCurrent();

            size_t bufferSize = 0;
            CUDA_CHECK(cudaGraphicsMapResources(1, &m_cudaGfxResource, stream));
            CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&m_devicePointer, &bufferSize, m_cudaGfxResource));
        }

        return (CUdeviceptr)m_devicePointer;
    }

    void Buffer::endCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        if (m_type == BufferType::GL_Interop) {
            makeCurrent();

            CUDA_CHECK(cudaGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
        }
    }

    void* Buffer::map() {
        if (m_mapped)
            throw std::runtime_error("This buffer is already mapped.");

        m_mapped = true;

        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P ||
            m_type == BufferType::GL_Interop) {
            makeCurrent();

            size_t size = (size_t)m_numElements * m_stride;
            m_mappedPointer = new uint8_t[size];

            void* devicePointer = m_devicePointer;
            if (m_type == BufferType::GL_Interop)
                devicePointer = (void*)beginCUDAAccess(0);

            CUDA_CHECK(cudaMemcpy(m_mappedPointer, devicePointer, size, cudaMemcpyDeviceToHost));

            if (m_type == BufferType::GL_Interop)
                endCUDAAccess(0);

            return m_mappedPointer;
        }
        else {
            return m_hostPointer;
        }
    }

    void Buffer::unmap() {
        if (!m_mapped)
            throw std::runtime_error("This buffer is not mapped.");

        m_mapped = false;

        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P ||
            m_type == BufferType::GL_Interop) {
            CUDAHAssert(m_mappedPointer, "This buffer is not mapped.");

            makeCurrent();

            size_t size = (size_t)m_numElements * m_stride;

            void* devicePointer = m_devicePointer;
            if (m_type == BufferType::GL_Interop)
                devicePointer = (void*)beginCUDAAccess(0);

            CUDA_CHECK(cudaMemcpy(devicePointer, m_mappedPointer, size, cudaMemcpyHostToDevice));

            if (m_type == BufferType::GL_Interop)
                endCUDAAccess(0);

            delete[] m_mappedPointer;
            m_mappedPointer = nullptr;
        }
    }
}
