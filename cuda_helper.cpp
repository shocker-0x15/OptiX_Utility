#include "cuda_helper.h"

#ifdef CUDAHPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
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
        m_cudaContext(nullptr),
        m_hostPointer(nullptr), m_devicePointer(0), m_mappedPointer(nullptr),
        m_GLBufferID(0), m_cudaGfxResource(nullptr),
        m_initialized(false), m_mapped(false) {
    }

    Buffer::~Buffer() {
        if (m_initialized)
            finalize();
    }

    Buffer::Buffer(Buffer &&b) {
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_cudaContext = b.m_cudaContext;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;
    }

    Buffer &Buffer::operator=(Buffer &&b) {
        finalize();

        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_cudaContext = b.m_cudaContext;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;

        return *this;
    }


    
    void Buffer::initialize(CUcontext context, BufferType type, int32_t numElements, int32_t stride, uint32_t glBufferID) {
        if (m_initialized)
            throw std::runtime_error("Buffer is already initialized.");

        m_cudaContext = context;
        m_type = type;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        // If using GL Interop, expect that the active device is also the display device.
        if (m_type == BufferType::GL_Interop) {
            CUdevice currentDevice;
            int32_t isDisplayDevice;
            CUDADRV_CHECK(cuCtxGetDevice(&currentDevice));
            CUDADRV_CHECK(cuDeviceGetAttribute(&isDisplayDevice, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, currentDevice));
            if (!isDisplayDevice)
                throw std::runtime_error("GL Interop is only available on the display device.");
        }

        m_numElements = numElements;
        m_stride = stride;

        m_GLBufferID = glBufferID;

        if (m_type == BufferType::Device || m_type == BufferType::P2P)
            CUDADRV_CHECK(cuMemAlloc(&m_devicePointer, m_numElements * m_stride));

        if (m_type == BufferType::GL_Interop)
            CUDADRV_CHECK(cuGraphicsGLRegisterBuffer(&m_cudaGfxResource, m_GLBufferID, CU_GRAPHICS_REGISTER_FLAGS_NONE));

        if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemHostAlloc(&m_hostPointer, m_numElements * m_stride, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
            CUDADRV_CHECK(cuMemHostGetDevicePointer(&m_devicePointer, m_hostPointer, 0));
        }

        m_initialized = true;
    }

    void Buffer::finalize() {
        if (!m_initialized)
            return;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        if (m_mapped)
            unmap();

        if (m_type == BufferType::ZeroCopy) {
            CUDADRV_CHECK(cuMemFreeHost(m_hostPointer));
            m_devicePointer = 0;
            m_hostPointer = nullptr;
        }

        if (m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuGraphicsUnregisterResource(m_cudaGfxResource));
            m_devicePointer = 0;
        }

        if (m_type == BufferType::Device || m_type == BufferType::P2P) {
            CUDADRV_CHECK(cuMemFree(m_devicePointer));
            m_devicePointer = 0;
        }

        m_mappedPointer = nullptr;;
        m_stride = 0;
        m_numElements = 0;

        m_initialized = false;
    }

    void Buffer::resize(int32_t numElements, int32_t stride) {
        if (!m_initialized)
            throw std::runtime_error("Buffer is not initialized.");
        if (m_type == BufferType::GL_Interop)
            throw std::runtime_error("Resize for GL-interop buffer is not supported.");
        if (stride < m_stride)
            throw std::runtime_error("New stride must be >= the current stride.");

        if (numElements == m_numElements && stride == m_stride)
            return;

        Buffer newBuffer;
        newBuffer.initialize(m_cudaContext, m_type, numElements, stride, m_GLBufferID);

        int32_t numElementsToCopy = std::min(m_numElements, numElements);
        if (stride == m_stride) {
            size_t numBytesToCopy = static_cast<size_t>(numElementsToCopy) * m_stride;
            CUDADRV_CHECK(cuMemcpyDtoD(newBuffer.m_devicePointer, m_devicePointer, numBytesToCopy));
        }
        else {
            auto src = map<const uint8_t>();
            auto dst = newBuffer.map<uint8_t>();
            for (int i = 0; i < numElementsToCopy; ++i) {
                std::memset(dst, 0, stride);
                std::memcpy(dst, src, m_stride);
            }
            newBuffer.unmap();
            unmap();
        }

        *this = std::move(newBuffer);
    }



    CUdeviceptr Buffer::beginCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        if (m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            size_t bufferSize = 0;
            CUDADRV_CHECK(cuGraphicsMapResources(1, &m_cudaGfxResource, stream));
            CUDADRV_CHECK(cuGraphicsResourceGetMappedPointer(&m_devicePointer, &bufferSize, m_cudaGfxResource));
        }

        return (CUdeviceptr)m_devicePointer;
    }

    void Buffer::endCUDAAccess(CUstream stream) {
        if (m_type != BufferType::GL_Interop)
            throw std::runtime_error("This is not an OpenGL-interop buffer.");

        if (m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            CUDADRV_CHECK(cuGraphicsUnmapResources(1, &m_cudaGfxResource, stream));
        }
    }

    void* Buffer::map() {
        if (m_mapped)
            throw std::runtime_error("This buffer is already mapped.");

        m_mapped = true;

        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P ||
            m_type == BufferType::GL_Interop) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            size_t size = (size_t)m_numElements * m_stride;
            m_mappedPointer = new uint8_t[size];

            CUdeviceptr devicePointer = m_devicePointer;
            if (m_type == BufferType::GL_Interop)
                devicePointer = beginCUDAAccess(0);

            CUDADRV_CHECK(cuMemcpyDtoH(m_mappedPointer, devicePointer, size));

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

            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            size_t size = (size_t)m_numElements * m_stride;

            CUdeviceptr devicePointer = m_devicePointer;
            if (m_type == BufferType::GL_Interop)
                devicePointer = beginCUDAAccess(0);

            CUDADRV_CHECK(cuMemcpyHtoD(devicePointer, m_mappedPointer, size));

            if (m_type == BufferType::GL_Interop)
                endCUDAAccess(0);

            delete[] m_mappedPointer;
            m_mappedPointer = nullptr;
        }
    }
}
