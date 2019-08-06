#include "cuda_helper.h"

#ifdef CUDAHPlatform_Windows_MSVC
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#endif


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
        m_hostPointer(nullptr), m_devicePointer(nullptr),
        m_GLBufferID(0), m_cudaGfxResource(nullptr) {
    }

    Buffer::~Buffer() {
    }



    void Buffer::makeCurrent() {
        CUDA_CHECK(cudaSetDevice(m_deviceIndex));
    }


    
    void Buffer::initialize(BufferType type, int32_t width, int32_t height, int32_t stride, uint32_t glBufferID) {
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

        m_width = width;
        m_height = height;
        m_dimension = 2;
        m_stride = stride;

        m_GLBufferID = glBufferID;

        makeCurrent();

        if (m_type == BufferType::Device || m_type == BufferType::P2P)
            CUDA_CHECK(cudaMalloc(&m_devicePointer, m_width * m_height * m_stride));

        if (m_type == BufferType::GL_Interop)
            CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cudaGfxResource, m_GLBufferID, cudaGraphicsMapFlagsWriteDiscard));

        if (m_type == BufferType::ZeroCopy) {
            CUDA_CHECK(cudaHostAlloc(&m_hostPointer, m_width * m_height * m_stride, cudaHostAllocPortable | cudaHostAllocMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(&m_devicePointer, m_hostPointer, 0));
        }
    }
}
