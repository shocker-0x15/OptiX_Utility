#include "cuda_util.h"

#ifdef CUDAHPlatform_Windows_MSVC
#   include <Windows.h>
#   undef near
#   undef far
#   undef min
#   undef max
#endif



namespace cudau {
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
        m_cudaContext = b.m_cudaContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;
    }

    Buffer &Buffer::operator=(Buffer &&b) {
        finalize();

        m_cudaContext = b.m_cudaContext;
        m_type = b.m_type;
        m_numElements = b.m_numElements;
        m_stride = b.m_stride;
        m_hostPointer = b.m_hostPointer;
        m_devicePointer = b.m_devicePointer;
        m_mappedPointer = b.m_mappedPointer;
        m_GLBufferID = b.m_GLBufferID;
        m_cudaGfxResource = b.m_cudaGfxResource;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;

        return *this;
    }


    
    void Buffer::initialize(CUcontext context, BufferType type,
                            uint32_t numElements, uint32_t stride, uint32_t glBufferID) {
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

        m_mappedPointer = nullptr;
        m_stride = 0;
        m_numElements = 0;

        m_initialized = false;
    }

    void Buffer::resize(uint32_t numElements, uint32_t stride) {
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

        uint32_t numElementsToCopy = std::min(m_numElements, numElements);
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

    Buffer Buffer::copy() const {
        if (m_GLBufferID != 0)
            throw std::runtime_error("Copying OpenGL buffer is not supported.");

        Buffer ret;
        ret.initialize(m_cudaContext, m_type, m_numElements, m_stride, m_GLBufferID);

        size_t size = m_stride * m_numElements;
        if (m_type == BufferType::Device ||
            m_type == BufferType::P2P) {
            CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

            CUDADRV_CHECK(cuMemcpyDtoD(ret.m_devicePointer, m_devicePointer, size));
        }
        else {
            std::memcpy(ret.m_hostPointer, m_hostPointer, size);
        }

        return ret;
    }



    static bool isBCFormat(ArrayElementType elemType) {
        return (elemType == cudau::ArrayElementType::BC1_UNorm ||
                elemType == cudau::ArrayElementType::BC2_UNorm ||
                elemType == cudau::ArrayElementType::BC3_UNorm ||
                elemType == cudau::ArrayElementType::BC4_UNorm ||
                elemType == cudau::ArrayElementType::BC4_SNorm ||
                elemType == cudau::ArrayElementType::BC5_UNorm ||
                elemType == cudau::ArrayElementType::BC5_SNorm ||
                elemType == cudau::ArrayElementType::BC6H_UF16 ||
                elemType == cudau::ArrayElementType::BC6H_SF16 ||
                elemType == cudau::ArrayElementType::BC7_UNorm);
    }

    static CUresourceViewFormat getResourceViewFormat(ArrayElementType elemType, uint32_t numChannels) {
        switch (elemType) {
        case cudau::ArrayElementType::UInt8:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_UINT_1X8;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UINT_2X8;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UINT_4X8;
            break;
        case cudau::ArrayElementType::Int8:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_SINT_1X8;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SINT_2X8;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SINT_4X8;
            break;
        case cudau::ArrayElementType::UInt16:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_UINT_1X16;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UINT_2X16;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UINT_4X16;
            break;
        case cudau::ArrayElementType::Int16:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_SINT_1X16;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SINT_2X16;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SINT_4X16;
            break;
        case cudau::ArrayElementType::UInt32:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_UINT_1X32;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UINT_2X32;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UINT_4X32;
            break;
        case cudau::ArrayElementType::Int32:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_SINT_1X32;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SINT_2X32;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SINT_4X32;
            break;
        case cudau::ArrayElementType::Float16:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_FLOAT_1X16;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_FLOAT_2X16;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_FLOAT_4X16;
            break;
        case cudau::ArrayElementType::Float32:
            if (numChannels == 1)
                return CU_RES_VIEW_FORMAT_FLOAT_1X32;
            else if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_FLOAT_2X32;
            else if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_FLOAT_4X32;
            break;
        case cudau::ArrayElementType::BC1_UNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC1;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC2;
            break;
        case cudau::ArrayElementType::BC3_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC3;
            break;
        case cudau::ArrayElementType::BC4_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC4_SNorm:
            if (numChannels == 2)
                return CU_RES_VIEW_FORMAT_SIGNED_BC4;
            break;
        case cudau::ArrayElementType::BC5_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC5_SNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC5;
            break;
        case cudau::ArrayElementType::BC6H_UF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC6H_SF16:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_SIGNED_BC6H;
            break;
        case cudau::ArrayElementType::BC7_UNorm:
            if (numChannels == 4)
                return CU_RES_VIEW_FORMAT_UNSIGNED_BC7;
            break;
        default:
            break;
        }
        CUDAUAssert_ShouldNotBeCalled();
        return CU_RES_VIEW_FORMAT_NONE;
    }



    Array::Array() :
        m_cudaContext(nullptr),
        m_array(0), m_mappedPointer(nullptr),
        m_writable(false), m_cubemap(false), m_layered(false),
        m_initialized(false), m_mapped(false) {
    }

    Array::~Array() {
        if (m_initialized)
            finalize();
    }

    Array::Array(Array &&b) {
        m_cudaContext = b.m_cudaContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        m_array = b.m_array;
        m_mappedPointer = b.m_mappedPointer;
        m_writable = b.m_writable;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;
    }

    Array &Array::operator=(Array &&b) {
        finalize();

        m_cudaContext = b.m_cudaContext;
        m_width = b.m_width;
        m_height = b.m_height;
        m_depth = b.m_depth;
        m_stride = b.m_stride;
        m_elemType = b.m_elemType;
        m_numChannels = b.m_numChannels;
        m_array = b.m_array;
        m_mappedPointer = b.m_mappedPointer;
        m_writable = b.m_writable;
        m_cubemap = b.m_cubemap;
        m_layered = b.m_layered;
        m_initialized = b.m_initialized;
        m_mapped = b.m_mapped;

        b.m_initialized = false;

        return *this;
    }


    
    void Array::initialize(CUcontext context, ArrayElementType elemType, uint32_t numChannels,
                           uint32_t width, uint32_t height, uint32_t depth,
                           bool writable, bool cubemap, bool layered) {
        if (m_initialized)
            throw std::runtime_error("Array is already initialized.");
        if (numChannels != 1 && numChannels != 2 && numChannels != 4)
            throw std::runtime_error("numChannels must be 1, 2, or 4.");
        if (elemType >= ArrayElementType::BC1_UNorm &&
            elemType <= ArrayElementType::BC7_UNorm &&
            numChannels != 1)
            throw std::runtime_error("numChannels must be 1 for BC format.");

        m_cudaContext = context;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        //if (height == 0 && depth == 0 && cubemap == false && layered == false)
        //    m_arrayType = ArrayType::E_1D;
        //else if (depth == 0 && cubemap == false && layered == false)
        //    m_arrayType = ArrayType::E_2D;
        //else if (cubemap == false && layered == false)
        //    m_arrayType = ArrayType::E_3D;
        //else if (height == 0 && cubemap == false && layered == true)
        //    m_arrayType = ArrayType::E_1DLayered;
        //else if (cubemap == false && layered == true)
        //    m_arrayType = ArrayType::E_2DLayered;
        //else if (cubemap == true && layered == false)
        //    m_arrayType = ArrayType::Cubemap;
        //else if (cubemap == true && layered == true)
        //    m_arrayType = ArrayType::CubemapLayered;

        m_width = width;
        m_height = height;
        m_depth = depth;
        m_elemType = elemType;
        m_numChannels = numChannels;
        m_writable = writable;
        m_cubemap = cubemap;
        m_layered = layered;

        CUDA_ARRAY3D_DESCRIPTOR arrayDesc = {};
        if (writable)
            arrayDesc.Flags |= CUDA_ARRAY3D_SURFACE_LDST;
        if (layered)
            arrayDesc.Flags |= CUDA_ARRAY3D_LAYERED;
        if (cubemap)
            arrayDesc.Flags |= CUDA_ARRAY3D_CUBEMAP;
        switch (m_elemType) {
        case cudau::ArrayElementType::UInt8:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::UInt16:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::UInt32:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Int8:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT8;
            m_stride = 1;
            break;
        case cudau::ArrayElementType::Int16:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT16;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Int32:
            arrayDesc.Format = CU_AD_FORMAT_SIGNED_INT32;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::Float16:
            arrayDesc.Format = CU_AD_FORMAT_HALF;
            m_stride = 2;
            break;
        case cudau::ArrayElementType::Float32:
            arrayDesc.Format = CU_AD_FORMAT_FLOAT;
            m_stride = 4;
            break;
        case cudau::ArrayElementType::BC1_UNorm:
        case cudau::ArrayElementType::BC4_UNorm:
        case cudau::ArrayElementType::BC4_SNorm:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            m_numChannels = 2;
            numChannels = 2;
            m_width >>= 2;
            m_height >>= 2;
            break;
        case cudau::ArrayElementType::BC2_UNorm:
        case cudau::ArrayElementType::BC3_UNorm:
        case cudau::ArrayElementType::BC5_UNorm:
        case cudau::ArrayElementType::BC5_SNorm:
        case cudau::ArrayElementType::BC6H_UF16:
        case cudau::ArrayElementType::BC6H_SF16:
        case cudau::ArrayElementType::BC7_UNorm:
            arrayDesc.Format = CU_AD_FORMAT_UNSIGNED_INT32;
            m_stride = 4;
            m_numChannels = 4;
            numChannels = 4;
            m_width >>= 2;
            m_height >>= 2;
            break;
        default:
            CUDAUAssert_ShouldNotBeCalled();
            break;
        }
        arrayDesc.Width = m_width;
        arrayDesc.Height = m_height;
        arrayDesc.Depth = m_depth;
        arrayDesc.NumChannels = numChannels;
        m_stride *= numChannels;

        // Is cuArray3DCreate the upper compatible to cuArrayCreate?
        //CUDADRV_CHECK(cuArrayCreate(&m_array, &arrayDesc));
        CUDADRV_CHECK(cuArray3DCreate(&m_array, &arrayDesc));

        m_initialized = true;
    }

    void Array::finalize() {
        if (!m_initialized)
            return;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        CUDADRV_CHECK(cuArrayDestroy(m_array));

        m_initialized = false;
    }

    void Array::resize(uint32_t length) {
        if (m_height > 0 || m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        CUDAUAssert_NotImplemented();
    }

    void Array::resize(uint32_t width, uint32_t height) {
        if (m_depth > 0)
            throw std::runtime_error("Array dimension cannot be changed.");
        
        if (width == m_width && height == m_height)
            return;

        Array newArray;
        newArray.initialize(m_cudaContext, m_elemType, m_numChannels, width, height, m_height,
                            m_writable, m_cubemap, m_layered);

        size_t sizePerRow = std::min(m_width, width) * static_cast<size_t>(m_stride);

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = std::max<size_t>(1, std::min(m_height, height));
        params.Depth = std::max<size_t>(1, m_depth);

        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = m_array;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = m_array;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        *this = std::move(newArray);
    }

    void Array::resize(uint32_t width, uint32_t height, uint32_t depth) {
        CUDAUAssert_NotImplemented();
    }



    void* Array::map() {
        if (m_mapped)
            throw std::runtime_error("This buffer is already mapped.");

        m_mapped = true;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        size_t sizePerRow = m_width * static_cast<size_t>(m_stride);
        size_t size = std::max<size_t>(1, m_depth) * std::max<size_t>(1, m_height) * sizePerRow;
        m_mappedPointer = new uint8_t[size];

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = std::max<size_t>(1, m_height);
        params.Depth = std::max<size_t>(1, m_depth);

        params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        params.srcArray = m_array;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcDevice, srcHeight, srcHost, srcLOD, srcPitch are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_HOST;
        params.dstHost = m_mappedPointer;
        params.dstPitch = sizePerRow;
        params.dstHeight = m_height;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstArray, dstDevice, dstLOD are not used in this case.

        CUDADRV_CHECK(cuMemcpy3D(&params));

        return m_mappedPointer;
    }

    void Array::unmap() {
        if (!m_mapped)
            throw std::runtime_error("This buffer is not mapped.");

        m_mapped = false;

        CUDADRV_CHECK(cuCtxSetCurrent(m_cudaContext));

        size_t sizePerRow = m_width * static_cast<size_t>(m_stride);

        CUDA_MEMCPY3D params = {};
        params.WidthInBytes = sizePerRow;
        params.Height = std::max<size_t>(1, m_height);
        params.Depth = std::max<size_t>(1, m_depth);

        params.srcMemoryType = CU_MEMORYTYPE_HOST;
        params.srcHost = m_mappedPointer;
        params.srcPitch = sizePerRow;
        params.srcHeight = m_height;
        params.srcXInBytes = 0;
        params.srcY = 0;
        params.srcZ = 0;
        // srcArray, srcDevice, srcLOD are not used in this case.

        params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
        params.dstArray = m_array;
        params.dstXInBytes = 0;
        params.dstY = 0;
        params.dstZ = 0;
        // dstDevice, dstHeight, dstHost, dstLOD, dstPitch are not used in this case.

        CUDADRV_CHECK(cuMemcpy3D(&params));

        delete[] m_mappedPointer;
        m_mappedPointer = nullptr;
    }



    CUDA_RESOURCE_VIEW_DESC Array::getResourceViewDesc() const {
        CUDA_RESOURCE_VIEW_DESC ret = {};
        bool isBC = isBCFormat(m_elemType);
        ret.format = getResourceViewFormat(m_elemType, m_numChannels);
        ret.width = (isBC ? 4 : 1) * m_width;
        ret.height = (isBC ? 4 : 1) * m_height;
        ret.depth = m_depth;
        ret.firstMipmapLevel = 0;
        ret.lastMipmapLevel = 0;
        if (m_layered) {
            CUDAUAssert_NotImplemented();
        }
        else {
            ret.firstLayer = 0;
            ret.lastLayer = 0;
        }

        return ret;
    }
}
