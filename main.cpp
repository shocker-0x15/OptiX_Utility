// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define HP_Platform_Windows
#    if defined(_MSC_VER)
#        define HP_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define HP_Platform_macOS
#endif

#if defined(HP_Platform_Windows_MSVC)
#   define NOMINMAX
#   define _USE_MATH_DEFINES
#   include <Windows.h>
#   undef near
#   undef far
#   undef RGB
#endif



#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>

#include <fstream>
#include <sstream>
#include <filesystem>

#include <GL/gl3w.h>

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "GLToolkit.h"
#include "cuda_helper.h"

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include "shared.h"



#ifdef _DEBUG
#   define ENABLE_ASSERT
#endif

#ifdef HP_Platform_Windows_MSVC
static void devPrintf(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    char str[4096];
    vsnprintf_s(str, sizeof(str), _TRUNCATE, fmt, args);
    va_end(args);
    OutputDebugString(str);
}
#else
#   define devPrintf(fmt, ...) printf(fmt, ##__VA_ARGS__);
#endif

#if 1
#   define hpprintf(fmt, ...) do { devPrintf(fmt, ##__VA_ARGS__); printf(fmt, ##__VA_ARGS__); } while (0)
#else
#   define hpprintf(fmt, ...) printf(fmt, ##__VA_ARGS__)
#endif

#ifdef ENABLE_ASSERT
#   define Assert(expr, fmt, ...) if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } 0
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")

template <typename T, size_t size>
constexpr size_t lengthof(const T(&array)[size]) {
    return size;
}



#define OPTIX_CHECK(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

#define OPTIX_CHECK_LOG(call) \
    do { \
        OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n" \
               << "Log: " << log << (logSize > sizeof(log) ? "<TRUNCATED>" : "") \
               << "\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)



template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SBTRecord {
    uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSBTRecord = SBTRecord<Shared::RayGenData>;
using MissSBTRecord = SBTRecord<Shared::MissData>;
using HitGroupSBTRecord = SBTRecord<Shared::HitGroupData>;
using EmptySBTRecord = SBTRecord<int32_t>;



class AccelerationStructure {
    OptixDeviceContext m_context;
    std::vector<OptixBuildInput> m_children;

    OptixAccelBuildOptions m_buildOptions;

    size_t m_accelBufferSize;
    CUDAHelper::Buffer m_accelBuffer;
    CUDAHelper::Buffer m_accelTempBuffer;

    CUDAHelper::Buffer m_compactedSizeOnDevice;
    size_t m_compactedSize;
    OptixAccelEmitDesc m_propertyCompactedSize;
    CUDAHelper::Buffer m_compactedAccelBuffer;

    OptixTraversableHandle m_handle;
    OptixTraversableHandle m_compactedHandle;
    struct {
        unsigned int m_available : 1;
        unsigned int m_compactedAvailable : 1;
    };

public:
    void initialize(OptixDeviceContext context, const OptixBuildInput* children, uint32_t numChildren,
                    bool preferFastTrace, bool allowUpdate, bool enableCompaction) {
        m_context = context;

        m_children.resize(numChildren);
        std::copy_n(children, numChildren, m_children.data());

        std::memset(&m_buildOptions, 0, sizeof(m_buildOptions));
        m_buildOptions.buildFlags = ((preferFastTrace ? OPTIX_BUILD_FLAG_PREFER_FAST_TRACE : 0) |
                                     (allowUpdate ? OPTIX_BUILD_FLAG_ALLOW_UPDATE : 0) |
                                     (enableCompaction ? OPTIX_BUILD_FLAG_ALLOW_COMPACTION : 0));
        //m_buildOptions.motionOptions
        m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes bufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &m_buildOptions, m_children.data(), m_children.size(),
                                                 &bufferSizes));

        m_accelBufferSize = bufferSizes.outputSizeInBytes;
        m_accelTempBuffer.initialize(CUDAHelper::BufferType::Device, std::max(bufferSizes.tempSizeInBytes, bufferSizes.tempUpdateSizeInBytes), 1, 0);

        m_compactedSizeOnDevice.initialize(CUDAHelper::BufferType::Device, 1, sizeof(size_t), 0);

        std::memset(&m_propertyCompactedSize, 0, sizeof(m_propertyCompactedSize));
        m_propertyCompactedSize.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        m_propertyCompactedSize.result = m_compactedSizeOnDevice.getDevicePointer();

        m_available = false;
        m_compactedAvailable = false;
    }

    void finalize() {
        if (m_compactedAvailable)
            m_compactedAccelBuffer.finalize();
        m_compactedSizeOnDevice.finalize();

        m_accelTempBuffer.finalize();
        if (m_available)
            m_accelBuffer.finalize();

        m_children.clear();

        m_context = nullptr;
    }

    void rebuild(CUstream stream) {
        bool compactionEnabled = (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_available)
            m_accelBuffer.initialize(CUDAHelper::BufferType::Device, m_accelBufferSize, 1, 0);

        m_buildOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        OPTIX_CHECK(optixAccelBuild(m_context, stream, &m_buildOptions, m_children.data(), m_children.size(),
                                    m_accelTempBuffer.getDevicePointer(), m_accelTempBuffer.size(),
                                    m_accelBuffer.getDevicePointer(), m_accelBuffer.size(),
                                    &m_handle,
                                    compactionEnabled ? &m_propertyCompactedSize : nullptr, compactionEnabled ? 1 : 0));

        m_available = true;
        m_compactedHandle = 0;
        m_compactedAvailable = false;
    }

    void compaction(CUstream rebuildOrUpdateStream, CUstream stream) {
        bool compactionEnabled = (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_available || m_compactedAvailable || !compactionEnabled)
            return;

        // JP: リビルド・アップデートの完了を待ってコンパクション後のサイズ情報を取得。
        CUDA_CHECK(cudaStreamSynchronize(rebuildOrUpdateStream));
        CUDA_CHECK(cudaMemcpy(&m_compactedSize, (void*)m_propertyCompactedSize.result, sizeof(m_compactedSize), cudaMemcpyDeviceToHost));
        // JP: 以下になるべき？
        // CUDA_CHECK(cudaMemcpyAsync(&m_compactedSize, (void*)m_propertyCompactedSize.result, sizeof(m_compactedSize), cudaMemcpyDeviceToHost, rebuildStream));

        if (m_compactedSize < m_accelBuffer.size()) {
            m_compactedAccelBuffer.initialize(CUDAHelper::BufferType::Device, m_compactedSize, 1, 0);

            OPTIX_CHECK(optixAccelCompact(m_context, stream, m_handle, m_compactedAccelBuffer.getDevicePointer(), m_compactedAccelBuffer.size(),
                                          &m_compactedHandle));

            m_compactedAvailable = true;
        }
    }

    void removeUncompacted(CUstream compactionStream) {
        bool compactionEnabled = (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION) != 0;

        if (!m_compactedAvailable || !compactionEnabled)
            return;

        // JP: コンパクションの完了を待ってバッファーを解放。
        CUDA_CHECK(cudaStreamSynchronize(compactionStream));
        m_accelBuffer.finalize();

        m_handle = 0;
        m_available = false;
    }

    void update(CUstream stream) {
        bool updateEnabled = (m_buildOptions.buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE) != 0;

        // Should this be an assert?
        if ((!m_available && !m_compactedAvailable) || !updateEnabled)
            return;

        const CUDAHelper::Buffer &accelBuffer = m_compactedAvailable ? m_compactedAccelBuffer : m_accelBuffer;
        OptixTraversableHandle &handle = m_compactedAvailable ? m_compactedHandle : m_handle;

        m_buildOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
        OPTIX_CHECK(optixAccelBuild(m_context, stream, &m_buildOptions, m_children.data(), m_children.size(),
                                    m_accelTempBuffer.getDevicePointer(), m_accelTempBuffer.size(),
                                    accelBuffer.getDevicePointer(), accelBuffer.size(),
                                    &handle,
                                    nullptr, 0));
    }

    OptixTraversableHandle getHandle() const {
        if (m_compactedAvailable)
            return m_compactedHandle;
        if (m_available)
            return m_handle;
        return 0;
    }
};



class TriangleMesh {
    struct MaterialGroup {
        CUDAHelper::Buffer triangleBuffer;
        uint32_t* buildInputFlags;
        Shared::MaterialData material;
    };

    CUdeviceptr m_vertexBuffers[1];
    CUDAHelper::Buffer m_vertexBuffer;
    std::vector<MaterialGroup> m_materialGroups;
    std::vector<OptixBuildInput> m_buildInputs;
    uint32_t m_numSBTRecords;

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;
public:
    TriangleMesh() : m_numSBTRecords(0) {}

    void setVertexBuffer(const Shared::Vertex* vertices, uint32_t numVertices) {
        m_vertexBuffer.initialize(CUDAHelper::BufferType::Device, numVertices, sizeof(vertices[0]), 0);
        auto verticesD = (Shared::Vertex*)m_vertexBuffer.map();
        std::copy_n(vertices, numVertices, verticesD);
        m_vertexBuffer.unmap();

        // JP: モーションキーごとのバーテックスバッファー配列。
        m_vertexBuffers[0] = m_vertexBuffer.getDevicePointer();
    }

    void addMaterialGroup(const Shared::Triangle* triangles, uint32_t numTriangles, const Shared::MaterialData &matData) {
        m_materialGroups.push_back(MaterialGroup());
        m_buildInputs.push_back(OptixBuildInput{});

        MaterialGroup &group = m_materialGroups.back();
        {
            group.triangleBuffer.initialize(CUDAHelper::BufferType::Device, numTriangles, sizeof(triangles[0]), 0);
            auto trianglesD = (Shared::Triangle*)group.triangleBuffer.map();
            std::copy_n(triangles, numTriangles, trianglesD);
            group.triangleBuffer.unmap();

            group.material = matData;
        }

        OptixBuildInput &buildInput = m_buildInputs.back();
        {
            std::memset(&buildInput, 0, sizeof(buildInput));

            buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            OptixBuildInputTriangleArray &triArray = buildInput.triangleArray;

            triArray.vertexBuffers = m_vertexBuffers;
            triArray.numVertices = m_vertexBuffer.numElements();
            triArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triArray.vertexStrideInBytes = m_vertexBuffer.stride();

            triArray.indexBuffer = group.triangleBuffer.getDevicePointer();
            triArray.numIndexTriplets = numTriangles;
            triArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
            triArray.indexStrideInBytes = group.triangleBuffer.stride();
            triArray.primitiveIndexOffset = 0;

            // JP: per-Primitive SBT Recordは使用しないこととする。
            triArray.numSbtRecords = 1;
            triArray.sbtIndexOffsetBuffer = 0; // No per-primitive record
            triArray.sbtIndexOffsetSizeInBytes = 0; // No effect
            triArray.sbtIndexOffsetStrideInBytes = 0; // No effect

            triArray.preTransform = 0;

            group.buildInputFlags = new uint32_t[triArray.numSbtRecords];
            for (int i = 0; i < triArray.numSbtRecords; ++i)
                group.buildInputFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
            triArray.flags = group.buildInputFlags;

            m_numSBTRecords += triArray.numSbtRecords;
        }
    }

    void getBuildInfo(const OptixBuildInput** buildInputs, uint32_t* numBuildInputs, uint32_t* numSBTRecords) const {
        *buildInputs = m_buildInputs.data();
        *numBuildInputs = m_buildInputs.size();
        *numSBTRecords = m_numSBTRecords;
    }
};



namespace filesystem = std::experimental::filesystem;
static filesystem::path getExecutableDirectory() {
    static filesystem::path ret;

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

static std::string readTxtFile(const filesystem::path& filepath) {
    std::ifstream ifs;
    ifs.open(filepath, std::ios::in);
    if (ifs.fail())
        return "";

    std::stringstream sstream;
    sstream << ifs.rdbuf();

    return std::string(sstream.str());
};



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



static void optixLogCallBack(uint32_t level, const char* tag, const char* message, void* cbdata) {
    hpprintf("[%2u][%12s]: %s\n", level, tag, message);
}



float sRGB_degamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
};



int32_t mainFunc(int32_t argc, const char* argv[]) {
    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);

    int32_t renderTargetSizeX = 1280;
    int32_t renderTargetSizeY = 720;

    // JP: ウインドウの初期化。
    //     HiDPIディスプレイに対応する。
    // EN: Initialize a window.
    //     Support Hi-DPI display.
    float contentScaleX, contentScaleY;
    glfwGetMonitorContentScale(monitor, &contentScaleX, &contentScaleY);
    float UIScaling = contentScaleX;
    GLFWwindow* window = glfwCreateWindow((int32_t)(renderTargetSizeX * UIScaling), (int32_t)(renderTargetSizeY * UIScaling), "OptiX 7 + GLFW + ImGui", NULL, NULL);
    glfwSetWindowUserPointer(window, nullptr);
    if (!window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        return -1;
    }

    int32_t curFBWidth;
    int32_t curFBHeight;
    glfwGetFramebufferSize(window, &curFBWidth, &curFBHeight);

    glfwMakeContextCurrent(window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_FRAMEBUFFER_SRGB);
    GLTK::errorCheck();

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGuiStyle guiStyle, guiStyleWithGamma;
    ImGui::StyleColorsDark(&guiStyle);
    guiStyleWithGamma = guiStyle;
    const auto degamma = [](const ImVec4 &color) {
        return ImVec4(sRGB_degamma_s(color.x),
                      sRGB_degamma_s(color.y),
                      sRGB_degamma_s(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        guiStyleWithGamma.Colors[i] = degamma(guiStyleWithGamma.Colors[i]);
    }
    ImGui::GetStyle() = guiStyleWithGamma;

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    //     OptiXが定義する構造体(例：OptixPipelineCompileOptions)は将来の拡張に備えてゼロで初期化しておく必要がある。
    // EN: Settings for OptiX context and pipeline.
    //     Structs (e.g. OptixPipelineCompileOptions) defined by OptiX should be initialized with zeroes for future extensions.
    
    OptixDeviceContext optixContext = nullptr;
    {
        // JP: CUDAの初期化。
        //     ゼロは現在のCUDAコンテキストを意味する。
        // EN: initialize CUDA.
        //     Zero means taking the current CUDA ontext.
        CUDA_CHECK(cudaFree(0));
        CUcontext cudaContext = 0;

        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &optixLogCallBack;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cudaContext, &options, &optixContext));
    }



    // JP: パイプライン中のモジュール、そしてパイプライン自体に共通なコンパイルオプションの設定。
    // EN: Set a pipeline compile options common among modules in the pipeline and the pipeline itself.
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.numPayloadValues = 3;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "plp";
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;

    char log[2048];
    size_t logSize;



    OptixModule module = nullptr;
    {
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

        const std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/kernel.ptx");

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(optixContext, 
                                                 &moduleCompileOptions,
                                                 &pipelineCompileOptions,
                                                 ptx.c_str(), ptx.size(),
                                                 log, &logSize,
                                                 &module));
    }



    std::vector<OptixProgramGroup> programGroups;
    
    OptixProgramGroup pgRayGen = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        progGroupDesc.raygen.module = module;
        progGroupDesc.raygen.entryFunctionName = "__raygen__fill";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgRayGen));

        programGroups.push_back(pgRayGen);
    }

    OptixProgramGroup pgMissSearchRay = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        progGroupDesc.miss.module = module;
        progGroupDesc.miss.entryFunctionName = "__miss__searchRay";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgMissSearchRay));

        programGroups.push_back(pgMissSearchRay);
    }

    // JP: 空のMissプログラム
    // EN: Dummy Miss Program
    OptixProgramGroup pgMissEmpty = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgMissEmpty));

        programGroups.push_back(pgMissEmpty);
    }

    OptixProgramGroup pgHitGroupSearchRay = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        progGroupDesc.hitgroup.moduleCH = module;
        progGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__shading";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgHitGroupSearchRay));

        programGroups.push_back(pgHitGroupSearchRay);
    }

    OptixProgramGroup pgHitGroupVisibilityRay = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        progGroupDesc.hitgroup.moduleAH = module;
        progGroupDesc.hitgroup.entryFunctionNameAH = "__anyhit__visibility";

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgHitGroupVisibilityRay));

        programGroups.push_back(pgHitGroupVisibilityRay);
    }



    OptixPipeline pipeline = nullptr;
    {
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipelineLinkOptions.overrideUsesMotionBlur = false;
        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(optixContext,
                                            &pipelineCompileOptions,
                                            &pipelineLinkOptions,
                                            programGroups.data(), programGroups.size(),
                                            log, &logSize,
                                            &pipeline));
    }

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // JP: デフォルトストリーム
    // EN: default stream
    CUstream stream = 0;
    //CUDA_CHECK(cudaStreamCreate(&stream));



    // ----------------------------------------------------------------
    // JP: シーンデータを読み込み、GPU転送する。
    //     シェーダーバインディングテーブルのセットアップしAcceleration Structureを構築する。
    // EN: Read scene data and transfer it to GPU.
    //     Setup a shader binding table and build acceleration structures.
    
    TriangleMesh meshCornellBox;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(1, 0) },
            // back wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 0) },
            // ceiling
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(1, 0) },
            // left wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 1) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 1) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 0) },
            // right wall
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 0) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
            // back wall
            { 4, 5, 6 }, { 4, 6, 7 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        // JP: 頂点バッファーは共通にしてみる。
        meshCornellBox.setVertexBuffer(vertices, lengthof(vertices));

        Shared::MaterialData mat;
        
        // JP: インデックスバッファーは別々にしてみる。
        // floor, back wall, ceiling
        mat.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
        meshCornellBox.addMaterialGroup(triangles + 0, 6, mat);
        // left wall
        mat.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
        meshCornellBox.addMaterialGroup(triangles + 6, 2, mat);
        // right wall
        mat.albedo = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
        meshCornellBox.addMaterialGroup(triangles + 8, 2, mat);
    }

    TriangleMesh meshAreaLight;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.5f, 0.0f, -0.5f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.5f, 0.0f, 0.5f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.5f, 0.0f, 0.5f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.5f, 0.0f, -0.5f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        meshAreaLight.setVertexBuffer(vertices, lengthof(vertices));

        Shared::MaterialData mat;
        mat.albedo = make_float3(1, 1, 1);
        meshAreaLight.addMaterialGroup(triangles + 0, 2, mat);
    }



    // R: Radiance Ray, V: Visibility Ray
    //
    // | RG   |
    // +------+------+
    // | MS R | MS V |
    // +------+------+------+------+------+------+------+------+
    // |    0 |    1 |    2 |    3 |    4 |    5 |    6 |    7 |
    // +------+------+------+------+------+------+------+------+
    // | HG R | HG V | HG R | HG V | HG R | HG V | HG R | HG V |
    // | b.i. 0-0    | b.i. 1-0    | b.i. 2-0    | b.i. 0-0    |
    // | GAS 0                                   | GAS 1       |



    std::vector<OptixInstance> instances;
    uint32_t sbtOffset = 0;

    AccelerationStructure gasCornellBox;
    {
        const OptixBuildInput* buildInputs;
        uint32_t numBuildInputs;
        uint32_t numSBTRecords;
        meshCornellBox.getBuildInfo(&buildInputs, &numBuildInputs, &numSBTRecords);

        gasCornellBox.initialize(optixContext, buildInputs, numBuildInputs,
                                 true, false, true);

        gasCornellBox.rebuild(stream);
        gasCornellBox.compaction(stream, stream);
        gasCornellBox.removeUncompacted(stream);

        instances.push_back(OptixInstance());
        OptixInstance &inst = instances.back();
        std::memset(&inst, 0, sizeof(inst));
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.instanceId = 0;
        inst.visibilityMask = 0xFF;
        inst.traversableHandle = gasCornellBox.getHandle();
        float tfCornellBox[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0
        };
        std::copy_n(tfCornellBox, 12, inst.transform);
        inst.sbtOffset = sbtOffset;

        sbtOffset += numSBTRecords * Shared::NumRayTypes;
    }

    AccelerationStructure gasLight;
    {
        const OptixBuildInput* buildInputs;
        uint32_t numBuildInputs;
        uint32_t numSBTRecords;
        meshAreaLight.getBuildInfo(&buildInputs, &numBuildInputs, &numSBTRecords);

        gasLight.initialize(optixContext, buildInputs, numBuildInputs,
                            true, false, true);

        gasLight.rebuild(stream);
        gasLight.compaction(stream, stream);
        gasLight.removeUncompacted(stream);

        instances.push_back(OptixInstance());
        OptixInstance &inst = instances.back();
        inst.flags = OPTIX_INSTANCE_FLAG_NONE;
        inst.instanceId = 0;
        inst.visibilityMask = 0xFF;
        inst.traversableHandle = gasLight.getHandle();
        float tfCornellBox[] = {
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0.99f
        };
        std::copy_n(tfCornellBox, 12, inst.transform);
        inst.sbtOffset = sbtOffset;

        sbtOffset += numSBTRecords * Shared::NumRayTypes;
    }

    CUDAHelper::Buffer instanceBuffer;
    instanceBuffer.initialize(CUDAHelper::BufferType::Device, instances.size(), sizeof(instances[0]), 0);
    auto instancesD = (OptixInstance*)instanceBuffer.map();
    std::copy_n(instances.data(), instances.size(), instancesD);
    instanceBuffer.unmap();

    OptixBuildInput iasBuildInput;
    std::memset(&iasBuildInput, 0, sizeof(iasBuildInput));
    iasBuildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    OptixBuildInputInstanceArray &instArray = iasBuildInput.instanceArray;
    instArray.instances = instanceBuffer.getDevicePointer();
    instArray.numInstances = instances.size();

    AccelerationStructure iasScene;
    iasScene.initialize(optixContext, &iasBuildInput, 1, false, true, true);

    iasScene.rebuild(stream);
    iasScene.compaction(stream, stream);
    iasScene.removeUncompacted(stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));



    OptixShaderBindingTable sbt = {};
    {
        CUDAHelper::Buffer rayGenSBTRBuffer;
        rayGenSBTRBuffer.initialize(CUDAHelper::BufferType::Device, 1, sizeof(RayGenSBTRecord), 0);

        MissSBTRecord searchRayMissSBTR;
        OPTIX_CHECK(optixSbtRecordPackHeader(pgMissSearchRay, &searchRayMissSBTR));
        searchRayMissSBTR.data.bgRadiance = make_float3(0, 0, 0.25f);

        MissSBTRecord visibilityRayMissSBTR;
        OPTIX_CHECK(optixSbtRecordPackHeader(pgMissEmpty, &visibilityRayMissSBTR));

        CUDAHelper::Buffer missSBTRBuffer;
        missSBTRBuffer.initialize(CUDAHelper::BufferType::Device, Shared::NumRayTypes, sizeof(MissSBTRecord), 0);
        {
            auto missSBTRs = (MissSBTRecord*)missSBTRBuffer.map();
            missSBTRs[Shared::RayType_Search] = searchRayMissSBTR;
            missSBTRs[Shared::RayType_Visibility] = visibilityRayMissSBTR;
            missSBTRBuffer.unmap();
        }

        CUDAHelper::Buffer hitGroupSBTRBuffer;
        hitGroupSBTRBuffer.initialize(CUDAHelper::BufferType::Device, Shared::NumRayTypes, sizeof(HitGroupSBTRecord), 0);

        sbt.raygenRecord = rayGenSBTRBuffer.getDevicePointer();

        sbt.missRecordBase = missSBTRBuffer.getDevicePointer();
        sbt.missRecordStrideInBytes = sizeof(MissSBTRecord);
        sbt.missRecordCount = missSBTRBuffer.numElements();

        sbt.hitgroupRecordBase = hitGroupSBTRBuffer.getDevicePointer();
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSBTRecord);
        sbt.hitgroupRecordCount = hitGroupSBTRBuffer.numElements();
    }

    // END: Read scene data and transfer it to GPU.
    //      Setup a shader binding table and build acceleration structures.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer object.
    GLTK::Buffer outputBufferGL;
    GLTK::BufferTexture outputTexture;
    CUDAHelper::Buffer outputBufferCUDA;
    outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(float) * 4, renderTargetSizeX * renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);
    outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGBA32F);
    outputBufferCUDA.initialize(CUDAHelper::BufferType::GL_Interop, renderTargetSizeX, renderTargetSizeY, sizeof(float) * 4, outputBufferGL.getRawHandle());


    
    // JP: Hi-DPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
    // EN: Create a low-resolution frame buffer to avoid too much rendering load caused by Hi-DPI display.
    GLTK::FrameBuffer frameBuffer;
    frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    GLTK::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    const filesystem::path exeDir = getExecutableDirectory();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    GLTK::GraphicsShader drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "shaders/drawOptiXResult.frag"));

    // JP: アップスケール用のシェーダー。
    // EN: Shader for upscale.
    GLTK::GraphicsShader scaleShader;
    scaleShader.initializeVSPS(readTxtFile(exeDir / "shaders/scale.vert"),
                               readTxtFile(exeDir / "shaders/scale.frag"));

    // JP: アップスケール用のサンプラー。
    //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
    // EN: Sampler for upscaling.
    //     It seems to require to bind a sampler even when using texelFetch() which is independent from the sampler settings.
    GLTK::Sampler scaleSampler;
    scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



    Shared::PipelineLaunchParameters plp;
    plp.topGroup = 0;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.outputBuffer = (float4*)outputBufferCUDA.getDevicePointer();

    CUdeviceptr plpOnDevice;
    CUDA_CHECK(cudaMalloc((void**)&plpOnDevice, sizeof(plp)));

    RayGenSBTRecord rayGenSBTR;
    OPTIX_CHECK(optixSbtRecordPackHeader(pgRayGen, &rayGenSBTR));
    rayGenSBTR.data.camera.fovY = 60 * M_PI / 180;
    rayGenSBTR.data.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;



    uint64_t frameIndex = 0;
    int32_t requestedSize[2];
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        bool resized = false;
        int32_t newFBWidth;
        int32_t newFBHeight;
        glfwGetFramebufferSize(window, &newFBWidth, &newFBHeight);
        if (newFBWidth != curFBWidth || newFBHeight != curFBHeight) {
            curFBWidth = newFBWidth;
            curFBHeight = newFBHeight;

            renderTargetSizeX = curFBWidth / UIScaling;
            renderTargetSizeY = curFBHeight / UIScaling;
            requestedSize[0] = renderTargetSizeX;
            requestedSize[1] = renderTargetSizeY;

            outputBufferCUDA.finalize();
            outputTexture.finalize();
            outputBufferGL.finalize();
            outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(float) * 4, renderTargetSizeX * renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);
            outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGBA32F);
            outputBufferCUDA.initialize(CUDAHelper::BufferType::GL_Interop, renderTargetSizeX, renderTargetSizeY, sizeof(float) * 4, outputBufferGL.getRawHandle());

            frameBuffer.finalize();
            frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

            // EN: update the pipeline parameters.
            plp.imageSize.x = renderTargetSizeX;
            plp.imageSize.y = renderTargetSizeY;
            plp.outputBuffer = (float4*)outputBufferCUDA.getDevicePointer();

            resized = true;
        }



        CUDA_CHECK(cudaMemcpyAsync((void*)sbt.raygenRecord, &rayGenSBTR, sizeof(rayGenSBTR), cudaMemcpyHostToDevice, stream));

        plp.outputBuffer = (float4*)outputBufferCUDA.beginCUDAAccess(stream);

        CUDA_CHECK(cudaMemcpyAsync((void*)plpOnDevice, &plp, sizeof(plp), cudaMemcpyHostToDevice, stream));
        OPTIX_CHECK(optixLaunch(pipeline, stream, plpOnDevice, sizeof(plp), &sbt, renderTargetSizeX, renderTargetSizeY, 1));

        outputBufferCUDA.endCUDAAccess(stream);



        {
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImGui::ShowDemoWindow();

            // ----------------------------------------------------------------
            // JP: OptiXの出力とImGuiの描画。

            frameBuffer.bind(GLTK::FrameBuffer::Target::ReadDraw);

            glViewport(0, 0, frameBuffer.getWidth(), frameBuffer.getHeight());
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClearDepth(1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            {
                drawOptiXResultShader.useProgram();

                glUniform1i(0, (int32_t)renderTargetSizeX); GLTK::errorCheck();

                glActiveTexture(GL_TEXTURE0); GLTK::errorCheck();
                outputTexture.bind();

                vertexArrayForFullScreen.bind();
                glDrawArrays(GL_TRIANGLES, 0, 3); GLTK::errorCheck();
                vertexArrayForFullScreen.unbind();

                outputTexture.unbind();
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

            frameBuffer.unbind();

            // END: draw OptiX's output and ImGui.
            // ----------------------------------------------------------------
        }

        // ----------------------------------------------------------------
        // JP: スケーリング

        glEnable(GL_FRAMEBUFFER_SRGB);
        GLTK::errorCheck();

        glViewport(0, 0, curFBWidth, curFBHeight);

        scaleShader.useProgram();

        glUniform1f(0, UIScaling);

        glActiveTexture(GL_TEXTURE0);
        GLTK::Texture2D& srcFBTex = frameBuffer.getRenderTargetTexture();
        srcFBTex.bind();
        scaleSampler.bindToTextureUnit(0);

        vertexArrayForFullScreen.bind();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        vertexArrayForFullScreen.unbind();

        srcFBTex.unbind();

        // END: scaling
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);

        ++frameIndex;
    }



    CUDA_CHECK(cudaFree((void*)plpOnDevice));



    scaleSampler.finalize();
    scaleShader.finalize();
    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    frameBuffer.finalize();

    outputBufferCUDA.finalize();
    outputTexture.finalize();
    outputBufferGL.finalize();

    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    OPTIX_CHECK(optixProgramGroupDestroy(pgHitGroupVisibilityRay));
    OPTIX_CHECK(optixProgramGroupDestroy(pgHitGroupSearchRay));
    OPTIX_CHECK(optixProgramGroupDestroy(pgMissSearchRay));
    OPTIX_CHECK(optixProgramGroupDestroy(pgRayGen));

    OPTIX_CHECK(optixModuleDestroy(module));

    OPTIX_CHECK(optixDeviceContextDestroy(optixContext));

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    
    glfwTerminate();

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
    }

    return 0;
}
