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
#include <vector>
#include <filesystem>
#include <random>

#include <GL/gl3w.h>

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "GLToolkit.h"
#include "cuda_helper.h"
#include "optix_util.h"
#include <cuda_runtime.h>

#include "shared.h"
#include "stopwatch.h"

#include "ext/tiny_obj_loader.h"



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



struct SceneContext {
    optix::Scene optixScene;
    CUDAHelper::TypedBuffer<Shared::GeometryData> geometryDataBuffer;
    uint32_t geometryID;
};

class TriangleMesh {
    CUcontext m_cudaContext;
    SceneContext* m_sceneContext;

    struct MaterialGroup {
        CUDAHelper::TypedBuffer<Shared::Triangle>* triangleBuffer;
        optix::Material material;
        optix::GeometryInstance geometryInstance;
    };

    CUDAHelper::TypedBuffer<Shared::Vertex> m_vertexBuffer;
    std::vector<MaterialGroup> m_materialGroups;

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;
public:
    TriangleMesh(CUcontext cudaContext, SceneContext* sceneContext) :
        m_cudaContext(cudaContext), m_sceneContext(sceneContext) {}

    void destroy() {
        for (auto it = m_materialGroups.rbegin(); it != m_materialGroups.rend(); ++it) {
            MaterialGroup &matGroup = *it;
            matGroup.geometryInstance.destroy();
            matGroup.triangleBuffer->finalize();
            delete matGroup.triangleBuffer;
        }
        m_materialGroups.clear();

        m_vertexBuffer.finalize();
    }

    void setVertexBuffer(const Shared::Vertex* vertices, uint32_t numVertices) {
        m_vertexBuffer.initialize(m_cudaContext, CUDAHelper::BufferType::Device, numVertices);
        auto verticesOnHost = m_vertexBuffer.map();
        std::copy_n(vertices, numVertices, verticesOnHost);
        m_vertexBuffer.unmap();
    }

    void addMaterialGroup(const Shared::Triangle* triangles, uint32_t numTriangles, optix::Material &material) {
        m_materialGroups.push_back(MaterialGroup());

        MaterialGroup &group = m_materialGroups.back();

        auto triangleBuffer = new CUDAHelper::TypedBuffer<Shared::Triangle>();
        group.triangleBuffer = triangleBuffer;
        triangleBuffer->initialize(m_cudaContext, CUDAHelper::BufferType::Device, numTriangles);
        Shared::Triangle* trianglesOnHost = triangleBuffer->map();
        std::copy_n(triangles, numTriangles, trianglesOnHost);
        triangleBuffer->unmap();

        group.material = material;

        Shared::GeometryData* geomDataPtr = m_sceneContext->geometryDataBuffer.map();
        Shared::GeometryData &recordData = geomDataPtr[m_sceneContext->geometryID];
        recordData.vertexBuffer = m_vertexBuffer.getDevicePointer();
        recordData.triangleBuffer = triangleBuffer->getDevicePointer();
        m_sceneContext->geometryDataBuffer.unmap();

        optix::GeometryInstance geomInst = m_sceneContext->optixScene.createGeometryInstance();
        geomInst.setVertexBuffer(&m_vertexBuffer);
        geomInst.setTriangleBuffer(triangleBuffer);
        geomInst.setUserData(m_sceneContext->geometryID);
        geomInst.setNumMaterials(1, nullptr);
        geomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInst.setMaterial(0, 0, material);
        ++m_sceneContext->geometryID;

        group.geometryInstance = geomInst;
    }

    void addToGAS(optix::GeometryAccelerationStructure* gas) {
        for (int i = 0; i < m_materialGroups.size(); ++i)
            gas->addChild(m_materialGroups[i].geometryInstance);
    }
};



static std::filesystem::path getExecutableDirectory() {
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

static std::string readTxtFile(const std::filesystem::path& filepath) {
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
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUstream cuStream[2];
    CUDAHelper::Timer cudaRenderTimer[2];
    CUDAHelper::Timer cudaPostProcessTimer[2];
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[0], 0));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[1], 0));
    cudaRenderTimer[0].initialize(cuContext);
    cudaRenderTimer[1].initialize(cuContext);
    cudaPostProcessTimer[0].initialize(cuContext);
    cudaPostProcessTimer[1].initialize(cuContext);
    
    optix::Context optixContext = optix::Context::create(cuContext);

    optix::Pipeline pipeline = optixContext.createPipeline();

    pipeline.setPipelineOptions(6, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/kernel.ptx");
    optix::Module moduleOptiX = pipeline.createModuleFromPTXString(ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                                                                   OPTIX_COMPILE_OPTIMIZATION_DEFAULT, OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO);

    optix::Module emptyModule;

    optix::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, "__raygen__pathtracing");
    //optix::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optix::ProgramGroup searchRayMissProgram = pipeline.createMissProgram(moduleOptiX, "__miss__searchRay");
    optix::ProgramGroup visibilityRayMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optix::ProgramGroup searchRayHitProgramGroup = pipeline.createHitProgramGroup(moduleOptiX, "__closesthit__shading", emptyModule, nullptr, emptyModule, nullptr);
    optix::ProgramGroup visibilityRayHitProgramGroup = pipeline.createHitProgramGroup(emptyModule, nullptr, moduleOptiX, "__anyhit__visibility", emptyModule, nullptr);

    pipeline.setMaxTraceDepth(2);
    pipeline.link(OPTIX_COMPILE_DEBUG_LEVEL_FULL, false);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, searchRayMissProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, visibilityRayMissProgram);

    CUmodule modulePostProcess;
    CUDADRV_CHECK(cuModuleLoad(&modulePostProcess, (getExecutableDirectory() / "ptxes/post_process.ptx").string().c_str()));
    CUfunction kernelPostProcess;
    CUDADRV_CHECK(cuModuleGetFunction(&kernelPostProcess, modulePostProcess, "postProcess"));

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    CUDAHelper::TypedBuffer<Shared::MaterialData> materialDataBuffer;
    materialDataBuffer.initialize(cuContext, CUDAHelper::BufferType::Device, 128);
    uint32_t materialID = 0;

    Shared::MaterialData* matData = materialDataBuffer.map();

    uint32_t matGrayWallIndex = materialID++;
    optix::Material matGray = optixContext.createMaterial();
    matGray.setHitGroup(Shared::RayType_Search, searchRayHitProgramGroup);
    matGray.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matGray.setUserData(matGrayWallIndex);
    Shared::MaterialData matGrayWallData;
    matGrayWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    matData[matGrayWallIndex] = matGrayWallData;

    uint32_t matLeftWallIndex = materialID++;
    optix::Material matLeft = optixContext.createMaterial();
    matLeft.setHitGroup(Shared::RayType_Search, searchRayHitProgramGroup);
    matLeft.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matLeft.setUserData(matLeftWallIndex);
    Shared::MaterialData matLeftWallData;
    matLeftWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    matData[matLeftWallIndex] = matLeftWallData;

    uint32_t matRightWallIndex = materialID++;
    optix::Material matRight = optixContext.createMaterial();
    matRight.setHitGroup(Shared::RayType_Search, searchRayHitProgramGroup);
    matRight.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matRight.setUserData(matRightWallIndex);
    Shared::MaterialData matRightWallData;
    matRightWallData.albedo = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    matData[matRightWallIndex] = matRightWallData;

    uint32_t matLightIndex = materialID++;
    optix::Material matLight = optixContext.createMaterial();
    matLight.setHitGroup(Shared::RayType_Search, searchRayHitProgramGroup);
    matLight.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matLight.setUserData(matLightIndex);
    Shared::MaterialData matLightData;
    matLightData.albedo = make_float3(1, 1, 1);
    matData[matLightIndex] = matLightData;

    materialDataBuffer.unmap();



    optix::Scene scene = optixContext.createScene();
    
    SceneContext sceneContext;
    sceneContext.optixScene = scene;
    sceneContext.geometryDataBuffer.initialize(cuContext, CUDAHelper::BufferType::Device, 128);
    sceneContext.geometryID = 0;
    
    TriangleMesh meshCornellBox(cuContext, &sceneContext);
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
        // EN: Share the vertex buffer among walls.
        meshCornellBox.setVertexBuffer(vertices, lengthof(vertices));

        Shared::MaterialData mat;
        
        // JP: インデックスバッファーは別々にしてみる。
        // EN: Use separated index buffers among walls.
        // floor, back wall, ceiling
        meshCornellBox.addMaterialGroup(triangles + 0, 6, matGray);
        // left wall
        meshCornellBox.addMaterialGroup(triangles + 6, 2, matLeft);
        // right wall
        meshCornellBox.addMaterialGroup(triangles + 8, 2, matRight);
    }

    TriangleMesh meshAreaLight(cuContext, &sceneContext);
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        meshAreaLight.setVertexBuffer(vertices, lengthof(vertices));

        meshAreaLight.addMaterialGroup(triangles + 0, 2, matLight);
    }

    TriangleMesh meshObject(cuContext, &sceneContext);
    std::vector<Shared::Vertex> objectVertices;
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "data/subd_cube.obj");

        constexpr float scale = 0.3f;
        
        // Record unified unique vertices.
        std::map<std::tuple<int32_t, int32_t>, Shared::Vertex> unifiedVertexMap;
        for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = shapes[sIdx];
            size_t idxOffset = 0;
            for (int fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                for (int vIdx = 0; vIdx < numFaceVertices; ++vIdx) {
                    tinyobj::index_t idx = shape.mesh.indices[idxOffset + vIdx];
                    auto key = std::make_tuple(idx.vertex_index, idx.normal_index);
                    unifiedVertexMap[key] = Shared::Vertex{
                        make_float3(scale * attrib.vertices[3 * idx.vertex_index + 0],
                        scale * attrib.vertices[3 * idx.vertex_index + 1],
                        scale * attrib.vertices[3 * idx.vertex_index + 2]),
                        make_float3(0, 0, 0),
                        make_float2(0, 0)
                    };
                }

                idxOffset += numFaceVertices;
            }
        }

        // Assign a vertex index to each of unified unique unifiedVertexMap.
        std::map<std::tuple<int32_t, int32_t>, uint32_t> vertexIndices;
        objectVertices.resize(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            objectVertices[vertexIndex] = kv.second;
            vertexIndices[kv.first] = vertexIndex++;
        }

        // Calculate triangle index buffer.
        std::vector<Shared::Triangle> triangles;
        for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = shapes[sIdx];
            size_t idxOffset = 0;
            for (int fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                tinyobj::index_t idx0 = shape.mesh.indices[idxOffset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[idxOffset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[idxOffset + 2];
                auto key0 = std::make_tuple(idx0.vertex_index, idx0.normal_index);
                auto key1 = std::make_tuple(idx1.vertex_index, idx1.normal_index);
                auto key2 = std::make_tuple(idx2.vertex_index, idx2.normal_index);

                triangles.push_back(Shared::Triangle{
                    vertexIndices.at(key0),
                    vertexIndices.at(key1),
                    vertexIndices.at(key2) });

                idxOffset += numFaceVertices;
            }
        }

        for (int tIdx = 0; tIdx < triangles.size(); ++tIdx) {
            const Shared::Triangle &tri = triangles[tIdx];
            Shared::Vertex &v0 = objectVertices[tri.index0];
            Shared::Vertex &v1 = objectVertices[tri.index1];
            Shared::Vertex &v2 = objectVertices[tri.index2];
            float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
            v0.normal = v0.normal + gn;
            v1.normal = v1.normal + gn;
            v2.normal = v2.normal + gn;
        }
        for (int vIdx = 0; vIdx < objectVertices.size(); ++vIdx) {
            Shared::Vertex &v = objectVertices[vIdx];
            v.normal = normalize(v.normal);
        }

        meshObject.setVertexBuffer(objectVertices.data(), objectVertices.size());

        meshObject.addMaterialGroup(triangles.data(), triangles.size(), matGray);
    }



    uint32_t travID = 0;
    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    CUDAHelper::Buffer asBuildScratchMem;
    CUDAHelper::TypedBuffer<OptixTraversableHandle> travHandleBuffer;
    travHandleBuffer.initialize(cuContext, CUDAHelper::BufferType::Device, 128);
    OptixTraversableHandle* travHandles = travHandleBuffer.map();

    // JP: コーネルボックスと面光源にサンプルとして敢えて別々のGASを使う。
    // EN: Use different GAS for the Cornell box and the area light
    //     on purpose as sample.
    
    uint32_t gasCornellBoxIndex = travID++;
    optix::GeometryAccelerationStructure gasCornellBox = scene.createGeometryAccelerationStructure();
    CUDAHelper::Buffer gasCornellBoxMem;
    gasCornellBox.setConfiguration(true, false, false);
    gasCornellBox.setNumMaterialSets(1);
    gasCornellBox.setNumRayTypes(0, Shared::NumRayTypes);
    meshCornellBox.addToGAS(&gasCornellBox);
    gasCornellBox.prepareForBuild(&asMemReqs);
    gasCornellBoxMem.initialize(cuContext, CUDAHelper::BufferType::Device, asMemReqs.outputSizeInBytes, 1, 0);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    uint32_t gasAreaLightIndex = travID++;
    optix::GeometryAccelerationStructure gasAreaLight = scene.createGeometryAccelerationStructure();
    CUDAHelper::Buffer gasAreaLightMem;
    gasAreaLight.setConfiguration(true, false, false);
    gasAreaLight.setNumMaterialSets(1);
    gasAreaLight.setNumRayTypes(0, Shared::NumRayTypes);
    meshAreaLight.addToGAS(&gasAreaLight);
    gasAreaLight.prepareForBuild(&asMemReqs);
    gasAreaLightMem.initialize(cuContext, CUDAHelper::BufferType::Device, asMemReqs.outputSizeInBytes, 1, 0);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    uint32_t gasObjectIndex = travID++;
    optix::GeometryAccelerationStructure gasObject = scene.createGeometryAccelerationStructure();
    CUDAHelper::Buffer gasObjectMem;
    gasObject.setConfiguration(false, true, false);
    gasObject.setNumMaterialSets(1);
    gasObject.setNumRayTypes(0, Shared::NumRayTypes);
    meshObject.addToGAS(&gasObject);
    gasObject.prepareForBuild(&asMemReqs);
    gasObjectMem.initialize(cuContext, CUDAHelper::BufferType::Device, asMemReqs.outputSizeInBytes, 1, 0);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    //     スクラッチバッファーは共用する。
    // EN: Build geometry acceleration structures.
    //     Share the scratch buffer among them.
    asBuildScratchMem.initialize(cuContext, CUDAHelper::BufferType::Device, maxSizeOfScratchBuffer, 1, 0);
    travHandles[gasCornellBoxIndex] = gasCornellBox.rebuild(cuStream[0], gasCornellBoxMem, asBuildScratchMem);
    travHandles[gasAreaLightIndex] = gasAreaLight.rebuild(cuStream[0], gasAreaLightMem, asBuildScratchMem);
    travHandles[gasObjectIndex] = gasObject.rebuild(cuStream[0], gasObjectMem, asBuildScratchMem);



    uint32_t iasSceneIndex = travID++;
    optix::InstanceAccelerationStructure iasScene = scene.createInstanceAccelerationStructure();
    CUDAHelper::Buffer iasSceneMem;
    iasScene.setConfiguration(false, true, true);
    iasScene.addChild(gasCornellBox);
    float tfAreaLight[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.99f,
        0, 0, 1, 0
    };
    iasScene.addChild(gasAreaLight, 0, tfAreaLight);
    float tfObject[] = {
        1, 0, 0, 0,
        0, 1, 0, -0.5f,
        0, 0, 1, 0
    };
    iasScene.addChild(gasObject, 0, tfObject);
    iasScene.prepareForBuild(&asMemReqs);
    iasSceneMem.initialize(cuContext, CUDAHelper::BufferType::Device, asMemReqs.outputSizeInBytes, 1, 0);
    if (asMemReqs.tempSizeInBytes >= asBuildScratchMem.sizeInBytes()) {
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);
    }

    // JP: Instance Acceleration Structureをビルドする。
    // EN: Build the instance acceleration structure.
    travHandles[iasSceneIndex] = iasScene.rebuild(cuStream[0], iasSceneMem, asBuildScratchMem);

    travHandleBuffer.unmap();
    CUDADRV_CHECK(cuStreamSynchronize(cuStream[0]));
    asBuildScratchMem.finalize();



    pipeline.setScene(scene);

    // END: 
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer object.
    GLTK::Buffer outputBufferGL;
    GLTK::BufferTexture outputTexture;
    CUDAHelper::Buffer outputBufferCUDA;
    outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, sizeof(float) * 4, renderTargetSizeX * renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);
    outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGBA32F);
    outputBufferCUDA.initialize(cuContext, CUDAHelper::BufferType::GL_Interop, renderTargetSizeX * renderTargetSizeY, sizeof(float) * 4, outputBufferGL.getRawHandle());


    
    // JP: Hi-DPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
    // EN: Create a low-resolution frame buffer to avoid too much rendering load caused by Hi-DPI display.
    GLTK::FrameBuffer frameBuffer;
    frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_SRGB8, GL_DEPTH_COMPONENT32);
    // sRGB8を指定しないとなぜか精度問題が発生したが、むしろRGB8が本来なら正しい気がする。



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    GLTK::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    const std::filesystem::path exeDir = getExecutableDirectory();

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



    CUDAHelper::TypedBuffer<Shared::PCG32RNG> rngBuffer;
    rngBuffer.initialize(cuContext, CUDAHelper::BufferType::Device, renderTargetSizeX * renderTargetSizeY);
    const auto initializeRNGSeeds = [](CUDAHelper::Buffer &buffer) {
        std::mt19937_64 rng(591842031321323413);

        auto seeds = buffer.map<uint64_t>();
        for (int i = 0; i < buffer.numElements(); ++i)
            seeds[i] = rng();
        buffer.unmap();
    };
    initializeRNGSeeds(rngBuffer);

    CUDAHelper::TypedBuffer<float4> accumBuffer;
    accumBuffer.initialize(cuContext, CUDAHelper::BufferType::Device, renderTargetSizeX * renderTargetSizeY);



    Shared::PipelineLaunchParameters plp;
    plp.travHandles = travHandleBuffer.getDevicePointer();
    plp.materialData = materialDataBuffer.getDevicePointer();
    plp.geomInstData = sceneContext.geometryDataBuffer.getDevicePointer();
    plp.topGroupIndex = iasSceneIndex;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.numAccumFrames = 1;
    plp.rngBuffer = rngBuffer.getDevicePointer();
    plp.accumBuffer = accumBuffer.getDevicePointer();
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    StopWatch<> sw;
    std::mt19937_64 rng(3092384202);
    std::uniform_real_distribution u01;
    
    uint64_t frameIndex = 0;
    int32_t requestedSize[2];
    float frameTime = 0;
    float renderCmdTime = 0;
    float postProcessCmdTime = 0;
    float syncTime = 0;
    float swapTime = 0;
    float dummyTime = 0;
    while (!glfwWindowShouldClose(window)) {
        sw.start();

        uint32_t bufferIndex = frameIndex % 2;

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
            outputBufferCUDA.initialize(cuContext, CUDAHelper::BufferType::GL_Interop, renderTargetSizeX * renderTargetSizeY, sizeof(float) * 4, outputBufferGL.getRawHandle());

            frameBuffer.finalize();
            frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_SRGB8, GL_DEPTH_COMPONENT32);

            accumBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            rngBuffer.resize(renderTargetSizeX * renderTargetSizeY);
            initializeRNGSeeds(rngBuffer);

            // EN: update the pipeline parameters.
            plp.imageSize.x = renderTargetSizeX;
            plp.imageSize.y = renderTargetSizeY;
            plp.numAccumFrames = 1;
            plp.rngBuffer = rngBuffer.getDevicePointer();
            plp.accumBuffer = accumBuffer.getDevicePointer();
            plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        CUstream &curCuStream = cuStream[bufferIndex];
        CUDAHelper::Timer &curRenderTimer = cudaRenderTimer[bufferIndex];
        CUDAHelper::Timer &curPostProcessTimer = cudaPostProcessTimer[bufferIndex];
        
        // JP: 前フレームの処理が完了するのを待つ。
        // EN: Wait the previous frame processing to finish.
        sw.start();
        CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
        syncTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;

        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float renderTime = frameIndex >= 2 ? curRenderTimer.report() : 0.0f;
            float postProcessTime = frameIndex >= 2 ? curPostProcessTimer.report() : 0.0f;
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("GPU:");
            ImGui::Text("  Render: %.2f [ms]", renderTime);
            ImGui::Text("  Post Process: %.2f [ms]", postProcessTime);
            {
                static float times[100];
                constexpr uint32_t numTimes = lengthof(times);
                constexpr uint32_t numAccums = 1;
                static float accTime = 0;
                static uint32_t plotStartPos = -1;
                accTime += (renderTime + postProcessTime);
                if ((frameIndex + 1) % numAccums == 0) {
                    plotStartPos = (plotStartPos + 1) % numTimes;
                    times[(plotStartPos + numTimes - 1) % numTimes] = accTime / numAccums;
                    accTime = 0;
                }
                ImGui::PlotLines("GPU Time", times, numTimes, plotStartPos, nullptr, 0, 50.0f, ImVec2(0, 50));
            }

            ImGui::Text("CPU:");
            ImGui::Text("  Frame: %.3f [ms]", frameTime);
            ImGui::Text("  Sync: %.3f [ms]", syncTime);
            ImGui::Text("  Render: %.3f [ms]", renderCmdTime);
            ImGui::Text("  Post Process: %.3f [ms]", postProcessCmdTime);
            ImGui::Text("  Swap: %.3f [ms]", swapTime);
            ImGui::Text("  Dummy: %.3f [ms]", dummyTime);
            {
                static float times[100];
                constexpr uint32_t numTimes = lengthof(times);
                constexpr uint32_t numAccums = 1;
                static float accTime = 0;
                static uint32_t plotStartPos = -1;
                accTime += frameTime;
                if ((frameIndex + 1) % numAccums == 0) {
                    plotStartPos = (plotStartPos + 1) % numTimes;
                    times[(plotStartPos + numTimes - 1) % numTimes] = accTime / numAccums;
                    accTime = 0;
                }
                ImGui::PlotLines("CPU Time", times, numTimes, plotStartPos, nullptr, 0, 50.0f, ImVec2(0, 50));
            }
            {
                static float times[100];
                constexpr uint32_t numTimes = lengthof(times);
                static uint32_t plotStartPos = -1;
                plotStartPos = (plotStartPos + 1) % numTimes;
                times[(plotStartPos + numTimes - 1) % numTimes] = std::sinf(2 * M_PI * (frameIndex % 60) / 60.0f);
                ImGui::PlotLines("Sin Curve", times, numTimes, plotStartPos, nullptr, -1.0f, 1.0f, ImVec2(0, 50));
            }

            static float cpuDummyLoad = 0.0f;
            ImGui::SliderFloat("Dummy CPU Load", &cpuDummyLoad, 0.0f, 33.3333f);
            static float dummyProb = 0.0f;
            ImGui::SliderFloat("Probability", &dummyProb, 0.0f, 100.0f);
            sw.start();
            if (cpuDummyLoad > 0.0f && u01(rng) < dummyProb * 0.01f)
                Sleep(cpuDummyLoad);
            dummyTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;

            ImGui::End();

            if (frameIndex >= 1 && frameIndex <= 10) {
                hpprintf("CPU Time Frame %u: %.3f [ms]\n", frameIndex - 1, frameTime);
                hpprintf("  Sync: %.3f [ms]\n", syncTime);
                hpprintf("  Render: %.3f [ms]\n", renderCmdTime);
                hpprintf("  Post Process: %.3f [ms]\n", postProcessCmdTime);
                hpprintf("  Swap: %.3f [ms]\n", swapTime);
                hpprintf("  Dummy: %.3f [ms]\n", dummyTime);
            }
        }

        bool sceneEdited = false;
        {
            ImGui::Begin("Materials", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::ColorEdit3("Left Wall", reinterpret_cast<float*>(&matLeftWallData.albedo),
                                  ImGuiColorEditFlags_DisplayHSV |
                                  ImGuiColorEditFlags_Float)) {
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matLeftWallIndex),
                                                &matLeftWallData, sizeof(matLeftWallData),
                                                curCuStream));
                sceneEdited = true;
            }
            if (ImGui::ColorEdit3("Right Wall", reinterpret_cast<float*>(&matRightWallData.albedo),
                                  ImGuiColorEditFlags_DisplayHSV |
                                  ImGuiColorEditFlags_Float)) {
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matRightWallIndex),
                                                &matRightWallData, sizeof(matRightWallData),
                                                curCuStream));
                sceneEdited = true;
            }
            if (ImGui::ColorEdit3("Other Walls", reinterpret_cast<float*>(&matGrayWallData.albedo),
                                  ImGuiColorEditFlags_DisplayHSV |
                                  ImGuiColorEditFlags_Float)) {
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matGrayWallIndex),
                                                &matGrayWallData, sizeof(matGrayWallData),
                                                curCuStream));
                sceneEdited = true;
            }

            ImGui::End();
        }

        if (sceneEdited)
            plp.numAccumFrames = 1;

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curCuStream));

        // Render
        sw.start();
        curRenderTimer.start(curCuStream);
        pipeline.launch(curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curRenderTimer.stop(curCuStream);
        renderCmdTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;

        // Post Process
        sw.start();
        curPostProcessTimer.start(curCuStream);
        const uint32_t blockSize = 8;
        uint32_t dimX = (renderTargetSizeX + blockSize - 1) / blockSize;
        uint32_t dimY = (renderTargetSizeY + blockSize - 1) / blockSize;
        CUDAHelper::callKernel(curCuStream, kernelPostProcess, dim3(dimX, dimY), dim3(blockSize, blockSize), 0,
                               accumBuffer.getDevicePointer(), renderTargetSizeX, renderTargetSizeY, plp.numAccumFrames,
                               outputBufferCUDA.beginCUDAAccess(curCuStream));
        outputBufferCUDA.endCUDAAccess(curCuStream);
        curPostProcessTimer.stop(curCuStream);
        postProcessCmdTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;
        ++plp.numAccumFrames;



        {
            ImGui::Render();

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

        sw.start();
        glfwSwapBuffers(window);
        swapTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;

        ++frameIndex;
        frameTime = sw.getMeasurement(sw.stop(), StopWatch<>::DurationType::Microseconds) * 1e-3f;
        sw.clearAllMeasurements();
    }



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();
    rngBuffer.finalize();
    
    scaleSampler.finalize();
    scaleShader.finalize();
    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    frameBuffer.finalize();

    outputBufferCUDA.finalize();
    outputTexture.finalize();
    outputBufferGL.finalize();

    travHandleBuffer.finalize();

    iasSceneMem.finalize();
    iasScene.destroy();

    gasObjectMem.finalize();
    gasObject.destroy();
    gasAreaLightMem.finalize();
    gasAreaLight.destroy();
    gasCornellBoxMem.finalize();
    gasCornellBox.destroy();

    meshObject.destroy();
    meshAreaLight.destroy();
    meshCornellBox.destroy();

    sceneContext.geometryDataBuffer.finalize();

    scene.destroy();

    matLight.destroy();
    matRight.destroy();
    matLeft.destroy();
    matGray.destroy();

    materialDataBuffer.finalize();

    CUDADRV_CHECK(cuModuleUnload(modulePostProcess));

    visibilityRayHitProgramGroup.destroy();
    searchRayHitProgramGroup.destroy();

    visibilityRayMissProgram.destroy();
    searchRayMissProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    cudaPostProcessTimer[1].finalize();
    cudaPostProcessTimer[0].finalize();
    cudaRenderTimer[1].finalize();
    cudaRenderTimer[0].finalize();
    CUDADRV_CHECK(cuStreamDestroy(cuStream[1]));
    CUDADRV_CHECK(cuStreamDestroy(cuStream[0]));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

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
