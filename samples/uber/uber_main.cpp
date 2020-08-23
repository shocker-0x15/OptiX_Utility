// This sample is a spaghetti test place for the author.
// No specific theme, No clean code. Someday, may disappears.

#include "uber_shared.h"

// Include glfw3.h after our OpenGL definitions
#include "../common/GLToolkit.h"
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include "../../ext/tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"
#include "../common/dds_loader.h"



struct KeyState {
    uint64_t timesLastChanged[5];
    bool statesLastChanged[5];
    uint32_t lastIndex;

    KeyState() : lastIndex(0) {
        for (int i = 0; i < 5; ++i) {
            timesLastChanged[i] = 0;
            statesLastChanged[i] = false;
        }
    }

    void recordStateChange(bool state, uint64_t time) {
        bool lastState = statesLastChanged[lastIndex];
        if (state == lastState)
            return;

        lastIndex = (lastIndex + 1) % 5;
        statesLastChanged[lastIndex] = !lastState;
        timesLastChanged[lastIndex] = time;
    }

    bool getState(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return statesLastChanged[(lastIndex + goBack + 5) % 5];
    }

    uint64_t getTime(int32_t goBack = 0) const {
        Assert(goBack >= -4 && goBack <= 0, "goBack must be in the range [-4, 0].");
        return timesLastChanged[(lastIndex + goBack + 5) % 5];
    }
};

KeyState g_keyForward;
KeyState g_keyBackward;
KeyState g_keyLeftward;
KeyState g_keyRightward;
KeyState g_keyUpward;
KeyState g_keyDownward;
KeyState g_keyTiltLeft;
KeyState g_keyTiltRight;
KeyState g_keyFasterPosMovSpeed;
KeyState g_keySlowerPosMovSpeed;
KeyState g_buttonRotate;
double g_mouseX;
double g_mouseY;

float g_cameraPositionalMovingSpeed;
float g_cameraDirectionalMovingSpeed;
float g_cameraTiltSpeed;
Quaternion g_cameraOrientation;
Quaternion g_tempCameraOrientation;
float3 g_cameraPosition;



constexpr cudau::BufferType g_bufferType = cudau::BufferType::Device;

struct SceneContext {
    optixu::Scene optixScene;
    Shared::ProgDecodeHitPoint decodeHitPointTriangle;
    cudau::TypedBuffer<Shared::GeometryData> geometryDataBuffer;
    uint32_t geometryID;
};

class TriangleMesh {
    CUcontext m_cuContext;
    SceneContext* m_sceneContext;

    struct MaterialGroup {
        cudau::TypedBuffer<Shared::Triangle>* triangleBuffer;
        optixu::Material material;
        optixu::GeometryInstance geometryInstance;
    };

    cudau::TypedBuffer<Shared::Vertex> m_vertexBuffer;
    std::vector<MaterialGroup> m_materialGroups;

    TriangleMesh(const TriangleMesh &) = delete;
    TriangleMesh &operator=(const TriangleMesh &) = delete;
public:
    TriangleMesh(CUcontext cudaContext, SceneContext* sceneContext) :
        m_cuContext(cudaContext), m_sceneContext(sceneContext) {}

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
        m_vertexBuffer.initialize(m_cuContext, g_bufferType, numVertices);
        m_vertexBuffer.transfer(vertices, numVertices);
    }

    const cudau::TypedBuffer<Shared::Vertex> &getVertexBuffer() const {
        return m_vertexBuffer;
    }

    uint32_t addMaterialGroup(const Shared::Triangle* triangles, uint32_t numTriangles, optixu::Material &material) {
        m_materialGroups.push_back(MaterialGroup());

        MaterialGroup &group = m_materialGroups.back();

        auto triangleBuffer = new cudau::TypedBuffer<Shared::Triangle>();
        group.triangleBuffer = triangleBuffer;
        triangleBuffer->initialize(m_cuContext, g_bufferType, numTriangles);
        triangleBuffer->transfer(triangles, numTriangles);

        group.material = material;

        Shared::GeometryData* geomDataPtr = m_sceneContext->geometryDataBuffer.map();
        Shared::GeometryData &recordData = geomDataPtr[m_sceneContext->geometryID];
        recordData.vertexBuffer = m_vertexBuffer.getDevicePointer();
        recordData.triangleBuffer = triangleBuffer->getDevicePointer();
        recordData.decodeHitPointFunc = m_sceneContext->decodeHitPointTriangle;
        m_sceneContext->geometryDataBuffer.unmap();

        optixu::GeometryInstance geomInst = m_sceneContext->optixScene.createGeometryInstance();
        geomInst.setVertexBuffer(&m_vertexBuffer);
        geomInst.setTriangleBuffer(triangleBuffer);
        geomInst.setUserData(m_sceneContext->geometryID);
        geomInst.setNumMaterials(1, nullptr);
        geomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInst.setMaterial(0, 0, material);
        ++m_sceneContext->geometryID;

        group.geometryInstance = geomInst;

        return static_cast<uint32_t>(m_materialGroups.size()) - 1;
    }

    void setMatrial(uint32_t matSetIdx, uint32_t matGroupIdx, optixu::Material &material) {
        MaterialGroup &group = m_materialGroups[matGroupIdx];
        group.geometryInstance.setMaterial(matSetIdx, 0, material);
    }

    const cudau::TypedBuffer<Shared::Triangle> &getTriangleBuffer(uint32_t matGroupIdx) const {
        return *m_materialGroups[matGroupIdx].triangleBuffer;
    }

    void addToGAS(optixu::GeometryAccelerationStructure* gas) {
        for (int i = 0; i < m_materialGroups.size(); ++i)
            gas->addChild(m_materialGroups[i].geometryInstance);
    }
};



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



int32_t main(int32_t argc, const char* argv[]) try {
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
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(window, [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

        switch (button) {
        case GLFW_MOUSE_BUTTON_MIDDLE: {
            devPrintf("Mouse Middle\n");
            g_buttonRotate.recordStateChange(action == GLFW_PRESS, frameIndex);
            break;
        }
        default:
            break;
        }
                               });
    glfwSetCursorPosCallback(window, [](GLFWwindow* window, double x, double y) {
        g_mouseX = x;
        g_mouseY = y;
                             });
    glfwSetKeyCallback(window, [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
        uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);
        ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);

        switch (key) {
        case GLFW_KEY_W: {
            g_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_S: {
            g_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_A: {
            g_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_D: {
            g_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_R: {
            g_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_F: {
            g_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_Q: {
            g_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_E: {
            g_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_T: {
            g_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        case GLFW_KEY_G: {
            g_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, frameIndex);
            break;
        }
        default:
            break;
        }
                       });

    g_cameraPositionalMovingSpeed = 0.01f;
    g_cameraDirectionalMovingSpeed = 0.0015f;
    g_cameraTiltSpeed = 0.025f;
    g_cameraPosition = make_float3(0, 0, 3.2f);
    g_cameraOrientation = qRotateY(M_PI);

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    hpprintf("Setup OptiX context and pipeline.\n");

    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer deform;
        cudau::Timer updateGAS;
        cudau::Timer updateIAS;
        cudau::Timer render;
        cudau::Timer postProcess;
        bool animated;

        void initialize(CUcontext context) {
            frame.initialize(context);
            deform.initialize(context);
            updateGAS.initialize(context);
            updateIAS.initialize(context);
            render.initialize(context);
            postProcess.initialize(context);
            animated = false;
        }
        void finalize() {
            postProcess.finalize();
            render.finalize();
            updateIAS.finalize();
            updateGAS.finalize();
            deform.finalize();
            frame.finalize();
        }
    };

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUstream cuStream[2];
    GPUTimer gpuTimer[2];
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[0], 0));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[1], 0));
    gpuTimer[0].initialize(cuContext);
    gpuTimer[1].initialize(cuContext);

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    pipeline.setPipelineOptions(6, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                                DEBUG_SELECT((OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                              OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                              OPTIX_EXCEPTION_FLAG_DEBUG),
                                             OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "uber/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("pathtracing"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup searchRayMissProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("searchRay"));
    optixu::ProgramGroup visibilityRayMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    // JP: これらのグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: These are for ray-triangle hit groups, so we don't need custom intersection program.
    optixu::ProgramGroup searchRayDiffuseHitProgramGroup = pipeline.createHitProgramGroup(moduleOptiX, RT_CH_NAME_STR("shading_diffuse"), emptyModule, nullptr, emptyModule, nullptr);
    optixu::ProgramGroup searchRaySpecularHitProgramGroup = pipeline.createHitProgramGroup(moduleOptiX, RT_CH_NAME_STR("shading_specular"), emptyModule, nullptr, emptyModule, nullptr);
    optixu::ProgramGroup visibilityRayHitProgramGroup = pipeline.createHitProgramGroup(emptyModule, nullptr, moduleOptiX, RT_AH_NAME_STR("visibility"), emptyModule, nullptr);

    // JP: これらのグループはレイとカスタムプリミティブの交差判定用なのでIntersectionプログラムを渡す必要がある。
    // EN: These are for ray-custom primitive hit groups, so we need a custom intersection program.
    optixu::ProgramGroup searchRayDiffuseCustomHitProgramGroup = pipeline.createHitProgramGroup(moduleOptiX, RT_CH_NAME_STR("shading_diffuse"),
                                                                                                emptyModule, nullptr,
                                                                                                moduleOptiX, RT_IS_NAME_STR("custom_primitive"));
    optixu::ProgramGroup visibilityRayCustomHitProgramGroup = pipeline.createHitProgramGroup(emptyModule, nullptr,
                                                                                             moduleOptiX, RT_AH_NAME_STR("visibility"),
                                                                                             moduleOptiX, RT_IS_NAME_STR("custom_primitive"));

    uint32_t nextCallableProgramIndex = 0;
    uint32_t callableProgramSampleTextureIndex = nextCallableProgramIndex++;
    optixu::ProgramGroup callableProgramSampleTexture = pipeline.createCallableProgramGroup(moduleOptiX, RT_DC_NAME_STR("sampleTexture"), emptyModule, nullptr);
    uint32_t callableProgramDecodeHitPointTriangleIndex = nextCallableProgramIndex++;
    optixu::ProgramGroup callableProgramDecodeHitPointTriangle = pipeline.createCallableProgramGroup(moduleOptiX, RT_DC_NAME_STR("decodeHitPointTriangle"), emptyModule, nullptr);
    uint32_t callableProgramDecodeHitPointSphereIndex = nextCallableProgramIndex++;
    optixu::ProgramGroup callableProgramDecodeHitPointSphere = pipeline.createCallableProgramGroup(moduleOptiX, RT_DC_NAME_STR("decodeHitPointSphere"), emptyModule, nullptr);

    pipeline.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, searchRayMissProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, visibilityRayMissProgram);

    pipeline.setNumCallablePrograms(nextCallableProgramIndex);
    pipeline.setCallableProgram(callableProgramSampleTextureIndex, callableProgramSampleTexture);
    pipeline.setCallableProgram(callableProgramDecodeHitPointTriangleIndex, callableProgramDecodeHitPointTriangle);
    pipeline.setCallableProgram(callableProgramDecodeHitPointSphereIndex, callableProgramDecodeHitPointSphere);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    pipeline.setShaderBindingTable(&shaderBindingTable);

    OptixStackSizes stackSizes;

    rayGenProgram.getStackSize(&stackSizes);
    uint32_t cssRG = stackSizes.cssRG;

    searchRayMissProgram.getStackSize(&stackSizes);
    uint32_t cssMS = stackSizes.cssMS;

    searchRayDiffuseHitProgramGroup.getStackSize(&stackSizes);
    uint32_t cssCHSearchRayDiffuse = stackSizes.cssCH;
    searchRaySpecularHitProgramGroup.getStackSize(&stackSizes);
    uint32_t cssCHSearchRaySpecular = stackSizes.cssCH;
    visibilityRayHitProgramGroup.getStackSize(&stackSizes);
    uint32_t cssAHVisibilityRay = stackSizes.cssAH;

    uint32_t dssDC = 0;
    callableProgramSampleTexture.getStackSize(&stackSizes);
    dssDC = std::max(dssDC, stackSizes.dssDC);
    callableProgramDecodeHitPointTriangle.getStackSize(&stackSizes);
    dssDC = std::max(dssDC, stackSizes.dssDC);
    callableProgramDecodeHitPointSphere.getStackSize(&stackSizes);
    dssDC = std::max(dssDC, stackSizes.dssDC);

    uint32_t dcStackSizeFromTrav = 0; // This sample doesn't call a direct callable during traversal.
    uint32_t dcStackSizeFromState = dssDC;
    // Possible Program Paths:
    // RG - CH(SearchRay/Diffuse) - AH(VisibilityRay)
    // RG - CH(SearchRay/Specular)
    // RG - MS(SearchRay)
    uint32_t ccStackSize =
        cssRG +
        std::max(std::max(cssCHSearchRayDiffuse + cssAHVisibilityRay,
                          cssCHSearchRaySpecular),
                 cssMS);
    pipeline.setStackSize(dcStackSizeFromTrav, dcStackSizeFromState, ccStackSize, 2);

    CUmodule modulePostProcess;
    CUDADRV_CHECK(cuModuleLoad(&modulePostProcess, (getExecutableDirectory() / "uber/ptxes/post_process.ptx").string().c_str()));
    cudau::Kernel kernelPostProcess(modulePostProcess, "postProcess", cudau::dim3(8, 8), 0);

    CUmodule moduleDeform;
    CUDADRV_CHECK(cuModuleLoad(&moduleDeform, (getExecutableDirectory() / "uber/ptxes/deform.ptx").string().c_str()));
    cudau::Kernel kernelDeform(moduleDeform, "deform", cudau::dim3(32), 0);
    cudau::Kernel kernelAccumulateVertexNormals(moduleDeform, "accumulateVertexNormals", cudau::dim3(32), 0);
    cudau::Kernel kernelNormalizeVertexNormals(moduleDeform, "normalizeVertexNormals", cudau::dim3(32), 0);

    CUmodule moduleBoundingBoxProgram;
    CUDADRV_CHECK(cuModuleLoad(&moduleBoundingBoxProgram, (getExecutableDirectory() / "uber/ptxes/sphere_bounding_box.ptx").string().c_str()));
    cudau::Kernel kernelCalculateBoundingBoxesForSpheres(moduleBoundingBoxProgram, "calculateBoundingBoxesForSpheres", cudau::dim3(32), 0);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: テクスチャー・マテリアルのセットアップ。
    // EN: Setup materials.

    hpprintf("Setup materials.\n");

    cudau::TypedBuffer<CUtexObject> textureObjectBuffer;
    textureObjectBuffer.initialize(cuContext, g_bufferType, 128);
    uint32_t textureID = 0;

#define USE_BLOCK_COMPRESSED_TEXTURE

    CUtexObject* textureObjects = textureObjectBuffer.map();

    cudau::TextureSampler texSampler;
    texSampler.setFilterMode(cudau::TextureFilterMode::Point,
                             cudau::TextureFilterMode::Point);
    texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
    texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    cudau::Array arrayCheckerBoard;
    {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
        int32_t width, height, mipCount;
        size_t* sizes;
        dds::Format format;
        uint8_t** ddsData = dds::load("../../data/checkerboard_line.DDS", &width, &height, &mipCount, &sizes, &format);

        arrayCheckerBoard.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                                       cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                       width, height, 1/*mipCount*/);
        for (int i = 0; i < arrayCheckerBoard.getNumMipmapLevels(); ++i)
            arrayCheckerBoard.transfer<uint8_t>(ddsData[i], sizes[i], i);

        dds::free(ddsData, mipCount, sizes);
#else
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load("../../data/checkerboard_line.png", &width, &height, &n, 4);
        arrayCheckerBoard.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                                       cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                       width, height, 1);
        arrayCheckerBoard.transfer<uint8_t>(linearImageData, width * height * 4);
        stbi_image_free(linearImageData);
#endif
    }
    uint32_t texCheckerBoardIndex = textureID++;
    textureObjects[texCheckerBoardIndex] = texSampler.createTextureObject(arrayCheckerBoard);

    cudau::Array arrayGrid;
    {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
        int32_t width, height, mipCount;
        size_t* sizes;
        dds::Format format;
        uint8_t** ddsData = dds::load("../../data/grid.DDS", &width, &height, &mipCount, &sizes, &format);

        arrayGrid.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                               cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                               width, height, 1/*mipCount*/);
        for (int i = 0; i < arrayGrid.getNumMipmapLevels(); ++i)
            arrayGrid.transfer<uint8_t>(ddsData[i], sizes[i], i);

        dds::free(ddsData, mipCount, sizes);
#else
        int32_t width, height, n;
        uint8_t* linearImageData = stbi_load("../../data/grid.png", &width, &height, &n, 4);
        arrayGrid.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                               cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                               width, height, 1);
        arrayGrid.transfer<uint8_t>(linearImageData, width * height * 4);
        stbi_image_free(linearImageData);
#endif
    }
    uint32_t texGridIndex = textureID++;
    textureObjects[texGridIndex] = texSampler.createTextureObject(arrayGrid);

    const char* textureNames[] = {
        "Checkerboard",
        "Grid"
    };

    textureObjectBuffer.unmap();



    cudau::TypedBuffer<Shared::MaterialData> materialDataBuffer;
    materialDataBuffer.initialize(cuContext, g_bufferType, 128);
    uint32_t materialID = 0;

    Shared::MaterialData* matData = materialDataBuffer.map();

    uint32_t matGrayWallIndex = materialID++;
    optixu::Material matGray = optixContext.createMaterial();
    matGray.setHitGroup(Shared::RayType_Search, searchRayDiffuseHitProgramGroup);
    matGray.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matGray.setUserData(matGrayWallIndex);
    Shared::MaterialData matGrayWallData;
    matGrayWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    matData[matGrayWallIndex] = matGrayWallData;

    uint32_t matFloorIndex = materialID++;
    optixu::Material matFloor = optixContext.createMaterial();
    matFloor.setHitGroup(Shared::RayType_Search, searchRayDiffuseHitProgramGroup);
    matFloor.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matFloor.setUserData(matFloorIndex);
    Shared::MaterialData matFloorData;
    matFloorData.albedo = make_float3(0, 0, 0);
    matFloorData.program = callableProgramSampleTextureIndex;
    matFloorData.texID = texCheckerBoardIndex;
    matData[matFloorIndex] = matFloorData;

    uint32_t matLeftWallIndex = materialID++;
    optixu::Material matLeft = optixContext.createMaterial();
    matLeft.setHitGroup(Shared::RayType_Search, searchRayDiffuseHitProgramGroup);
    matLeft.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matLeft.setUserData(matLeftWallIndex);
    Shared::MaterialData matLeftWallData;
    matLeftWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    matData[matLeftWallIndex] = matLeftWallData;

    uint32_t matRightWallIndex = materialID++;
    optixu::Material matRight = optixContext.createMaterial();
    matRight.setHitGroup(Shared::RayType_Search, searchRayDiffuseHitProgramGroup);
    matRight.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matRight.setUserData(matRightWallIndex);
    Shared::MaterialData matRightWallData;
    matRightWallData.albedo = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    matData[matRightWallIndex] = matRightWallData;

    uint32_t matLightIndex = materialID++;
    optixu::Material matLight = optixContext.createMaterial();
    matLight.setHitGroup(Shared::RayType_Search, searchRayDiffuseHitProgramGroup);
    matLight.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matLight.setUserData(matLightIndex);
    Shared::MaterialData matLightData;
    matLightData.albedo = make_float3(1, 1, 1);
    matData[matLightIndex] = matLightData;

    uint32_t matObject0Index = materialID++;
    optixu::Material matObject0 = optixContext.createMaterial();
    matObject0.setHitGroup(Shared::RayType_Search, searchRaySpecularHitProgramGroup);
    matObject0.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matObject0.setUserData(matObject0Index);
    Shared::MaterialData matObject0Data;
    matObject0Data.albedo = make_float3(1, 0.5f, 0);
    matData[matObject0Index] = matObject0Data;

    uint32_t matObject1Index = materialID++;
    optixu::Material matObject1 = optixContext.createMaterial();
    matObject1.setHitGroup(Shared::RayType_Search, searchRaySpecularHitProgramGroup);
    matObject1.setHitGroup(Shared::RayType_Visibility, visibilityRayHitProgramGroup);
    matObject1.setUserData(matObject1Index);
    Shared::MaterialData matObject1Data;
    matObject1Data.albedo = make_float3(0, 0.5f, 1);
    matData[matObject1Index] = matObject1Data;

    uint32_t matCustomPrimObjectIndex = materialID++;
    optixu::Material matCustomPrimObject = optixContext.createMaterial();
    matCustomPrimObject.setHitGroup(Shared::RayType_Search, searchRayDiffuseCustomHitProgramGroup);
    matCustomPrimObject.setHitGroup(Shared::RayType_Visibility, visibilityRayCustomHitProgramGroup);
    matCustomPrimObject.setUserData(matCustomPrimObjectIndex);
    Shared::MaterialData matCustomPrimObjectData;
    matCustomPrimObjectData.program = callableProgramSampleTextureIndex;
    matCustomPrimObjectData.texID = texGridIndex;
    matData[matCustomPrimObjectIndex] = matCustomPrimObjectData;

    materialDataBuffer.unmap();

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    hpprintf("Setup a scene.\n");

    optixu::Scene scene = optixContext.createScene();
    
    SceneContext sceneContext;
    sceneContext.optixScene = scene;
    sceneContext.decodeHitPointTriangle = static_cast<Shared::ProgDecodeHitPoint>(callableProgramDecodeHitPointTriangleIndex);
    sceneContext.geometryDataBuffer.initialize(cuContext, g_bufferType, 128);
    sceneContext.geometryID = 0;
    
    TriangleMesh meshCornellBox(cuContext, &sceneContext);
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
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
        // floor
        meshCornellBox.addMaterialGroup(triangles + 0, 2, matFloor);
        // back wall, ceiling
        meshCornellBox.addMaterialGroup(triangles + 2, 4, matGray);
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
    uint32_t objectMatGroupIndex;
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "../../data/subd_cube.obj");

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
        std::vector<Shared::Vertex> orgObjectVertices(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            orgObjectVertices[vertexIndex] = kv.second;
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
            Shared::Vertex &v0 = orgObjectVertices[tri.index0];
            Shared::Vertex &v1 = orgObjectVertices[tri.index1];
            Shared::Vertex &v2 = orgObjectVertices[tri.index2];
            float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
            v0.normal = v0.normal + gn;
            v1.normal = v1.normal + gn;
            v2.normal = v2.normal + gn;
        }
        for (int vIdx = 0; vIdx < orgObjectVertices.size(); ++vIdx) {
            Shared::Vertex &v = orgObjectVertices[vIdx];
            v.normal = normalize(v.normal);
        }

        meshObject.setVertexBuffer(orgObjectVertices.data(), orgObjectVertices.size());

        objectMatGroupIndex = meshObject.addMaterialGroup(triangles.data(), triangles.size(), matObject0);
        meshObject.setMatrial(1, objectMatGroupIndex, matObject1);
    }
    cudau::TypedBuffer<Shared::Vertex> orgObjectVertexBuffer = meshObject.getVertexBuffer().copy();

    // JP: カスタムプリミティブによるGeometryInstanceのセットアップ。
    // EN: Setup a geometry instance with custom primitives.
    optixu::GeometryInstance customPrimInstance = scene.createGeometryInstance(true);
    cudau::TypedBuffer<AABB> customPrimAABBs;
    cudau::TypedBuffer<Shared::SphereParameter> customPrimParameters;
    {
        constexpr uint32_t numPrimitives = 25;
        customPrimAABBs.initialize(cuContext, g_bufferType, numPrimitives);
        customPrimParameters.initialize(cuContext, g_bufferType, numPrimitives);

        Shared::SphereParameter* params = customPrimParameters.map();
        std::mt19937 rng(1290527201);
        std::uniform_real_distribution u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::SphereParameter &param = params[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = 0.3f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param.center = make_float3(x, y, z);
            param.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);
            param.texCoordMultiplier = 10;
        }
        customPrimParameters.unmap();

        static_assert(sizeof(AABB) == sizeof(OptixAabb),
                      "Custom AABB buffer must obey the same format as OptixAabb.");
        customPrimInstance.setCustomPrimitiveAABBBuffer(reinterpret_cast<cudau::TypedBuffer<OptixAabb>*>(&customPrimAABBs));
        customPrimInstance.setNumMaterials(1, nullptr);
        customPrimInstance.setMaterial(0, 0, matCustomPrimObject);
        customPrimInstance.setUserData(sceneContext.geometryID);

        Shared::GeometryData* geomDataPtr = sceneContext.geometryDataBuffer.map();
        Shared::GeometryData &recordData = geomDataPtr[sceneContext.geometryID];
        recordData.aabbBuffer = customPrimAABBs.getDevicePointer();
        recordData.paramBuffer = customPrimParameters.getDevicePointer();
        recordData.decodeHitPointFunc = static_cast<Shared::ProgDecodeHitPoint>(callableProgramDecodeHitPointSphereIndex);
        sceneContext.geometryDataBuffer.unmap();

        ++sceneContext.geometryID;
    }



    uint32_t travID = 0;
    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;
    cudau::TypedBuffer<OptixTraversableHandle> travHandleBuffer;
    travHandleBuffer.initialize(cuContext, g_bufferType, 128);
    OptixTraversableHandle* travHandles = travHandleBuffer.map();

    // JP: コーネルボックスと面光源にサンプルとして敢えて別々のGASを使う。
    // EN: Use different GAS for the Cornell box and the area light
    //     on purpose as sample.
    
    uint32_t gasCornellBoxIndex = travID++;
    optixu::GeometryAccelerationStructure gasCornellBox = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasCornellBoxMem;
    cudau::Buffer gasCornellBoxCompactedMem;
    gasCornellBox.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    gasCornellBox.setNumMaterialSets(1);
    gasCornellBox.setNumRayTypes(0, Shared::NumRayTypes);
    meshCornellBox.addToGAS(&gasCornellBox);
    gasCornellBox.prepareForBuild(&asMemReqs);
    gasCornellBoxMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    uint32_t gasAreaLightIndex = travID++;
    optixu::GeometryAccelerationStructure gasAreaLight = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasAreaLightMem;
    cudau::Buffer gasAreaLightCompactedMem;
    gasAreaLight.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    gasAreaLight.setNumMaterialSets(1);
    gasAreaLight.setNumRayTypes(0, Shared::NumRayTypes);
    meshAreaLight.addToGAS(&gasAreaLight);
    gasAreaLight.prepareForBuild(&asMemReqs);
    gasAreaLightMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    uint32_t gasObjectIndex = travID++;
    optixu::GeometryAccelerationStructure gasObject = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasObjectMem;
    gasObject.setConfiguration(optixu::ASTradeoff::PreferFastBuild, true, false, false);
    gasObject.setNumMaterialSets(2);
    gasObject.setNumRayTypes(0, Shared::NumRayTypes);
    gasObject.setNumRayTypes(1, Shared::NumRayTypes);
    meshObject.addToGAS(&gasObject);
    gasObject.prepareForBuild(&asMemReqs);
    gasObjectMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer,
                                      std::max(asMemReqs.tempSizeInBytes, asMemReqs.tempUpdateSizeInBytes));

    // JP: カスタムプリミティブ用のGASを作成する。
    // EN: Create a GAS for custom primitives.
    uint32_t gasCustomPrimObjectIndex = travID++;
    optixu::GeometryAccelerationStructure gasCustomPrimObject = scene.createGeometryAccelerationStructure(true);
    cudau::Buffer gasCustomPrimObjectMem;
    gasCustomPrimObject.setConfiguration(optixu::ASTradeoff::PreferFastBuild, true, false, false);
    gasCustomPrimObject.setNumMaterialSets(1);
    gasCustomPrimObject.setNumRayTypes(0, Shared::NumRayTypes);
    gasCustomPrimObject.addChild(customPrimInstance);
    gasCustomPrimObject.prepareForBuild(&asMemReqs);
    gasCustomPrimObjectMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer,
                                      std::max(asMemReqs.tempSizeInBytes, asMemReqs.tempUpdateSizeInBytes));

    // JP: カスタムプリミティブのAABBを計算するカーネルを実行。
    // EN: Execute a kernel to compute AABBs of custom primitives.
    cudau::dim3 dimBB = kernelCalculateBoundingBoxesForSpheres.calcGridDim(customPrimAABBs.numElements());
    kernelCalculateBoundingBoxesForSpheres(cuStream[0], dimBB,
                                           customPrimParameters.getDevicePointer(), customPrimAABBs.getDevicePointer(), customPrimAABBs.numElements());

    // JP: Geometry Acceleration Structureをビルドする。
    //     スクラッチバッファーは共用する。
    // EN: Build geometry acceleration structures.
    //     Share the scratch buffer among them.
    asBuildScratchMem.initialize(cuContext, g_bufferType, maxSizeOfScratchBuffer, 1);
    travHandles[gasCornellBoxIndex] = gasCornellBox.rebuild(cuStream[0], gasCornellBoxMem, asBuildScratchMem);
    travHandles[gasAreaLightIndex] = gasAreaLight.rebuild(cuStream[0], gasAreaLightMem, asBuildScratchMem);
    travHandles[gasObjectIndex] = gasObject.rebuild(cuStream[0], gasObjectMem, asBuildScratchMem);
    travHandles[gasCustomPrimObjectIndex] = gasCustomPrimObject.rebuild(cuStream[0], gasCustomPrimObjectMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    // EN: Perform compaction for static meshes.
    size_t compactedASSize;
    gasCornellBox.prepareForCompact(&compactedASSize);
    gasCornellBoxCompactedMem.initialize(cuContext, cudau::BufferType::Device, compactedASSize, 1);
    gasAreaLight.prepareForCompact(&compactedASSize);
    gasAreaLightCompactedMem.initialize(cuContext, cudau::BufferType::Device, compactedASSize, 1);
    travHandles[gasCornellBoxIndex] = gasCornellBox.compact(cuStream[0], gasCornellBoxCompactedMem);
    travHandles[gasAreaLightIndex] = gasAreaLight.compact(cuStream[0], gasAreaLightCompactedMem);
    gasCornellBox.removeUncompacted();
    gasAreaLight.removeUncompacted();



    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, g_bufferType, hitGroupSbtSize, 1);
    
    // JP: GASからインスタンスを作成する。
    // EN: Make instances from GASs.

    optixu::Instance instCornellBox = scene.createInstance();
    instCornellBox.setChild(gasCornellBox);

    optixu::Instance instAreaLight = scene.createInstance();
    float tfAreaLight[] = {
    1, 0, 0, 0,
    0, 1, 0, 0.99f,
    0, 0, 1, 0
    };
    instAreaLight.setChild(gasAreaLight);
    instAreaLight.setTransform(tfAreaLight);

    // JP: オブジェクトのインスタンスを2つ作成するが、
    //     ひとつはマテリアルセット0、もうひとつは1にする。
    // EN: Create two instances using the object but
    //     the one with material set 0, the other with 1.
    optixu::Instance instObject0 = scene.createInstance();
    instObject0.setChild(gasObject, 0);
    optixu::Instance instObject1 = scene.createInstance();
    instObject1.setChild(gasObject, 1);

    optixu::Instance instCustomPrimObject = scene.createInstance();
    float tfCustomPrimObject[] = {
    1, 0, 0, 0,
    0, 1, 0, -0.8f,
    0, 0, 1, 0
    };
    instCustomPrimObject.setChild(gasCustomPrimObject);
    instCustomPrimObject.setTransform(tfCustomPrimObject);



    // JP: Instance Acceleration Structureの準備。
    // EN: Prepare the instance acceleration structure.
    uint32_t iasSceneIndex = travID++;
    optixu::InstanceAccelerationStructure iasScene = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasSceneMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    uint32_t numInstances;
    iasScene.setConfiguration(optixu::ASTradeoff::PreferFastBuild, true, false);
    iasScene.addChild(instCornellBox);
    iasScene.addChild(instAreaLight);
    iasScene.addChild(instObject0);
    iasScene.addChild(instObject1);
    iasScene.addChild(instCustomPrimObject);
    iasScene.prepareForBuild(&asMemReqs, &numInstances);
    instanceBuffer.initialize(cuContext, g_bufferType, numInstances);
    iasSceneMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    size_t tempBufferForIAS = std::max(asMemReqs.tempSizeInBytes, asMemReqs.tempUpdateSizeInBytes);
    if (tempBufferForIAS >= asBuildScratchMem.sizeInBytes()) {
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, tempBufferForIAS);
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);
    }

    // JP: Instance Acceleration Structureをビルドする。
    // EN: Build the instance acceleration structure.
    travHandles[iasSceneIndex] = iasScene.rebuild(cuStream[0], instanceBuffer, iasSceneMem, asBuildScratchMem);

    travHandleBuffer.unmap();
    CUDADRV_CHECK(cuStreamSynchronize(cuStream[0]));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    hpprintf("Setup resources for composite.\n");
    
    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    GLTK::Texture2D outputTexture;
    cudau::Array outputArray;
    outputTexture.initialize(renderTargetSizeX, renderTargetSizeY, GLTK::SizedInternalFormat::RGBA32F); GLTK::errorCheck();
    outputArray.initializeFromGLTexture2D(cuContext, outputTexture.getRawHandle(),
                                          cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);


    
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
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "uber/shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "uber/shaders/drawOptiXResult.frag"));

    // JP: アップスケール用のシェーダー。
    // EN: Shader for upscale.
    GLTK::GraphicsShader scaleShader;
    scaleShader.initializeVSPS(readTxtFile(exeDir / "uber/shaders/scale.vert"),
                               readTxtFile(exeDir / "uber/shaders/scale.frag"));

    // JP: アップスケール用のサンプラー。
    //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
    // EN: Sampler for upscaling.
    //     It seems to require to bind a sampler even when using texelFetch() which is independent from the sampler settings.
    GLTK::Sampler scaleSampler;
    scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> rngBuffer;
    rngBuffer.initialize(cuContext, g_bufferType, renderTargetSizeX, renderTargetSizeY);
    const auto initializeRNGSeeds = [&renderTargetSizeX, &renderTargetSizeY](optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> &buffer) {
        std::mt19937_64 rng(591842031321323413);

        buffer.map();
        for (int y = 0; y < renderTargetSizeY; ++y)
            for (int x = 0; x < renderTargetSizeX; ++x)
                buffer(x, y).setState(rng());
        buffer.unmap();
    };
    initializeRNGSeeds(rngBuffer);

#if defined(USE_NATIVE_BLOCK_BUFFER2D)
    cudau::Array arrayAccumBuffer;
    arrayAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                  renderTargetSizeX, renderTargetSizeY, 1);
#else
    optixu::HostBlockBuffer2D<float4, 1> accumBuffer;
    accumBuffer.initialize(cuContext, g_bufferType, renderTargetSizeX, renderTargetSizeY);
#endif



    Shared::PipelineLaunchParameters plp;
    plp.travHandles = travHandleBuffer.getDevicePointer();
    plp.materialData = materialDataBuffer.getDevicePointer();
    plp.geomInstData = sceneContext.geometryDataBuffer.getDevicePointer();
    plp.travIndex = iasSceneIndex;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.numAccumFrames = 1;
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
    plp.accumBuffer = arrayAccumBuffer.getSurfaceObject(0);
#else
    plp.accumBuffer = accumBuffer.getBlockBuffer2D();
#endif
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;
    plp.matLightIndex = matLightIndex;
    plp.textures = textureObjectBuffer.getDevicePointer();

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(&hitGroupSBT);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    hpprintf("Render loop.\n");

    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputBufferSurfaceHolder.initialize(&outputArray);
    
    StopWatchHiRes<> sw;
    std::mt19937_64 rng(3092384202);
    std::uniform_real_distribution<float> u01;
    
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    uint64_t animFrameIndex = 0;
    int32_t requestedSize[2];
    struct CPUTimeRecord {
        float frameTime;
        float frameBeginTime;
        float syncTime;
        float imGuiTime;
        float updateIASTime;
        float renderCmdTime;
        float postProcessCmdTime;
        float guiCmdTime;
        float swapTime;
        float dummyTime;

        CPUTimeRecord() :
            frameTime(0.0f),
            frameBeginTime(0.0f),
            syncTime(0.0f),
            imGuiTime(0.0f),
            updateIASTime(0.0f),
            renderCmdTime(0.0f),
            postProcessCmdTime(0.0f),
            guiCmdTime(0.0f),
            swapTime(0.0f),
            dummyTime(0.0f) {}
    };
    CPUTimeRecord cpuTimeRecords[600];
    uint32_t cpuTimeRecordIndex = 0;
    while (true) {
        CPUTimeRecord &cpuTimeRecord = cpuTimeRecords[cpuTimeRecordIndex];

        sw.start();

        sw.start(); // Frame Begin
        uint32_t bufferIndex = frameIndex % 2;

        if (glfwWindowShouldClose(window))
            break;
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

            outputTexture.finalize();
            outputTexture.initialize(renderTargetSizeX, renderTargetSizeY, GLTK::SizedInternalFormat::RGBA32F);
            outputArray.finalize();
            outputArray.initializeFromGLTexture2D(cuContext, outputTexture.getRawHandle(),
                                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            frameBuffer.finalize();
            frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_SRGB8, GL_DEPTH_COMPONENT32);

#if defined(USE_NATIVE_BLOCK_BUFFER2D)
            arrayAccumBuffer.resize(renderTargetSizeX, renderTargetSizeY);
#else
            accumBuffer.resize(renderTargetSizeX, renderTargetSizeY);
#endif
            rngBuffer.resize(renderTargetSizeX, renderTargetSizeY);
            initializeRNGSeeds(rngBuffer);

            // EN: update the pipeline parameters.
            plp.imageSize.x = renderTargetSizeX;
            plp.imageSize.y = renderTargetSizeY;
            plp.numAccumFrames = 1;
            plp.rngBuffer = rngBuffer.getBlockBuffer2D();
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
            plp.accumBuffer = arrayAccumBuffer.getSurfaceObject(0);
#else
            plp.accumBuffer = accumBuffer.getBlockBuffer2D();
#endif
            plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        cpuTimeRecord.frameBeginTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;



        bool operatingCamera;
        bool cameraIsActuallyMoving;
        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState& a, const KeyState& b) {
                int32_t dir = 0;
                if (a.getState() == true) {
                    if (b.getState() == true)
                        dir = 0;
                    else
                        dir = 1;
                }
                else {
                    if (b.getState() == true)
                        dir = -1;
                    else
                        dir = 0;
                }
                return dir;
            };

            int32_t trackZ = decideDirection(g_keyForward, g_keyBackward);
            int32_t trackX = decideDirection(g_keyLeftward, g_keyRightward);
            int32_t trackY = decideDirection(g_keyUpward, g_keyDownward);
            int32_t tiltZ = decideDirection(g_keyTiltRight, g_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(g_keyFasterPosMovSpeed, g_keySlowerPosMovSpeed);

            g_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            g_cameraPositionalMovingSpeed = std::min(std::max(g_cameraPositionalMovingSpeed, 1e-6f), 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double g_prevMouseX = g_mouseX, g_prevMouseY = g_mouseY;
            if (g_buttonRotate.getState() == true) {
                if (g_buttonRotate.getTime() == frameIndex) {
                    lastX = g_mouseX;
                    lastY = g_mouseY;
                }
                else {
                    deltaX = g_mouseX - lastX;
                    deltaY = g_mouseY - lastY;
                }
            }

            float deltaAngle = std::sqrt(deltaX * deltaX + deltaY * deltaY);
            float3 axis = make_float3(deltaY, -deltaX, 0);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = make_float3(1, 0, 0);

            g_cameraOrientation = g_cameraOrientation * qRotateZ(g_cameraTiltSpeed * tiltZ);
            g_tempCameraOrientation = g_cameraOrientation * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition += g_tempCameraOrientation.toMatrix3x3() * (g_cameraPositionalMovingSpeed * make_float3(trackX, trackY, trackZ));
            if (g_buttonRotate.getState() == false && g_buttonRotate.getTime() == frameIndex) {
                g_cameraOrientation = g_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            operatingCamera = (g_keyForward.getState() || g_keyBackward.getState() ||
                               g_keyLeftward.getState() || g_keyRightward.getState() ||
                               g_keyUpward.getState() || g_keyDownward.getState() ||
                               g_keyTiltLeft.getState() || g_keyTiltRight.getState() ||
                               g_buttonRotate.getState());
            cameraIsActuallyMoving = (trackZ != 0 || trackX != 0 || trackY != 0 ||
                                      tiltZ != 0 || (g_mouseX != g_prevMouseX) || (g_mouseY != g_prevMouseY))
                && operatingCamera;

            g_prevMouseX = g_mouseX;
            g_prevMouseY = g_mouseY;

            plp.camera.position = g_cameraPosition;
            plp.camera.orientation = g_tempCameraOrientation.toMatrix3x3();
        }



        CUstream &curCuStream = cuStream[bufferIndex];
        GPUTimer &curGPUTimer = gpuTimer[bufferIndex];
        
        // JP: 前フレームの処理が完了するのを待つ。
        // EN: Wait the previous frame processing to finish.
        sw.start(); // Sync
        CUDADRV_CHECK(cuStreamSynchronize(curCuStream));
        cpuTimeRecord.syncTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;

        // JP: 非同期実行を確かめるためにCPU側にダミー負荷を与える。
        // EN: Have dummy load on CPU to verify asynchronous execution.
        static float cpuDummyLoad = 15.0f;
        static float dummyProb = 0.0f;
        sw.start(); // Dummy Load
        if (cpuDummyLoad > 0.0f && u01(rng) < dummyProb * 0.01f)
            std::this_thread::sleep_for(std::chrono::microseconds(static_cast<uint64_t>(cpuDummyLoad * 1000)));
        cpuTimeRecord.dummyTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;



        sw.start(); // ImGui
        {
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float cudaFrameTime = frameIndex >= 2 ? curGPUTimer.frame.report() : 0.0f;
            float deformTime = (frameIndex >= 2 && curGPUTimer.animated) ? curGPUTimer.deform.report() : 0.0f;
            float updateGASTime = (frameIndex >= 2 && curGPUTimer.animated) ? curGPUTimer.updateGAS.report() : 0.0f;
            float updateIASTime = (frameIndex >= 2 && curGPUTimer.animated) ? curGPUTimer.updateIAS.report() : 0.0f;
            float renderTime = frameIndex >= 2 ? curGPUTimer.render.report() : 0.0f;
            float postProcessTime = frameIndex >= 2 ? curGPUTimer.postProcess.report() : 0.0f;
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime);
            ImGui::Text("  Deform: %.3f [ms]", deformTime);
            ImGui::Text("  Update GAS: %.3f [ms]", updateGASTime);
            ImGui::Text("  Update IAS: %.3f [ms]", updateIASTime);
            ImGui::Text("  Render: %.3f [ms]", renderTime);
            ImGui::Text("  Post Process: %.3f [ms]", postProcessTime);
            {
                static float times[100];
                constexpr uint32_t numTimes = lengthof(times);
                constexpr uint32_t numAccums = 1;
                static float accTime = 0;
                static uint32_t plotStartPos = -1;
                accTime += cudaFrameTime;
                if ((frameIndex + 1) % numAccums == 0) {
                    plotStartPos = (plotStartPos + 1) % numTimes;
                    times[(plotStartPos + numTimes - 1) % numTimes] = accTime / numAccums;
                    accTime = 0;
                }
                ImGui::PlotLines("CUDA/OptiX GPU Time", times, numTimes, plotStartPos, nullptr, 0, 50.0f, ImVec2(0, 50));
            }

            const CPUTimeRecord prevCpuTimeRecord = cpuTimeRecords[(cpuTimeRecordIndex + lengthof(cpuTimeRecords) - 1) % lengthof(cpuTimeRecords)];
            ImGui::Text("CPU %.3f [ms]:", prevCpuTimeRecord.frameTime);
            ImGui::Text("  Begin: %.3f [ms]", prevCpuTimeRecord.frameBeginTime);
            ImGui::Text("  Sync: %.3f [ms]", prevCpuTimeRecord.syncTime);
            ImGui::Text("  ImGui: %.3f [ms]", prevCpuTimeRecord.imGuiTime);
            ImGui::Text("  Update IAS: %.3f [ms]", prevCpuTimeRecord.updateIASTime);
            ImGui::Text("  Render: %.3f [ms]", prevCpuTimeRecord.renderCmdTime);
            ImGui::Text("  Post Process: %.3f [ms]", prevCpuTimeRecord.postProcessCmdTime);
            ImGui::Text("  GUI: %.3f [ms]", prevCpuTimeRecord.guiCmdTime);
            ImGui::Text("  Swap: %.3f [ms]", prevCpuTimeRecord.swapTime);
            ImGui::Text("  Dummy: %.3f [ms]", prevCpuTimeRecord.dummyTime);
            {
                static float times[100];
                constexpr uint32_t numTimes = lengthof(times);
                constexpr uint32_t numAccums = 1;
                static float accTime = 0;
                static uint32_t plotStartPos = -1;
                accTime += prevCpuTimeRecord.frameTime;
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
            if (ImGui::Button("Dump History")) {
                for (int i = 0; i < lengthof(cpuTimeRecords); ++i) {
                    const CPUTimeRecord &record = cpuTimeRecords[(cpuTimeRecordIndex + i) % lengthof(cpuTimeRecords)];
                    hpprintf("CPU Time Frame %llu: %.3f [ms]\n", frameIndex - lengthof(cpuTimeRecords) + i, record.frameTime);
                    hpprintf("  Begin: %.3f [ms]\n", record.frameBeginTime);
                    hpprintf("  Sync: %.3f [ms]\n", record.syncTime);
                    hpprintf("  ImGui: %.3f [ms]\n", record.imGuiTime);
                    hpprintf("  Update IAS: %.3f [ms]\n", record.updateIASTime);
                    hpprintf("  Render: %.3f [ms]\n", record.renderCmdTime);
                    hpprintf("  Post Process: %.3f [ms]\n", record.postProcessCmdTime);
                    hpprintf("  GUI: %.3f [ms]\n", record.guiCmdTime);
                    hpprintf("  Swap: %.3f [ms]\n", record.swapTime);
                    hpprintf("  Dummy: %.3f [ms]\n", record.dummyTime);
                }
            }

            ImGui::SliderFloat("Dummy CPU Load", &cpuDummyLoad, 0.0f, 33.3333f);
            ImGui::SliderFloat("Probability", &dummyProb, 0.0f, 100.0f);

            ImGui::End();
        }



        static bool enablePeriodicGASRebuild = true;
        static int32_t gasRebuildInterval = 30;
        static bool enablePeriodicIASRebuild = true;
        static int32_t iasRebuildInterval = 30;
        {
            ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Checkbox("Enable GAS Rebuild", &enablePeriodicGASRebuild);
            ImGui::SliderInt("GAS Rebuild Interval", &gasRebuildInterval, 1, 60);
            ImGui::Checkbox("Enable IAS Rebuild", &enablePeriodicIASRebuild);
            ImGui::SliderInt("IAS Rebuild Interval", &iasRebuildInterval, 1, 60);

            ImGui::End();
        }



        {
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&plp.camera.position));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / M_PI;
            rollPitchYaw[1] *= 180 / M_PI;
            rollPitchYaw[2] *= 180 / M_PI;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw, 3))
                g_cameraOrientation = qFromEulerAngles(rollPitchYaw[0] * M_PI / 180,
                                                       rollPitchYaw[1] * M_PI / 180,
                                                       rollPitchYaw[2] * M_PI / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);

            ImGui::End();
        }



        static bool play = true;
        bool playStep = false;
        bool sceneEdited = false;

        // JP: マテリアルの編集。
        // EN; Edit materials.
        {
            ImGui::Begin("Scene", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::Button(play ? "Stop" : "Play"))
                play = !play;
            ImGui::SameLine();
            if (ImGui::Button("Step")) {
                playStep = true;
                play = false;
            }

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
            if (ImGui::ColorEdit3("Object 0", reinterpret_cast<float*>(&matObject0Data.albedo),
                                  ImGuiColorEditFlags_DisplayHSV |
                                  ImGuiColorEditFlags_Float)) {
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matObject0Index),
                                                &matObject0Data, sizeof(matObject0Data),
                                                curCuStream));
                sceneEdited = true;
            }
            if (ImGui::ColorEdit3("Object 1", reinterpret_cast<float*>(&matObject1Data.albedo),
                                  ImGuiColorEditFlags_DisplayHSV |
                                  ImGuiColorEditFlags_Float)) {
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matObject1Index),
                                                &matObject1Data, sizeof(matObject1Data),
                                                curCuStream));
                sceneEdited = true;
            }
            static int32_t floorTexID;
            floorTexID = matFloorData.texID;
            if (ImGui::Combo("Floor", &floorTexID, textureNames, lengthof(textureNames))) {
                matFloorData.texID = floorTexID;
                CUDADRV_CHECK(cuMemcpyHtoDAsync(materialDataBuffer.getCUdeviceptrAt(matFloorIndex),
                                                &matFloorData, sizeof(matFloorData),
                                                curCuStream));
                sceneEdited = true;
            }

            ImGui::End();
        }

        cpuTimeRecord.imGuiTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;



        curGPUTimer.frame.start(curCuStream);

        sw.start();
        curGPUTimer.animated = false;
        if (play || playStep) {
            curGPUTimer.animated = true;

            // JP: ジオメトリの非剛体変形。
            // EN: Non-rigid deformation of a geometry.
            curGPUTimer.deform.start(curCuStream);
            cudau::dim3 dimDeform = kernelDeform.calcGridDim(orgObjectVertexBuffer.numElements());
            kernelDeform(curCuStream, dimDeform,
                         orgObjectVertexBuffer.getDevicePointer(), meshObject.getVertexBuffer().getDevicePointer(), orgObjectVertexBuffer.numElements(),
                         0.5f * std::sinf(2 * M_PI * (animFrameIndex % 690) / 690.0f));
            const cudau::TypedBuffer<Shared::Triangle> &triangleBuffer = meshObject.getTriangleBuffer(objectMatGroupIndex);
            cudau::dim3 dimAccum = kernelAccumulateVertexNormals.calcGridDim(triangleBuffer.numElements());
            kernelAccumulateVertexNormals(curCuStream, dimAccum,
                                          meshObject.getVertexBuffer().getDevicePointer(),
                                          triangleBuffer.getDevicePointer(), triangleBuffer.numElements());
            kernelNormalizeVertexNormals(curCuStream, dimDeform,
                                         meshObject.getVertexBuffer().getDevicePointer(), orgObjectVertexBuffer.numElements());
            curGPUTimer.deform.stop(curCuStream);

            // JP: 変形したジオメトリを基にGASをアップデート。
            //     たまにリビルドを実行するが、ここでは頂点情報以外変化しないため、
            //     メモリサイズの再計算や再確保は不要。
            // EN: Update the GAS based on the deformed geometry.
            //     It sometimes performs rebuild, but all the information except for vertices doesn't change here
            //     so neither recalculation of nor reallocating memory is not required.
            curGPUTimer.updateGAS.start(curCuStream);
            OptixTraversableHandle gasHandle;
            if (enablePeriodicGASRebuild && animFrameIndex % gasRebuildInterval == 0)
                gasHandle = gasObject.rebuild(curCuStream, gasObjectMem, asBuildScratchMem);
            else
                gasHandle = gasObject.update(curCuStream, asBuildScratchMem);
            curGPUTimer.updateGAS.stop(curCuStream);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(travHandleBuffer.getCUdeviceptrAt(gasObjectIndex),
                                            &gasHandle, sizeof(gasHandle),
                                            curCuStream));

            // JP: インスタンスのトランスフォーム。
            // EN: Transform instances.

            Matrix3x3 sr0 =
                scale3x3(0.25 + 0.2f * std::sinf(2 * M_PI * (animFrameIndex % 660) / 660.0f)) *
                rotateY3x3(2 * M_PI * (animFrameIndex % 180) / 180.0f) *
                rotateX3x3(2 * M_PI * (animFrameIndex % 300) / 300.0f) *
                rotateZ3x3(2 * M_PI * (animFrameIndex % 420) / 420.0f);
            float tfObject0[] = {
                sr0.m00, sr0.m10, sr0.m20, 0.75f * std::sinf(2 * M_PI * (animFrameIndex % 360) / 360.0f),
                sr0.m01, sr0.m11, sr0.m21, 0,
                sr0.m02, sr0.m12, sr0.m22, 0.75f * std::cosf(2 * M_PI * (animFrameIndex % 360) / 360.0f)
            };
            instObject0.setTransform(tfObject0);

            Matrix3x3 sr1 =
                scale3x3(0.333f + 0.125f * std::sinf(2 * M_PI * (animFrameIndex % 780) / 780.0f + M_PI / 2)) *
                rotateY3x3(2 * M_PI * (animFrameIndex % 660) / 660.0f) *
                rotateX3x3(2 * M_PI * (animFrameIndex % 330) / 330.0f) *
                rotateZ3x3(2 * M_PI * (animFrameIndex % 570) / 570.0f);
            float tfObject1[] = {
                sr1.m00, sr1.m10, sr1.m20, 0.5f * std::sinf(2 * M_PI * (animFrameIndex % 180) / 180.0f + M_PI),
                sr1.m01, sr1.m11, sr1.m21, 0.25f * std::sinf(2 * M_PI * (animFrameIndex % 90) / 90.0f),
                sr1.m02, sr1.m12, sr1.m22, 0.5f * std::cosf(2 * M_PI * (animFrameIndex % 180) / 180.0f + M_PI)
            };
            instObject1.setTransform(tfObject1);

            // JP: IASをアップデート。
            // EN: Update the IAS.
            curGPUTimer.updateIAS.start(curCuStream);
            OptixTraversableHandle iasHandle;
            if (enablePeriodicIASRebuild && animFrameIndex % iasRebuildInterval == 0)
                iasHandle = iasScene.rebuild(curCuStream, instanceBuffer, iasSceneMem, asBuildScratchMem);
            else
                iasHandle = iasScene.update(curCuStream, asBuildScratchMem);
            curGPUTimer.updateIAS.stop(curCuStream);
            CUDADRV_CHECK(cuMemcpyHtoDAsync(travHandleBuffer.getCUdeviceptrAt(iasSceneIndex),
                                            &iasHandle, sizeof(iasHandle),
                                            curCuStream));

            ++animFrameIndex;
        }
        cpuTimeRecord.updateIASTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;



        if (play || playStep || sceneEdited || cameraIsActuallyMoving)
            plp.numAccumFrames = 1;

        // Render
        sw.start();
        curGPUTimer.render.start(curCuStream);
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curCuStream));
        pipeline.launch(curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
        curGPUTimer.render.stop(curCuStream);
        cpuTimeRecord.renderCmdTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;

        // Post Process
        sw.start();
        curGPUTimer.postProcess.start(curCuStream);
        cudau::dim3 dimPostProcess = kernelPostProcess.calcGridDim(renderTargetSizeX, renderTargetSizeY);
        outputBufferSurfaceHolder.beginCUDAAccess(curCuStream);
        kernelPostProcess(curCuStream, dimPostProcess,
#if defined(USE_NATIVE_BLOCK_BUFFER2D)
                          arrayAccumBuffer.getSurfaceObject(0),
#else
                          accumBuffer.getBlockBuffer2D(),
#endif
                          renderTargetSizeX, renderTargetSizeY, plp.numAccumFrames,
                          outputBufferSurfaceHolder.getNext());
        outputBufferSurfaceHolder.endCUDAAccess(curCuStream);
        curGPUTimer.postProcess.stop(curCuStream);
        cpuTimeRecord.postProcessCmdTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;
        ++plp.numAccumFrames;

        curGPUTimer.frame.stop(curCuStream);



        sw.start();

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

        cpuTimeRecord.guiCmdTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;

        sw.start();
        glfwSwapBuffers(window);
        cpuTimeRecord.swapTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;

        ++frameIndex;
        cpuTimeRecord.frameTime = sw.getMeasurement(sw.stop(), StopWatchDurationType::Microseconds) * 1e-3f;

        cpuTimeRecordIndex = (cpuTimeRecordIndex + 1) % lengthof(cpuTimeRecords);
        sw.clearAllMeasurements();
    }

    outputBufferSurfaceHolder.finalize();



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



#if defined(USE_NATIVE_BLOCK_BUFFER2D)
    arrayAccumBuffer.finalize();
#else
    accumBuffer.finalize();
#endif
    rngBuffer.finalize();
    
    scaleSampler.finalize();
    scaleShader.finalize();
    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    frameBuffer.finalize();

    outputArray.finalize();
    outputTexture.finalize();

    instanceBuffer.finalize();
    iasSceneMem.finalize();
    iasScene.destroy();

    instCustomPrimObject.destroy();
    instObject1.destroy();
    instObject0.destroy();
    instAreaLight.destroy();
    instCornellBox.destroy();

    hitGroupSBT.finalize();

    asBuildScratchMem.finalize();

    gasCustomPrimObjectMem.finalize();
    gasCustomPrimObject.destroy();
    gasObjectMem.finalize();
    gasObject.destroy();
    gasAreaLightCompactedMem.finalize();
    gasAreaLightMem.finalize();
    gasAreaLight.destroy();
    gasCornellBoxCompactedMem.finalize();
    gasCornellBoxMem.finalize();
    gasCornellBox.destroy();

    travHandleBuffer.finalize();

    customPrimParameters.finalize();
    customPrimAABBs.finalize();
    customPrimInstance.destroy();

    orgObjectVertexBuffer.finalize();

    meshObject.destroy();
    meshAreaLight.destroy();
    meshCornellBox.destroy();

    sceneContext.geometryDataBuffer.finalize();

    scene.destroy();

    matObject0.destroy();
    matObject1.destroy();
    matLight.destroy();
    matRight.destroy();
    matLeft.destroy();
    matFloor.destroy();
    matGray.destroy();

    materialDataBuffer.finalize();

    textureObjects = textureObjectBuffer.map();
    for (int i = textureID - 1; i >= 0; --i)
        CUDADRV_CHECK(cuTexObjectDestroy(textureObjects[i]));
    textureObjectBuffer.unmap();
    arrayGrid.finalize();
    arrayCheckerBoard.finalize();

    textureObjectBuffer.finalize();

    CUDADRV_CHECK(cuModuleUnload(moduleBoundingBoxProgram));
    CUDADRV_CHECK(cuModuleUnload(moduleDeform));
    CUDADRV_CHECK(cuModuleUnload(modulePostProcess));

    shaderBindingTable.finalize();

    callableProgramDecodeHitPointSphere.destroy();
    callableProgramDecodeHitPointTriangle.destroy();
    callableProgramSampleTexture.destroy();

    visibilityRayCustomHitProgramGroup.destroy();
    searchRayDiffuseCustomHitProgramGroup.destroy();

    visibilityRayHitProgramGroup.destroy();
    searchRaySpecularHitProgramGroup.destroy();
    searchRayDiffuseHitProgramGroup.destroy();

    visibilityRayMissProgram.destroy();
    searchRayMissProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    gpuTimer[1].finalize();
    gpuTimer[0].finalize();
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
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
