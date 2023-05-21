/*

JP: このサンプルはディスプレイスメントマッピングによる高密度ジオメトリを効率的に表現するための
    Displacement Micro-Map (DMM)の使用方法を示します。
    DMMは三角形メッシュにおけるハイトマップなどによる凹凸情報を三角形ごとにコンパクトに格納したものです。
    GASの生成時に追加情報として渡すことで三角形メッシュに高密度な凹凸を較的省メモリに追加することができます。
    逆に粗いメッシュにDMMを付加することで、通常の三角形メッシュよりも遥かに省メモリなGASで同様のジオメトリを
    表現することができますしGASのビルドも高速になります。
    OptiXのAPIにはDMM自体の生成処理は含まれていないため、何らかの手段を用いて生成する必要があります。
    DMM生成処理はテクスチャーとメッシュ間のマッピング、テクスチャー自体が静的な場合オフラインで予め行うことも可能です。
    このサンプルにはDMMの生成処理も含まれていますが、
    おそらくDisplacement Micro-Map SDK [1]などのツールを使うほうが良いでしょう。

    --no-dmm: DMMを無効化する。
    --visualize ***: 可視化モードを切り替える。
      - final: 最終レンダリング。
      - barycentric: 重心座標の可視化。ベース三角形の形状を確認できる。
      - micro-barycentric: マイクロ三角形の重心座標の可視化。
      - subdiv-level: 細分割レベルの可視化。
      - normal: 法線ベクトルの可視化。
    --max-compressed-format ***: DMMのエンコードを強制的に指定する。
      - none: 強制しない。(自動的に決定される)
      - 64utris: DMMあたり64マイクロ三角形64Bのフォーマットを使う。
      - 256utris: DMMあたり256マイクロ三角形128Bのフォーマットを使う。
      - 1024utris: DMMあたり1024マイクロ三角形128Bのフォーマットを使う。
    --max-subdiv-level *: DMMの最大分割レベル。
    --subdiv-level-bias *: DMMの分割レベルへのバイアス。
    --displacement-bias *: ベース頂点のディスプレイスメントベクター方向への事前の移動量。
    --displacement-scale *: ベース頂点のディスプレイスメントベクター方向の変位スケール。
    --no-index-buffer: DMM用のインデックスバッファーを使用しない。

EN: This sample shows how to use Displacement Micro-Map (DMM) with which high-definition geometry by
    displacement mapping can be efficiently represented.
    DMM is a data structure which compactly stores per-triangle displacement information by height map or
    something else for a triangle mesh.
    Providing DMM as additional information when building a GAS adds high frequency geometric detail to
    a triangle mesh with relatively low additional memory.
    In other words, a geometry with similar complexity as a normal triangle mesh can be represented with
    a GAS with much less memory by adding DMM to a coarse mesh. This makes GAS build faster also.
    OptiX API doesn't provide generation of DMM itself, so DMM generation by some means is required.
    DMM generation can be offline pre-computation if the mapping between a texture and a mesh and
    the texture itself are static.
    This sample provides DMM generation also, but you may want to use a tool like Displacement Micro-Map SDK [1].

    [1] Displacement Micro-Map SDK: https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/

    --no-dmm: Disable DMM.
    --visualize ***: You can change visualizing mode.
      - final: Final rendering.
      - barycentric: Visualize barycentric coordinates, can be used to see the shapes of base triangles.
      - micro-barycentric: Visualize barycentric coordinates of micro-triangles.
      - subdiv-level: Visualize subdivision levels.
      - normal: Visualize normal vectors.
    --max-compressed-format ***: Forcefully specify a DMM encoding.
      - none: Do not force (encodings are automatically determined)
      - 64utris: Use an encoding with 64 micro triangles, 64B per triangle.
      - 256utris: Use an encoding with 256 micro triangles, 128B per triangle.
      - 1024utris: Use an encoding with 1024 micro triangles, 128B per triangle.
    --max-subdiv-level *: The maximum DMM subdivision level.
    --subdiv-level-bias *: The bias to DMM subdivision level.
    --displacement-bias *: The amount of pre-movement of base vertices along displacement vectors.
    --displacement-scale *: The amount of displacement of base vertices along displacement vectors.
    --no-index-buffer: Specify not to use index buffers for DMM.

*/

#include "displacement_micro_map_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

// Include glfw3.h after our OpenGL definitions
#include "../common/gl_util.h"
#include <GLFW/glfw3.h>

#include "../common/imgui_more.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"



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



static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path exeDir = getExecutableDirectory();

    bool takeScreenShot = false;
    bool useDMM = true;
    auto visualizationMode = Shared::VisualizationMode_Final;
    shared::DMMEncoding maxCompressedFormat = shared::DMMEncoding_None;
    shared::DMMSubdivLevel maxDmmSubDivLevel = shared::DMMSubdivLevel_5;
    int32_t dmmSubdivLevelBias = 0;
    bool useDmmIndexBuffer = true;
    float displacementBias = 0.0f;
    float displacementScale = 1.0f;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot") {
            takeScreenShot = true;
        }
        else if (arg == "--no-dmm") {
            useDMM = false;
        }
        else if (arg == "--visualize") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --visualize is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "final")
                visualizationMode = Shared::VisualizationMode_Final;
            else if (visType == "barycentric")
                visualizationMode = Shared::VisualizationMode_Barycentric;
            else if (visType == "micro-barycentric")
                visualizationMode = Shared::VisualizationMode_MicroBarycentric;
            else if (visType == "subdiv-level")
                visualizationMode = Shared::VisualizationMode_SubdivLevel;
            else if (visType == "normal")
                visualizationMode = Shared::VisualizationMode_Normal;
            else
                throw std::runtime_error("Argument for --visualize is invalid.");
            argIdx += 1;
        }
        else if (arg == "--max-compressed-format") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --max-compressed-format is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "none")
                maxCompressedFormat = shared::DMMEncoding_None;
            else if (visType == "64utris")
                maxCompressedFormat = shared::DMMEncoding_64B_per_64MicroTris;
            else if (visType == "256utris")
                maxCompressedFormat = shared::DMMEncoding_128B_per_256MicroTris;
            else if (visType == "1024utris")
                maxCompressedFormat = shared::DMMEncoding_128B_per_1024MicroTris;
            else
                throw std::runtime_error("Argument for --max-compressed-format is invalid.");
            argIdx += 1;
        }
        else if (arg == "--max-subdiv-level") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --max-subdiv-level is not complete.");
            int32_t level = std::atoi(argv[argIdx + 1]);
            if (level < 0 || level > shared::DMMSubdivLevel_5)
                throw std::runtime_error("Invalid DMM subdivision level.");
            maxDmmSubDivLevel = static_cast<shared::DMMSubdivLevel>(level);
            argIdx += 1;
        }
        else if (arg == "--subdiv-level-bias") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --subdiv-level-bias is not complete.");
            dmmSubdivLevelBias = std::atoi(argv[argIdx + 1]);
            argIdx += 1;
        }
        else if (arg == "--no-index-buffer") {
            useDmmIndexBuffer = false;
        }
        else if (arg == "--displacement-bias") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --displacement-bias is not complete.");
            displacementBias = std::atof(argv[argIdx + 1]);
            argIdx += 1;
        }
        else if (arg == "--displacement-scale") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --displacement-scale is not complete.");
            displacementScale = std::atof(argv[argIdx + 1]);
            argIdx += 1;
        }
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }



    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    constexpr bool enableGLDebugCallback = DEBUG_SELECT(true, false);

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    // EN: Create an OpenGL 4.6 core profile context.
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    if constexpr (enableGLDebugCallback)
        glfwWindowHint(GLFW_OPENGL_DEBUG_CONTEXT, GL_TRUE);

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
    GLFWwindow* window = glfwCreateWindow(
        static_cast<int32_t>(renderTargetSizeX * UIScaling),
        static_cast<int32_t>(renderTargetSizeY * UIScaling),
        "OptiX Utility - Displacement Micro Map", NULL, NULL);
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

    if constexpr (enableGLDebugCallback) {
        glu::enableDebugCallback(true);
        glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_NOTIFICATION, 0, nullptr, false);
    }

    // END: Initialize OpenGL and GLFW.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: 入力コールバックの設定。
    // EN: Set up input callbacks.

    glfwSetMouseButtonCallback(
        window,
        [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

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
    glfwSetCursorPosCallback(
        window,
        [](GLFWwindow* window, double x, double y) {
            g_mouseX = x;
            g_mouseY = y;
        });
    glfwSetKeyCallback(
        window,
        [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            uint64_t &frameIndex = *(uint64_t*)glfwGetWindowUserPointer(window);

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
    g_cameraPosition = make_float3(0, 6.0f, 6.0f);
    g_cameraOrientation = qRotateY(pi_v<float>) * qRotateX(pi_v<float> / 4.0f);

    // END: Set up input callbacks.
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
    guiStyle.DisabledAlpha = 0.1f;
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
    StreamChain<2> streamChain;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    streamChain.initialize(cuContext);
    CUstream stream = streamChain.waitAvailableAndGetCurrentStream();

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: Displacement Micro-Mapを使う場合、プリミティブ種別のフラグを適切に設定する必要がある。
    // EN: Appropriately setting primitive type flags is required when using displacement micro-map.
    pipeline.setPipelineOptions(
        std::max(Shared::PrimaryRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_NONE/*DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE)*/,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE,
        optixu::UseMotionBlur::No);

    const std::vector<char> optixIr =
        readBinaryFile(exeDir / "displacement_micro_map/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT/*DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT)*/,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE/*DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE)*/);

    optixu::Module emptyModule;

    optixu::Program rayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

    optixu::Program missProgram = pipeline.createMissProgram(
        moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));

    pipeline.link(2);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    constexpr bool useBlockCompressedTexture = true;

    optixu::Material defaultMat = optixContext.createMaterial();
    defaultMat.setHitGroup(Shared::RayType_Primary, shadingHitProgramGroup);
    defaultMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer dmmBuildScratchMem;
    cudau::Buffer asBuildScratchMem;

    constexpr optixu::IndexSize dmmIndexSize = optixu::IndexSize::k2Bytes;

    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        struct MaterialGroup {
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            optixu::GeometryInstance optixGeomInst;
            cudau::Array texArray;
            CUtexObject texObj;
            cudau::Array heightTexArray;
            CUtexObject heightTexObj;

            optixu::DisplacementMicroMapArray optixDmmArray;
            cudau::Buffer dmmArrayMem;

            // JP: これらはDMM Arrayがビルドされた時点で不要になる。
            // EN: These are disposable once the DMM array is built.
            cudau::Buffer rawDmmArray;
            cudau::TypedBuffer<OptixDisplacementMicromapDesc> dmmDescs;

            // JP: これらはDMM Arrayが関連づくGASがビルドされた時点で不要になる。
            // EN: These are disposable once the GAS to which the DMM array associated is built.
            cudau::Buffer dmmIndexBuffer;
            cudau::TypedBuffer<float2> dmmVertexBiasAndScaleBuffer;
            cudau::TypedBuffer<OptixDisplacementMicromapFlags> dmmTriangleFlagsBuffer;
        };
        std::vector<MaterialGroup> matGroups;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            for (auto it = matGroups.rbegin(); it != matGroups.rend(); ++it) {
                it->dmmTriangleFlagsBuffer.finalize();
                it->dmmVertexBiasAndScaleBuffer.finalize();
                it->dmmArrayMem.finalize();
                it->dmmIndexBuffer.finalize();
                it->dmmDescs.finalize();
                it->rawDmmArray.finalize();
                it->optixDmmArray.destroy();

                if (it->heightTexObj) {
                    CUDADRV_CHECK(cuTexObjectDestroy(it->heightTexObj));
                    it->heightTexArray.finalize();
                }
                if (it->texObj) {
                    CUDADRV_CHECK(cuTexObjectDestroy(it->texObj));
                    it->texArray.finalize();
                }
                it->triangleBuffer.finalize();
                it->optixGeomInst.destroy();
            }
            vertexBuffer.finalize();
        }
    };

    Geometry floor;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floor.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        floor.optixGas = scene.createGeometryAccelerationStructure();
        floor.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        floor.optixGas.setNumMaterialSets(1);
        floor.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        Geometry::MaterialGroup group;
        {
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = floor.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.texture = 0;
            geomData.albedo = float3(0.8f, 0.8f, 0.8f);

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(floor.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            floor.optixGas.addChild(group.optixGeomInst);
            floor.matGroups.push_back(std::move(group));
        }

        floor.optixGas.prepareForBuild(&asMemReqs);
        floor.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry displacedMesh;
    {
        std::filesystem::path filePath = R"(../../data/stanford_bunny_309_faces_smooth.obj)";
        std::filesystem::path fileDir = filePath.parent_path();

        std::vector<obj::Vertex> vertices;
        std::vector<obj::MaterialGroup> matGroups;
        std::vector<obj::Material> materials;
        obj::load(filePath, &vertices, &matGroups, &materials);

        displacedMesh.vertexBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            reinterpret_cast<Shared::Vertex*>(vertices.data()), vertices.size());

        // JP: DMMを適用するジオメトリやそれを含むGASは通常の三角形用のもので良い。
        // EN: Geometry and GAS to which DMM applied are ones for ordinary triangle mesh.
        displacedMesh.optixGas = scene.createGeometryAccelerationStructure();
        displacedMesh.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        displacedMesh.optixGas.setNumMaterialSets(1);
        displacedMesh.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        uint32_t maxNumTrianglesPerGroup = 0;
        for (int groupIdx = 0; groupIdx < matGroups.size(); ++groupIdx) {
            const obj::MaterialGroup &srcGroup = matGroups[groupIdx];
            maxNumTrianglesPerGroup = std::max(
                maxNumTrianglesPerGroup,
                static_cast<uint32_t>(srcGroup.triangles.size()));
        }

        size_t scratchMemSizeForDMM = getScratchMemSizeForDMMGenerator(maxNumTrianglesPerGroup);
        cudau::Buffer scratchMemForDMM;
        scratchMemForDMM.initialize(cuContext, cudau::BufferType::Device, scratchMemSizeForDMM, 1);

        for (int groupIdx = 0; groupIdx < matGroups.size(); ++groupIdx) {
            const obj::MaterialGroup &srcGroup = matGroups[groupIdx];
            const obj::Material &srcMat = materials[srcGroup.materialIndex];
            const uint32_t numTriangles = srcGroup.triangles.size();

            Geometry::MaterialGroup group;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device,
                reinterpret_cast<const Shared::Triangle*>(srcGroup.triangles.data()),
                numTriangles);

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = displacedMesh.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.albedo = float3(srcMat.diffuse[0], srcMat.diffuse[1], srcMat.diffuse[2]);
            if (!srcMat.diffuseTexPath.empty()) {
                int32_t width, height, n;
                uint8_t* linearImageData = stbi_load(
                    srcMat.diffuseTexPath.string().c_str(),
                    &width, &height, &n, 4);
                group.texArray.initialize2D(
                    cuContext, cudau::ArrayElementType::UInt8, 4,
                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                    width, height, 1);
                group.texArray.write<uint8_t>(linearImageData, width * height * 4);
                stbi_image_free(linearImageData);

                cudau::TextureSampler texSampler;
                texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
                texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
                texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);
                texSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
                texSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
                group.texObj = texSampler.createTextureObject(group.texArray);
                geomData.texture = group.texObj;
            }
            if (!srcMat.bumpTexPath.empty()) {
                int32_t width, height, n;
                uint8_t* linearImageData = stbi_load(
                    srcMat.bumpTexPath.string().c_str(),
                    &width, &height, &n, 1);
                group.heightTexArray.initialize2D(
                    cuContext, cudau::ArrayElementType::UInt8, 1,
                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                    width, height, 1);
                group.heightTexArray.write<uint8_t>(linearImageData, width * height);
                stbi_image_free(linearImageData);

                cudau::TextureSampler texSampler;
                texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
                texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
                texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);
                texSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
                texSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
                group.heightTexObj = texSampler.createTextureObject(group.heightTexArray);
                //geomData.heightTexture = group.heightTexObj;
            }

            // JP: まずは各三角形のDMMフォーマットを決定する。
            // EN: Fisrt, determine the DMM format of each triangle.
            DMMGeneratorContext dmmContext;
            uint32_t histInDMMArray[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels];
            uint32_t histInMesh[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels];
            uint64_t rawDmmArraySize = 0;
            if (useDMM) {
                initializeDMMGeneratorContext(
                    exeDir / "displacement_micro_map/ptxes",
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, position),
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, texCoord),
                    sizeof(Shared::Vertex), vertices.size(),
                    group.triangleBuffer.getCUdeviceptr(), sizeof(Shared::Triangle), numTriangles,
                    group.heightTexObj,
                    make_uint2(group.heightTexArray.getWidth(), group.heightTexArray.getHeight()), 1, 0,
                    maxCompressedFormat,
                    shared::DMMSubdivLevel_0, maxDmmSubDivLevel, dmmSubdivLevelBias,
                    useDmmIndexBuffer, 1 << static_cast<uint32_t>(dmmIndexSize),
                    scratchMemForDMM.getCUdeviceptr(), scratchMemForDMM.sizeInBytes(),
                    &dmmContext);

                countDMMFormats(dmmContext, histInDMMArray, histInMesh, &rawDmmArraySize);
            }

            std::vector<OptixDisplacementMicromapUsageCount> dmmUsageCounts;
            hpprintf("(%u tris): DMM %s\n",
                     numTriangles, rawDmmArraySize > 0 ? "Enabled" : "Disabled");
            hpprintf("DMM Array Size: %llu [bytes]\n", rawDmmArraySize);
            if (rawDmmArraySize > 0) {
                uint32_t numDmms = 0;
                std::vector<OptixDisplacementMicromapHistogramEntry> dmmHistogramEntries;
                hpprintf("Histogram in DMM Array, Mesh\n");
                hpprintf("         None    : %5u, %5u\n",
                            histInDMMArray[shared::DMMEncoding_None][0],
                            histInMesh[shared::DMMEncoding_None][0]);
                for (int enc = shared::DMMEncoding_64B_per_64MicroTris; enc <= shared::DMMEncoding_128B_per_1024MicroTris; ++enc) {
                    for (int level = shared::DMMSubdivLevel_0; level <= shared::DMMSubdivLevel_5; ++level) {
                        uint32_t countInDmmArray = histInDMMArray[enc][level];
                        uint32_t countInMesh = histInMesh[enc][level];
                        hpprintf("  Enc %u - Level %u: %5u, %5u\n", enc, level, countInDmmArray, countInMesh);

                        if (countInDmmArray > 0) {
                            OptixDisplacementMicromapHistogramEntry histEntry;
                            histEntry.count = countInDmmArray;
                            histEntry.format = static_cast<OptixDisplacementMicromapFormat>(enc);
                            histEntry.subdivisionLevel = level;
                            dmmHistogramEntries.push_back(histEntry);

                            numDmms += histInDMMArray[enc][level];
                        }

                        if (countInMesh > 0) {
                            OptixDisplacementMicromapUsageCount histEntry;
                            histEntry.count = countInMesh;
                            histEntry.format = static_cast<OptixDisplacementMicromapFormat>(enc);
                            histEntry.subdivisionLevel = level;
                            dmmUsageCounts.push_back(histEntry);
                        }
                    }
                }
                hpprintf("\n");

                group.optixDmmArray = scene.createDisplacementMicroMapArray();

                OptixMicromapBufferSizes dmmArraySizes;
                group.optixDmmArray.setConfiguration(OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE);
                group.optixDmmArray.computeMemoryUsage(
                    dmmHistogramEntries.data(), dmmHistogramEntries.size(), &dmmArraySizes);
                group.dmmArrayMem.initialize(
                    cuContext, cudau::BufferType::Device, dmmArraySizes.outputSizeInBytes, 1);

                // JP: このサンプルではASビルド用のスクラッチメモリをDMMビルドにも再利用する。
                // EN: This sample reuses the scratch memory for AS builds also for DMM builds.
                maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dmmArraySizes.tempSizeInBytes);



                group.rawDmmArray.initialize(cuContext, cudau::BufferType::Device, rawDmmArraySize, 1);
                group.dmmDescs.initialize(cuContext, cudau::BufferType::Device, numDmms);
                geomData.dmmDescBuffer = group.dmmDescs.getDevicePointer();
                if (useDmmIndexBuffer) {
                    group.dmmIndexBuffer.initialize(
                        cuContext, cudau::BufferType::Device,
                        numTriangles, 1 << static_cast<uint32_t>(dmmIndexSize));
                    geomData.dmmIndexBuffer = group.dmmIndexBuffer.getCUdeviceptr();
                    geomData.dmmIndexSize = 1 << static_cast<uint32_t>(dmmIndexSize);
                }
                group.optixDmmArray.setBuffers(group.rawDmmArray, group.dmmDescs, group.dmmArrayMem);

                group.dmmTriangleFlagsBuffer.initialize(cuContext, cudau::BufferType::Device, numTriangles);

                // JP: 各三角形のDMMを生成する。
                // EN: Generate an DMM for each triangle.
                generateDMMArray(
                    dmmContext,
                    group.rawDmmArray, group.dmmDescs, group.dmmIndexBuffer,
                    group.dmmTriangleFlagsBuffer);

                /*
                JP: 頂点ごとにディスプレイスメントのスケールと事前移動量を指定できる。
                    DMMに記録されているマイクロ頂点ごとの変位量と併せて、ディスプレイスメント適用後
                    のメッシュを最小限に含むように調節することでより高効率かつ高精度なレイトレースが可能となる。
                    が、このサンプルではシンプルにグローバルな値を指定する。
                EN: Specify displacement scale and the amount of pre-movement per vertex.
                    These amounts should be adjusted along with displacement amounts per micro-vertices in DMM
                    so that these tightly encapsulates the diplaced mesh for faster and more precise ray tracing.
                    However, this sample simply specifies globally uniform values.
                */
                std::vector<float2> vertexBiasAndScaleBuffer(
                    vertices.size(), float2(displacementBias, displacementScale));
                group.dmmVertexBiasAndScaleBuffer.initialize(
                    cuContext, cudau::BufferType::Device, vertexBiasAndScaleBuffer);
            }

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(displacedMesh.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            // JP: DMM ArrayをGeometryInstanceにセットする。
            // EN: Set the DMM array to the geometry instance.
            if (useDMM && group.optixDmmArray &&
                visualizationMode != Shared::VisualizationMode_Barycentric)
                group.optixGeomInst.setDisplacementMicroMapArray(
                    // JP: 頂点ごとのディスプレイスメント方向として法線ベクトルを再利用する。
                    // EN: Reuse the normal vectors as displacement directions per vertex.
                    optixu::BufferView(
                        displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, normal),
                        displacedMesh.vertexBuffer.numElements(),
                        sizeof(Shared::Vertex)),
                    group.dmmVertexBiasAndScaleBuffer,
                    group.dmmTriangleFlagsBuffer,
                    group.optixDmmArray, dmmUsageCounts.data(), dmmUsageCounts.size(),
                    useDmmIndexBuffer ? group.dmmIndexBuffer : optixu::BufferView(),
                    dmmIndexSize, 0,
                    OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3,
                    OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            displacedMesh.optixGas.addChild(group.optixGeomInst);
            displacedMesh.matGroups.push_back(std::move(group));
        }

        displacedMesh.optixGas.prepareForBuild(&asMemReqs);
        displacedMesh.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを基にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floor.optixGas);

    optixu::Instance displacedMeshInst = scene.createInstance();
    displacedMeshInst.setChild(displacedMesh.optixGas);
    float xfm[] = {
        0.05f, 0.0f, 0.0f, 0,
        0.0f, 0.05f, 0.0f, 0,
        0.0f, 0.0f, 0.05f, 0,
    };
    displacedMeshInst.setTransform(xfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(floorInst);
    ias.addChild(displacedMeshInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Displacement Micro-Map Arrayをビルドする。
    // EN: Build displacement micro-map arrays.
    for (int i = 0; i < displacedMesh.matGroups.size(); ++i) {
        const Geometry::MaterialGroup &group = displacedMesh.matGroups[i];
        if (!group.optixDmmArray)
            continue;

        group.optixDmmArray.rebuild(stream, asBuildScratchMem);
    }



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    floor.optixGas.rebuild(stream, floor.gasMem, asBuildScratchMem);
    displacedMesh.optixGas.rebuild(stream, displacedMesh.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        Geometry* geom;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &floor, 0, 0 },
        { &displacedMesh, 0, 0 },
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.geom->optixGas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.geom->optixGas.compact(
            stream,
            optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].geom->optixGas.removeUncompacted();
        gasList[i].geom->gasMem.finalize();
    }



    // JP: IASビルド時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when building an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    OptixTraversableHandle travHandle = ias.rebuild(stream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    glu::Texture2D outputTexture;
    cudau::Array outputArray;
    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
    outputArray.initializeFromGLTexture2D(
        cuContext, outputTexture.getHandle(),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    outputBufferSurfaceHolder.initialize({ &outputArray });

    glu::Sampler outputSampler;
    outputSampler.initialize(
        glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
        glu::Sampler::WrapMode::Repeat, glu::Sampler::WrapMode::Repeat);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    glu::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    glu::GraphicsProgram drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(
        readTxtFile(exeDir / "displacement_micro_map/shaders/drawOptiXResult.vert"),
        readTxtFile(exeDir / "displacement_micro_map/shaders/drawOptiXResult.frag"));


    
    const auto computeHaltonSequence = []
    (uint32_t base, uint32_t idx) {
        const float recBase = 1.0f / base;
        float ret = 0.0f;
        float scale = 1.0f;
        while (idx) {
            scale *= recBase;
            ret += (idx % base) * scale;
            idx /= base;
        }
        return ret;
    };
    float2 subPixelOffsets[64];
    for (int i = 0; i < lengthof(subPixelOffsets); ++i)
        subPixelOffsets[i] = float2(computeHaltonSequence(2, i), computeHaltonSequence(3, i));

    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.lightDirection = normalize(float3(-2, 5, 2));
    plp.lightRadiance = float3(7.5f, 7.5f, 7.5f);
    plp.envRadiance = float3(0.10f, 0.13f, 0.9f);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    while (true) {
        uint32_t bufferIndex = frameIndex % 2;

        if (glfwWindowShouldClose(window))
            break;
        glfwPollEvents();

        CUstream curStream = streamChain.waitAvailableAndGetCurrentStream();

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

            glFinish();
            streamChain.waitAllWorkDone();

            outputTexture.finalize();
            outputTexture.initialize(GL_RGBA32F, renderTargetSizeX, renderTargetSizeY, 1);
            outputArray.finalize();
            outputArray.initializeFromGLTexture2D(
                cuContext, outputTexture.getHandle(),
                cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

            outputArray.resize(renderTargetSizeX, renderTargetSizeY);
            plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



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
            g_cameraPositionalMovingSpeed = std::clamp(g_cameraPositionalMovingSpeed, 1e-6f, 1e+6f);

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
            g_tempCameraOrientation =
                g_cameraOrientation
                * qRotate(g_cameraDirectionalMovingSpeed * deltaAngle, axis);
            g_cameraPosition +=
                g_tempCameraOrientation.toMatrix3x3()
                * (g_cameraPositionalMovingSpeed * make_float3(trackX, trackY, trackZ));
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



        // Camera Window
        {
            ImGui::Begin("Camera & Rendering", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&g_cameraPosition));
            static float rollPitchYaw[3];
            g_tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw))
                g_cameraOrientation = qFromEulerAngles(
                    rollPitchYaw[0] * pi_v<float> / 180,
                    rollPitchYaw[1] * pi_v<float> / 180,
                    rollPitchYaw[2] * pi_v<float> / 180);
            ImGui::Text("Pos. Speed (T/G): %g", g_cameraPositionalMovingSpeed);

            ImGui::End();
        }

        bool visModeChanged = false;
        {
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("Buffer to Display");
            visModeChanged |= ImGui::RadioButtonE(
                "Final", &visualizationMode, Shared::VisualizationMode_Final);
            visModeChanged |= ImGui::RadioButtonE(
                "Barycentric", &visualizationMode, Shared::VisualizationMode_Barycentric);
            visModeChanged |= ImGui::RadioButtonE(
                "Micro-Barycentric", &visualizationMode, Shared::VisualizationMode_MicroBarycentric);
            visModeChanged |= ImGui::RadioButtonE(
                "Subdivision Level", &visualizationMode, Shared::VisualizationMode_SubdivLevel);
            visModeChanged |= ImGui::RadioButtonE(
                "Normal", &visualizationMode, Shared::VisualizationMode_Normal);

            ImGui::End();
        }



        bool firstAccumFrame =
            cameraIsActuallyMoving ||
            resized ||
            frameIndex == 0 ||
            visModeChanged;
        bool isNewSequence = resized || frameIndex == 0;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;

        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        // Render
        {
            //curGPUTimer.render.start(curStream);

            plp.colorAccumBuffer = outputBufferSurfaceHolder.getNext();
            plp.visualizationMode = visualizationMode;
            plp.subPixelOffset = subPixelOffsets[numAccumFrames % static_cast<uint32_t>(lengthof(subPixelOffsets))];
            plp.sampleIndex = numAccumFrames;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
            pipeline.launch(curStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
            ++numAccumFrames;

            //curGPUTimer.render.stop(curStream);
        }

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);



        // ----------------------------------------------------------------
        // JP: OptiXによる描画結果を表示用レンダーターゲットにコピーする。
        // EN: Copy the OptiX rendering results to the display render target.

        if (visualizationMode == Shared::VisualizationMode_Final) {
            glEnable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyleWithGamma;
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = guiStyle;
        }

        glViewport(0, 0, curFBWidth, curFBHeight);

        glUseProgram(drawOptiXResultShader.getHandle());

        glUniform2ui(0, curFBWidth, curFBHeight);

        glBindTextureUnit(0, outputTexture.getHandle());
        glBindSampler(0, outputSampler.getHandle());

        glBindVertexArray(vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        // END: Copy the OptiX rendering results to the display render target.
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);
        streamChain.swap();

        ++frameIndex;
    }

    streamChain.waitAllWorkDone();



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    displacedMeshInst.destroy();
    floorInst.destroy();

    displacedMesh.finalize();
    floor.finalize();

    scene.destroy();

    defaultMat.destroy();



    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    streamChain.finalize();
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
