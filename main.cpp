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
    char str[1024];
    vsprintf_s(str, fmt, args);
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
struct SBTRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) uint8_t header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RayGenSBTRecord = SBTRecord<Shared::RayGenData>;
using EmptySBTRecord = SBTRecord<int32_t>;



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
    pipelineCompileOptions.numPayloadValues = 0;
    pipelineCompileOptions.numAttributeValues = 0;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "plp";
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;

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
    }

    // JP: 空のMissプログラム
    // EN: Dummy Miss Program
    OptixProgramGroup pgMiss = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgMiss));
    }

    // JP: 空のヒットグループ
    // EN: Dummy Hit Group
    OptixProgramGroup pgHitGroup = nullptr;
    {
        OptixProgramGroupOptions programGroupOptions = {};

        OptixProgramGroupDesc progGroupDesc = {};
        progGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixProgramGroupCreate(optixContext,
                                                &progGroupDesc, 1, // num program groups
                                                &programGroupOptions,
                                                log, &logSize,
                                                &pgHitGroup));
    }



    OptixPipeline pipeline = nullptr;
    {
        OptixProgramGroup programGroups[] = { pgRayGen };

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 0;
        pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
        pipelineLinkOptions.overrideUsesMotionBlur = false;
        logSize = sizeof(log);
        OPTIX_CHECK_LOG(optixPipelineCreate(optixContext,
                                            &pipelineCompileOptions,
                                            &pipelineLinkOptions,
                                            programGroups, lengthof(programGroups),
                                            log, &logSize,
                                            &pipeline));
    }



    OptixShaderBindingTable sbt = {};
    {
        CUdeviceptr rayGenSBTBuffer;
        CUDA_CHECK(cudaMalloc((void**)&rayGenSBTBuffer, sizeof(RayGenSBTRecord)));
        // JP: RayGen用レコードはこのサンプルでは毎フレーム更新する。
        // EN: This sample updates a record for raygen every frame.
        //CUDA_CHECK(cudaMemcpy((void*)rayGenSBTBuffer, &rayGenSBTR, sizeof(rayGenSBTR), cudaMemcpyHostToDevice));

        EmptySBTRecord missSBTR;
        OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss, &missSBTR));
        missSBTR.data = 0;

        CUdeviceptr missSBTBuffer;
        CUDA_CHECK(cudaMalloc((void**)&missSBTBuffer, sizeof(missSBTR)));
        CUDA_CHECK(cudaMemcpy((void*)missSBTBuffer, &missSBTR, sizeof(missSBTR), cudaMemcpyHostToDevice));

        EmptySBTRecord hitGroupSBTR;
        OPTIX_CHECK(optixSbtRecordPackHeader(pgHitGroup, &hitGroupSBTR));
        hitGroupSBTR.data = 0;

        CUdeviceptr hitGroupSBTBuffer;
        CUDA_CHECK(cudaMalloc((void**)&hitGroupSBTBuffer, sizeof(hitGroupSBTR)));
        CUDA_CHECK(cudaMemcpy((void*)hitGroupSBTBuffer, &hitGroupSBTR, sizeof(hitGroupSBTR), cudaMemcpyHostToDevice));

        sbt.raygenRecord = rayGenSBTBuffer;
        sbt.missRecordBase = missSBTBuffer;
        sbt.missRecordStrideInBytes = sizeof(missSBTR);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = hitGroupSBTBuffer;
        sbt.hitgroupRecordStrideInBytes = sizeof(hitGroupSBTR);
        sbt.hitgroupRecordCount = 1;
    }

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------


    
    // JP: デフォルトストリーム
    // EN: default stream
    CUstream stream = 0;
    //CUDA_CHECK(cudaStreamCreate(&stream));



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
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.outputBuffer = (float4*)outputBufferCUDA.getDevicePointer();

    CUdeviceptr plpOnDevice;
    CUDA_CHECK(cudaMalloc((void**)&plpOnDevice, sizeof(plp)));

    RayGenSBTRecord rayGenSBTR;
    OPTIX_CHECK(optixSbtRecordPackHeader(pgRayGen, &rayGenSBTR));



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



        rayGenSBTR.data = { 1.0f, 1.0f, (frameIndex % 300) / 300.0f };
        CUDA_CHECK(cudaMemcpyAsync((void*)sbt.raygenRecord, &rayGenSBTR, sizeof(rayGenSBTR), cudaMemcpyHostToDevice, stream));

        plp.outputBuffer = (float4*)outputBufferCUDA.mapOnDevice(stream);

        CUDA_CHECK(cudaMemcpyAsync((void*)plpOnDevice, &plp, sizeof(plp), cudaMemcpyHostToDevice, stream));
        OPTIX_CHECK(optixLaunch(pipeline, stream, plpOnDevice, sizeof(plp), &sbt, renderTargetSizeX, renderTargetSizeY, 1));

        outputBufferCUDA.unmapOnDevice(stream);



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

    CUDA_CHECK(cudaFree((void*)sbt.hitgroupRecordBase));
    CUDA_CHECK(cudaFree((void*)sbt.missRecordBase));
    CUDA_CHECK(cudaFree((void*)sbt.raygenRecord));

    OPTIX_CHECK(optixPipelineDestroy(pipeline));

    OPTIX_CHECK(optixProgramGroupDestroy(pgHitGroup));
    OPTIX_CHECK(optixProgramGroupDestroy(pgMiss));
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
