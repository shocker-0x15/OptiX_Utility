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



// Platform defines
#if defined(_WIN32) || defined(_WIN64)
#    define HP_Platform_Windows
#    if defined(_MSC_VER)
#        define HP_Platform_Windows_MSVC
#    endif
#elif defined(__APPLE__)
#    define HP_Platform_macOS
#endif

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



float sRGB_degamma_s(float value) {
    Assert(value >= 0, "Input value must be equal to or greater than 0: %g", value);
    if (value <= 0.04045f)
        return value / 12.92f;
    return std::pow((value + 0.055f) / 1.055f, 2.4f);
};



int32_t main(int32_t argc, const char* argv[]) {
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        return -1;
    }

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();

    // JP: OpenGL 4.6 Core Profileのコンテキストを作成する。
    const uint32_t OpenGLMajorVersion = 4;
    const uint32_t OpenGLMinorVersion = 6;
    const char* glsl_version = "#version 460";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, OpenGLMajorVersion);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, OpenGLMinorVersion);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SRGB_CAPABLE, GLFW_TRUE);
#if defined(HP_Platform_macOS)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    int32_t renderTargetSizeX = 1280;
    int32_t renderTargetSizeY = 720;

    // JP: ウインドウの初期化。
    //     HiDPIディスプレイに対応する。
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
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        return -1;
    }

    glEnable(GL_FRAMEBUFFER_SRGB);
    GLTK::errorCheck();



    // Setup ImGui binding
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
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



    GLTK::Buffer outputBufferGL;
    GLTK::BufferTexture outputTexture;
    CUDAHelper::Buffer outputBufferCUDA;
    outputBufferGL.initialize(GLTK::Buffer::Target::ArrayBuffer, 16, renderTargetSizeX * renderTargetSizeY, nullptr, GLTK::Buffer::Usage::StreamDraw);
    outputBufferCUDA.initialize(CUDAHelper::BufferType::GL_Interop, renderTargetSizeX, renderTargetSizeY, 16, outputBufferGL.getRawHandle());
    outputTexture.initialize(outputBufferGL, GLTK::SizedInternalFormat::RGB32F);
    
    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    GLTK::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    const filesystem::path exeDir = getExecutableDirectory();

    // JP: HiDPIディスプレイで過剰なレンダリング負荷になってしまうため低解像度フレームバッファーを作成する。
    GLTK::FrameBuffer frameBuffer;
    frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

    // JP: アップスケール用のシェーダー。
    GLTK::GraphicsShader scaleShader;
    scaleShader.initializeVSPS(readTxtFile(exeDir / "shaders/scale.vert"),
                               readTxtFile(exeDir / "shaders/scale.frag"));

    // JP: アップスケール用のサンプラー。
    //     texelFetch()を使う場合には設定値は無関係。だがバインドは必要な様子。
    GLTK::Sampler scaleSampler;
    scaleSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest, GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



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

            frameBuffer.finalize();

            frameBuffer.initialize(renderTargetSizeX, renderTargetSizeY, GL_RGBA8, GL_DEPTH_COMPONENT32);

            resized = true;
        }



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



    scaleSampler.finalize();
    scaleShader.finalize();
    frameBuffer.finalize();

    vertexArrayForFullScreen.finalize();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    
    glfwTerminate();

    return 0;
}