#include "gui_common.h"

void GUIFramework::initialize(const InitialConfig &initConfig) {
    // ----------------------------------------------------------------
    // JP: OpenGL, GLFWの初期化。
    // EN: Initialize OpenGL and GLFW.

    glfwSetErrorCallback(
        [](int32_t error, const char* description) {
            hpprintf("Error %d: %s\n", error, description);
        });
    if (!glfwInit()) {
        hpprintf("Failed to initialize GLFW.\n");
        exit(-1);
    }

    m_primaryMonitor = glfwGetPrimaryMonitor();

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

    m_windowContentRenderWidth = initConfig.windowContentRenderWidth;
    m_windowContentRenderHeight = initConfig.windowContentRenderHeight;

    float primaryContentScaleX, primaryContentScaleY;
    glfwGetMonitorContentScale(m_primaryMonitor, &primaryContentScaleX, &primaryContentScaleY);
    m_windowContentScale = primaryContentScaleX;
    m_windowContentWidth = static_cast<int32_t>(m_windowContentRenderWidth * m_windowContentScale);
    m_windowContentHeight = static_cast<int32_t>(m_windowContentRenderHeight * m_windowContentScale);

    // JP: ウインドウの初期化。
    //     HiDPIディスプレイに対応する。
    // EN: Initialize a window.
    //     Support Hi-DPI display.
    m_window = glfwCreateWindow(
        m_windowContentWidth, m_windowContentHeight,
        initConfig.windowTitle, NULL, NULL);
    glfwSetWindowUserPointer(m_window, this);
    if (!m_window) {
        hpprintf("Failed to create a GLFW window.\n");
        glfwTerminate();
        exit(-1);
    }

    glfwMakeContextCurrent(m_window);

    glfwSwapInterval(1); // Enable vsync



    // JP: gl3wInit()は何らかのOpenGLコンテキストが作られた後に呼ぶ必要がある。
    // EN: gl3wInit() must be called after some OpenGL context has been created.
    int32_t gl3wRet = gl3wInit();
    if (!gl3wIsSupported(OpenGLMajorVersion, OpenGLMinorVersion)) {
        hpprintf("gl3w doesn't support OpenGL %u.%u\n", OpenGLMajorVersion, OpenGLMinorVersion);
        glfwTerminate();
        exit(-1);
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
        m_window,
        [](GLFWwindow* window, int32_t button, int32_t action, int32_t mods) {
            auto framework = reinterpret_cast<GUIFramework*>(glfwGetWindowUserPointer(window));
            framework->mouseButtonCallback(button, action, mods);
        });
    glfwSetCursorPosCallback(
        m_window,
        [](GLFWwindow* window, double x, double y) {
            auto framework = reinterpret_cast<GUIFramework*>(glfwGetWindowUserPointer(window));
            framework->cursorPosCallback(x, y);
        });
    glfwSetKeyCallback(
        m_window,
        [](GLFWwindow* window, int32_t key, int32_t scancode, int32_t action, int32_t mods) {
            auto framework = reinterpret_cast<GUIFramework*>(glfwGetWindowUserPointer(window));
            framework->keyCallback(key, scancode, action, mods);
        });

    m_cameraPositionalMovingSpeed = initConfig.cameraMovingSpeed;
    m_cameraDirectionalMovingSpeed = 0.0015f;
    m_cameraTiltSpeed = 0.025f;
    m_cameraPosition = initConfig.cameraPosition;
    m_cameraOrientation = initConfig.cameraOrientation;

    // END: Set up input callbacks.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: ImGuiの初期化。
    // EN: Initialize ImGui.

    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Setup style
    // JP: ガンマ補正が有効なレンダーターゲットで、同じUIの見た目を得るためにデガンマされたスタイルも用意する。
    // EN: Prepare a degamma-ed style to have the identical UI appearance on gamma-corrected render target.
    ImGui::StyleColorsDark(&m_guiStyleBase);
    m_guiStyleBase.DisabledAlpha = 0.1f;
    m_guiStyleWithGammaBase = m_guiStyleBase;
    const auto degamma = []
    (const ImVec4 &color) {
        return ImVec4(sRGB_degamma_s(color.x),
                      sRGB_degamma_s(color.y),
                      sRGB_degamma_s(color.z),
                      color.w);
    };
    for (int i = 0; i < ImGuiCol_COUNT; ++i) {
        m_guiStyleWithGammaBase.Colors[i] = degamma(m_guiStyleWithGammaBase.Colors[i]);
    }
    m_guiStyle = m_guiStyleBase;
    m_guiStyleWithGamma = m_guiStyleWithGammaBase;
    m_guiStyle.ScaleAllSizes(m_windowContentScale);
    m_guiStyleWithGamma.ScaleAllSizes(m_windowContentScale);
    ImGui::GetStyle() = m_guiStyleWithGamma;

    // END: Initialize ImGui.
    // ----------------------------------------------------------------



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    m_vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    m_drawOptiXResultShader.initializeVSPS(
        readTxtFile(initConfig.resourceDir / "shaders/drawOptiXResult.vert"),
        readTxtFile(initConfig.resourceDir / "shaders/drawOptiXResult.frag"));
    
    m_outputTexture.initialize(GL_RGBA32F, m_windowContentRenderWidth, m_windowContentRenderHeight, 1);

    m_outputSampler.initialize(
        glu::Sampler::MinFilter::Nearest, glu::Sampler::MagFilter::Nearest,
        glu::Sampler::WrapMode::ClampToEdge, glu::Sampler::WrapMode::ClampToEdge);



    m_streamChain.initialize(initConfig.cuContext);
}

void GUIFramework::finalize() {
    m_streamChain.finalize();

    m_outputSampler.finalize();
    m_outputTexture.finalize();
    m_drawOptiXResultShader.finalize();
    m_vertexArrayForFullScreen.finalize();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwTerminate();
}



void GUIFramework::mouseButtonCallback(int32_t button, int32_t action, int32_t mods) {
    switch (button) {
    case GLFW_MOUSE_BUTTON_MIDDLE: {
        devPrintf("Mouse Middle\n");
        m_buttonRotate.recordStateChange(action == GLFW_PRESS, m_frameIndex);
        break;
    }
    default:
        break;
    }
}

void GUIFramework::cursorPosCallback(double x, double y) {
    m_mouseX = x / m_windowContentScale;
    m_mouseY = y / m_windowContentScale;
}

void GUIFramework::keyCallback(int32_t key, int32_t scancode, int32_t action, int32_t mods) {
    switch (key) {
    case GLFW_KEY_W: {
        m_keyForward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_S: {
        m_keyBackward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_A: {
        m_keyLeftward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_D: {
        m_keyRightward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_R: {
        m_keyUpward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_F: {
        m_keyDownward.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_Q: {
        m_keyTiltLeft.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_E: {
        m_keyTiltRight.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_T: {
        m_keyFasterPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    case GLFW_KEY_G: {
        m_keySlowerPosMovSpeed.recordStateChange(action == GLFW_PRESS || action == GLFW_REPEAT, m_frameIndex);
        break;
    }
    default:
        break;
    }
}



void GUIFramework::run(
    const std::function<ReturnValuesToRenderLoop(const RunArguments &args)> &updateAndRender,
    const std::function<void(CUstream, uint64_t, int32_t, int32_t)> &resizeCallback) {
    m_frameIndex = 0;

    while (true) {
        if (glfwWindowShouldClose(m_window))
            break;

        glfwPollEvents();

        CUstream curStream = m_streamChain.waitAvailableAndGetCurrentStream();

        // JP: リサイズ"完了"時にのみコールバックするためにGLFWのコールバックは使わない。
        // EN: Don't use the GLFW callback to callback only when resize "completes".
        bool resized = false;
        float newContentScaleX, newContentScaleY;
        glfwGetWindowContentScale(m_window, &newContentScaleX, &newContentScaleY);
        int32_t newFBWidth, newFBHeight;
        glfwGetFramebufferSize(m_window, &newFBWidth, &newFBHeight);
        if (newFBWidth != m_windowContentWidth || newFBHeight != m_windowContentHeight ||
            newContentScaleX != m_windowContentScale) {
            m_windowContentScale = newContentScaleX;
            m_windowContentWidth = newFBWidth;
            m_windowContentHeight = newFBHeight;
            m_windowContentRenderWidth = static_cast<int32_t>(newFBWidth / m_windowContentScale);
            m_windowContentRenderHeight = static_cast<int32_t>(newFBHeight / m_windowContentScale);

            glFinish();
            m_streamChain.waitAllWorkDone();

            m_outputTexture.finalize();
            m_outputTexture.initialize(GL_RGBA32F, m_windowContentRenderWidth, m_windowContentRenderHeight, 1);
            resizeCallback(curStream, m_frameIndex, m_windowContentRenderWidth, m_windowContentRenderHeight);

            m_guiStyle = m_guiStyleBase;
            m_guiStyleWithGamma = m_guiStyleWithGammaBase;
            m_guiStyle.ScaleAllSizes(m_windowContentScale);
            m_guiStyleWithGamma.ScaleAllSizes(m_windowContentScale);
            ImGui::GetIO().FontGlobalScale = m_windowContentScale;

            resized = true;
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();



        RunArguments args(
            m_mouseX,
            m_mouseY,
            m_cameraPositionalMovingSpeed,
            m_cameraDirectionalMovingSpeed,
            m_cameraTiltSpeed,
            m_cameraPosition,
            m_cameraOrientation);
        args.curStream = curStream;
        args.outputTexture = &m_outputTexture;
        args.windowContentRenderWidth = m_windowContentRenderWidth;
        args.windowContentRenderHeight = m_windowContentRenderHeight;
        args.frameIndex = m_frameIndex;
        args.resized = resized;

        static bool operatedCameraOnPrevFrame = false;
        {
            const auto decideDirection = [](const KeyState &a, const KeyState &b) {
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

            int32_t trackZ = decideDirection(m_keyForward, m_keyBackward);
            int32_t trackX = decideDirection(m_keyLeftward, m_keyRightward);
            int32_t trackY = decideDirection(m_keyUpward, m_keyDownward);
            int32_t tiltZ = decideDirection(m_keyTiltRight, m_keyTiltLeft);
            int32_t adjustPosMoveSpeed = decideDirection(m_keyFasterPosMovSpeed, m_keySlowerPosMovSpeed);

            m_cameraPositionalMovingSpeed *= 1.0f + 0.02f * adjustPosMoveSpeed;
            m_cameraPositionalMovingSpeed = std::clamp(m_cameraPositionalMovingSpeed, 1e-6f, 1e+6f);

            static double deltaX = 0, deltaY = 0;
            static double lastX, lastY;
            static double m_prevMouseX = m_mouseX, m_prevMouseY = m_mouseY;
            if (m_buttonRotate.getState() == true) {
                if (m_buttonRotate.getTime() == m_frameIndex) {
                    lastX = m_mouseX;
                    lastY = m_mouseY;
                }
                else {
                    deltaX = m_mouseX - lastX;
                    deltaY = m_mouseY - lastY;
                }
            }

            float deltaAngle = static_cast<float>(std::sqrt(deltaX * deltaX + deltaY * deltaY));
            float3 axis(static_cast<float>(deltaY), -static_cast<float>(deltaX), 0.0f);
            axis /= deltaAngle;
            if (deltaAngle == 0.0f)
                axis = float3(1, 0, 0);

            m_cameraOrientation = m_cameraOrientation * qRotateZ(m_cameraTiltSpeed * tiltZ);
            m_tempCameraOrientation = m_cameraOrientation *
                qRotate(m_cameraDirectionalMovingSpeed * deltaAngle, axis);
            m_cameraPosition += m_tempCameraOrientation.toMatrix3x3() *
                (m_cameraPositionalMovingSpeed
                 * float3(static_cast<float>(trackX), static_cast<float>(trackY), static_cast<float>(trackZ)));
            if (m_buttonRotate.getState() == false && m_buttonRotate.getTime() == m_frameIndex) {
                m_cameraOrientation = m_tempCameraOrientation;
                deltaX = 0;
                deltaY = 0;
            }

            args.tempCameraOrientation = m_tempCameraOrientation;

            args.operatingCamera =
                (m_keyForward.getState() || m_keyBackward.getState() ||
                 m_keyLeftward.getState() || m_keyRightward.getState() ||
                 m_keyUpward.getState() || m_keyDownward.getState() ||
                 m_keyTiltLeft.getState() || m_keyTiltRight.getState() ||
                 m_buttonRotate.getState());
            args.cameraIsActuallyMoving =
                (trackZ != 0 || trackX != 0 || trackY != 0 ||
                 tiltZ != 0 || (m_mouseX != m_prevMouseX) || (m_mouseY != m_prevMouseY))
                && args.operatingCamera;

            m_prevMouseX = m_mouseX;
            m_prevMouseY = m_mouseY;
        }



        ReturnValuesToRenderLoop ret = updateAndRender(args);
        if (ret.finish)
            break;



        // ----------------------------------------------------------------
        // JP: OptiXによる描画結果を表示用レンダーターゲットにコピーする。
        // EN: Copy the OptiX rendering results to the display render target.

        if (ret.enable_sRGB) {
            glEnable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = m_guiStyleWithGamma;
        }
        else {
            glDisable(GL_FRAMEBUFFER_SRGB);
            ImGui::GetStyle() = m_guiStyle;
        }

        glViewport(0, 0, m_windowContentWidth, m_windowContentHeight);

        glUseProgram(m_drawOptiXResultShader.getHandle());

        glUniform1f(0, m_windowContentScale);
        glUniform2i(1, m_windowContentWidth, m_windowContentHeight);

        glBindTextureUnit(0, m_outputTexture.getHandle());
        glBindSampler(0, m_outputSampler.getHandle());

        glBindVertexArray(m_vertexArrayForFullScreen.getHandle());
        glDrawArrays(GL_TRIANGLES, 0, 3);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glDisable(GL_FRAMEBUFFER_SRGB);

        // END: Copy the OptiX rendering results to the display render target.
        // ----------------------------------------------------------------

        glfwSwapBuffers(m_window);
        m_streamChain.swap();

        ++m_frameIndex;
    }

    glFinish();
    m_streamChain.waitAllWorkDone();
}
