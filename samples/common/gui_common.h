#pragma once

#include "common.h"

// Include glfw3.h after our OpenGL definitions
#include "gl_util.h"
#include <GLFW/glfw3.h>

#include "imgui_more.h"
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



struct InitialConfig {
    const char* windowTitle;
    std::filesystem::path resourceDir;
    int32_t windowContentRenderWidth;
    int32_t windowContentRenderHeight;
    float3 cameraPosition;
    Quaternion cameraOrientation;
    float cameraMovingSpeed;
    CUcontext cuContext;
};

struct RunArguments {
    CUstream curStream;
    const glu::Texture2D* outputTexture;
    int32_t windowContentRenderWidth;
    int32_t windowContentRenderHeight;
    uint64_t frameIndex;

    double &mouseX;
    double &mouseY;
    float &cameraPositionalMovingSpeed;
    float &cameraDirectionalMovingSpeed;
    float &cameraTiltSpeed;
    float3 &cameraPosition;
    Quaternion &cameraOrientation;
    Quaternion tempCameraOrientation;
    bool operatingCamera;
    bool cameraIsActuallyMoving;
    bool resized;

    RunArguments(
        double &_mouseX, double &_mouseY,
        float &_cameraPositionalMovingSpeed,
        float &_cameraDirectionalMovingSpeed,
        float &_cameraTiltSpeed,
        float3 &_cameraPosition, Quaternion &_cameraOrientation) :
        mouseX(_mouseX), mouseY(_mouseY),
        cameraPositionalMovingSpeed(_cameraPositionalMovingSpeed),
        cameraDirectionalMovingSpeed(_cameraDirectionalMovingSpeed),
        cameraTiltSpeed(_cameraTiltSpeed),
        cameraPosition(_cameraPosition),
        cameraOrientation(_cameraOrientation) {}
};

struct ReturnValuesToRenderLoop {
    bool enable_sRGB;
    bool finish;
};

class GUIFramework {
    KeyState m_keyForward;
    KeyState m_keyBackward;
    KeyState m_keyLeftward;
    KeyState m_keyRightward;
    KeyState m_keyUpward;
    KeyState m_keyDownward;
    KeyState m_keyTiltLeft;
    KeyState m_keyTiltRight;
    KeyState m_keyFasterPosMovSpeed;
    KeyState m_keySlowerPosMovSpeed;
    KeyState m_buttonRotate;
    double m_mouseX;
    double m_mouseY;

    float m_cameraPositionalMovingSpeed;
    float m_cameraDirectionalMovingSpeed;
    float m_cameraTiltSpeed;
    Quaternion m_cameraOrientation;
    Quaternion m_tempCameraOrientation;
    float3 m_cameraPosition;

    int32_t m_windowContentWidth;
    int32_t m_windowContentHeight;
    float m_windowContentScale;
    int32_t m_windowContentRenderWidth;
    int32_t m_windowContentRenderHeight;
    uint64_t m_frameIndex;

    GLFWmonitor* m_primaryMonitor;
    GLFWwindow* m_window;

    ImGuiStyle m_guiStyleBase;
    ImGuiStyle m_guiStyleWithGammaBase;
    ImGuiStyle m_guiStyle;
    ImGuiStyle m_guiStyleWithGamma;

    glu::VertexArray m_vertexArrayForFullScreen;
    glu::GraphicsProgram m_drawOptiXResultShader;
    glu::Texture2D m_outputTexture;
    glu::Sampler m_outputSampler;

    StreamChain<2> m_streamChain;

    void mouseButtonCallback(int32_t button, int32_t action, int32_t mods);
    void cursorPosCallback(double x, double y);
    void keyCallback(int32_t key, int32_t scancode, int32_t action, int32_t mods);

public:
    void initialize(const InitialConfig &initConfig);
    void finalize();

    const glu::Texture2D &getOutputTexture() const {
        return m_outputTexture;
    }

    void run(
        const std::function<ReturnValuesToRenderLoop(const RunArguments &args)> &updateAndRender,
        const std::function<void(CUstream, uint64_t, int32_t, int32_t)> &resizeCallback);
};