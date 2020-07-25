#include "scene_edit_shared.h"

// Include glfw3.h after our OpenGL definitions
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "../common/imgui_file_dialog.h"

#include "../ext/tiny_obj_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../ext/stb_image.h"
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

using VertexBufferRef = std::shared_ptr<cudau::TypedBuffer<Shared::Vertex>>;

struct OptiXEnv;

struct GeometryInstance {
    OptiXEnv* optixEnv;
    std::string name;
    optixu::GeometryInstance optixGeomInst;
    VertexBufferRef vertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
    uint32_t geomInstIndex;
    bool dataTransfered = false;
    bool selected;

    static void finalize(GeometryInstance* p);
};
using GeometryInstanceRef = std::shared_ptr<GeometryInstance>;

struct GeometryGroup {
    OptiXEnv* optixEnv;
    std::string name;
    optixu::GeometryAccelerationStructure optixGAS;
    std::vector<GeometryInstanceRef> geomInsts;
    cudau::Buffer optixGasMem;
    bool selected;

    static void finalize(GeometryGroup* p);
};
using GeometryGroupRef = std::shared_ptr<GeometryGroup>;

struct Instance {
    OptiXEnv* optixEnv;
    std::string name;
    optixu::Instance optixInst;
    GeometryGroupRef geomGroup;
    float3 position;
    float rollPitchYaw[3];
    bool selected;

    static void finalize(Instance* p);
};
using InstanceRef = std::shared_ptr<Instance>;

struct Group {
    OptiXEnv* optixEnv;
    std::string name;
    optixu::InstanceAccelerationStructure optixIAS;
    std::vector<InstanceRef> insts;
    cudau::Buffer optixIasMem;
    cudau::TypedBuffer<OptixInstance> optixInstanceBuffer;
    bool selected;

    static void finalize(Group* p);
};
using GroupRef = std::shared_ptr<Group>;

struct OptiXEnv {
    CUcontext cuContext;
    optixu::Context context;
    optixu::Material material;
    optixu::Scene scene;
    cudau::TypedBuffer<Shared::GeometryData> geometryDataBuffer;
    SlotFinder geometryInstSlotFinder;

    uint32_t geomInstSerialID;
    uint32_t gasSerialID;
    uint32_t instSerialID;
    uint32_t iasSerialID;
    std::map<uint32_t, GeometryInstanceRef> geomInsts;
    std::map<uint32_t, GeometryGroupRef> geomGroups;
    std::map<uint32_t, InstanceRef> insts;
    std::map<uint32_t, GroupRef> groups;

    cudau::Buffer asScratchBuffer;

    cudau::Buffer shaderBindingTable[2]; // double buffering
};

void GeometryInstance::finalize(GeometryInstance* p) {
    p->optixGeomInst.destroy();
    p->triangleBuffer.finalize();
    p->optixEnv->geometryInstSlotFinder.setNotInUse(p->geomInstIndex);
    delete p;
}
void GeometryGroup::finalize(GeometryGroup* p) {
    p->optixGasMem.finalize();
    p->optixGAS.destroy();
    delete p;
}
void Instance::finalize(Instance* p) {
    p->optixInst.destroy();
    delete p;
}
void Group::finalize(Group* p) {
    p->optixInstanceBuffer.finalize();
    p->optixIasMem.finalize();
    p->optixIAS.destroy();
    delete p;
}

void loadObjFile(const std::filesystem::path &filepath, OptiXEnv* optixEnv) {
    std::vector<Shared::Vertex> vertices;
    std::vector<std::vector<Shared::Triangle>> groups;
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.string().c_str(),
                                    nullptr);

        constexpr float scale = 1.0f;

        // Record unified unique vertices.
        std::map<std::tuple<int32_t, int32_t, int32_t>, Shared::Vertex> unifiedVertexMap;
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
                    auto key = std::make_tuple(idx.vertex_index, idx.normal_index, idx.texcoord_index);
                    unifiedVertexMap[key] = Shared::Vertex{
                        float3(scale * attrib.vertices[3 * idx.vertex_index + 0],
                               scale * attrib.vertices[3 * idx.vertex_index + 1],
                               scale * attrib.vertices[3 * idx.vertex_index + 2]),
                        float3(0, 0, 0),
                        float2(attrib.texcoords[2 * idx.texcoord_index + 0],
                               1 - attrib.texcoords[2 * idx.texcoord_index + 1])
                    };
                }

                idxOffset += numFaceVertices;
            }
        }

        // Assign a vertex index to each of unified unique unifiedVertexMap.
        std::map<std::tuple<int32_t, int32_t, int32_t>, uint32_t> vertexIndices;
        vertices.resize(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            vertices[vertexIndex] = kv.second;
            vertexIndices[kv.first] = vertexIndex++;
        }
        unifiedVertexMap.clear();

        uint32_t numTriangles = 0;
        for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
            std::vector<Shared::Triangle> triangles;

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
                auto key0 = std::make_tuple(idx0.vertex_index, idx0.normal_index, idx0.texcoord_index);
                auto key1 = std::make_tuple(idx1.vertex_index, idx1.normal_index, idx1.texcoord_index);
                auto key2 = std::make_tuple(idx2.vertex_index, idx2.normal_index, idx2.texcoord_index);

                Shared::Triangle triangle = Shared::Triangle{
                                    vertexIndices.at(key0),
                                    vertexIndices.at(key1),
                                    vertexIndices.at(key2) };
                triangles.push_back(triangle);

                Shared::Vertex &v0 = vertices[triangle.index0];
                Shared::Vertex &v1 = vertices[triangle.index1];
                Shared::Vertex &v2 = vertices[triangle.index2];

                float3 gn = normalize(cross(v1.position - v0.position,
                                            v2.position - v0.position));
                if (!allFinite(gn))
                    gn = float3(0, 0, 1);
                v0.normal += gn;
                v1.normal += gn;
                v2.normal += gn;

                idxOffset += numFaceVertices;
            }

            numTriangles += triangles.size();
            groups.push_back(std::move(triangles));
        }
        vertexIndices.clear();
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            Shared::Vertex &v = vertices[vIdx];
            v.normal = normalize(v.normal);
            if (!allFinite(v.normal))
                v.normal = float3(0, 0, 1);
        }
    }

    VertexBufferRef vertexBuffer = make_shared_with_deleter<cudau::TypedBuffer<Shared::Vertex>>(
        [](cudau::TypedBuffer<Shared::Vertex>* p) {
            p->finalize();
        });
    vertexBuffer->initialize(optixEnv->cuContext, g_bufferType, vertices);

    std::string basename = filepath.stem().string();
    for (int i = 0; i < groups.size(); ++i) {
        char name[256];
        sprintf_s(name, "%s-%d", basename.c_str(), i);
        GeometryInstanceRef geomInst = make_shared_with_deleter<GeometryInstance>(GeometryInstance::finalize);
        uint32_t geomInstIndex = optixEnv->geometryInstSlotFinder.getFirstAvailableSlot();
        optixEnv->geometryInstSlotFinder.setInUse(geomInstIndex);
        geomInst->optixEnv = optixEnv;
        geomInst->name = name;
        geomInst->vertexBuffer = vertexBuffer;
        geomInst->triangleBuffer.initialize(optixEnv->cuContext, g_bufferType, groups[i]);
        geomInst->geomInstIndex = geomInstIndex;
        geomInst->optixGeomInst = optixEnv->scene.createGeometryInstance();
        geomInst->optixGeomInst.setVertexBuffer(&*vertexBuffer);
        geomInst->optixGeomInst.setTriangleBuffer(&geomInst->triangleBuffer);
        geomInst->optixGeomInst.setNumMaterials(1, nullptr);
        geomInst->optixGeomInst.setMaterial(0, 0, optixEnv->material);
        geomInst->optixGeomInst.setUserData(geomInstIndex);
        geomInst->dataTransfered = false;
        geomInst->selected = false;

        optixEnv->geomInsts[optixEnv->geomInstSerialID++] = geomInst;
    }
}




static void glfw_error_callback(int32_t error, const char* description) {
    hpprintf("Error %d: %s\n", error, description);
}



int32_t mainFunc(int32_t argc, const char* argv[]) {
    const std::filesystem::path exeDir = getExecutableDirectory();

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
    GLFWwindow* window = glfwCreateWindow(static_cast<int32_t>(renderTargetSizeX * UIScaling),
                                          static_cast<int32_t>(renderTargetSizeY * UIScaling),
                                          "OptiX Utility - Scene Edit", NULL, NULL);
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

    io.Fonts->AddFontDefault();

    std::filesystem::path fontPath = exeDir / "fonts/RictyDiminished-Regular.ttf";
    ImFont* fontForFileDialog = io.Fonts->AddFontFromFileTTF(fontPath.u8string().c_str(), 14.0f, nullptr,
                                                             io.Fonts->GetGlyphRangesJapanese());
    if (fontForFileDialog == nullptr)
        hpprintf("Font Not Found!: %s\n", fontPath.u8string().c_str());

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

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUstream cuStream[2];
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[0], 0));
    CUDADRV_CHECK(cuStreamCreate(&cuStream[1], 0));

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false,
                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS |
                                OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                DEBUG_SELECT((OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW |
                                              OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                              OPTIX_EXCEPTION_FLAG_DEBUG),
                                             OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(exeDir / "scene_edit/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: これらのグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: These are for ray-triangle hit groups, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroup0 = pipeline.createHitProgramGroup(moduleOptiX, RT_CH_NAME_STR("closesthit0"),
                                                                           emptyModule, nullptr,
                                                                           emptyModule, nullptr);

    pipeline.setMaxTraceDepth(1);
    pipeline.link(DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE),
                  false);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    OptiXEnv optixEnv;
    optixEnv.cuContext = cuContext;
    optixEnv.context = optixContext;

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixEnv.material = optixContext.createMaterial();
    optixEnv.material.setHitGroup(Shared::RayType_Primary, hitProgramGroup0);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    constexpr uint32_t MaxNumGeometryInstances = 512;
    constexpr uint32_t MaxNumInstances = 512;
    
    optixEnv.scene = optixContext.createScene();
    optixEnv.geometryDataBuffer.initialize(cuContext, g_bufferType, MaxNumGeometryInstances);
    optixEnv.geometryInstSlotFinder.initialize(MaxNumGeometryInstances);
    optixEnv.geomInstSerialID = 0;
    optixEnv.gasSerialID = 0;
    optixEnv.instSerialID = 0;
    optixEnv.iasSerialID = 0;
    optixEnv.asScratchBuffer.initialize(cuContext, g_bufferType, 32 * 1024 * 1024, 1);

    size_t sbtSize = 512; // set a dummy as initial size.
    optixEnv.shaderBindingTable[0].initialize(cuContext, g_bufferType, sbtSize, 1);
    optixEnv.shaderBindingTable[1].initialize(cuContext, g_bufferType, sbtSize, 1);

    // END: Setup a scene.
    // ----------------------------------------------------------------



    // JP: OpenGL用バッファーオブジェクトからCUDAバッファーを生成する。
    // EN: Create a CUDA buffer from an OpenGL buffer instObject0.
    GLTK::Texture2D outputTexture;
    cudau::Array outputArray;
    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputTexture.initialize(renderTargetSizeX, renderTargetSizeY, GLTK::SizedInternalFormat::RGBA32F);
    GLTK::errorCheck();
    outputArray.initializeFromGLTexture2D(cuContext, outputTexture.getRawHandle(),
                                          cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    outputBufferSurfaceHolder.initialize(&outputArray);

    GLTK::Sampler outputSampler;
    outputSampler.initialize(GLTK::Sampler::MinFilter::Nearest, GLTK::Sampler::MagFilter::Nearest,
                             GLTK::Sampler::WrapMode::Repeat, GLTK::Sampler::WrapMode::Repeat);



    // JP: フルスクリーンクアッド(or 三角形)用の空のVAO。
    // EN: Empty VAO for full screen qud (or triangle).
    GLTK::VertexArray vertexArrayForFullScreen;
    vertexArrayForFullScreen.initialize();

    // JP: OptiXの結果をフレームバッファーにコピーするシェーダー。
    // EN: Shader to copy OptiX result to a frame buffer.
    GLTK::GraphicsShader drawOptiXResultShader;
    drawOptiXResultShader.initializeVSPS(readTxtFile(exeDir / "scene_edit/shaders/drawOptiXResult.vert"),
                                         readTxtFile(exeDir / "scene_edit/shaders/drawOptiXResult.frag"));



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = 0;
    plp.geomInstData = optixEnv.geometryDataBuffer.getDevicePointer();
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

    pipeline.setScene(optixEnv.scene);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));


    
    FileDialog fileDialog;
    fileDialog.setFont(fontForFileDialog);
    fileDialog.setFlags(FileDialog::Flag_FileSelection);
    //fileDialog.setFlags(FileDialog::Flag_FileSelection |
    //                    FileDialog::Flag_DirectorySelection |
    //                    FileDialog::Flag_MultipleSelection);
    
    uint64_t frameIndex = 0;
    glfwSetWindowUserPointer(window, &frameIndex);
    int32_t requestedSize[2];
    bool sbtLayoutUpdated = true;
    uint32_t sbtIndex = 0;
    cudau::Buffer* curShaderBindingTable = &optixEnv.shaderBindingTable[sbtIndex];
    OptixTraversableHandle curTravHandle = 0;
    while (true) {
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

            outputArray.resize(renderTargetSizeX, renderTargetSizeY);

            // EN: update the pipeline parameters.
            plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
            plp.camera.aspect = (float)renderTargetSizeX / renderTargetSizeY;

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
        
        // JP: 前フレームの処理が完了するのを待つ。
        // EN: Wait the previous frame processing to finish.
        CUDADRV_CHECK(cuStreamSynchronize(curCuStream));



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

        {
            ImGui::Begin("Scene", nullptr,
                         ImGuiWindowFlags_None);

            if (ImGui::Button("Open"))
                fileDialog.show();
            if (fileDialog.drawAndGetResult() == FileDialog::Result::Result_OK) {
                static std::vector<std::filesystem::directory_entry> entries;
                fileDialog.calcEntries(&entries);
                
                loadObjFile(entries[0], &optixEnv);
            }

            static int32_t travIndex = -1;
            static std::vector<std::string> traversableNames;
            static std::vector<OptixTraversableHandle> traversables;
            if (ImGui::Combo("Target", &travIndex,
                             [](void* data, int idx, const char** out_text) {
                                 if (idx < 0)
                                     return false;
                                 auto nameList = reinterpret_cast<std::string*>(data);
                                 *out_text = nameList[idx].c_str();
                                 return true;
                             }, traversableNames.data(), traversables.size())) {
                curTravHandle = traversables[travIndex];
            }

            bool traversablesUpdated = false;

            if (ImGui::BeginTabBar("Scene", ImGuiTabBarFlags_None)) {
                const auto ImGui_PushDisabledStyle = []() {
                    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.2f);
                };
                const auto ImGui_PopDisabledStyle = []() {
                    ImGui::PopStyleVar();
                };

                if (ImGui::BeginTabItem("Geom Inst")) {
                    static bool allSelected = false;
                    static std::set<uint32_t> selectedItems;
                    if (ImGui::BeginTable("##geomInstList", 5,
                                          ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_ScrollFreezeTopRow,
                                          ImVec2(0, 300))) {
                        ImGui::TableSetupColumn("CheckAll",
                                                ImGuiTableColumnFlags_WidthFixed |
                                                ImGuiTableColumnFlags_NoResize);
                        ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("#Prims", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
                        {
                            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(ImGui::TableGetColumnName(0));
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                            if (ImGui::Checkbox("##check", &allSelected)) {
                                if (allSelected) {
                                    for (const auto &kv : optixEnv.geomInsts) {
                                        kv.second->selected = true;
                                        selectedItems.insert(kv.first);
                                    }
                                }
                                else {
                                    for (const auto &kv : optixEnv.geomInsts)
                                        kv.second->selected = false;
                                    selectedItems.clear();
                                }
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            ImGui::TableHeader(ImGui::TableGetColumnName(1));
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TableHeader(ImGui::TableGetColumnName(2));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TableHeader(ImGui::TableGetColumnName(3));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TableHeader(ImGui::TableGetColumnName(4));
                        }
                        for (const auto &kv : optixEnv.geomInsts) {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(kv.first);
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                            if (ImGui::Checkbox("##check", &kv.second->selected)) {
                                if (kv.second->selected)
                                    selectedItems.insert(kv.first);
                                else
                                    selectedItems.erase(kv.first);
                                allSelected = selectedItems.size() == optixEnv.geomInsts.size();
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            char sid[32];
                            sprintf_s(sid, "%u", kv.first);
                            ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%s", kv.second->name.c_str());

                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%u", kv.second->triangleBuffer.numElements());

                            ImGui::TableSetColumnIndex(4);
                            ImGui::Text("%u", kv.second.use_count() - 1);
                        }
                        ImGui::EndTable();
                    }

                    bool enabled;

                    enabled = selectedItems.size() > 0;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create a GAS")) {
                        if (enabled) {
                            uint32_t serialID = optixEnv.gasSerialID++;
                            GeometryGroupRef geomGroup = make_shared_with_deleter<GeometryGroup>(GeometryGroup::finalize);
                            geomGroup->optixEnv = &optixEnv;
                            char name[256];
                            sprintf_s(name, "GAS-%u", serialID);
                            geomGroup->name = name;
                            geomGroup->optixGAS = optixEnv.scene.createGeometryAccelerationStructure();
                            geomGroup->optixGAS.setConfiguration(false, false, false, false);
                            geomGroup->optixGAS.setNumMaterialSets(1);
                            geomGroup->optixGAS.setNumRayTypes(0, Shared::NumRayTypes);
                            geomGroup->selected = false;

                            for (uint32_t sID : selectedItems) {
                                const GeometryInstanceRef &geomInst = optixEnv.geomInsts.at(sID);
                                geomGroup->geomInsts.push_back(geomInst);
                                geomGroup->optixGAS.addChild(geomInst->optixGeomInst);
                                if (!geomInst->dataTransfered) {
                                    Shared::GeometryData geomData;
                                    geomData.vertexBuffer = geomInst->vertexBuffer->getDevicePointer();
                                    geomData.triangleBuffer = geomInst->triangleBuffer.getDevicePointer();
                                    CUDADRV_CHECK(cuMemcpyHtoDAsync(optixEnv.geometryDataBuffer.getCUdeviceptrAt(geomInst->geomInstIndex),
                                                                    &geomData, sizeof(geomData), curCuStream));
                                }
                            }

                            optixEnv.scene.generateShaderBindingTableLayout(&sbtSize);
                            sbtLayoutUpdated = true;

                            OptixAccelBufferSizes bufferSizes;
                            geomGroup->optixGAS.prepareForBuild(&bufferSizes);
                            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1);
                            geomGroup->optixGasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
                            geomGroup->optixGAS.rebuild(curCuStream, geomGroup->optixGasMem, optixEnv.asScratchBuffer);

                            optixEnv.geomGroups[serialID] = geomGroup;
                            traversablesUpdated = true;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    ImGui::SameLine();
                    bool allUnused = selectedItems.size() > 0;
                    for (const auto &sid : selectedItems) {
                        allUnused &= optixEnv.geomInsts.at(sid).use_count() == 1;
                        if (!allUnused)
                            break;
                    }
                    enabled = allUnused && selectedItems.size() > 0;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (enabled) {
                            for (const auto &sid : selectedItems)
                                optixEnv.geomInsts.erase(sid);
                            selectedItems.clear();
                            allSelected = false;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Geom Group")) {
                    static bool allSelected = false;
                    static std::set<uint32_t> selectedItems;
                    if (ImGui::BeginTable("##geomGroupList", 5,
                                          ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_ScrollFreezeTopRow,
                                          ImVec2(0, 300))) {
                        ImGui::TableSetupColumn("CheckAll",
                                                ImGuiTableColumnFlags_WidthFixed |
                                                ImGuiTableColumnFlags_NoResize);
                        ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("#GeomInsts", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
                        {
                            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(ImGui::TableGetColumnName(0));
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                            if (ImGui::Checkbox("##check", &allSelected)) {
                                if (allSelected) {
                                    for (const auto &kv : optixEnv.geomGroups) {
                                        kv.second->selected = true;
                                        selectedItems.insert(kv.first);
                                    }
                                }
                                else {
                                    for (const auto &kv : optixEnv.geomGroups)
                                        kv.second->selected = false;
                                    selectedItems.clear();
                                }
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            ImGui::TableHeader(ImGui::TableGetColumnName(1));
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TableHeader(ImGui::TableGetColumnName(2));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TableHeader(ImGui::TableGetColumnName(3));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TableHeader(ImGui::TableGetColumnName(4));
                        }
                        for (const auto &kv : optixEnv.geomGroups) {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(kv.first);
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                            if (ImGui::Checkbox("##check", &kv.second->selected)) {
                                if (kv.second->selected)
                                    selectedItems.insert(kv.first);
                                else
                                    selectedItems.erase(kv.first);
                                allSelected = selectedItems.size() == optixEnv.geomGroups.size();
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            char sid[32];
                            sprintf_s(sid, "%u", kv.first);
                            ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%s", kv.second->name.c_str());

                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%u", static_cast<uint32_t>(kv.second->geomInsts.size()));

                            ImGui::TableSetColumnIndex(4);
                            ImGui::Text("%u", kv.second.use_count() - 1);
                        }
                        ImGui::EndTable();
                    }

                    bool enabled;

                    enabled = selectedItems.size() == 1;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create an Instance")) {
                        if (enabled) {
                            uint32_t serialID = optixEnv.instSerialID++;
                            InstanceRef inst = make_shared_with_deleter<Instance>(Instance::finalize);
                            inst->optixEnv = &optixEnv;
                            char name[256];
                            sprintf_s(name, "Instance-%u", serialID);
                            inst->name = name;
                            inst->optixInst = optixEnv.scene.createInstance();
                            inst->selected = false;
                            inst->position = float3(0.0f, 0.0f, 0.0f);
                            inst->rollPitchYaw[0] = 0.0f;
                            inst->rollPitchYaw[1] = 0.0f;
                            inst->rollPitchYaw[2] = 0.0f;

                            Matrix3x3 rotMat = qFromEulerAngles(inst->rollPitchYaw[0],
                                                                inst->rollPitchYaw[1],
                                                                inst->rollPitchYaw[2]).toMatrix3x3();

                            const GeometryGroupRef &geomGroup = optixEnv.geomGroups.at(*selectedItems.begin());
                            inst->geomGroup = geomGroup;
                            inst->optixInst.setGAS(geomGroup->optixGAS);
                            float tr[] = {
                                rotMat.m00, rotMat.m01, rotMat.m02, inst->position.x,
                                rotMat.m10, rotMat.m11, rotMat.m12, inst->position.y,
                                rotMat.m20, rotMat.m21, rotMat.m22, inst->position.z,
                            };
                            inst->optixInst.setTransform(tr);

                            optixEnv.insts[serialID] = inst;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    ImGui::SameLine();
                    bool allUnused = selectedItems.size() > 0;
                    for (const auto &sid : selectedItems) {
                        allUnused &= optixEnv.geomGroups.at(sid).use_count() == 1;
                        if (!allUnused)
                            break;
                    }
                    enabled = allUnused && selectedItems.size() > 0;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (enabled) {
                            for (const auto &sid : selectedItems)
                                optixEnv.geomGroups.erase(sid);
                            selectedItems.clear();
                            allSelected = false;
                            traversablesUpdated = true;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Inst")) {
                    static bool allSelected = false;
                    static std::set<uint32_t> selectedItems;
                    if (ImGui::BeginTable("##instList", 5,
                                          ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_ScrollFreezeTopRow,
                                          ImVec2(0, 300))) {

                        ImGui::TableSetupColumn("CheckAll",
                                                ImGuiTableColumnFlags_WidthFixed |
                                                ImGuiTableColumnFlags_NoResize);
                        ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("GAS", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
                        {
                            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(ImGui::TableGetColumnName(0));
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                            if (ImGui::Checkbox("##check", &allSelected)) {
                                if (allSelected) {
                                    for (const auto &kv : optixEnv.insts) {
                                        kv.second->selected = true;
                                        selectedItems.insert(kv.first);
                                    }
                                }
                                else {
                                    for (const auto &kv : optixEnv.insts)
                                        kv.second->selected = false;
                                    selectedItems.clear();
                                }
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            ImGui::TableHeader(ImGui::TableGetColumnName(1));
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TableHeader(ImGui::TableGetColumnName(2));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TableHeader(ImGui::TableGetColumnName(3));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TableHeader(ImGui::TableGetColumnName(4));
                        }
                        for (const auto &kv : optixEnv.insts) {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(kv.first);
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                            if (ImGui::Checkbox("##check", &kv.second->selected)) {
                                if (kv.second->selected)
                                    selectedItems.insert(kv.first);
                                else
                                    selectedItems.erase(kv.first);
                                allSelected = selectedItems.size() == optixEnv.insts.size();
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            char sid[32];
                            sprintf_s(sid, "%u", kv.first);
                            ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%s", kv.second->name.c_str());

                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%s", kv.second->geomGroup->name.c_str());

                            ImGui::TableSetColumnIndex(4);
                            ImGui::Text("%u", kv.second.use_count() - 1);
                        }
                        ImGui::EndTable();
                    }

                    bool enabled;

                    enabled = selectedItems.size() > 0;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Create an IAS")) {
                        if (enabled) {
                            uint32_t serialID = optixEnv.iasSerialID++;
                            GroupRef group = make_shared_with_deleter<Group>(Group::finalize);
                            group->optixEnv = &optixEnv;
                            char name[256];
                            sprintf_s(name, "IAS-%u", serialID);
                            group->name = name;
                            group->optixIAS = optixEnv.scene.createInstanceAccelerationStructure();
                            group->optixIAS.setConfiguration(false, false, false);
                            group->selected = false;

                            for (uint32_t sID : selectedItems) {
                                const InstanceRef &inst = optixEnv.insts.at(sID);
                                group->insts.push_back(inst);
                                group->optixIAS.addChild(inst->optixInst);
                            }

                            OptixAccelBufferSizes bufferSizes;
                            uint32_t numInstances;
                            group->optixIAS.prepareForBuild(&bufferSizes, &numInstances);
                            if (bufferSizes.tempSizeInBytes >= optixEnv.asScratchBuffer.sizeInBytes())
                                optixEnv.asScratchBuffer.resize(bufferSizes.tempSizeInBytes, 1);
                            group->optixIasMem.initialize(optixEnv.cuContext, g_bufferType, bufferSizes.outputSizeInBytes, 1);
                            group->optixInstanceBuffer.initialize(optixEnv.cuContext, g_bufferType, numInstances);
                            group->optixIAS.rebuild(curCuStream, group->optixInstanceBuffer, group->optixIasMem, optixEnv.asScratchBuffer);

                            optixEnv.groups[serialID] = group;
                            traversablesUpdated = true;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    bool allUnused = selectedItems.size() > 0;
                    for (const auto &sid : selectedItems) {
                        allUnused &= optixEnv.insts.at(sid).use_count() == 1;
                        if (!allUnused)
                            break;
                    }
                    ImGui::SameLine();
                    if (!allUnused || selectedItems.size() == 0)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (allUnused && selectedItems.size() > 0) {
                            for (const auto &sid : selectedItems)
                                optixEnv.insts.erase(sid);
                            selectedItems.clear();
                            allSelected = false;
                        }
                    }
                    if (!allUnused || selectedItems.size() == 0)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                if (ImGui::BeginTabItem("Group")) {
                    static bool allSelected = false;
                    static std::set<uint32_t> selectedItems;
                    if (ImGui::BeginTable("##groupList", 5,
                                          ImGuiTableFlags_Borders |
                                          ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_ScrollY |
                                          ImGuiTableFlags_ScrollFreezeTopRow,
                                          ImVec2(0, 300))) {
                        ImGui::TableSetupColumn("CheckAll",
                                                ImGuiTableColumnFlags_WidthFixed |
                                                ImGuiTableColumnFlags_NoResize);
                        ImGui::TableSetupColumn("SID", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthStretch);
                        ImGui::TableSetupColumn("#insts", ImGuiTableColumnFlags_WidthFixed);
                        ImGui::TableSetupColumn("Used", ImGuiTableColumnFlags_WidthFixed);
                        {
                            ImGui::TableNextRow(ImGuiTableRowFlags_Headers);

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(ImGui::TableGetColumnName(0));
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 2));
                            if (ImGui::Checkbox("##check", &allSelected)) {
                                if (allSelected) {
                                    for (const auto &kv : optixEnv.groups) {
                                        kv.second->selected = true;
                                        selectedItems.insert(kv.first);
                                    }
                                }
                                else {
                                    for (const auto &kv : optixEnv.groups)
                                        kv.second->selected = false;
                                    selectedItems.clear();
                                }
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            ImGui::TableHeader(ImGui::TableGetColumnName(1));
                            ImGui::TableSetColumnIndex(2);
                            ImGui::TableHeader(ImGui::TableGetColumnName(2));
                            ImGui::TableSetColumnIndex(3);
                            ImGui::TableHeader(ImGui::TableGetColumnName(3));
                            ImGui::TableSetColumnIndex(4);
                            ImGui::TableHeader(ImGui::TableGetColumnName(4));
                        }
                        for (const auto &kv : optixEnv.groups) {
                            ImGui::TableNextRow();

                            ImGui::TableSetColumnIndex(0);
                            ImGui::PushID(kv.first);
                            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(ImGui::GetStyle().FramePadding.x, 1));
                            if (ImGui::Checkbox("##check", &kv.second->selected)) {
                                if (kv.second->selected)
                                    selectedItems.insert(kv.first);
                                else
                                    selectedItems.erase(kv.first);
                                allSelected = selectedItems.size() == optixEnv.groups.size();
                            }
                            ImGui::PopStyleVar();
                            ImGui::PopID();

                            ImGui::TableSetColumnIndex(1);
                            char sid[32];
                            sprintf_s(sid, "%u", kv.first);
                            ImGui::Selectable(sid, false, ImGuiSelectableFlags_None);

                            ImGui::TableSetColumnIndex(2);
                            ImGui::Text("%s", kv.second->name.c_str());

                            ImGui::TableSetColumnIndex(3);
                            ImGui::Text("%u", static_cast<uint32_t>(kv.second->insts.size()));

                            ImGui::TableSetColumnIndex(4);
                            ImGui::Text("%u", kv.second.use_count() - 1);
                        }
                        ImGui::EndTable();
                    }

                    bool enabled;

                    enabled = selectedItems.size() > 0;
                    if (!enabled)
                        ImGui_PushDisabledStyle();
                    if (ImGui::Button("Remove")) {
                        if (enabled) {
                            for (const auto &sid : selectedItems)
                                optixEnv.groups.erase(sid);
                            selectedItems.clear();
                            allSelected = false;
                            traversablesUpdated = true;
                        }
                    }
                    if (!enabled)
                        ImGui_PopDisabledStyle();

                    ImGui::EndTabItem();
                }
                ImGui::EndTabBar();



                if (traversablesUpdated) {
                    traversables.clear();
                    traversableNames.clear();
                    for (const auto &kv : optixEnv.groups) {
                        const GroupRef &group = kv.second;
                        traversables.push_back(group->optixIAS.getHandle());
                        traversableNames.push_back(group->name);
                    }
                    for (const auto &kv : optixEnv.geomGroups) {
                        const GeometryGroupRef &group = kv.second;
                        traversables.push_back(group->optixGAS.getHandle());
                        traversableNames.push_back(group->name);
                    }

                    travIndex = -1;
                    for (int i = 0; i < traversables.size(); ++i) {
                        if (traversables[i] == curTravHandle) {
                            travIndex = i;
                            break;
                        }
                    }
                }
            }

            ImGui::End();
        }



        outputBufferSurfaceHolder.beginCUDAAccess(curCuStream);

        plp.travHandle = curTravHandle;
        plp.resultBuffer = outputBufferSurfaceHolder.getNext();

        // Render
        if (sbtLayoutUpdated) {
            curShaderBindingTable->resize(sbtSize, 1, curCuStream);
            pipeline.setHitGroupShaderBindingTable(curShaderBindingTable);
            sbtLayoutUpdated = false;
        }
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curCuStream));
        pipeline.launch(curCuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);

        outputBufferSurfaceHolder.endCUDAAccess(curCuStream);



        // ----------------------------------------------------------------
        // JP: 

        glEnable(GL_FRAMEBUFFER_SRGB);
        GLTK::errorCheck();

        glViewport(0, 0, curFBWidth, curFBHeight);

        drawOptiXResultShader.useProgram();

        glUniform2ui(0, curFBWidth, curFBHeight);

        glActiveTexture(GL_TEXTURE0);
        outputTexture.bind();
        outputSampler.bindToTextureUnit(0);

        vertexArrayForFullScreen.bind();
        glDrawArrays(GL_TRIANGLES, 0, 3);
        vertexArrayForFullScreen.unbind();

        outputTexture.unbind();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // END: 
        // ----------------------------------------------------------------

        glfwSwapBuffers(window);

        ++frameIndex;
    }

    CUDADRV_CHECK(cuMemFree(plpOnDevice));

    drawOptiXResultShader.finalize();
    vertexArrayForFullScreen.finalize();

    outputSampler.finalize();
    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();
    outputTexture.finalize();

    optixEnv.asScratchBuffer.finalize();
    optixEnv.shaderBindingTable[1].finalize();
    optixEnv.shaderBindingTable[0].finalize();
    optixEnv.geometryInstSlotFinder.finalize();
    optixEnv.geometryDataBuffer.finalize();
    optixEnv.scene.destroy();

    optixEnv.material.destroy();

    hitProgramGroup0.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

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
