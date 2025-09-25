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
#   define DEBUG_SELECT(A, B) A
#else
#   define DEBUG_SELECT(A, B) B
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
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>



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

#ifdef ENABLE_ASSERT
#   if defined(__CUDA_ARCH__)
#       define Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); assert(false); } } while (0)
#   else
#       define Assert(expr, fmt, ...) do { if (!(expr)) { devPrintf("%s @%s: %u:\n", #expr, __FILE__, __LINE__); devPrintf(fmt"\n", ##__VA_ARGS__); abort(); } } while (0)
#   endif
#else
#   define Assert(expr, fmt, ...)
#endif

#define Assert_ShouldNotBeCalled() Assert(false, "Should not be called!")
#define Assert_NotImplemented() Assert(false, "Not implemented yet!")



#include "../../optix_util.cpp"

#include "shared.h"

#include "gtest/gtest.h"

#define EXPECT_EXCEPTION(call) \
    do { \
        bool caught = false; \
        try { \
            call; \
        } \
        catch (std::exception &) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, true); \
    } \
    while (0)

#define EXPECT_EXCEPTION_RET(ret, call) \
    do { \
        bool caught = false; \
        try { \
            ret = call; \
        } \
        catch (std::exception &) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, true); \
    } \
    while (0)



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
}

std::vector<char> readBinaryFile(const std::filesystem::path &filepath) {
    std::vector<char> ret;

    std::ifstream ifs;
    ifs.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
    if (ifs.fail())
        return std::move(ret);

    std::streamsize fileSize = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    ret.resize(fileSize);
    ifs.read(ret.data(), fileSize);

    return std::move(ret);
}



static CUcontext cuContext;
static CUstream cuStream;



TEST(ContextTest, ContextCreation) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        context.destroy();

        context = optixu::Context::create(0);
        context.destroy();

        //context = optixu::Context::create(reinterpret_cast<CUcontext>(~static_cast<uintptr_t>(0))), false);
        context.destroy();

        context = optixu::Context::create(cuContext, 0, optixu::EnableValidation::No);
        context.destroy();
        context = optixu::Context::create(cuContext, 1, optixu::EnableValidation::No);
        context.destroy();
        context = optixu::Context::create(cuContext, 2, optixu::EnableValidation::No);
        context.destroy();
        context = optixu::Context::create(cuContext, 3, optixu::EnableValidation::No);
        context.destroy();
        context = optixu::Context::create(cuContext, 4, optixu::EnableValidation::No);
        context.destroy();
        context = optixu::Context::create(cuContext, 4, optixu::EnableValidation::Yes);
        context.destroy();

        // JP: コールバックレベルが範囲外。
        EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 5, optixu::EnableValidation::No));
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}

TEST(ContextTest, ContextBasic) {
    try {
        // JP: 適当なCUDAコンテキストに対してoptixuのコンテキストを生成。
        optixu::Context context = optixu::Context::create(cuContext);
        {
            // JP: 共通処理。
            {
                const char* nameA = "ABCDE";
                context.setName(nameA);
                EXPECT_STREQ(context.getName(), nameA);

                const char* nameB = "";
                context.setName(nameB);
                EXPECT_STREQ(context.getName(), nameB);
            }

            CUcontext retCuContext = context.getCUcontext();
            EXPECT_EQ(retCuContext, cuContext);

            // JP: コールバックの登録。
            {
                struct CallbackUserData {
                    uint32_t value;
                };
                const auto callback = [](uint32_t level, const char* tag, const char* message, void* cbdata) {
                };
                CallbackUserData cbUserData;
                context.setLogCallback(callback, &cbUserData, 0);
                context.setLogCallback(callback, &cbUserData, 1);
                context.setLogCallback(callback, &cbUserData, 2);
                context.setLogCallback(callback, &cbUserData, 3);
                context.setLogCallback(callback, &cbUserData, 4);
                // JP: コールバックレベルとして範囲外の値を設定。
                EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 5));
                // JP: コールバックリセット。
                context.setLogCallback(nullptr, nullptr, 0);
            }

            // JP: 各オブジェクトの生成。
            {
                optixu::Pipeline pipeline = context.createPipeline();
                EXPECT_NE(pipeline, optixu::Pipeline());
                pipeline.destroy();

                optixu::Material material = context.createMaterial();
                EXPECT_NE(material, optixu::Material());
                material.destroy();

                optixu::Scene scene = context.createScene();
                EXPECT_NE(scene, optixu::Scene());
                scene.destroy();

                optixu::Denoiser denoiser;

                denoiser = context.createDenoiser(
                    OPTIX_DENOISER_MODEL_KIND_LDR, optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes, OPTIX_DENOISER_ALPHA_MODE_COPY);
                EXPECT_NE(denoiser, optixu::Denoiser());
                denoiser.destroy();

                denoiser = context.createDenoiser(
                    OPTIX_DENOISER_MODEL_KIND_HDR, optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes, OPTIX_DENOISER_ALPHA_MODE_COPY);
                EXPECT_NE(denoiser, optixu::Denoiser());
                denoiser.destroy();

                // JP: 無効なenumを使ってデノイザーを生成。
                EXPECT_EXCEPTION_RET(
                    denoiser,
                    context.createDenoiser(
                        static_cast<OptixDenoiserModelKind>(~0),
                        optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes, OPTIX_DENOISER_ALPHA_MODE_COPY));
            }
        }
        context.destroy();



        // JP: CUDAのデフォルトコンテキストに対してoptixuのコンテキストを生成。
        context = optixu::Context::create(0);
        {
            EXPECT_EQ(context.getCUcontext(), reinterpret_cast<CUcontext>(0));
        }
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(MaterialTest, MaterialBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext, 4, optixu::EnableValidation::Yes);

        optixu::Pipeline pipeline0 = context.createPipeline();
        optixu::PipelineOptions pipelineOptions0;
        pipelineOptions0.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions0.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions0.launchParamsVariableName = "plp";
        pipelineOptions0.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions0.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions0.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions0.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline0.setPipelineOptions(pipelineOptions0);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline0.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        optixu::HitProgramGroup hitProgramGroup0 = pipeline0.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch0"),
            emptyModule, nullptr);

        optixu::HitProgramGroup hitProgramGroup1 = pipeline0.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch1"),
            emptyModule, nullptr);



        optixu::Material mat0 = context.createMaterial();

        // JP: 共通処理。
        {
            EXPECT_EQ(mat0.getContext(), context);

            // JP: 普通の名前の設定と取得。
            const char* nameA = "ABCDE";
            mat0.setName(nameA);
            EXPECT_STREQ(mat0.getName(), nameA);

            // JP: 空白の名前の設定と取得。
            const char* nameB = "";
            mat0.setName(nameB);
            EXPECT_STREQ(mat0.getName(), nameB);
        }

        // JP: ヒットグループ関連。
        {
            optixu::HitProgramGroup retHitProgramGroup;

            // JP: まずは普通に設定と取得。
            mat0.setHitGroup(0, hitProgramGroup0);
            EXPECT_EQ(mat0.getHitGroup(pipeline0, 0), hitProgramGroup0);

            // JP: 同じレイインデックスに対して設定と取得。
            mat0.setHitGroup(0, hitProgramGroup1);
            EXPECT_EQ(mat0.getHitGroup(pipeline0, 0), hitProgramGroup1);

            // JP: 別のレイインデックスに対して設定と取得。
            mat0.setHitGroup(2, hitProgramGroup0);
            EXPECT_EQ(mat0.getHitGroup(pipeline0, 2), hitProgramGroup0);

            // JP: 未設定のレイインデックスから取得。
            EXPECT_EXCEPTION_RET(retHitProgramGroup, mat0.getHitGroup(pipeline0, 1));
        }

        // JP: ユーザーデータ関連。
        {
            uint32_t udSize, udAlignment;

            // JP: まずは普通に設定と取得。
            struct UserData0 {
                uint32_t a;
                float2 b;
            };
            UserData0 ud0 = {};
            ud0.a = 1;
            ud0.b = float2(2, 3);
            mat0.setUserData(ud0);
            UserData0 retUd0;
            mat0.getUserData(&retUd0, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud0));
            EXPECT_EQ(udAlignment, alignof(UserData0));
            EXPECT_EQ(std::memcmp(&ud0, &retUd0, sizeof(ud0)), 0);

            // JP: 限界サイズのユーザーデータの設定と取得。
            struct UserData1 {
                uint8_t a[optixu::s_maxMaterialUserDataSize];
            };
            UserData1 ud1 = {};
            for (int i = 0; i < sizeof(ud1.a); ++i)
                ud1.a[i] = i;
            mat0.setUserData(ud1);
            UserData1 retUd1;
            mat0.getUserData(&retUd1, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud1));
            EXPECT_EQ(udAlignment, alignof(UserData1));
            EXPECT_EQ(std::memcmp(&ud1, &retUd1, sizeof(ud1)), 0);

            // JP: 限界サイズを超えたユーザーデータの設定と取得。
            struct UserData2 {
                uint8_t a[optixu::s_maxMaterialUserDataSize + 2];
            };
            UserData2 ud2 = {};
            for (int i = 0; i < sizeof(ud2.a); ++i)
                ud2.a[i] = i;
            EXPECT_EXCEPTION(mat0.setUserData(ud2));
        }

        mat0.destroy();

        hitProgramGroup1.destroy();
        hitProgramGroup0.destroy();
        moduleOptiX.destroy();
        pipeline0.destroy();

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(SceneTest, SceneBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        optixu::Scene scene0 = context.createScene();

        // JP: 共通処理。
        {
            EXPECT_EQ(scene0.getContext(), context);

            // JP: 普通の名前の設定と取得。
            const char* nameA = "ABCDE";
            scene0.setName(nameA);
            EXPECT_STREQ(scene0.getName(), nameA);

            // JP: 空白の名前の設定と取得。
            const char* nameB = "";
            scene0.setName(nameB);
            EXPECT_STREQ(scene0.getName(), nameB);
        }

        // JP: オブジェクト生成前のSBTレイアウト関連。
        {
            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), false);

            scene0.markShaderBindingTableLayoutDirty();

            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), false);

            size_t sbtSize;
            scene0.generateShaderBindingTableLayout(&sbtSize);
            EXPECT_NE(sbtSize, 0);

            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), true);

            scene0.markShaderBindingTableLayoutDirty();

            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), false);
        }

        // JP: 各オブジェクトの生成。
        {
            optixu::GeometryInstance geomInstTri = scene0.createGeometryInstance();
            EXPECT_NE(geomInstTri, optixu::GeometryInstance());
            geomInstTri.destroy();

            optixu::GeometryInstance geomInstLinear =
                scene0.createGeometryInstance(optixu::GeometryType::LinearSegments);
            EXPECT_NE(geomInstLinear, optixu::GeometryInstance());
            geomInstLinear.destroy();

            optixu::GeometryInstance geomInstQuadratic =
                scene0.createGeometryInstance(optixu::GeometryType::QuadraticBSplines);
            EXPECT_NE(geomInstQuadratic, optixu::GeometryInstance());
            geomInstQuadratic.destroy();

            optixu::GeometryInstance geomInstCubic =
                scene0.createGeometryInstance(optixu::GeometryType::CubicBSplines);
            EXPECT_NE(geomInstCubic, optixu::GeometryInstance());
            geomInstCubic.destroy();

            optixu::GeometryInstance geomInstCustom =
                scene0.createGeometryInstance(optixu::GeometryType::CustomPrimitives);
            EXPECT_NE(geomInstCustom, optixu::GeometryInstance());
            geomInstCustom.destroy();

            EXPECT_EXCEPTION(scene0.createGeometryInstance(static_cast<optixu::GeometryType>(~0)));

            optixu::GeometryAccelerationStructure gasTri = scene0.createGeometryAccelerationStructure();
            EXPECT_NE(gasTri, optixu::GeometryAccelerationStructure());
            gasTri.destroy();

            optixu::GeometryAccelerationStructure gasLinear =
                scene0.createGeometryAccelerationStructure(optixu::GeometryType::LinearSegments);
            EXPECT_NE(gasLinear, optixu::GeometryAccelerationStructure());
            gasLinear.destroy();

            optixu::GeometryAccelerationStructure gasQuadratic =
                scene0.createGeometryAccelerationStructure(optixu::GeometryType::QuadraticBSplines);
            EXPECT_NE(gasQuadratic, optixu::GeometryAccelerationStructure());
            gasQuadratic.destroy();

            optixu::GeometryAccelerationStructure gasCubic =
                scene0.createGeometryAccelerationStructure(optixu::GeometryType::CubicBSplines);
            EXPECT_NE(gasCubic, optixu::GeometryAccelerationStructure());
            gasCubic.destroy();

            optixu::GeometryAccelerationStructure gasCustom =
                scene0.createGeometryAccelerationStructure(optixu::GeometryType::CustomPrimitives);
            EXPECT_NE(gasCustom, optixu::GeometryAccelerationStructure());
            gasCustom.destroy();

            EXPECT_EXCEPTION(scene0.createGeometryAccelerationStructure(static_cast<optixu::GeometryType>(~0)));

            optixu::Transform xfm = scene0.createTransform();
            EXPECT_NE(xfm, optixu::Transform());
            xfm.destroy();

            optixu::Instance inst = scene0.createInstance();
            EXPECT_NE(inst, optixu::Instance());
            inst.destroy();

            optixu::InstanceAccelerationStructure ias = scene0.createInstanceAccelerationStructure();
            EXPECT_NE(ias, optixu::InstanceAccelerationStructure());
            ias.destroy();
        }

        // JP: オブジェクト生成後のSBTレイアウト関連。
        {
            size_t sbtSize;
            scene0.generateShaderBindingTableLayout(&sbtSize);
            EXPECT_NE(sbtSize, 0);

            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), true);

            scene0.markShaderBindingTableLayoutDirty();

            EXPECT_EQ(scene0.shaderBindingTableLayoutIsReady(), false);
        }

        scene0.destroy();

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(GeometryInstanceTest, GeometryInstanceBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        optixu::Scene scene = context.createScene();

        optixu::GeometryInstance geomInstTri = scene.createGeometryInstance();
        optixu::GeometryInstance geomInstLinear =
            scene.createGeometryInstance(optixu::GeometryType::LinearSegments);
        optixu::GeometryInstance geomInstQuadratic =
            scene.createGeometryInstance(optixu::GeometryType::QuadraticBSplines);
        optixu::GeometryInstance geomInstCubic =
            scene.createGeometryInstance(optixu::GeometryType::CubicBSplines);
        optixu::GeometryInstance geomInstCustom =
            scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);

        // JP: 共通処理。
        {
            EXPECT_EQ(geomInstTri.getContext(), context);

            // JP: 普通の名前の設定と取得。
            const char* nameA = "ABCDE";
            geomInstTri.setName(nameA);
            EXPECT_STREQ(geomInstTri.getName(), nameA);

            // JP: 空白の名前の設定と取得。
            const char* nameB = "";
            geomInstTri.setName(nameB);
            EXPECT_STREQ(geomInstTri.getName(), nameB);
        }

        // JP: 三角形プリミティブ。
        {
            optixu::GeometryInstance geomInst = geomInstTri;

            // JP: まずはデフォルト値の取得。
            EXPECT_EQ(geomInst.getMotionStepCount(), 1);
            EXPECT_EQ(geomInst.getVertexFormat(), OPTIX_VERTEX_FORMAT_FLOAT3);
            EXPECT_EQ(geomInst.getVertexBuffer(), optixu::BufferView());
            EXPECT_EXCEPTION(geomInst.getWidthBuffer());
            OptixIndicesFormat retIndicesFormat;
            EXPECT_EQ(geomInst.getTriangleBuffer(&retIndicesFormat), optixu::BufferView());
            EXPECT_EQ(retIndicesFormat, OPTIX_INDICES_FORMAT_NONE);
            EXPECT_EXCEPTION(geomInst.getSegmentIndexBuffer());
            EXPECT_EXCEPTION(geomInst.getCustomPrimitiveAABBBuffer());
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), 0);
            optixu::BufferView retMatIndexBuffer;
            optixu::IndexSize retMatIndexSize;
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, optixu::IndexSize::None);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setMotionStepCount(numMotionSteps);
            EXPECT_EQ(geomInst.getMotionStepCount(), numMotionSteps);

            OptixVertexFormat vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            geomInst.setVertexFormat(vertexFormat);
            EXPECT_EQ(geomInst.getVertexFormat(), vertexFormat);

            struct Vertex {
                float3 position;
            };
            std::vector<optixu::BufferView> vertexBuffers(numMotionSteps);
            uint32_t numVertices = 512;
            for (int step = 0; step < numMotionSteps; ++step) {
                vertexBuffers[step] =
                    optixu::BufferView(static_cast<CUdeviceptr>(step), numVertices, sizeof(Vertex));
                geomInst.setVertexBuffer(vertexBuffers[step], step);
                EXPECT_EQ(geomInst.getVertexBuffer(step), vertexBuffers[step]);
            }
            EXPECT_EXCEPTION(geomInst.setVertexBuffer(optixu::BufferView(), numMotionSteps));
            EXPECT_EXCEPTION(geomInst.getVertexBuffer(numMotionSteps));

            EXPECT_EXCEPTION(geomInst.setWidthBuffer(optixu::BufferView()));

            uint32_t numPrimitives = 128;
            optixu::BufferView triangleBuffer;
            triangleBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t) * 3);
            geomInst.setTriangleBuffer(triangleBuffer);
            EXPECT_EQ(geomInst.getTriangleBuffer(&retIndicesFormat), triangleBuffer);
            EXPECT_EQ(retIndicesFormat, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);

            triangleBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint16_t) * 3);
            geomInst.setTriangleBuffer(triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);
            EXPECT_EQ(geomInst.getTriangleBuffer(&retIndicesFormat), triangleBuffer);
            EXPECT_EQ(retIndicesFormat, OPTIX_INDICES_FORMAT_UNSIGNED_SHORT3);

            EXPECT_EXCEPTION(geomInst.setSegmentIndexBuffer(optixu::BufferView()));
            EXPECT_EXCEPTION(geomInst.setCustomPrimitiveAABBBuffer(optixu::BufferView()));

            uint32_t primIndexOffset = 55555;
            geomInst.setPrimitiveIndexOffset(primIndexOffset);
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), primIndexOffset);

            uint32_t numMaterials = 3;
            optixu::BufferView matIndexBuffer;
            optixu::IndexSize matIndexSize;
            matIndexSize = optixu::IndexSize::k4Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = optixu::IndexSize::k2Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = optixu::IndexSize::k4Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint16_t));
            EXPECT_EXCEPTION(geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize));

            for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
                geomInst.setGeometryFlags(matIdx, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
                EXPECT_EQ(geomInst.getGeometryFlags(matIdx), OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
            }
            EXPECT_EXCEPTION(
                geomInst.setGeometryFlags(numMaterials, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL));
            EXPECT_EXCEPTION(geomInst.getGeometryFlags(numMaterials));

            // JP: ユーザーデータ関連。
            uint32_t udSize, udAlignment;

            // JP: まずは普通に設定と取得。
            struct UserData0 {
                uint32_t a;
                float2 b;
            };
            UserData0 ud0 = {};
            ud0.a = 1;
            ud0.b = float2(2, 3);
            geomInst.setUserData(ud0);
            UserData0 retUd0;
            geomInst.getUserData(&retUd0, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud0));
            EXPECT_EQ(udAlignment, alignof(UserData0));
            EXPECT_EQ(std::memcmp(&ud0, &retUd0, sizeof(ud0)), 0);

            // JP: 限界サイズのユーザーデータの設定と取得。
            struct UserData1 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize];
            };
            UserData1 ud1 = {};
            for (int i = 0; i < sizeof(ud1.a); ++i)
                ud1.a[i] = i;
            geomInst.setUserData(ud1);
            UserData1 retUd1;
            geomInst.getUserData(&retUd1, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud1));
            EXPECT_EQ(udAlignment, alignof(UserData1));
            EXPECT_EQ(std::memcmp(&ud1, &retUd1, sizeof(ud1)), 0);

            // JP: 限界サイズを超えたユーザーデータの設定と取得。
            struct UserData2 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize + 2];
            };
            UserData2 ud2 = {};
            for (int i = 0; i < sizeof(ud2.a); ++i)
                ud2.a[i] = i;
            EXPECT_EXCEPTION(geomInst.setUserData(ud2));
        }

        // JP: 曲線プリミティブ。
        optixu::GeometryInstance curveGeomInsts[] = {
            geomInstLinear, geomInstQuadratic, geomInstCubic
        };
        for (int curveDim = 1; curveDim <= 3; ++curveDim) {
            optixu::GeometryInstance geomInst = curveGeomInsts[curveDim - 1];

            // JP: まずはデフォルト値の取得。
            EXPECT_EQ(geomInst.getMotionStepCount(), 1);
            EXPECT_EXCEPTION(geomInst.getVertexFormat());
            EXPECT_EQ(geomInst.getVertexBuffer(), optixu::BufferView());
            EXPECT_EQ(geomInst.getWidthBuffer(), optixu::BufferView());
            EXPECT_EXCEPTION(geomInst.getTriangleBuffer());
            EXPECT_EQ(geomInst.getSegmentIndexBuffer(), optixu::BufferView());
            EXPECT_EXCEPTION(geomInst.getCustomPrimitiveAABBBuffer());
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), 0);
            optixu::BufferView retMatIndexBuffer;
            optixu::IndexSize retMatIndexSize;
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, optixu::IndexSize::None);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setMotionStepCount(numMotionSteps);
            EXPECT_EQ(geomInst.getMotionStepCount(), numMotionSteps);

            EXPECT_EXCEPTION(geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3));

            struct Vertex {
                float3 position;
            };
            std::vector<optixu::BufferView> vertexBuffers(numMotionSteps);
            std::vector<optixu::BufferView> widthBuffers(numMotionSteps);
            uint32_t numVertices = 512;
            for (int step = 0; step < numMotionSteps; ++step) {
                vertexBuffers[step] =
                    optixu::BufferView(static_cast<CUdeviceptr>(step), numVertices, sizeof(Vertex));
                widthBuffers[step] =
                    optixu::BufferView(static_cast<CUdeviceptr>(step), numVertices, sizeof(float));
                geomInst.setVertexBuffer(vertexBuffers[step], step);
                geomInst.setWidthBuffer(widthBuffers[step], step);
                EXPECT_EQ(geomInst.getVertexBuffer(step), vertexBuffers[step]);
                EXPECT_EQ(geomInst.getWidthBuffer(step), widthBuffers[step]);
            }
            EXPECT_EXCEPTION(geomInst.setVertexBuffer(optixu::BufferView(), numMotionSteps));
            EXPECT_EXCEPTION(geomInst.getVertexBuffer(numMotionSteps));
            EXPECT_EXCEPTION(geomInst.setWidthBuffer(optixu::BufferView(), numMotionSteps));
            EXPECT_EXCEPTION(geomInst.getWidthBuffer(numMotionSteps));

            EXPECT_EXCEPTION(geomInst.setTriangleBuffer(optixu::BufferView()));

            uint32_t numSegments = 128;
            optixu::BufferView segmentIndexBuffer;
            segmentIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numSegments, sizeof(uint32_t));
            geomInst.setSegmentIndexBuffer(segmentIndexBuffer);
            EXPECT_EQ(geomInst.getSegmentIndexBuffer(), segmentIndexBuffer);

            EXPECT_EXCEPTION(geomInst.setCustomPrimitiveAABBBuffer(optixu::BufferView()));

            uint32_t primIndexOffset = 55555;
            geomInst.setPrimitiveIndexOffset(primIndexOffset);
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), primIndexOffset);

            uint32_t numMaterials = 1;
            optixu::BufferView matIndexBuffer;
            optixu::IndexSize matIndexSize;
            matIndexSize = optixu::IndexSize::k4Bytes;
            matIndexBuffer = optixu::BufferView(static_cast<CUdeviceptr>(12345678), numSegments, sizeof(uint32_t));
            EXPECT_EXCEPTION(geomInst.setMaterialCount(2, matIndexBuffer, matIndexSize));

            for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
                geomInst.setGeometryFlags(matIdx, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
                EXPECT_EQ(geomInst.getGeometryFlags(matIdx), OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
            }
            EXPECT_EXCEPTION(
                geomInst.setGeometryFlags(numMaterials, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL));
            EXPECT_EXCEPTION(geomInst.getGeometryFlags(numMaterials));

            // JP: ユーザーデータ関連。
            uint32_t udSize, udAlignment;

            // JP: まずは普通に設定と取得。
            struct UserData0 {
                uint32_t a;
                float2 b;
            };
            UserData0 ud0 = {};
            ud0.a = 1;
            ud0.b = float2(2, 3);
            geomInst.setUserData(ud0);
            UserData0 retUd0;
            geomInst.getUserData(&retUd0, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud0));
            EXPECT_EQ(udAlignment, alignof(UserData0));
            EXPECT_EQ(std::memcmp(&ud0, &retUd0, sizeof(ud0)), 0);

            // JP: 限界サイズのユーザーデータの設定と取得。
            struct UserData1 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize];
            };
            UserData1 ud1 = {};
            for (int i = 0; i < sizeof(ud1.a); ++i)
                ud1.a[i] = i;
            geomInst.setUserData(ud1);
            UserData1 retUd1;
            geomInst.getUserData(&retUd1, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud1));
            EXPECT_EQ(udAlignment, alignof(UserData1));
            EXPECT_EQ(std::memcmp(&ud1, &retUd1, sizeof(ud1)), 0);

            // JP: 限界サイズを超えたユーザーデータの設定と取得。
            struct UserData2 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize + 2];
            };
            UserData2 ud2 = {};
            for (int i = 0; i < sizeof(ud2.a); ++i)
                ud2.a[i] = i;
            EXPECT_EXCEPTION(geomInst.setUserData(ud2));
        }

        // JP: ユーザー定義プリミティブ。
        {
            optixu::GeometryInstance geomInst = geomInstCustom;

            // JP: まずはデフォルト値の取得。
            EXPECT_EQ(geomInst.getMotionStepCount(), 1);
            EXPECT_EXCEPTION(geomInst.getVertexFormat());
            EXPECT_EXCEPTION(geomInst.getVertexBuffer());
            EXPECT_EXCEPTION(geomInst.getWidthBuffer());
            EXPECT_EXCEPTION(geomInst.getTriangleBuffer());
            EXPECT_EXCEPTION(geomInst.getSegmentIndexBuffer());
            EXPECT_EQ(geomInst.getCustomPrimitiveAABBBuffer(), optixu::BufferView());
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), 0);
            optixu::BufferView retMatIndexBuffer;
            optixu::IndexSize retMatIndexSize;
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, optixu::IndexSize::None);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setMotionStepCount(numMotionSteps);
            EXPECT_EQ(geomInst.getMotionStepCount(), numMotionSteps);

            EXPECT_EXCEPTION(geomInst.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3));
            EXPECT_EXCEPTION(geomInst.setVertexBuffer(optixu::BufferView(), 0));
            EXPECT_EXCEPTION(geomInst.setWidthBuffer(optixu::BufferView(), 0));
            EXPECT_EXCEPTION(geomInst.setTriangleBuffer(optixu::BufferView()));
            EXPECT_EXCEPTION(geomInst.setSegmentIndexBuffer(optixu::BufferView()));

            struct Primitive {
                OptixAabb aabb;
            };
            uint32_t numPrimitives = 128;
            std::vector<optixu::BufferView> aabbBuffers(numMotionSteps);
            for (int step = 0; step < numMotionSteps; ++step) {
                aabbBuffers[step] =
                    optixu::BufferView(static_cast<CUdeviceptr>(step), numPrimitives, sizeof(Primitive));
                geomInst.setCustomPrimitiveAABBBuffer(aabbBuffers[step], step);
                EXPECT_EQ(geomInst.getCustomPrimitiveAABBBuffer(step), aabbBuffers[step]);
            }

            uint32_t primIndexOffset = 55555;
            geomInst.setPrimitiveIndexOffset(primIndexOffset);
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), primIndexOffset);

            uint32_t numMaterials = 3;
            optixu::BufferView matIndexBuffer;
            optixu::IndexSize matIndexSize;
            matIndexSize = optixu::IndexSize::k4Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = optixu::IndexSize::k2Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getMaterialCount(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = optixu::IndexSize::k4Bytes;
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint16_t));
            EXPECT_EXCEPTION(geomInst.setMaterialCount(numMaterials, matIndexBuffer, matIndexSize));

            for (int matIdx = 0; matIdx < numMaterials; ++matIdx) {
                geomInst.setGeometryFlags(matIdx, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
                EXPECT_EQ(geomInst.getGeometryFlags(matIdx), OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL);
            }
            EXPECT_EXCEPTION(
                geomInst.setGeometryFlags(numMaterials, OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL));
            EXPECT_EXCEPTION(geomInst.getGeometryFlags(numMaterials));

            // JP: ユーザーデータ関連。
            uint32_t udSize, udAlignment;

            // JP: まずは普通に設定と取得。
            struct UserData0 {
                uint32_t a;
                float2 b;
            };
            UserData0 ud0 = {};
            ud0.a = 1;
            ud0.b = float2(2, 3);
            geomInst.setUserData(ud0);
            UserData0 retUd0;
            geomInst.getUserData(&retUd0, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud0));
            EXPECT_EQ(udAlignment, alignof(UserData0));
            EXPECT_EQ(std::memcmp(&ud0, &retUd0, sizeof(ud0)), 0);

            // JP: 限界サイズのユーザーデータの設定と取得。
            struct UserData1 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize];
            };
            UserData1 ud1 = {};
            for (int i = 0; i < sizeof(ud1.a); ++i)
                ud1.a[i] = i;
            geomInst.setUserData(ud1);
            UserData1 retUd1;
            geomInst.getUserData(&retUd1, &udSize, &udAlignment);
            EXPECT_EQ(udSize, sizeof(ud1));
            EXPECT_EQ(udAlignment, alignof(UserData1));
            EXPECT_EQ(std::memcmp(&ud1, &retUd1, sizeof(ud1)), 0);

            // JP: 限界サイズを超えたユーザーデータの設定と取得。
            struct UserData2 {
                uint8_t a[optixu::s_maxGeometryInstanceUserDataSize + 2];
            };
            UserData2 ud2 = {};
            for (int i = 0; i < sizeof(ud2.a); ++i)
                ud2.a[i] = i;
            EXPECT_EXCEPTION(geomInst.setUserData(ud2));
        }

        geomInstCustom.destroy();
        geomInstCubic.destroy();
        geomInstQuadratic.destroy();
        geomInstLinear.destroy();
        geomInstTri.destroy();
        
        scene.destroy();

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(PipelineTest, PipelineBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        optixu::Pipeline pipeline = context.createPipeline();

        // JP: 共通処理。
        {
            EXPECT_EQ(pipeline.getContext(), context);

            // JP: 普通の名前の設定と取得。
            const char* nameA = "Pipeline_Test";
            pipeline.setName(nameA);
            EXPECT_STREQ(pipeline.getName(), nameA);

            // JP: 空白の名前の設定と取得。
            const char* nameB = "";
            pipeline.setName(nameB);
            EXPECT_STREQ(pipeline.getName(), nameB);
        }

        // JP: パイプラインオプションの設定。
        {
            optixu::PipelineOptions options;
            options.payloadCountInDwords = 8;
            options.attributeCountInDwords = 4;
            options.launchParamsVariableName = "plp";
            options.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
            options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            options.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            options.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            options.useMotionBlur = optixu::UseMotionBlur::No;
            options.useOpacityMicroMaps = optixu::UseOpacityMicroMaps::No;
            options.allowClusteredGeometry = optixu::AllowClusteredGeometry::No;

            pipeline.setPipelineOptions(options);
        }

        // JP: レイタイプとコール可能プログラムの数の設定。
        {
            pipeline.setMissRayTypeCount(2);
            pipeline.setCallableProgramCount(1);

            size_t sbtSize;
            pipeline.generateShaderBindingTableLayout(&sbtSize);
            EXPECT_NE(sbtSize, 0);
        }

        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(ModuleTest, ModuleBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline.setPipelineOptions(pipelineOptions);

        // JP: OptixIRからのモジュール作成をテスト。
        {
            const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
            EXPECT_GT(optixIr.size(), 0);

            optixu::Module module = pipeline.createModuleFromOptixIR(
                optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
                DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

            EXPECT_NE(module, optixu::Module());

            // JP: 共通処理。
            {
                EXPECT_EQ(module.getContext(), context);

                const char* nameA = "Test_Module";
                module.setName(nameA);
                EXPECT_STREQ(module.getName(), nameA);

                const char* nameB = "";
                module.setName(nameB);
                EXPECT_STREQ(module.getName(), nameB);
            }

            module.destroy();
        }

        // JP: 空のモジュールテスト。
        {
            optixu::Module emptyModule;
            EXPECT_EQ(emptyModule, optixu::Module());
        }

        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(ProgramTest, ProgramBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        // JP: レイ生成プログラムのテスト。
        {
            optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
            EXPECT_NE(rayGenProgram, optixu::Program());

            // JP: 共通処理。
            {
                EXPECT_EQ(rayGenProgram.getContext(), context);

                const char* nameA = "RayGen_Program";
                rayGenProgram.setName(nameA);
                EXPECT_STREQ(rayGenProgram.getName(), nameA);

                const char* nameB = "";
                rayGenProgram.setName(nameB);
                EXPECT_STREQ(rayGenProgram.getName(), nameB);
            }

            // JP: スタックサイズの取得。
            uint32_t stackSize = rayGenProgram.getStackSize();
            EXPECT_GE(stackSize, 0);

            // JP: パイプラインでの有効/無効の設定。
            rayGenProgram.setActiveInPipeline(true);
            rayGenProgram.setActiveInPipeline(false);

            rayGenProgram.destroy();
        }

        // JP: ミスプログラムのテスト。
        {
            optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
            EXPECT_NE(missProgram, optixu::Program());

            uint32_t stackSize = missProgram.getStackSize();
            EXPECT_GE(stackSize, 0);

            missProgram.destroy();
        }

        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(HitProgramGroupTest, HitProgramGroupBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        // JP: 三角形用ヒットプログラムグループのテスト。
        {
            optixu::HitProgramGroup hitGroup = pipeline.createHitProgramGroupForTriangleIS(
                moduleOptiX, RT_CH_NAME_STR("ch0"),
                emptyModule, nullptr);
            EXPECT_NE(hitGroup, optixu::HitProgramGroup());

            // JP: 共通処理。
            {
                EXPECT_EQ(hitGroup.getContext(), context);

                const char* nameA = "HitGroup_Triangle";
                hitGroup.setName(nameA);
                EXPECT_STREQ(hitGroup.getName(), nameA);

                const char* nameB = "";
                hitGroup.setName(nameB);
                EXPECT_STREQ(hitGroup.getName(), nameB);
            }

            // JP: スタックサイズの取得。
            uint32_t chStackSize = hitGroup.getCHStackSize();
            uint32_t ahStackSize = hitGroup.getAHStackSize();
            uint32_t isStackSize = hitGroup.getISStackSize();
            EXPECT_GE(chStackSize, 0);
            EXPECT_GE(ahStackSize, 0);
            EXPECT_GE(isStackSize, 0);

            // JP: パイプラインでの有効/無効の設定。
            hitGroup.setActiveInPipeline(true);
            hitGroup.setActiveInPipeline(false);

            hitGroup.destroy();
        }

        // JP: 空のヒットプログラムグループのテスト。
        {
            optixu::HitProgramGroup emptyHitGroup = pipeline.createEmptyHitProgramGroup();
            EXPECT_NE(emptyHitGroup, optixu::HitProgramGroup());

            uint32_t chStackSize = emptyHitGroup.getCHStackSize();
            uint32_t ahStackSize = emptyHitGroup.getAHStackSize();
            uint32_t isStackSize = emptyHitGroup.getISStackSize();
            EXPECT_GE(chStackSize, 0);
            EXPECT_GE(ahStackSize, 0);
            EXPECT_GE(isStackSize, 0);

            emptyHitGroup.destroy();
        }

        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(CallableProgramGroupTest, CallableProgramGroupBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        // JP: コール可能プログラムグループのテスト。
        {
            optixu::CallableProgramGroup callableGroup = pipeline.createCallableProgramGroup(
                emptyModule, nullptr,
                emptyModule, nullptr);
            EXPECT_NE(callableGroup, optixu::CallableProgramGroup());

            // JP: 共通処理。
            {
                EXPECT_EQ(callableGroup.getContext(), context);

                const char* nameA = "Callable_Group";
                callableGroup.setName(nameA);
                EXPECT_STREQ(callableGroup.getName(), nameA);

                const char* nameB = "";
                callableGroup.setName(nameB);
                EXPECT_STREQ(callableGroup.getName(), nameB);
            }

            // JP: スタックサイズの取得。
            uint32_t dcStackSize = callableGroup.getDCStackSize();
            uint32_t ccStackSize = callableGroup.getCCStackSize();
            EXPECT_GE(dcStackSize, 0);
            EXPECT_GE(ccStackSize, 0);

            // JP: パイプラインでの有効/無効の設定。
            callableGroup.setActiveInPipeline(true);
            callableGroup.setActiveInPipeline(false);

            callableGroup.destroy();
        }

        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(TransformTest, TransformBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        optixu::Transform transform = scene.createTransform();

        // JP: 共通処理。
        {
            EXPECT_EQ(transform.getContext(), context);

            const char* nameA = "Test_Transform";
            transform.setName(nameA);
            EXPECT_STREQ(transform.getName(), nameA);

            const char* nameB = "";
            transform.setName(nameB);
            EXPECT_STREQ(transform.getName(), nameB);
        }

        // JP: 静的変換の設定とテスト。
        {
            float matrix[12] = {
                1.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 2.0f,
                0.0f, 0.0f, 1.0f, 3.0f
            };

            transform.setStaticTransform(matrix);

            optixu::TransformType type;
            uint32_t numKeys;
            transform.getConfiguration(&type, &numKeys);
            EXPECT_EQ(type, optixu::TransformType::Static);
            EXPECT_EQ(numKeys, 1);

            float retrievedMatrix[12];
            transform.getStaticTransform(retrievedMatrix);
            for (int i = 0; i < 12; ++i) {
                EXPECT_FLOAT_EQ(matrix[i], retrievedMatrix[i]);
            }
        }

        // JP: モーション変換の設定とテスト。
        {
            uint32_t numKeys = 3;
            size_t transformSize;
            transform.setConfiguration(optixu::TransformType::MatrixMotion, numKeys, &transformSize);
            EXPECT_GT(transformSize, 0);

            transform.setMotionOptions(0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);

            for (uint32_t keyIdx = 0; keyIdx < numKeys; ++keyIdx) {
                float matrix[12] = {
                    1.0f, 0.0f, 0.0f, static_cast<float>(keyIdx),
                    0.0f, 1.0f, 0.0f, static_cast<float>(keyIdx),
                    0.0f, 0.0f, 1.0f, static_cast<float>(keyIdx)
                };
                transform.setMatrixMotionKey(keyIdx, matrix);

                float retrievedMatrix[12];
                transform.getMatrixMotionKey(keyIdx, retrievedMatrix);
                for (int i = 0; i < 12; ++i) {
                    EXPECT_FLOAT_EQ(matrix[i], retrievedMatrix[i]);
                }
            }

            float timeBegin, timeEnd;
            OptixMotionFlags flags;
            transform.getMotionOptions(&timeBegin, &timeEnd, &flags);
            EXPECT_FLOAT_EQ(timeBegin, 0.0f);
            EXPECT_FLOAT_EQ(timeEnd, 1.0f);
            EXPECT_EQ(flags, OPTIX_MOTION_FLAG_NONE);
        }

        // JP: Dirty状態のテスト。
        {
            transform.markDirty();
            EXPECT_EQ(transform.isReady(), false);
        }

        transform.destroy();
        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(InstanceTest, InstanceBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        optixu::Instance instance = scene.createInstance();

        // JP: 共通処理。
        {
            EXPECT_EQ(instance.getContext(), context);

            const char* nameA = "Test_Instance";
            instance.setName(nameA);
            EXPECT_STREQ(instance.getName(), nameA);

            const char* nameB = "";
            instance.setName(nameB);
            EXPECT_STREQ(instance.getName(), nameB);
        }

        // JP: インスタンスのプロパティ設定とテスト。
        {
            // JP: IDの設定と取得。
            uint32_t testID = 42;
            instance.setID(testID);
            EXPECT_EQ(instance.getID(), testID);

            // JP: 可視性マスクの設定と取得。
            uint32_t visibilityMask = 0xFF;
            instance.setVisibilityMask(visibilityMask);
            EXPECT_EQ(instance.getVisibilityMask(), visibilityMask);

            // JP: フラグの設定と取得。
            OptixInstanceFlags flags = OPTIX_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE;
            instance.setFlags(flags);
            EXPECT_EQ(instance.getFlags(), flags);

            // JP: 変換行列の設定と取得。
            float transform[12] = {
                1.0f, 0.0f, 0.0f, 1.0f,
                0.0f, 1.0f, 0.0f, 2.0f,
                0.0f, 0.0f, 1.0f, 3.0f
            };
            instance.setTransform(transform);
            
            float retrievedTransform[12];
            instance.getTransform(retrievedTransform);
            for (int i = 0; i < 12; ++i) {
                EXPECT_FLOAT_EQ(transform[i], retrievedTransform[i]);
            }

            // JP: マテリアルセットインデックスの設定と取得。
            uint32_t matSetIdx = 1;
            instance.setMaterialSetIndex(matSetIdx);
            EXPECT_EQ(instance.getMaterialSetIndex(), matSetIdx);
        }

        instance.destroy();
        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(BufferViewTest, BufferViewBasic) {
    try {
        // JP: デフォルトコンストラクターのテスト。
        {
            optixu::BufferView defaultView;
            EXPECT_EQ(defaultView.getCUdeviceptr(), 0);
            EXPECT_EQ(defaultView.numElements(), 0);
            EXPECT_EQ(defaultView.stride(), 0);
            EXPECT_EQ(defaultView.sizeInBytes(), 0);
            EXPECT_EQ(defaultView.isValid(), false);
        }

        // JP: パラメーター付きコンストラクターのテスト。
        {
            CUdeviceptr devicePtr = reinterpret_cast<CUdeviceptr>(0x12345678);
            size_t numElements = 100;
            uint32_t stride = 16;

            optixu::BufferView bufferView(devicePtr, numElements, stride);
            EXPECT_EQ(bufferView.getCUdeviceptr(), devicePtr);
            EXPECT_EQ(bufferView.numElements(), numElements);
            EXPECT_EQ(bufferView.stride(), stride);
            EXPECT_EQ(bufferView.sizeInBytes(), numElements * stride);
            EXPECT_EQ(bufferView.isValid(), true);

            // JP: インデックス指定でのアドレス取得テスト。
            uint32_t index = 5;
            CUdeviceptr expectedPtr = devicePtr + stride * index;
            EXPECT_EQ(bufferView.getCUdeviceptrAt(index), expectedPtr);
        }

        // JP: 等価演算子のテスト。
        {
            CUdeviceptr ptr1 = reinterpret_cast<CUdeviceptr>(0x12345678);
            CUdeviceptr ptr2 = reinterpret_cast<CUdeviceptr>(0x87654321);
            
            optixu::BufferView view1(ptr1, 100, 16);
            optixu::BufferView view2(ptr1, 100, 16);
            optixu::BufferView view3(ptr2, 100, 16);
            optixu::BufferView view4(ptr1, 200, 16);
            optixu::BufferView view5(ptr1, 100, 32);

            EXPECT_EQ(view1, view2);
            EXPECT_NE(view1, view3);
            EXPECT_NE(view1, view4);
            EXPECT_NE(view1, view5);
        }
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(GeometryAccelerationStructureTest, GeometryAccelerationStructureBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();

        // JP: 共通処理。
        {
            EXPECT_EQ(gas.getContext(), context);

            const char* nameA = "Test_GAS";
            gas.setName(nameA);
            EXPECT_STREQ(gas.getName(), nameA);

            const char* nameB = "";
            gas.setName(nameB);
            EXPECT_STREQ(gas.getName(), nameB);
        }

        // JP: 設定のテスト。
        {
            gas.setConfiguration(
                optixu::ASTradeoff::PreferFastTrace,
                optixu::AllowUpdate::Yes,
                optixu::AllowCompaction::Yes,
                optixu::AllowRandomVertexAccess::No);

            optixu::ASTradeoff tradeoff;
            optixu::AllowUpdate allowUpdate;
            optixu::AllowCompaction allowCompaction;
            optixu::AllowRandomVertexAccess allowRandomVertexAccess;
            gas.getConfiguration(&tradeoff, &allowUpdate, &allowCompaction, &allowRandomVertexAccess);

            EXPECT_EQ(tradeoff, optixu::ASTradeoff::PreferFastTrace);
            EXPECT_EQ(allowUpdate, optixu::AllowUpdate::Yes);
            EXPECT_EQ(allowCompaction, optixu::AllowCompaction::Yes);
            EXPECT_EQ(allowRandomVertexAccess, optixu::AllowRandomVertexAccess::No);
        }

        // JP: 子要素の追加とテスト。
        {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();
            gas.addChild(geomInst);
            EXPECT_EQ(gas.getChildCount(), 1);

            EXPECT_EQ(gas.findChildIndex(geomInst), 0);
            EXPECT_EQ(gas.getChild(0), geomInst);

            gas.removeChildAt(0);
            EXPECT_EQ(gas.getChildCount(), 0);

            gas.addChild(geomInst);
            gas.clearChildren();
            EXPECT_EQ(gas.getChildCount(), 0);

            geomInst.destroy();
        }

        // JP: Dirty状態のテスト。
        {
            gas.markDirty();
            EXPECT_EQ(gas.isReady(), false);
        }

        gas.destroy();
        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(InstanceAccelerationStructureTest, InstanceAccelerationStructureBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();

        // JP: 共通処理。
        {
            EXPECT_EQ(ias.getContext(), context);

            const char* nameA = "Test_IAS";
            ias.setName(nameA);
            EXPECT_STREQ(ias.getName(), nameA);

            const char* nameB = "";
            ias.setName(nameB);
            EXPECT_STREQ(ias.getName(), nameB);
        }

        // JP: 設定のテスト。
        {
            ias.setConfiguration(
                optixu::ASTradeoff::PreferFastBuild,
                optixu::AllowUpdate::Yes,
                optixu::AllowCompaction::No,
                optixu::AllowRandomInstanceAccess::Yes);

            optixu::ASTradeoff tradeoff;
            optixu::AllowUpdate allowUpdate;
            optixu::AllowCompaction allowCompaction;
            optixu::AllowRandomInstanceAccess allowRandomInstanceAccess;
            ias.getConfiguration(&tradeoff, &allowUpdate, &allowCompaction, &allowRandomInstanceAccess);

            EXPECT_EQ(tradeoff, optixu::ASTradeoff::PreferFastBuild);
            EXPECT_EQ(allowUpdate, optixu::AllowUpdate::Yes);
            EXPECT_EQ(allowCompaction, optixu::AllowCompaction::No);
            EXPECT_EQ(allowRandomInstanceAccess, optixu::AllowRandomInstanceAccess::Yes);
        }

        // JP: モーションオプションのテスト。
        {
            uint32_t numKeys = 5;
            float timeBegin = 0.5f;
            float timeEnd = 1.5f;
            OptixMotionFlags flags = OPTIX_MOTION_FLAG_START_VANISH;

            ias.setMotionOptions(numKeys, timeBegin, timeEnd, flags);

            uint32_t retrievedNumKeys;
            float retrievedTimeBegin, retrievedTimeEnd;
            OptixMotionFlags retrievedFlags;
            ias.getMotionOptions(&retrievedNumKeys, &retrievedTimeBegin, &retrievedTimeEnd, &retrievedFlags);

            EXPECT_EQ(retrievedNumKeys, numKeys);
            EXPECT_FLOAT_EQ(retrievedTimeBegin, timeBegin);
            EXPECT_FLOAT_EQ(retrievedTimeEnd, timeEnd);
            EXPECT_EQ(retrievedFlags, flags);
        }

        // JP: 子要素の追加とテスト。
        {
            optixu::Instance instance = scene.createInstance();
            ias.addChild(instance);
            EXPECT_EQ(ias.getChildCount(), 1);

            EXPECT_EQ(ias.findChildIndex(instance), 0);
            EXPECT_EQ(ias.getChild(0), instance);

            ias.removeChildAt(0);
            EXPECT_EQ(ias.getChildCount(), 0);

            ias.addChild(instance);
            ias.clearChildren();
            EXPECT_EQ(ias.getChildCount(), 0);

            instance.destroy();
        }

        // JP: Dirty状態のテスト。
        {
            ias.markDirty();
            EXPECT_EQ(ias.isReady(), false);
        }

        ias.destroy();
        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(DenoiserTest, DenoiserBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        // JP: LDRデノイザーのテスト。
        {
            optixu::Denoiser denoiser = context.createDenoiser(
                OPTIX_DENOISER_MODEL_KIND_LDR,
                optixu::GuideAlbedo::Yes,
                optixu::GuideNormal::Yes,
                OPTIX_DENOISER_ALPHA_MODE_COPY);

            EXPECT_NE(denoiser, optixu::Denoiser());

            // JP: 共通処理。
            {
                EXPECT_EQ(denoiser.getContext(), context);

                const char* nameA = "LDR_Denoiser";
                denoiser.setName(nameA);
                EXPECT_STREQ(denoiser.getName(), nameA);

                const char* nameB = "";
                denoiser.setName(nameB);
                EXPECT_STREQ(denoiser.getName(), nameB);
            }

            // JP: デノイザーの準備テスト（ダミーパラメーター）。
            {
                optixu::DenoiserSizes sizes;
                uint32_t numTasks;
                denoiser.prepare(1024, 768, 1024, 768, &sizes, &numTasks);

                EXPECT_GT(sizes.stateSize, 0);
                EXPECT_GT(sizes.scratchSize, 0);
                EXPECT_EQ(numTasks, 1);
            }

            denoiser.destroy();
        }

        // JP: HDRデノイザーのテスト。
        {
            optixu::Denoiser hdrDenoiser = context.createDenoiser(
                OPTIX_DENOISER_MODEL_KIND_HDR,
                optixu::GuideAlbedo::No,
                optixu::GuideNormal::No,
                OPTIX_DENOISER_ALPHA_MODE_DENOISE);

            EXPECT_NE(hdrDenoiser, optixu::Denoiser());

            hdrDenoiser.destroy();
        }

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(OpacityMicroMapArrayTest, OpacityMicroMapArrayBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        optixu::OpacityMicroMapArray ommArray = scene.createOpacityMicroMapArray();

        // JP: 共通処理。
        {
            EXPECT_EQ(ommArray.getContext(), context);

            const char* nameA = "Test_OMM_Array";
            ommArray.setName(nameA);
            EXPECT_STREQ(ommArray.getName(), nameA);

            const char* nameB = "";
            ommArray.setName(nameB);
            EXPECT_STREQ(ommArray.getName(), nameB);
        }

        // JP: 設定のテスト。
        {
            OptixOpacityMicromapFlags config = OPTIX_OPACITY_MICROMAP_FLAG_NONE;
            ommArray.setConfiguration(config);
            EXPECT_EQ(ommArray.getConfiguration(), config);
        }

        // JP: Dirty状態のテスト。
        {
            ommArray.markDirty();
            EXPECT_EQ(ommArray.isReady(), false);
        }

        ommArray.destroy();
        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(ErrorHandlingTest, InvalidParametersTest) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        // JP: 無効なコールバックレベルでの例外テスト。
        {
            const auto callback = [](uint32_t level, const char* tag, const char* message, void* cbdata) {};
            EXPECT_EXCEPTION(context.setLogCallback(callback, nullptr, 10));
        }

        // JP: 無効なデノイザーモデルでの例外テスト。
        {
            optixu::Denoiser denoiser;
            EXPECT_EXCEPTION_RET(
                denoiser,
                context.createDenoiser(
                    static_cast<OptixDenoiserModelKind>(999),
                    optixu::GuideAlbedo::Yes,
                    optixu::GuideNormal::Yes,
                    OPTIX_DENOISER_ALPHA_MODE_COPY));
        }

        optixu::Scene scene = context.createScene();

        // JP: 無効なジオメトリタイプでの例外テスト。
        {
            EXPECT_EXCEPTION(scene.createGeometryInstance(static_cast<optixu::GeometryType>(999)));
            EXPECT_EXCEPTION(scene.createGeometryAccelerationStructure(static_cast<optixu::GeometryType>(999)));
        }

        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(EdgeCasesTest, EdgeCasesBasic) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        // JP: 空のオブジェクトの等価比較テスト。
        {
            optixu::Pipeline emptyPipeline1;
            optixu::Pipeline emptyPipeline2;
            optixu::Module emptyModule1;
            optixu::Module emptyModule2;
            optixu::Program emptyProgram1;
            optixu::Program emptyProgram2;

            EXPECT_EQ(emptyPipeline1, emptyPipeline2);
            EXPECT_EQ(emptyModule1, emptyModule2);
            EXPECT_EQ(emptyProgram1, emptyProgram2);
        }

        // JP: マテリアルのユーザーデータ限界サイズテスト。
        {
            optixu::Material material = context.createMaterial();

            // JP: 最大サイズのユーザーデータのテスト。
            struct MaxUserData {
                uint8_t data[optixu::s_maxMaterialUserDataSize];
            };
            MaxUserData maxData = {};
            for (size_t i = 0; i < sizeof(maxData.data); ++i) {
                maxData.data[i] = static_cast<uint8_t>(i & 0xFF);
            }

            material.setUserData(maxData);
            
            MaxUserData retrievedData;
            uint32_t size, alignment;
            material.getUserData(&retrievedData, &size, &alignment);
            EXPECT_EQ(size, sizeof(maxData));
            EXPECT_EQ(std::memcmp(&maxData, &retrievedData, sizeof(maxData)), 0);

            // JP: 限界を超えるサイズでの例外テスト。
            struct OversizedUserData {
                uint8_t data[optixu::s_maxMaterialUserDataSize + 1];
            };
            OversizedUserData oversizedData = {};
            EXPECT_EXCEPTION(material.setUserData(oversizedData));

            material.destroy();
        }

        // JP: ジオメトリインスタンスのユーザーデータ限界サイズテスト。
        {
            optixu::Scene scene = context.createScene();
            optixu::GeometryInstance geomInst = scene.createGeometryInstance();

            // JP: 最大サイズのユーザーデータのテスト。
            struct MaxUserData {
                uint8_t data[optixu::s_maxGeometryInstanceUserDataSize];
            };
            MaxUserData maxData = {};
            for (size_t i = 0; i < sizeof(maxData.data); ++i) {
                maxData.data[i] = static_cast<uint8_t>(i & 0xFF);
            }

            geomInst.setUserData(maxData);
            
            MaxUserData retrievedData;
            uint32_t size, alignment;
            geomInst.getUserData(&retrievedData, &size, &alignment);
            EXPECT_EQ(size, sizeof(maxData));
            EXPECT_EQ(std::memcmp(&maxData, &retrievedData, sizeof(maxData)), 0);

            // JP: 限界を超えるサイズでの例外テスト。
            struct OversizedUserData {
                uint8_t data[optixu::s_maxGeometryInstanceUserDataSize + 1];
            };
            OversizedUserData oversizedData = {};
            EXPECT_EXCEPTION(geomInst.setUserData(oversizedData));

            geomInst.destroy();
            scene.destroy();
        }

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(PipelineAdvancedTest, PipelineLinkingAndAdvancedFeatures) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        // JP: プログラムの作成。
        optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
        optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
        optixu::HitProgramGroup hitGroup = pipeline.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch0"), emptyModule, nullptr);
        optixu::CallableProgramGroup callableGroup = pipeline.createCallableProgramGroup(
            emptyModule, nullptr, emptyModule, nullptr);

        // JP: プログラムをパイプラインに設定。
        pipeline.setRayGenerationProgram(rayGenProgram);
        pipeline.setMissProgram(0, missProgram);
        pipeline.setCallableProgram(0, callableGroup);

        // JP: プログラムの取得テスト。
        {
            EXPECT_EQ(pipeline.getRayGenerationProgram(), rayGenProgram);
            EXPECT_EQ(pipeline.getMissProgram(0), missProgram);
            EXPECT_EQ(pipeline.getCallableProgram(0), callableGroup);
        }

        // JP: パイプラインのリンク。
        pipeline.link(2);

        // JP: スタックサイズの設定。
        pipeline.setStackSize(1024, 1024, 2048, 4);

        // JP: シェーダーバインディングテーブルのレイアウト生成。
        size_t sbtSize;
        pipeline.generateShaderBindingTableLayout(&sbtSize);
        EXPECT_GT(sbtSize, 0);

        // JP: ヒットグループSBTをdirty状態にする。
        pipeline.markHitGroupShaderBindingTableDirty();

        // JP: クリーンアップ。
        callableGroup.destroy();
        hitGroup.destroy();
        missProgram.destroy();
        rayGenProgram.destroy();
        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(SceneAdvancedTest, SceneWithMaterialsAndGeometry) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = 
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        // JP: ヒットプログラムグループの作成。
        optixu::HitProgramGroup hitGroupTriangle = pipeline.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch0"), emptyModule, nullptr);
        optixu::HitProgramGroup hitGroupCustom = pipeline.createHitProgramGroupForCustomIS(
            moduleOptiX, RT_CH_NAME_STR("ch1"), emptyModule, nullptr,
            moduleOptiX, RT_IS_NAME_STR("is"));

        // JP: マテリアルの作成とヒットグループの設定。
        optixu::Material material1 = context.createMaterial();
        optixu::Material material2 = context.createMaterial();

        material1.setHitGroup(0, hitGroupTriangle);
        material2.setHitGroup(0, hitGroupCustom);

        // JP: シーンの作成とジオメトリインスタンスの設定。
        optixu::Scene scene = context.createScene();
        pipeline.setScene(scene);
        EXPECT_EQ(pipeline.getScene(), scene);

        optixu::GeometryInstance triangleGeom = scene.createGeometryInstance();
        optixu::GeometryInstance customGeom = scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);

        // JP: ジオメトリインスタンスにマテリアルを設定。
        triangleGeom.setMaterial(0, 0, material1);
        customGeom.setMaterial(0, 0, material2);

        // JP: GASの作成と設定。
        optixu::GeometryAccelerationStructure gasTriangle = scene.createGeometryAccelerationStructure();
        optixu::GeometryAccelerationStructure gasCustom = scene.createGeometryAccelerationStructure(
            optixu::GeometryType::CustomPrimitives);

        gasTriangle.addChild(triangleGeom);
        gasCustom.addChild(customGeom);

        // JP: SBTレイアウトの生成。
        size_t sbtSize;
        scene.generateShaderBindingTableLayout(&sbtSize);
        EXPECT_GT(sbtSize, 0);
        EXPECT_TRUE(scene.shaderBindingTableLayoutIsReady());

        // JP: クリーンアップ。
        gasCustom.destroy();
        gasTriangle.destroy();
        customGeom.destroy();
        triangleGeom.destroy();
        scene.destroy();
        material2.destroy();
        material1.destroy();
        hitGroupCustom.destroy();
        hitGroupTriangle.destroy();
        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(GeometryInstanceAdvancedTest, GeometryInstanceComplexConfiguration) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Scene scene = context.createScene();

        // JP: 異なるジオメトリタイプのテスト。
        std::vector<optixu::GeometryType> geometryTypes = {
            optixu::GeometryType::Triangles,
            optixu::GeometryType::LinearSegments,
            optixu::GeometryType::QuadraticBSplines,
            optixu::GeometryType::CubicBSplines,
            optixu::GeometryType::CustomPrimitives,
            optixu::GeometryType::Spheres
        };

        for (auto geomType : geometryTypes) {
            optixu::GeometryInstance geomInst = scene.createGeometryInstance(geomType);

            // JP: 共通プロパティのテスト。
            {
                uint32_t motionSteps = 3;
                geomInst.setMotionStepCount(motionSteps);
                EXPECT_EQ(geomInst.getMotionStepCount(), motionSteps);

                uint32_t primitiveIndexOffset = 100;
                geomInst.setPrimitiveIndexOffset(primitiveIndexOffset);
                EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), primitiveIndexOffset);
            }

            // JP: ジオメトリタイプ固有の設定テスト。
            switch (geomType) {
                case optixu::GeometryType::Triangles: {
                    // JP: 三角形特有の設定。
                    OptixVertexFormat vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
                    geomInst.setVertexFormat(vertexFormat);
                    EXPECT_EQ(geomInst.getVertexFormat(), vertexFormat);

                    optixu::BufferView triangleBuffer(reinterpret_cast<CUdeviceptr>(0x1000), 100, sizeof(uint32_t) * 3);
                    geomInst.setTriangleBuffer(triangleBuffer, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);

                    OptixIndicesFormat retrievedFormat;
                    optixu::BufferView retrievedBuffer = geomInst.getTriangleBuffer(&retrievedFormat);
                    EXPECT_EQ(retrievedBuffer, triangleBuffer);
                    EXPECT_EQ(retrievedFormat, OPTIX_INDICES_FORMAT_UNSIGNED_INT3);
                    break;
                }
                case optixu::GeometryType::LinearSegments:
                case optixu::GeometryType::QuadraticBSplines:
                case optixu::GeometryType::CubicBSplines: {
                    // JP: カーブ特有の設定。
                    optixu::BufferView widthBuffer(reinterpret_cast<CUdeviceptr>(0x2000), 100, sizeof(float));
                    geomInst.setWidthBuffer(widthBuffer);
                    EXPECT_EQ(geomInst.getWidthBuffer(), widthBuffer);

                    optixu::BufferView segmentIndexBuffer(reinterpret_cast<CUdeviceptr>(0x3000), 50, sizeof(uint32_t));
                    geomInst.setSegmentIndexBuffer(segmentIndexBuffer);
                    EXPECT_EQ(geomInst.getSegmentIndexBuffer(), segmentIndexBuffer);
                    break;
                }
                case optixu::GeometryType::CustomPrimitives: {
                    // JP: カスタムプリミティブ特有の設定。
                    optixu::BufferView aabbBuffer(reinterpret_cast<CUdeviceptr>(0x4000), 25, sizeof(OptixAabb));
                    geomInst.setCustomPrimitiveAABBBuffer(aabbBuffer);
                    EXPECT_EQ(geomInst.getCustomPrimitiveAABBBuffer(), aabbBuffer);
                    break;
                }
                case optixu::GeometryType::Spheres: {
                    // JP: 球特有の設定。
                    optixu::BufferView radiusBuffer(reinterpret_cast<CUdeviceptr>(0x5000), 100, sizeof(float));
                    geomInst.setRadiusBuffer(radiusBuffer);
                    EXPECT_EQ(geomInst.getRadiusBuffer(), radiusBuffer);
                    break;
                }
                default:
                    break;
            }

            // JP: マテリアル設定のテスト。
            {
                uint32_t numMaterials = 2;
                if (geomType != optixu::GeometryType::LinearSegments &&
                    geomType != optixu::GeometryType::QuadraticBSplines &&
                    geomType != optixu::GeometryType::CubicBSplines) { // カーブ系は単一マテリアルのみ
                    
                    optixu::BufferView matIndexBuffer(reinterpret_cast<CUdeviceptr>(0x6000), 100, sizeof(uint32_t));
                    geomInst.setMaterialCount(numMaterials, matIndexBuffer, optixu::IndexSize::k4Bytes);

                    optixu::BufferView retrievedMatBuffer;
                    optixu::IndexSize retrievedIndexSize;
                    uint32_t retrievedMatCount = geomInst.getMaterialCount(&retrievedMatBuffer, &retrievedIndexSize);
                    
                    EXPECT_EQ(retrievedMatCount, numMaterials);
                    EXPECT_EQ(retrievedMatBuffer, matIndexBuffer);
                    EXPECT_EQ(retrievedIndexSize, optixu::IndexSize::k4Bytes);

                    // JP: 各マテリアルスロットのジオメトリフラグ設定。
                    for (uint32_t matIdx = 0; matIdx < numMaterials; ++matIdx) {
                        OptixGeometryFlags flags = static_cast<OptixGeometryFlags>(
                            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT | (matIdx == 0 ? OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL : 0));
                        geomInst.setGeometryFlags(matIdx, flags);
                        EXPECT_EQ(geomInst.getGeometryFlags(matIdx), flags);
                    }
                }
            }

            geomInst.destroy();
        }

        scene.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(CurveGeometryTest, CurveSpecificFunctionality) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);
        optixu::Pipeline pipeline = context.createPipeline();

        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = 8;
        pipelineOptions.attributeCountInDwords = 4;
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineOptions.supportedPrimitiveTypeFlags = 
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE |
            OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        // JP: カーブ用ヒットプログラムグループの作成。
        {
            optixu::HitProgramGroup quadraticCurveHitGroup = pipeline.createHitProgramGroupForCurveIS(
                OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
                OPTIX_CURVE_ENDCAP_DEFAULT,
                moduleOptiX, RT_CH_NAME_STR("ch0"),
                emptyModule, nullptr,
                optixu::ASTradeoff::Default,
                optixu::AllowUpdate::No,
                optixu::AllowCompaction::No,
                optixu::AllowRandomVertexAccess::No);

            EXPECT_NE(quadraticCurveHitGroup, optixu::HitProgramGroup());

            optixu::HitProgramGroup cubicCurveHitGroup = pipeline.createHitProgramGroupForCurveIS(
                OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
                OPTIX_CURVE_ENDCAP_ON,
                moduleOptiX, RT_CH_NAME_STR("ch1"),
                emptyModule, nullptr,
                optixu::ASTradeoff::PreferFastTrace,
                optixu::AllowUpdate::Yes,
                optixu::AllowCompaction::Yes,
                optixu::AllowRandomVertexAccess::Yes);

            EXPECT_NE(cubicCurveHitGroup, optixu::HitProgramGroup());

            cubicCurveHitGroup.destroy();
            quadraticCurveHitGroup.destroy();
        }

        // JP: 球プリミティブ用ヒットプログラムグループの作成。
        {
            optixu::HitProgramGroup sphereHitGroup = pipeline.createHitProgramGroupForSphereIS(
                moduleOptiX, RT_CH_NAME_STR("ch2"),
                emptyModule, nullptr,
                optixu::ASTradeoff::PreferFastBuild,
                optixu::AllowUpdate::Yes,
                optixu::AllowCompaction::No,
                optixu::AllowRandomVertexAccess::Yes);

            EXPECT_NE(sphereHitGroup, optixu::HitProgramGroup());

            sphereHitGroup.destroy();
        }

        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(IntegrationTest, CompleteWorkflowTest) {
    try {
        // JP: 完全なワークフローのテスト - Context からシーン構築まで。
        optixu::Context context = optixu::Context::create(cuContext, 4, optixu::EnableValidation::Yes);

        // JP: パイプラインの設定。
        optixu::Pipeline pipeline = context.createPipeline();
        optixu::PipelineOptions pipelineOptions;
        pipelineOptions.payloadCountInDwords = shared::Pipeline0Payload0Signature::numDwords;
        pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
        pipelineOptions.launchParamsVariableName = "plp";
        pipelineOptions.sizeOfLaunchParams = sizeof(shared::PipelineLaunchParameters0);
        pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
        pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
        pipelineOptions.supportedPrimitiveTypeFlags = 
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
        pipeline.setPipelineOptions(pipelineOptions);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        // JP: プログラムの作成。
        optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
        optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
        
        optixu::Module emptyModule;
        optixu::HitProgramGroup hitGroupTriangle = pipeline.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch0"), emptyModule, nullptr);
        optixu::HitProgramGroup hitGroupCustom = pipeline.createHitProgramGroupForCustomIS(
            moduleOptiX, RT_CH_NAME_STR("ch1"), emptyModule, nullptr,
            moduleOptiX, RT_IS_NAME_STR("is"));

        // JP: マテリアルの作成。
        optixu::Material materialTriangle = context.createMaterial();
        optixu::Material materialCustom = context.createMaterial();
        
        materialTriangle.setHitGroup(0, hitGroupTriangle);
        materialCustom.setHitGroup(0, hitGroupCustom);

        // JP: シーンとジオメトリの構築。
        optixu::Scene scene = context.createScene();
        
        // JP: 三角形ジオメトリ。
        optixu::GeometryInstance triangleGeom = scene.createGeometryInstance();
        triangleGeom.setVertexFormat(OPTIX_VERTEX_FORMAT_FLOAT3);
        triangleGeom.setMaterial(0, 0, materialTriangle);
        
        // JP: カスタムプリミティブジオメトリ。
        optixu::GeometryInstance customGeom = scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);
        customGeom.setMaterial(0, 0, materialCustom);

        // JP: GASの構築。
        optixu::GeometryAccelerationStructure gasTriangle = scene.createGeometryAccelerationStructure();
        gasTriangle.setConfiguration(
            optixu::ASTradeoff::Default,
            optixu::AllowUpdate::Yes,
            optixu::AllowCompaction::Yes,
            optixu::AllowRandomVertexAccess::No);
        gasTriangle.addChild(triangleGeom);

        optixu::GeometryAccelerationStructure gasCustom = scene.createGeometryAccelerationStructure(
            optixu::GeometryType::CustomPrimitives);
        gasCustom.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::No,
            optixu::AllowRandomVertexAccess::No);
        gasCustom.addChild(customGeom);

        // JP: インスタンスの作成。
        optixu::Instance instanceTriangle = scene.createInstance();
        instanceTriangle.setChild(gasTriangle);
        instanceTriangle.setID(1);
        instanceTriangle.setVisibilityMask(0xFF);

        optixu::Instance instanceCustom = scene.createInstance();
        instanceCustom.setChild(gasCustom);
        instanceCustom.setID(2);
        instanceCustom.setVisibilityMask(0xFF);

        // JP: IASの構築。
        optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
        ias.setConfiguration(
            optixu::ASTradeoff::Default,
            optixu::AllowUpdate::Yes,
            optixu::AllowCompaction::No,
            optixu::AllowRandomInstanceAccess::Yes);
        ias.addChild(instanceTriangle);
        ias.addChild(instanceCustom);

        // JP: パイプラインにプログラムを設定。
        pipeline.setRayGenerationProgram(rayGenProgram);
        pipeline.setMissProgram(0, missProgram);
        pipeline.setScene(scene);

        // JP: SBTレイアウトの生成。
        size_t pipelineSbtSize, sceneSbtSize;
        pipeline.generateShaderBindingTableLayout(&pipelineSbtSize);
        scene.generateShaderBindingTableLayout(&sceneSbtSize);

        EXPECT_GT(pipelineSbtSize, 0);
        EXPECT_GT(sceneSbtSize, 0);
        EXPECT_TRUE(scene.shaderBindingTableLayoutIsReady());

        // JP: 子要素の数をチェック。
        EXPECT_EQ(gasTriangle.getChildCount(), 1);
        EXPECT_EQ(gasCustom.getChildCount(), 1);
        EXPECT_EQ(ias.getChildCount(), 2);

        // JP: 検索テスト。
        EXPECT_EQ(gasTriangle.findChildIndex(triangleGeom), 0);
        EXPECT_EQ(gasCustom.findChildIndex(customGeom), 0);
        EXPECT_EQ(ias.findChildIndex(instanceTriangle), 0);
        EXPECT_EQ(ias.findChildIndex(instanceCustom), 1);

        // JP: パイプラインのリンク。
        pipeline.link(2);

        // JP: クリーンアップ（逆順）。
        ias.destroy();
        instanceCustom.destroy();
        instanceTriangle.destroy();
        gasCustom.destroy();
        gasTriangle.destroy();
        customGeom.destroy();
        triangleGeom.destroy();
        scene.destroy();
        materialCustom.destroy();
        materialTriangle.destroy();
        hitGroupCustom.destroy();
        hitGroupTriangle.destroy();
        missProgram.destroy();
        rayGenProgram.destroy();
        moduleOptiX.destroy();
        pipeline.destroy();
        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



TEST(RobustnessTest, StressTestAndCornerCases) {
    try {
        optixu::Context context = optixu::Context::create(cuContext);

        // JP: 大量のオブジェクト作成と破棄のテスト。
        {
            const int numObjects = 100;
            std::vector<optixu::Material> materials;
            std::vector<optixu::Pipeline> pipelines;

            // JP: 大量作成。
            for (int i = 0; i < numObjects; ++i) {
                materials.push_back(context.createMaterial());
                pipelines.push_back(context.createPipeline());
            }

            // JP: 全て有効であることを確認。
            for (int i = 0; i < numObjects; ++i) {
                EXPECT_NE(materials[i], optixu::Material());
                EXPECT_NE(pipelines[i], optixu::Pipeline());
                EXPECT_EQ(materials[i].getContext(), context);
                EXPECT_EQ(pipelines[i].getContext(), context);
            }

            // JP: 大量破棄。
            for (int i = 0; i < numObjects; ++i) {
                materials[i].destroy();
                pipelines[i].destroy();
            }
        }

        // JP: 名前の長さの限界テスト。
        {
            optixu::Material material = context.createMaterial();
            
            // JP: 非常に長い名前の設定。
            std::string longName(1000, 'A');
            material.setName(longName.c_str());
            EXPECT_STREQ(material.getName(), longName.c_str());

            // JP: 特殊文字を含む名前。
            std::string specialName = "テスト_Material_123!@#$%^&*()";
            material.setName(specialName.c_str());
            EXPECT_STREQ(material.getName(), specialName.c_str());

            material.destroy();
        }

        // JP: BufferViewの境界値テスト。
        {
            // JP: 最大値に近い値でのテスト。
            CUdeviceptr maxPtr = ~static_cast<CUdeviceptr>(0) - 1000;
            size_t maxElements = ~static_cast<size_t>(0) / 1000;
            uint32_t maxStride = ~static_cast<uint32_t>(0);

            optixu::BufferView largeBuffer(maxPtr, maxElements, maxStride);
            EXPECT_EQ(largeBuffer.getCUdeviceptr(), maxPtr);
            EXPECT_EQ(largeBuffer.numElements(), maxElements);
            EXPECT_EQ(largeBuffer.stride(), maxStride);
            EXPECT_TRUE(largeBuffer.isValid());

            // JP: サイズオーバーフローのテスト（結果は実装依存）。
            size_t totalSize = largeBuffer.sizeInBytes();
            // オーバーフローしても例外は投げられない（実装による）
        }

        // JP: 空オブジェクトの操作テスト。
        {
            optixu::Material emptyMaterial;
            optixu::Scene emptyScene;
            optixu::GeometryInstance emptyGeomInst;

            // JP: 空オブジェクト同士の比較。
            EXPECT_EQ(emptyMaterial, optixu::Material());
            EXPECT_EQ(emptyScene, optixu::Scene());
            EXPECT_EQ(emptyGeomInst, optixu::GeometryInstance());

            // JP: 空オブジェクトの破棄（何も起こらない）。
            emptyMaterial.destroy();
            emptyScene.destroy();
            emptyGeomInst.destroy();
        }

        context.destroy();
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}



int32_t main(int32_t argc, const char* argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char**>(argv));

    CUDADRV_CHECK(cuInit(0));
#if CUDA_VERSION < 13000
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
#else
    CUctxCreateParams cuCtxCreateParams = {};
    CUDADRV_CHECK(cuCtxCreate(&cuContext, &cuCtxCreateParams, 0, 0));
#endif
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    return RUN_ALL_TESTS();
}
