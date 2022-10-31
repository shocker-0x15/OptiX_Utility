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
                optixu::Context retContext = context.getContext();
                EXPECT_EQ(retContext, context);

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
                    OPTIX_DENOISER_MODEL_KIND_LDR, optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes);
                EXPECT_NE(denoiser, optixu::Denoiser());
                denoiser.destroy();

                denoiser = context.createDenoiser(
                    OPTIX_DENOISER_MODEL_KIND_HDR, optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes);
                EXPECT_NE(denoiser, optixu::Denoiser());
                denoiser.destroy();

                // JP: 無効なenumを使ってデノイザーを生成。
                EXPECT_EXCEPTION_RET(
                    denoiser,
                    context.createDenoiser(
                        static_cast<OptixDenoiserModelKind>(~0),
                        optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes),
                    true);
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
        optixu::Context context = optixu::Context::create(cuContext);

        optixu::Pipeline pipeline0 = context.createPipeline();
        pipeline0.setPipelineOptions(
            shared::Pipeline0Payload0Signature::numDwords,
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(shared::PipelineLaunchParameters0),
            optixu::UseMotionBlur::No,
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            OPTIX_EXCEPTION_FLAG_DEBUG,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::vector<char> optixIr = readBinaryFile(getExecutableDirectory() / "optixu_tests/ptxes/kernels_0.optixir");
        optixu::Module moduleOptiX = pipeline0.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        optixu::Module emptyModule;

        optixu::ProgramGroup hitProgramGroup0 = pipeline0.createHitProgramGroupForTriangleIS(
            moduleOptiX, RT_CH_NAME_STR("ch0"),
            emptyModule, nullptr);

        optixu::ProgramGroup hitProgramGroup1 = pipeline0.createHitProgramGroupForTriangleIS(
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
            optixu::ProgramGroup retHitProgramGroup;

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
            EXPECT_EQ(geomInst.getNumMotionSteps(), 1);
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
            uint32_t retMatIndexSize;
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, 0);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setNumMotionSteps(numMotionSteps);
            EXPECT_EQ(geomInst.getNumMotionSteps(), numMotionSteps);

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
            uint32_t matIndexSize;
            matIndexSize = sizeof(4);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = sizeof(2);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = sizeof(4);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint16_t));
            EXPECT_EXCEPTION(geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize));

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
            EXPECT_EQ(geomInst.getNumMotionSteps(), 1);
            EXPECT_EXCEPTION(geomInst.getVertexFormat());
            EXPECT_EQ(geomInst.getVertexBuffer(), optixu::BufferView());
            EXPECT_EQ(geomInst.getWidthBuffer(), optixu::BufferView());
            EXPECT_EXCEPTION(geomInst.getTriangleBuffer());
            EXPECT_EQ(geomInst.getSegmentIndexBuffer(), optixu::BufferView());
            EXPECT_EXCEPTION(geomInst.getCustomPrimitiveAABBBuffer());
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), 0);
            optixu::BufferView retMatIndexBuffer;
            uint32_t retMatIndexSize;
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, 0);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setNumMotionSteps(numMotionSteps);
            EXPECT_EQ(geomInst.getNumMotionSteps(), numMotionSteps);

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
            uint32_t matIndexSize;
            matIndexSize = sizeof(4);
            matIndexBuffer = optixu::BufferView(static_cast<CUdeviceptr>(12345678), numSegments, sizeof(uint32_t));
            EXPECT_EXCEPTION(geomInst.setNumMaterials(2, matIndexBuffer, matIndexSize));

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
            EXPECT_EQ(geomInst.getNumMotionSteps(), 1);
            EXPECT_EXCEPTION(geomInst.getVertexFormat());
            EXPECT_EXCEPTION(geomInst.getVertexBuffer());
            EXPECT_EXCEPTION(geomInst.getWidthBuffer());
            EXPECT_EXCEPTION(geomInst.getTriangleBuffer());
            EXPECT_EXCEPTION(geomInst.getSegmentIndexBuffer());
            EXPECT_EQ(geomInst.getCustomPrimitiveAABBBuffer(), optixu::BufferView());
            EXPECT_EQ(geomInst.getPrimitiveIndexOffset(), 0);
            optixu::BufferView retMatIndexBuffer;
            uint32_t retMatIndexSize;
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), 1);
            EXPECT_EQ(retMatIndexBuffer, optixu::BufferView());
            EXPECT_EQ(retMatIndexSize, 0);
            EXPECT_EQ(geomInst.getGeometryFlags(0), OPTIX_GEOMETRY_FLAG_NONE);
            EXPECT_EQ(geomInst.getMaterial(0, 0), optixu::Material());
            uint32_t userDataSize;
            uint32_t userDataAlignment;
            geomInst.getUserData(nullptr, &userDataSize, &userDataAlignment);
            EXPECT_EQ(userDataSize, 0);
            EXPECT_EQ(userDataAlignment, 1);



            uint32_t numMotionSteps = 5;
            geomInst.setNumMotionSteps(numMotionSteps);
            EXPECT_EQ(geomInst.getNumMotionSteps(), numMotionSteps);

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
            uint32_t matIndexSize;
            matIndexSize = sizeof(4);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = sizeof(2);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint32_t));
            geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize);
            EXPECT_EQ(geomInst.getNumMaterials(&retMatIndexBuffer, &retMatIndexSize), numMaterials);
            EXPECT_EQ(retMatIndexBuffer, matIndexBuffer);
            EXPECT_EQ(retMatIndexSize, matIndexSize);

            matIndexSize = sizeof(4);
            matIndexBuffer =
                optixu::BufferView(static_cast<CUdeviceptr>(12345678), numPrimitives, sizeof(uint16_t));
            EXPECT_EXCEPTION(geomInst.setNumMaterials(numMaterials, matIndexBuffer, matIndexSize));

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



int32_t main(int32_t argc, const char* argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char**>(argv));

    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    return RUN_ALL_TESTS();
}
