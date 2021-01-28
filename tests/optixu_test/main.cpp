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
        bool caught; \
        try { \
            call; \
            caught = false; \
        } \
        catch (std::exception &) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, true); \
    } \
    while (0)

#define EXPECT_EXCEPTION_RET(ret, call) \
    do { \
        bool caught; \
        try { \
            ret = call; \
            caught = false; \
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



static CUcontext cuContext;
static CUstream cuStream;



TEST(ContextTest, ContextCreation) {
    optixu::Context context;

    try {
        context = optixu::Context::create(cuContext);
        context.destroy();

        context = optixu::Context::create(0);
        context.destroy();

        //context = optixu::Context::create(reinterpret_cast<CUcontext>(~static_cast<uintptr_t>(0))), false);
        context.destroy();

        context = optixu::Context::create(cuContext, 0, false);
        context.destroy();
        context = optixu::Context::create(cuContext, 1, false);
        context.destroy();
        context = optixu::Context::create(cuContext, 2, false);
        context.destroy();
        context = optixu::Context::create(cuContext, 3, false);
        context.destroy();
        context = optixu::Context::create(cuContext, 4, false);
        context.destroy();
        context = optixu::Context::create(cuContext, 4, true);
        context.destroy();

        // JP: コールバックレベルが範囲外。
        EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 5, false));
    }
    catch (std::exception &ex) {
        printf("%s\n", ex.what());
        EXPECT_EQ(0, 1);
    }
}

TEST(ContextTest, ContextBasic) {
    optixu::Context context;

    try {
        // JP: 適当なCUDAコンテキストに対してoptixuのコンテキストを生成。
        context = optixu::Context::create(cuContext);
        {
            // ----------------------------------------------------------------
            // JP: 共通処理。

            optixu::Context retContext = context.getContext();
            EXPECT_EQ(retContext, context);

            const char* nameA = "ABCDE";
            context.setName(nameA);
            EXPECT_STREQ(context.getName(), nameA);

            const char* nameB = "";
            context.setName(nameB);
            EXPECT_STREQ(context.getName(), nameB);

            // END: 共通処理。
            // ----------------------------------------------------------------



            CUcontext retCuContext = context.getCUcontext();
            EXPECT_EQ(retCuContext, cuContext);



            // JP: コールバックの登録。
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



            // ----------------------------------------------------------------
            // JP: 各オブジェクトの生成。

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

            denoiser = context.createDenoiser(OPTIX_DENOISER_INPUT_RGB);
            EXPECT_NE(denoiser, optixu::Denoiser());
            denoiser.destroy();

            denoiser = context.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO);
            EXPECT_NE(denoiser, optixu::Denoiser());
            denoiser.destroy();

            denoiser = context.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL);
            EXPECT_NE(denoiser, optixu::Denoiser());
            denoiser.destroy();

            // JP: 向こうなenumを使ってデノイザーを生成。
            EXPECT_EXCEPTION_RET(denoiser, context.createDenoiser(static_cast<OptixDenoiserInputKind>(~0)), true);

            // END: 各オブジェクトの生成。
            // ----------------------------------------------------------------
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
    optixu::Context context;

    try {
        context = optixu::Context::create(cuContext);

        optixu::Pipeline pipeline0 = context.createPipeline();
        pipeline0.setPipelineOptions(
            optixu::calcSumDwords<Pipeline0Payload0Signature>(),
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(shared::PipelineLaunchParameters0),
            false,
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
            OPTIX_EXCEPTION_FLAG_DEBUG,
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/kernels_0.ptx");
        optixu::Module moduleOptiX = pipeline0.createModuleFromPTXString(
            ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
            OPTIX_COMPILE_DEBUG_LEVEL_NONE);

        optixu::Module emptyModule;

        optixu::ProgramGroup hitProgramGroup0 = pipeline0.createHitProgramGroupForBuiltinIS(
            OPTIX_PRIMITIVE_TYPE_TRIANGLE,
            moduleOptiX, RT_CH_NAME_STR("ch0"),
            emptyModule, nullptr);

        optixu::ProgramGroup hitProgramGroup1 = pipeline0.createHitProgramGroupForBuiltinIS(
            OPTIX_PRIMITIVE_TYPE_TRIANGLE,
            moduleOptiX, RT_CH_NAME_STR("ch1"),
            emptyModule, nullptr);



        optixu::Material mat0 = context.createMaterial();

        // ----------------------------------------------------------------
        // JP: 共通処理。

        EXPECT_EQ(mat0.getContext(), context);

        // JP: 普通の名前の設定と取得。
        const char* nameA = "ABCDE";
        mat0.setName(nameA);
        EXPECT_STREQ(mat0.getName(), nameA);

        // JP: 空白の名前の設定と取得。
        const char* nameB = "";
        mat0.setName(nameB);
        EXPECT_STREQ(mat0.getName(), nameB);

        // END: 共通処理。
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
        // JP: ヒットグループ関連。

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

        // END: ヒットグループ関連。
        // ----------------------------------------------------------------



        // ----------------------------------------------------------------
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

        // END: ユーザーデータ関連。
        // ----------------------------------------------------------------



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



int32_t main(int32_t argc, const char* argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char**>(argv));

    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    return RUN_ALL_TESTS();
}
