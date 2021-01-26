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

#define EXPECT_EXCEPTION(call, expect) \
    do { \
        bool caught; \
        try { \
            call; \
            caught = false; \
        } \
        catch (std::exception &) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, expect); \
    } \
    while (0)

#define EXPECT_EXCEPTION_RET(ret, call, expect) \
    do { \
        bool caught; \
        try { \
            ret = call; \
            caught = false; \
        } \
        catch (std::exception &) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, expect); \
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

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext), false);
    context.destroy();

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(0), false);
    context.destroy();

    //EXPECT_EXCEPTION_RET(context, optixu::Context::create(reinterpret_cast<CUcontext>(~static_cast<uintptr_t>(0))), false);
    //context.destroy();

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 0, false), false);
    context.destroy();
    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 1, false), false);
    context.destroy();
    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 2, false), false);
    context.destroy();
    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 3, false), false);
    context.destroy();
    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 4, false), false);
    context.destroy();
    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 4, true), false);
    context.destroy();

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext, 5, false), true);
}

TEST(ContextTest, ContextBasic) {
    optixu::Context context;

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext), false);
    {
        optixu::Context retContext;
        EXPECT_EXCEPTION_RET(retContext, context.getContext(), false);
        EXPECT_EQ(retContext, context);

        const char* retName = nullptr;

        const char* nameA = "ABCDE";
        EXPECT_EXCEPTION(context.setName(nameA), false);
        EXPECT_EXCEPTION_RET(retName, context.getName(), false);
        EXPECT_STREQ(retName, nameA);

        const char* nameB = "";
        EXPECT_EXCEPTION(context.setName(nameB), false);
        EXPECT_EXCEPTION_RET(retName, context.getName(), false);
        EXPECT_STREQ(retName, nameB);



        CUcontext retCuContext;
        EXPECT_EXCEPTION_RET(retCuContext, context.getCUcontext(), false);
        EXPECT_EQ(retCuContext, cuContext);



        struct CallbackUserData {
            uint32_t value;
        };
        const auto callback = [](uint32_t level, const char* tag, const char* message, void* cbdata) {
        };
        CallbackUserData cbUserData;
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 0), false);
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 1), false);
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 2), false);
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 3), false);
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 4), false);
        EXPECT_EXCEPTION(context.setLogCallback(callback, &cbUserData, 5), true);
        EXPECT_EXCEPTION(context.setLogCallback(nullptr, nullptr, 0), false);



        optixu::Pipeline pipeline;
        EXPECT_EXCEPTION_RET(pipeline, context.createPipeline(), false);
        EXPECT_NE(pipeline, optixu::Pipeline());
        pipeline.destroy();

        optixu::Material material;
        EXPECT_EXCEPTION_RET(material, context.createMaterial(), false);
        EXPECT_NE(material, optixu::Material());
        material.destroy();

        optixu::Scene scene;
        EXPECT_EXCEPTION_RET(scene, context.createScene(), false);
        EXPECT_NE(scene, optixu::Scene());
        scene.destroy();

        optixu::Denoiser denoiser;

        EXPECT_EXCEPTION_RET(denoiser, context.createDenoiser(OPTIX_DENOISER_INPUT_RGB), false);
        EXPECT_NE(denoiser, optixu::Denoiser());
        denoiser.destroy();

        EXPECT_EXCEPTION_RET(denoiser, context.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO), false);
        EXPECT_NE(denoiser, optixu::Denoiser());
        denoiser.destroy();

        EXPECT_EXCEPTION_RET(denoiser, context.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL), false);
        EXPECT_NE(denoiser, optixu::Denoiser());
        denoiser.destroy();

        EXPECT_EXCEPTION_RET(denoiser, context.createDenoiser(static_cast<OptixDenoiserInputKind>(~0)), true);
    }
    context.destroy();

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(0), false);
    {
        CUcontext retCuContext;
        EXPECT_EXCEPTION_RET(retCuContext, context.getCUcontext(), false);
        EXPECT_EQ(retCuContext, reinterpret_cast<CUcontext>(0));
    }
    context.destroy();
}



TEST(MaterialTest, MaterialBasic) {
    optixu::Context context;

    EXPECT_EXCEPTION_RET(context, optixu::Context::create(cuContext), false);

    optixu::Pipeline pipeline;
    EXPECT_EXCEPTION_RET(pipeline, context.createPipeline(), false);
    EXPECT_EXCEPTION(pipeline.setPipelineOptions(
        optixu::calcSumDwords<Pipeline0Payload0Signature>(),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(shared::PipelineLaunchParameters0),
        false,
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
        OPTIX_EXCEPTION_FLAG_DEBUG,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE), false);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "ptxes/kernels_0.ptx");
    optixu::Module moduleOptiX;
    EXPECT_EXCEPTION_RET(moduleOptiX,
                         pipeline.createModuleFromPTXString(
                             ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
                             OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
                             OPTIX_COMPILE_DEBUG_LEVEL_NONE), false);

    optixu::Module emptyModule;

    optixu::ProgramGroup hitProgramGroup0;
    EXPECT_EXCEPTION_RET(hitProgramGroup0,
                         pipeline.createHitProgramGroupForBuiltinIS(
                             OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                             moduleOptiX, RT_CH_NAME_STR("ch0"),
                             emptyModule, nullptr), false);

    optixu::ProgramGroup hitProgramGroup1;
    EXPECT_EXCEPTION_RET(hitProgramGroup1,
                         pipeline.createHitProgramGroupForBuiltinIS(
                             OPTIX_PRIMITIVE_TYPE_TRIANGLE,
                             moduleOptiX, RT_CH_NAME_STR("ch1"),
                             emptyModule, nullptr), false);

    hitProgramGroup1.destroy();
    hitProgramGroup0.destroy();
    moduleOptiX.destroy();
    pipeline.destroy();

    context.destroy();
}



int32_t main(int32_t argc, const char* argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char**>(argv));

    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    return RUN_ALL_TESTS();
}
