#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include "../../optix_util.cpp"
#include "../../cuda_util.h"

#include "gtest/gtest.h"

static CUcontext cuContext;
static CUstream cuStream;

#define EXPECT_EXCEPTION(call, expect) \
    do { \
        bool caught; \
        try { \
            call; \
            caught = false; \
        } \
        catch (std::exception &ex) { \
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
        catch (std::exception &ex) { \
            caught = true; \
        } \
        EXPECT_EQ(caught, expect); \
    } \
    while (0)



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

        CUcontext retCuContext;
        EXPECT_EXCEPTION_RET(retCuContext, context.getCUcontext(), false);
        EXPECT_EQ(retCuContext, cuContext);

        const char* retName = nullptr;

        const char* nameA = "ABCDE";
        EXPECT_EXCEPTION(context.setName(nameA), false);
        EXPECT_EXCEPTION_RET(retName, context.getName(), false);
        EXPECT_STREQ(retName, nameA);

        const char* nameB = "";
        EXPECT_EXCEPTION(context.setName(nameB), false);
        EXPECT_EXCEPTION_RET(retName, context.getName(), false);
        EXPECT_STREQ(retName, nameB);

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



int32_t main(int32_t argc, const char* argv[]) {
    ::testing::InitGoogleTest(&argc, const_cast<char**>(argv));

    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    return RUN_ALL_TESTS();
}
