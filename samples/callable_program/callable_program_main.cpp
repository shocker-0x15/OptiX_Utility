/*

JP: このサンプルはCallable Programの使用方法を示します。
    Callable Programは関数ポインターのように、プログラムが実行時に呼ぶ関数の実体を動的に変更することができます。
    OptiXには二種類のCallable Program、Direct CallableとContinuation Callableがあります。
    両者ともにシェーダー中でレイトレースを行うことはできますが、再帰的なシェーダーの呼び出しは
    Continuation Callable Programのみがサポートしています。一方で、Continuation Callable Programの呼び出しは
    スケジューラーが管理するためオーバーヘッドを生じる可能性があります。

EN: This sample shows how to use callble programs.
    Callable program is similar to function pointer, allows a program to dynamically change the instance
    of function called at runtime.
    There are two types of callable programs in OptiX: direct callables and continuation callables.
    Both types support tracing rays in a shader, but only continuation callable supports
    recursive shader calls. However, calling a continuation callable is handled by the scheduler,
    so it may result in additional overhead.

*/

#include "callable_program_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

int32_t main(int32_t argc, const char* argv[]) try {
    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
    // EN: This sample uses two-level AS (single-level instancing).
    pipeline.setPipelineOptions(
        std::max(Shared::SearchRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::vector<char> pathTracingOptixIr =
        readBinaryFile(getExecutableDirectory() / "callable_program/ptxes/path_tracing.optixir");
    optixu::Module pathTracingModule = pipeline.createModuleFromOptixIR(
        pathTracingOptixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program pathTracingRayGenProgram =
        pipeline.createRayGenProgram(pathTracingModule, RT_RG_NAME_STR("pathTracing"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(pathTracingModule, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(pathTracingModule, RT_MS_NAME_STR("miss"));
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        pathTracingModule, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        pathTracingModule, RT_AH_NAME_STR("visibility"));

#define PROCESS_CALLABLE_PROGRAMS \
    PROCESS_CALLABLE_PROGRAM(setUpLambertBRDF),\
    PROCESS_CALLABLE_PROGRAM(LambertBRDF_sampleF),\
    PROCESS_CALLABLE_PROGRAM(LambertBRDF_evaluateF),\
    PROCESS_CALLABLE_PROGRAM(setUpMirrorBRDF),\
    PROCESS_CALLABLE_PROGRAM(MirrorBRDF_sampleF),\
    PROCESS_CALLABLE_PROGRAM(MirrorBRDF_evaluateF),\
    PROCESS_CALLABLE_PROGRAM(setUpGlassBSDF),\
    PROCESS_CALLABLE_PROGRAM(GlassBSDF_sampleF),\
    PROCESS_CALLABLE_PROGRAM(GlassBSDF_evaluateF),

    enum CallableProgram : uint32_t {
#define PROCESS_CALLABLE_PROGRAM(name) CallableProgram_ ## name
        PROCESS_CALLABLE_PROGRAMS
#undef PROCESS_CALLABLE_PROGRAM
        NumCallablePrograms,
    };

    const char* callableProgramEntryPoints[] = {
#define PROCESS_CALLABLE_PROGRAM(name) RT_DC_NAME_STR(#name)
        PROCESS_CALLABLE_PROGRAMS
#undef PROCESS_CALLABLE_PROGRAM
    };

#undef PROCESS_CALLABLE_PROGRAMS

    const std::vector<char> bsdfsOptixIr =
        readBinaryFile(getExecutableDirectory() / "callable_program/ptxes/bsdfs.optixir");
    optixu::Module bsdfsModule = pipeline.createModuleFromOptixIR(
        bsdfsOptixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    // JP: Callable Programを作成し、パイプラインにセットする。
    // EN: Create callable programs and set them to the pipeline.
    pipeline.setNumCallablePrograms(NumCallablePrograms);
    for (int i = 0; i < NumCallablePrograms; ++i) {
        optixu::CallableProgramGroup program = pipeline.createCallableProgramGroup(
            bsdfsModule, callableProgramEntryPoints[i],
            emptyModule, nullptr);
        pipeline.setCallableProgram(i, program);
    }

    pipeline.link(2);

    pipeline.setRayGenerationProgram(pathTracingRayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());

    // JP: パイプラインが必要とする各種スタックサイズを計算する。
    // EN: Compute the stack sizes the pipeline requires.
    {
        // JP: このサンプルはトラバーサル中にDirect Callable Programを呼ばない。
        // EN: This sample doesn't call a direct callable program during traversal.
        constexpr uint32_t dcStackSizeFromTrav = 0;

        // JP: Direct Callable Programの最大のスタックサイズを計算する。
        // EN: Calculate the maximum stack size of a direct callable program.
        uint32_t dcStackSizeFromState = 0;
        for (int i = 0; i < NumCallablePrograms; ++i) {
            dcStackSizeFromState = std::max(
                dcStackSizeFromState, pipeline.getCallableProgram(i).getDCStackSize());
        }

        // Possible Program Paths:
        // RG - CH - AH
        // RG - MS
        uint32_t ccStackSize =
            pathTracingRayGenProgram.getStackSize() +
            std::max(
                {
                    shadingHitProgramGroup.getCHStackSize() + visibilityHitProgramGroup.getAHStackSize(),
                    missProgram.getStackSize(),
                });
        pipeline.setStackSize(dcStackSizeFromTrav, dcStackSizeFromState, ccStackSize, 1);
    }

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    // JP: このサンプルではマテリアルの差異はCallable Programとジオメトリに紐づけられたインデックス
    //     によって処理されるため、optixu::Materialは一種類。
    // EN: This sample handles material differences with callable programs and indices associated to geometries.
    //     Therefore there is only one optixu::Material.
    optixu::Material commonMaterial = optixContext.createMaterial();
    commonMaterial.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    commonMaterial.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);



    const auto createTexture = [&cuContext](const std::filesystem::path &filepath) {
        cudau::Array array;

        if (filepath.extension() == ".DDS") {
            int32_t width, height, mipCount;
            size_t* sizes;
            dds::Format format;
            uint8_t** ddsData = dds::load(
                filepath.string().c_str(),
                &width, &height, &mipCount, &sizes, &format);

            const auto translate = [](dds::Format srcFormat) {
                cudau::ArrayElementType dstFormat;
                switch (srcFormat) {
                case dds::Format::BC1_UNorm:
                    return cudau::ArrayElementType::BC1_UNorm;
                case dds::Format::BC1_UNorm_sRGB:
                    return cudau::ArrayElementType::BC1_UNorm_sRGB;
                case dds::Format::BC2_UNorm:
                    return cudau::ArrayElementType::BC2_UNorm;
                case dds::Format::BC2_UNorm_sRGB:
                    return cudau::ArrayElementType::BC2_UNorm_sRGB;
                case dds::Format::BC3_UNorm:
                    return cudau::ArrayElementType::BC3_UNorm;
                case dds::Format::BC3_UNorm_sRGB:
                    return cudau::ArrayElementType::BC3_UNorm_sRGB;
                case dds::Format::BC4_UNorm:
                    return cudau::ArrayElementType::BC4_UNorm;
                case dds::Format::BC4_SNorm:
                    return cudau::ArrayElementType::BC4_SNorm;
                case dds::Format::BC5_UNorm:
                    return cudau::ArrayElementType::BC5_UNorm;
                case dds::Format::BC5_SNorm:
                    return cudau::ArrayElementType::BC5_SNorm;
                case dds::Format::BC6H_UF16:
                    return cudau::ArrayElementType::BC6H_UF16;
                case dds::Format::BC6H_SF16:
                    return cudau::ArrayElementType::BC6H_SF16;
                case dds::Format::BC7_UNorm:
                    return cudau::ArrayElementType::BC7_UNorm;
                case dds::Format::BC7_UNorm_sRGB:
                    return cudau::ArrayElementType::BC7_UNorm_sRGB;
                default:
                    Assert_ShouldNotBeCalled();
                    return static_cast<cudau::ArrayElementType>(-1);
                }
            };

            array.initialize2D(
                cuContext, translate(format), 1,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, mipCount);
            for (int i = 0; i < mipCount; ++i)
                array.write<uint8_t>(ddsData[i], sizes[i], i);

            dds::free(ddsData, sizes);
        }
        else {
            int32_t width, height, n;
            uint8_t* linearImageData = stbi_load(filepath.string().c_str(), &width, &height, &n, 4);
            array.initialize2D(
                cuContext, cudau::ArrayElementType::UInt8, 4,
                cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                width, height, 1);
            array.write<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
        }

        return array;
    };

    constexpr bool useBlockCompressedTexture = true;

    struct Material {
        cudau::Array array;
        CUtexObject texObj;

        void finalize() {
            CUDADRV_CHECK(cuTexObjectDestroy(texObj));
            array.finalize();
        }
    };

    enum MaterialID {
        MaterialID_Floor = 0,
        MaterialID_FarSideWall,
        MaterialID_LeftWall,
        MaterialID_RightWall,
        MaterialID_Ceiling,
        MaterialID_AreaLight,
        MaterialID_Bunny0,
        MaterialID_Bunny1,
        MaterialID_Bunny2,
        NumMaterialIDs
    };

    std::vector<Material> materialsOnHost(NumMaterialIDs);
    cudau::TypedBuffer<Shared::MaterialData> materialBuffer;
    materialBuffer.initialize(cuContext, cudau::BufferType::Device, NumMaterialIDs);
    Shared::MaterialData* materials = materialBuffer.map();

    cudau::TextureSampler texSampler;
    texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
    texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Linear);
    texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
    texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

    // floor
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matOnHost.array = createTexture(
            useBlockCompressedTexture ?
            "../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.DDS" :
            "../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.jpg");
        matOnHost.texObj = texSampler.createTextureObject(matOnHost.array);
        matData.asMatte.texture = matOnHost.texObj;
        matData.asMatte.reflectance = float3(0.0f);

        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_Floor] = matData;
        materialsOnHost[MaterialID_Floor] = std::move(matOnHost);
    }

    // far side wall
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matOnHost.array = createTexture(
            useBlockCompressedTexture ?
            "../../data/TexturesCom_FabricPlain0077_1_seamless_S.DDS" :
            "../../data/TexturesCom_FabricPlain0077_1_seamless_S.jpg");
        matOnHost.texObj = texSampler.createTextureObject(matOnHost.array);
        matData.asMatte.texture = matOnHost.texObj;
        matData.asMatte.reflectance = float3(0.0f);

        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_FarSideWall] = matData;
        materialsOnHost[MaterialID_FarSideWall] = std::move(matOnHost);
    }

    // left wall
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMatte.texture = 0;
        matData.asMatte.reflectance =
            make_float3(sRGB_degamma_s(0.75f), sRGB_degamma_s(0.25f), sRGB_degamma_s(0.25f));
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_LeftWall] = matData;
        materialsOnHost[MaterialID_LeftWall] = std::move(matOnHost);
    }

    // right wall
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMatte.texture = 0;
        matData.asMatte.reflectance =
            make_float3(sRGB_degamma_s(0.25f), sRGB_degamma_s(0.25f), sRGB_degamma_s(0.75f));
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_RightWall] = matData;
        materialsOnHost[MaterialID_RightWall] = std::move(matOnHost);
    }

    // ceiling
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMatte.texture = 0;
        matData.asMatte.reflectance =
            make_float3(sRGB_degamma_s(0.75f), sRGB_degamma_s(0.75f), sRGB_degamma_s(0.75f));
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_Ceiling] = matData;
        materialsOnHost[MaterialID_Ceiling] = std::move(matOnHost);
    }

    // area light
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMatte.texture = 0;
        matData.asMatte.reflectance =
            make_float3(sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f));
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = true;

        materials[MaterialID_AreaLight] = matData;
        materialsOnHost[MaterialID_AreaLight] = std::move(matOnHost);
    }

    // bunny 0
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMatte.texture = 0;
        matData.asMatte.reflectance =
            make_float3(sRGB_degamma_s(0.5f), sRGB_degamma_s(0.5f), sRGB_degamma_s(0.5f));
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpLambertBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_LambertBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_LambertBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_Bunny0] = matData;
        materialsOnHost[MaterialID_Bunny0] = std::move(matOnHost);
    }

    // bunny 1
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asMirror.texture = 0;
        matData.asMirror.f0Reflectance = make_float3(0.5f, 0.5f, 0.5f);
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpMirrorBRDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_MirrorBRDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_MirrorBRDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_Bunny1] = matData;
        materialsOnHost[MaterialID_Bunny1] = std::move(matOnHost);
    }

    // bunny 2
    {
        Material matOnHost;
        Shared::MaterialData matData = {};

        matData.asGlass.ior = 1.5f;
        matData.setUpBSDF = Shared::SetUpBSDF(CallableProgram_setUpGlassBSDF);
        matData.sampleF = Shared::BSDF_sampleF(CallableProgram_GlassBSDF_sampleF);
        matData.evaluateF = Shared::BSDF_evaluateF(CallableProgram_GlassBSDF_evaluateF);
        matData.isEmitter = false;

        materials[MaterialID_Bunny2] = matData;
        materialsOnHost[MaterialID_Bunny2] = std::move(matOnHost);
    }

    materialBuffer.unmap();

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    struct Geometry {
        struct MaterialGroup {
            std::shared_ptr<cudau::TypedBuffer<Shared::Vertex>> vertexBuffer;
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            optixu::GeometryInstance optixGeomInst;
        };

        std::vector<MaterialGroup> matGroups;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            for (auto it = matGroups.rbegin(); it != matGroups.rend(); ++it) {
                it->triangleBuffer.finalize();
                it->vertexBuffer->finalize();
                it->optixGeomInst.destroy();
            }
            gasMem.finalize();
            optixGas.destroy();
        }
    };

    Geometry room;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3( 0,  1,  0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f,  1.0f), make_float3( 0,  1,  0), make_float2(0, 1) },
            { make_float3( 1.0f, -1.0f,  1.0f), make_float3( 0,  1,  0), make_float2(1, 1) },
            { make_float3( 1.0f, -1.0f, -1.0f), make_float3( 0,  1,  0), make_float2(1, 0) },
            // far side wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3( 0,  0,  1), make_float2(0, 2) },
            { make_float3(-1.0f,  1.0f, -1.0f), make_float3( 0,  0,  1), make_float2(0, 0) },
            { make_float3( 1.0f,  1.0f, -1.0f), make_float3( 0,  0,  1), make_float2(2, 0) },
            { make_float3( 1.0f, -1.0f, -1.0f), make_float3( 0,  0,  1), make_float2(2, 2) },
            // ceiling
            { make_float3(-1.0f,  1.0f, -1.0f), make_float3( 0, -1,  0), make_float2(0, 0) },
            { make_float3(-1.0f,  1.0f,  1.0f), make_float3( 0, -1,  0), make_float2(0, 1) },
            { make_float3( 1.0f,  1.0f,  1.0f), make_float3( 0, -1,  0), make_float2(1, 1) },
            { make_float3( 1.0f,  1.0f, -1.0f), make_float3( 0, -1,  0), make_float2(1, 0) },
            // left wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3( 1,  0,  0), make_float2(0, 0) },
            { make_float3(-1.0f,  1.0f, -1.0f), make_float3( 1,  0,  0), make_float2(0, 1) },
            { make_float3(-1.0f,  1.0f,  1.0f), make_float3( 1,  0,  0), make_float2(1, 1) },
            { make_float3(-1.0f, -1.0f,  1.0f), make_float3( 1,  0,  0), make_float2(1, 0) },
            // right wall
            { make_float3( 1.0f, -1.0f, -1.0f), make_float3(-1,  0,  0), make_float2(0, 0) },
            { make_float3( 1.0f,  1.0f, -1.0f), make_float3(-1,  0,  0), make_float2(0, 1) },
            { make_float3( 1.0f,  1.0f,  1.0f), make_float3(-1,  0,  0), make_float2(1, 1) },
            { make_float3( 1.0f, -1.0f,  1.0f), make_float3(-1,  0,  0), make_float2(1, 0) },
        };

        auto vertexBuffer = std::make_shared<cudau::TypedBuffer<Shared::Vertex>>(
            cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        // floor
        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 0, 1, 2 }, { 0, 2, 3 }
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_Floor;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            room.matGroups.push_back(std::move(group));
        }

        // far side wall
        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 4, 7, 6 }, { 4, 6, 5 }
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_FarSideWall;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            room.matGroups.push_back(std::move(group));
        }

        // left wall
        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 15, 12, 13 }, { 15, 13, 14 }
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_LeftWall;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            room.matGroups.push_back(std::move(group));
        }

        // right wall
        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 16, 19, 18 }, { 16, 18, 17 }
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_RightWall;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            room.matGroups.push_back(std::move(group));
        }

        // ceiling
        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 8, 11, 10 }, { 8, 10, 9 }
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_Ceiling;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            room.matGroups.push_back(std::move(group));
        }

        room.optixGas = scene.createGeometryAccelerationStructure();
        room.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        room.optixGas.setNumMaterialSets(1);
        room.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        for (uint32_t i = 0; i < room.matGroups.size(); ++i)
            room.optixGas.addChild(room.matGroups[i].optixGeomInst);
        room.optixGas.prepareForBuild(&asMemReqs);
        room.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry areaLight;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        auto vertexBuffer = std::make_shared<cudau::TypedBuffer<Shared::Vertex>>(
            cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        {
            Geometry::MaterialGroup group = {};

            Shared::Triangle triangles[] = {
                { 0, 1, 2 }, { 0, 2, 3 },
            };

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_AreaLight;

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            areaLight.matGroups.push_back(std::move(group));
        }

        areaLight.optixGas = scene.createGeometryAccelerationStructure();
        areaLight.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        areaLight.optixGas.setNumMaterialSets(1);
        areaLight.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        for (uint32_t i = 0; i < areaLight.matGroups.size(); ++i)
            areaLight.optixGas.addChild(areaLight.matGroups[i].optixGeomInst);
        areaLight.optixGas.prepareForBuild(&asMemReqs);
        areaLight.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry bunny;
    {
        std::vector<Shared::Vertex> vertices;
        std::vector<Shared::Triangle> triangles;
        {
            std::vector<obj::Vertex> objVertices;
            std::vector<obj::Triangle> objTriangles;
            obj::load("../../data/stanford_bunny_309_faces.obj", &objVertices, &objTriangles);

            vertices.resize(objVertices.size());
            for (int vIdx = 0; vIdx < objVertices.size(); ++vIdx) {
                const obj::Vertex &objVertex = objVertices[vIdx];
                vertices[vIdx] = Shared::Vertex{ objVertex.position, objVertex.normal, objVertex.texCoord };
            }
            static_assert(sizeof(Shared::Triangle) == sizeof(obj::Triangle),
                          "Assume triangle formats are the same.");
            triangles.resize(objTriangles.size());
            std::copy_n(reinterpret_cast<Shared::Triangle*>(objTriangles.data()),
                        triangles.size(), triangles.data());
        }

        auto vertexBuffer = std::make_shared<cudau::TypedBuffer<Shared::Vertex>>(
            cuContext, cudau::BufferType::Device, vertices);

        {
            Geometry::MaterialGroup group = {};

            group.vertexBuffer = vertexBuffer;
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

            Shared::GeometryInstanceData geomInstData = {};
            geomInstData.vertexBuffer = group.vertexBuffer->getROBuffer<enableBufferOobCheck>();
            geomInstData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
            geomInstData.matIndex = MaterialID_Bunny0; // base index

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(*vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, commonMaterial);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomInstData);

            bunny.matGroups.push_back(std::move(group));
        }

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        bunny.optixGas.setNumMaterialSets(1);
        bunny.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        for (uint32_t i = 0; i < bunny.matGroups.size(); ++i)
            bunny.optixGas.addChild(bunny.matGroups[i].optixGeomInst);
        bunny.optixGas.prepareForBuild(&asMemReqs);
        bunny.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(room.optixGas);

    float areaLightInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance areaLightInst = scene.createInstance();
    areaLightInst.setChild(areaLight.optixGas);
    areaLightInst.setTransform(areaLightInstXfm);

    float bunny0InstXfm[] = {
        0.006f, 0, 0, -0.6f,
        0, 0.006f, 0, -0.9f,
        0, 0, 0.006f, 0
    };
    optixu::Instance bunny0Inst = scene.createInstance();
    bunny0Inst.setChild(bunny.optixGas);
    bunny0Inst.setTransform(bunny0InstXfm);
    bunny0Inst.setID(0); // Use this field as an offset for the material buffer.

    float bunny1InstXfm[] = {
        0.006f, 0, 0, 0,
        0, 0.006f, 0, -0.9f,
        0, 0, 0.006f, 0
    };
    optixu::Instance bunny1Inst = scene.createInstance();
    bunny1Inst.setChild(bunny.optixGas);
    bunny1Inst.setTransform(bunny1InstXfm);
    bunny1Inst.setID(1); // Use this field as an offset for the material buffer.

    float bunny2InstXfm[] = {
        0.006f, 0, 0, 0.6f,
        0, 0.006f, 0, -0.9f,
        0, 0, 0.006f, 0
    };
    optixu::Instance bunny2Inst = scene.createInstance();
    bunny2Inst.setChild(bunny.optixGas);
    bunny2Inst.setTransform(bunny2InstXfm);
    bunny2Inst.setID(2); // Use this field as an offset for the material buffer.



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    ias.addChild(bunny0Inst);
    ias.addChild(bunny1Inst);
    ias.addChild(bunny2Inst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    room.optixGas.rebuild(cuStream, room.gasMem, asBuildScratchMem);
    areaLight.optixGas.rebuild(cuStream, areaLight.gasMem, asBuildScratchMem);
    bunny.optixGas.rebuild(cuStream, bunny.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        Geometry* geom;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &room, 0, 0 },
        { &areaLight, 0, 0 },
        { &bunny, 0, 0 }
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.geom->optixGas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.geom->optixGas.compact(cuStream, optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset,
                                                      info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].geom->optixGas.removeUncompacted();
        gasList[i].geom->gasMem.finalize();
    }



    // JP: IASビルド時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when building an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr uint32_t renderTargetSizeX = 1024;
    constexpr uint32_t renderTargetSizeY = 1024;
    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> rngBuffer;
    rngBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);
    {
        std::mt19937_64 rng(591842031321323413);

        rngBuffer.map();
        for (int y = 0; y < renderTargetSizeY; ++y)
            for (int x = 0; x < renderTargetSizeX; ++x)
                rngBuffer(x, y).setState(rng());
        rngBuffer.unmap();
    };

    cudau::Array accumBuffer;
    accumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        renderTargetSizeX, renderTargetSizeY, 1);



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.materialBuffer = materialBuffer.getROBuffer<enableBufferOobCheck>();
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.accumBuffer = accumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.16f);
    plp.camera.orientation = rotateY3x3(pi_v<float>);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));

    constexpr uint32_t numSamples = DEBUG_SELECT(8, 256);
    for (int frameIndex = 0; frameIndex < numSamples; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    saveImage("output.png", accumBuffer, true, true);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();
    rngBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    bunny2Inst.destroy();
    bunny1Inst.destroy();
    bunny0Inst.destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    bunny.finalize();    
    areaLight.finalize();
    room.finalize();

    scene.destroy();

    materialBuffer.finalize();
    for (int i = materialsOnHost.size() - 1; i >= 0; --i)
        materialsOnHost[i].finalize();



    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    pathTracingRayGenProgram.destroy();

    pathTracingModule.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
