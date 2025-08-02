/*

JP: このサンプルはテンポラルデノイザーの使用方法を示します。
    OptiXはノイズを低減するにあたってフレーム間の画像の安定性を考慮に入れたテンポラルデノイザーを提供しています。
    テンポラルデノイザーはアルベドや法線に加えて、前フレームのデノイズ済みビューティー、
    ピクセルごとの対応を示すフローチャンネルを補助画像として受け取ります。

    --upscale: アップスケールも実行する。

EN: This sample shows how to use the temporal denoiser.
    OptiX provides temporal denoiser taking the image stability between frames into account when denoising.
    The temporal denoiser takes the denoised beauty of the previous frame and a flow channel indicating
    per-pixel correspondance as auxiliary images in addition to albedo and normal.

    --upscale: Perform upscaling.

*/

#include "temporal_denoiser_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

#include "../common/gui_common.h"



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "temporal_denoiser";

    bool takeScreenShot = false;
    bool performUpscale = false;
    bool useLowResRendering;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot")
            takeScreenShot = true;
        else if (arg == "--upscale")
            performUpscale = true;
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

    useLowResRendering = performUpscale;



    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUstream stream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&stream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
    // EN: This sample uses two-level AS (single-level instancing).
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = std::max(
        Shared::SearchRayPayloadSignature::numDwords,
        Shared::VisibilityRayPayloadSignature::numDwords);
    pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(resourceDir / "ptxes/optix_path_tracing.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program pathTracingRayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("pathTracing"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));

    pipeline.link(2);

    pipeline.setRayGenerationProgram(pathTracingRayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setMissRayTypeCount(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

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

    optixu::Material ceilingMat = optixContext.createMaterial();
    ceilingMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    ceilingMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData ceilingMatData = {};
    ceilingMatData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    ceilingMat.setUserData(ceilingMatData);

    optixu::Material farSideWallMat = optixContext.createMaterial();
    farSideWallMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    farSideWallMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData farSideWallMatData = {};
    //farSideWallMatData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    cudau::Array farSideWallArray;
    {
        cudau::TextureSampler texSampler;
        texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
        texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        farSideWallArray = createTexture(
            useBlockCompressedTexture ?
            "../../data/TexturesCom_FabricPlain0077_1_seamless_S.DDS" :
            "../../data/TexturesCom_FabricPlain0077_1_seamless_S.jpg");
        farSideWallMatData.texture = texSampler.createTextureObject(farSideWallArray);
    }
    farSideWallMat.setUserData(farSideWallMatData);

    optixu::Material leftWallMat = optixContext.createMaterial();
    leftWallMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    leftWallMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData leftWallMatData = {};
    leftWallMatData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    leftWallMat.setUserData(leftWallMatData);

    optixu::Material rightWallMat = optixContext.createMaterial();
    rightWallMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    rightWallMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData rightWallMatData = {};
    rightWallMatData.albedo = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    rightWallMat.setUserData(rightWallMatData);

    optixu::Material floorMat = optixContext.createMaterial();
    floorMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    floorMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData floorMatData = {};
    cudau::Array floorArray;
    {
        cudau::TextureSampler texSampler;
        texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
        texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        floorArray = createTexture(
            useBlockCompressedTexture ?
            "../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.DDS" :
            "../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.jpg");
        floorMatData.texture = texSampler.createTextureObject(floorArray);
    }
    floorMat.setUserData(floorMatData);

    optixu::Material areaLightMat = optixContext.createMaterial();
    areaLightMat.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    areaLightMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    Shared::MaterialData areaLightMatData = {};
    areaLightMatData.albedo = make_float3(sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f));
    areaLightMatData.isEmitter = true;
    areaLightMat.setUserData(areaLightMatData);

    constexpr uint32_t NumBunnies = 100;
    std::vector<optixu::Material> bunnyMats(NumBunnies);
    for (int i = 0; i < NumBunnies; ++i) {
        bunnyMats[i] = optixContext.createMaterial();
        bunnyMats[i].setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
        bunnyMats[i].setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    }

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: このサンプルではデノイザーに焦点を当て、
    //     ほかをシンプルにするために1つのGASあたり1つのGeometryInstanceとする。
    // EN: Use one GeometryInstance per GAS for simplicty and
    //     to focus on denoiser in this sample.
    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        cudau::TypedBuffer<uint8_t> matIndexBuffer;
        optixu::GeometryInstance optixGeomInst;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            matIndexBuffer.finalize();
            triangleBuffer.finalize();
            vertexBuffer.finalize();
            optixGeomInst.destroy();
        }
    };

    Geometry room;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(1, 0) },
            // far side wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 2) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 0) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 2) },
            // ceiling
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(1, 0) },
            // left wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 1) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 1) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 0) },
            // right wall
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 0) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
            // far side wall
            { 4, 7, 6 }, { 4, 6, 5 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        uint8_t matIndices[] = {
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 4,
        };

        room.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        room.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
        room.matIndexBuffer.initialize(cuContext, cudau::BufferType::Device, matIndices, lengthof(matIndices));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = room.vertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = room.triangleBuffer.getROBuffer<enableBufferOobCheck>();

        room.optixGeomInst = scene.createGeometryInstance();
        room.optixGeomInst.setVertexBuffer(room.vertexBuffer);
        room.optixGeomInst.setTriangleBuffer(room.triangleBuffer);
        room.optixGeomInst.setMaterialCount(5, room.matIndexBuffer, optixu::IndexSize::k1Byte);
        room.optixGeomInst.setMaterial(0, 0, floorMat);
        room.optixGeomInst.setMaterial(0, 1, farSideWallMat);
        room.optixGeomInst.setMaterial(0, 2, ceilingMat);
        room.optixGeomInst.setMaterial(0, 3, leftWallMat);
        room.optixGeomInst.setMaterial(0, 4, rightWallMat);
        room.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setGeometryFlags(1, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setGeometryFlags(2, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setGeometryFlags(3, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setGeometryFlags(4, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setUserData(geomData);

        room.optixGas = scene.createGeometryAccelerationStructure();
        room.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        room.optixGas.setMaterialSetCount(1);
        room.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        room.optixGas.addChild(room.optixGeomInst);
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

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        areaLight.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLight.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLight.vertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = areaLight.triangleBuffer.getROBuffer<enableBufferOobCheck>();

        areaLight.optixGeomInst = scene.createGeometryInstance();
        areaLight.optixGeomInst.setVertexBuffer(areaLight.vertexBuffer);
        areaLight.optixGeomInst.setTriangleBuffer(areaLight.triangleBuffer);
        areaLight.optixGeomInst.setMaterialCount(1, optixu::BufferView());
        areaLight.optixGeomInst.setMaterial(0, 0, areaLightMat);
        areaLight.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLight.optixGeomInst.setUserData(geomData);

        areaLight.optixGas = scene.createGeometryAccelerationStructure();
        areaLight.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        areaLight.optixGas.setMaterialSetCount(1);
        areaLight.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        areaLight.optixGas.addChild(areaLight.optixGeomInst);
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

        bunny.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunny.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunny.vertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = bunny.triangleBuffer.getROBuffer<enableBufferOobCheck>();

        bunny.optixGeomInst = scene.createGeometryInstance();
        bunny.optixGeomInst.setVertexBuffer(bunny.vertexBuffer);
        bunny.optixGeomInst.setTriangleBuffer(bunny.triangleBuffer);
        bunny.optixGeomInst.setMaterialCount(1, optixu::BufferView());
        for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
            bunny.optixGeomInst.setMaterial(matSetIdx, 0, bunnyMats[matSetIdx]);
        bunny.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunny.optixGeomInst.setUserData(geomData);

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        bunny.optixGas.setMaterialSetCount(NumBunnies);
        for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
            bunny.optixGas.setRayTypeCount(matSetIdx, Shared::NumRayTypes);
        bunny.optixGas.addChild(bunny.optixGeomInst);
        bunny.optixGas.prepareForBuild(&asMemReqs);
        bunny.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    cudau::TypedBuffer<Shared::InstanceData> instDataBuffer;
    instDataBuffer.initialize(cuContext, cudau::BufferType::Device, 2 + NumBunnies);
    Shared::InstanceData* instData = instDataBuffer.map();
    uint32_t instID = 0;

    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(room.optixGas);
    roomInst.setID(instID);
    instData[instID] = Shared::InstanceData();
    ++instID;

    float areaLightInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance areaLightInst = scene.createInstance();
    areaLightInst.setChild(areaLight.optixGas);
    areaLightInst.setTransform(areaLightInstXfm);
    areaLightInst.setID(instID);
    instData[instID] = Shared::InstanceData(1.0f, Matrix3x3(), float3(0.0f, 0.9f, 0.0f));
    ++instID;

    struct MovingInstance {
        optixu::Instance inst;
        Shared::InstanceData instData;
        uint32_t ID;
        float scale_t;
        float scaleFreq;
        float scaleBase;
        float scaleAmp;
        float radius;
        float anglularPos_t;
        float angularPosFreq;
        float angularPosOffset;
        float y_t;
        float yBase;
        float yFreq;
        float yAmp;

        void setTransform() {
            float scale = scaleBase + scaleAmp * std::sin(2 * pi_v<float> * (scale_t / scaleFreq));

            float angle = 2 * pi_v<float> * (anglularPos_t / angularPosFreq) + angularPosOffset;
            float x = radius * std::cos(angle);
            float z = radius * std::sin(angle);

            float y = yBase + yAmp * std::sin(2 * pi_v<float> * (y_t / yFreq));

            float bunnyXfm[] = {
                scale, 0, 0, x,
                0, scale, 0, y,
                0, 0, scale, z,
            };
            inst.setTransform(bunnyXfm);

            instData.scale = scale;
            instData.rotation = Matrix3x3();
            instData.translation = float3(x, y, z);
        }

        void initializeState(
            float initScale_t, float _scaleFreq, float _scaleBase, float _scaleAmp,
            float _radius, float _angularPosFreq, float _angularPosOffset,
            float initY_t, float _yBase, float _yFreq, float _yAmp) {
            scale_t = initScale_t;
            scaleFreq = _scaleFreq;
            scaleBase = _scaleBase;
            scaleAmp = _scaleAmp;
            radius = _radius;
            anglularPos_t = 0.0f;
            angularPosFreq = _angularPosFreq;
            angularPosOffset = _angularPosOffset;
            y_t = initY_t;
            yBase = _yBase;
            yFreq = _yFreq;
            yAmp = _yAmp;

            scale_t = std::fmod(scale_t, scaleFreq);
            anglularPos_t = std::fmod(anglularPos_t, angularPosFreq);
            y_t = std::fmod(y_t, yFreq);
            setTransform();

            instData.prevScale = instData.scale;
            instData.prevRotation = instData.rotation;
            instData.prevTranslation = instData.translation;
        }

        void update(float dt) {
            instData.prevScale = instData.scale;
            instData.prevRotation = instData.rotation;
            instData.prevTranslation = instData.translation;

            scale_t = std::fmod(scale_t + dt, scaleFreq);
            anglularPos_t = std::fmod(anglularPos_t + dt, angularPosFreq);
            y_t = std::fmod(y_t + dt, yFreq);
            setTransform();
        }
    };

    std::vector<MovingInstance> bunnyInsts;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float angle = std::fmod(GoldenAngle * i, 2 * pi_v<float>);

        Shared::MaterialData matData;
        matData.albedo = sRGB_degamma(HSVtoRGB(
            angle / (2 * pi_v<float>),
            std::sqrt(r / 0.9f),
            1.0f));
        bunnyMats[i].setUserData(matData);

        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.003f + tt * 0.0006f;
        MovingInstance bunnyInst;
        bunnyInst.inst = scene.createInstance();
        bunnyInst.inst.setChild(bunny.optixGas, i);
        bunnyInst.inst.setID(instID);
        bunnyInst.ID = instID;
        bunnyInst.initializeState(
            0.0f, 1.0f, scale, 0.0f,
            r, 10.0f, angle,
            0.0f, -1 + (1 - tt), 1.0f, 0.0f);
        bunnyInsts.push_back(bunnyInst);
        ++instID;
    }

    instDataBuffer.unmap();



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastBuild);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    for (int i = 0; i < bunnyInsts.size(); ++i)
        ias.addChild(bunnyInsts[i].inst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    room.optixGas.rebuild(stream, room.gasMem, asBuildScratchMem);
    areaLight.optixGas.rebuild(stream, areaLight.gasMem, asBuildScratchMem);
    bunny.optixGas.rebuild(stream, bunny.gasMem, asBuildScratchMem);

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
        info.geom->optixGas.compact(
            stream,
            optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
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

    OptixTraversableHandle travHandle = ias.rebuild(stream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    cudau::Array beautyAccumBuffer;
    cudau::Array albedoAccumBuffer;
    cudau::Array normalAccumBuffer;
    cudau::TypedBuffer<float4> linearBeautyBuffer;
    cudau::TypedBuffer<float4> linearAlbedoBuffer;
    cudau::TypedBuffer<float4> linearNormalBuffer;
    cudau::TypedBuffer<float2> linearFlowBuffer;
    cudau::TypedBuffer<float4> linearDenoisedBeautyBuffers[2];

    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> rngBuffer;

    constexpr bool useTiledDenoising = false; // Change this to true to use tiled denoising.
    constexpr uint32_t tileWidth = useTiledDenoising ? 256 : 0;
    constexpr uint32_t tileHeight = useTiledDenoising ? 256 : 0;
    OptixDenoiserModelKind denoiserModel;
    optixu::Denoiser denoiser;
    std::vector<optixu::DenoisingTask> denoisingTasks;
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    cudau::Buffer internalGuideLayers[2];
    cudau::Buffer hdrNormalizer;

    denoiserModel = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV;
    if (performUpscale)
        denoiserModel = OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(
        &moduleCopyBuffers,
        (resourceDir / "ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyToLinearBuffers(
        moduleCopyBuffers, "copyToLinearBuffers", cudau::dim3(8, 8), 0);
    cudau::Kernel kernelVisualizeToOutputBuffer(
        moduleCopyBuffers, "visualizeToOutputBuffer", cudau::dim3(8, 8), 0);



    int32_t renderWidth = performUpscale ? 512 : 1024;
    int32_t renderHeight = performUpscale ? 512 : 1024;

    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = (float)renderWidth / renderHeight;
    plp.prevCamera = plp.camera;
    plp.instances = instDataBuffer.getDevicePointer();

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - Temporal Denoiser";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = (performUpscale ? 2 : 1) * renderWidth;
    initConfig.windowContentRenderHeight = (performUpscale ? 2 : 1) * renderHeight;
    initConfig.cameraPosition = make_float3(0, 0, 3.16f);
    initConfig.cameraOrientation = qRotateY(pi_v<float>);
    initConfig.cameraMovingSpeed = 0.01f;
    initConfig.cuContext = cuContext;

    GUIFramework framework;
    framework.initialize(initConfig);

    cudau::Array outputArray;

    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputBufferSurfaceHolder.initialize({ &outputArray });

    optixu::DenoiserInputBuffers inputBuffers = {};
    inputBuffers.beautyFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.albedoFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.normalFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
    inputBuffers.flowFormat = OPTIX_PIXEL_FORMAT_FLOAT2;

    const auto initializeRenderingResResources = [&]
    (int32_t width, int32_t height) {
        beautyAccumBuffer.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        albedoAccumBuffer.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);
        normalAccumBuffer.initialize2D(
            cuContext, cudau::ArrayElementType::Float32, 4,
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
            width, height, 1);

        linearBeautyBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            width * height);
        linearAlbedoBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            width * height);
        linearNormalBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            width * height);
        linearFlowBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            width * height);

        rngBuffer.initialize(cuContext, cudau::BufferType::Device, width, height);
        {
            std::mt19937_64 rng(591842031321323413);

            rngBuffer.map();
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    rngBuffer(x, y).setState(rng());
            rngBuffer.unmap();
        };

        plp.imageSize = int2(width, height);
        plp.rngBuffer = rngBuffer.getBlockBuffer2D();
        plp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
        plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
        plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
        plp.linearFlowBuffer = linearFlowBuffer.getDevicePointer();
        plp.camera.aspect = static_cast<float>(width) / height;

        inputBuffers.noisyBeauty = linearBeautyBuffer;
        inputBuffers.albedo = linearAlbedoBuffer;
        inputBuffers.normal = linearNormalBuffer;
        inputBuffers.flow = linearFlowBuffer;

        outputArray.initializeFromGLTexture2D(
            cuContext, framework.getOutputTexture().getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    };

    const auto initializeDenoiser = [&]
    (int32_t width, int32_t height, bool withUpscaleDenoiser) {
        int32_t denoiserScale = withUpscaleDenoiser ? 2 : 1;
        int32_t denoisedWidth = denoiserScale * width;
        int32_t denoisedHeight = denoiserScale * height;

        optixu::DenoiserSizes denoiserSizes;
        uint32_t numTasks;
        denoiser = optixContext.createDenoiser(
            denoiserModel,
            optixu::GuideAlbedo::Yes, optixu::GuideNormal::Yes, OPTIX_DENOISER_ALPHA_MODE_COPY);
        denoiser.prepare(
            width, height, tileWidth, tileHeight,
            &denoiserSizes, &numTasks);
        hpprintf("Denoiser State Buffer: %llu bytes\n", denoiserSizes.stateSize);
        hpprintf("Denoiser Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSize);
        hpprintf("Compute Normalizer Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSizeForComputeNormalizer);
        denoiserStateBuffer.initialize(cuContext, cudau::BufferType::Device, denoiserSizes.stateSize, 1);
        denoiserScratchBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            std::max(denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);

        denoisingTasks.resize(numTasks);
        denoiser.getTasks(denoisingTasks.data());

        denoiser.setupState(stream, denoiserStateBuffer, denoiserScratchBuffer);

        if (denoiserSizes.internalGuideLayerPixelSize > 0) {
            for (int bufIdx = 0; bufIdx < 2; ++bufIdx)
                internalGuideLayers[bufIdx].initialize(
                    cuContext, cudau::BufferType::Device,
                    denoisedWidth * denoisedHeight, denoiserSizes.internalGuideLayerPixelSize);
        }

        for (int bufIdx = 0; bufIdx < 2; ++bufIdx)
            linearDenoisedBeautyBuffers[bufIdx].initialize(
                cuContext, cudau::BufferType::Device,
                denoisedWidth * denoisedHeight);

        hdrNormalizer.initialize(cuContext, cudau::BufferType::Device, denoiserSizes.normalizerSize, 1);
    };

    const auto resizeRenderingResResources = [&]
    (int32_t width, int32_t height) {
        beautyAccumBuffer.resize(width, height);
        albedoAccumBuffer.resize(width, height);
        normalAccumBuffer.resize(width, height);

        linearBeautyBuffer.resize(width * height);
        linearAlbedoBuffer.resize(width * height);
        linearNormalBuffer.resize(width * height);
        linearFlowBuffer.resize(width * height);

        rngBuffer.resize(width, height);
        {
            std::mt19937_64 rng(591842031321323413);

            rngBuffer.map();
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    rngBuffer(x, y).setState(rng());
            rngBuffer.unmap();
        };

        plp.imageSize = int2(width, height);
        plp.rngBuffer = rngBuffer.getBlockBuffer2D();
        plp.beautyAccumBuffer = beautyAccumBuffer.getSurfaceObject(0);
        plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
        plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
        plp.linearFlowBuffer = linearFlowBuffer.getDevicePointer();
        plp.camera.aspect = static_cast<float>(width) / height;

        inputBuffers.noisyBeauty = linearBeautyBuffer;
        inputBuffers.albedo = linearAlbedoBuffer;
        inputBuffers.normal = linearNormalBuffer;
        inputBuffers.flow = linearFlowBuffer;

        outputArray.finalize();
        outputArray.initializeFromGLTexture2D(
            cuContext, framework.getOutputTexture().getHandle(),
            cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);
    };

    const auto resizeDenoiser = [&]
    (int32_t width, int32_t height, bool withUpscaleDenoiser, CUstream stream) {
        int32_t denoiserScale = withUpscaleDenoiser ? 2 : 1;
        int32_t denoisedWidth = denoiserScale * width;
        int32_t denoisedHeight = denoiserScale * height;

        optixu::DenoiserSizes denoiserSizes;
        uint32_t numTasks;
        denoiser.prepare(
            width, height, tileWidth, tileHeight,
            &denoiserSizes, &numTasks);
        hpprintf("Denoiser State Buffer: %llu bytes\n", denoiserSizes.stateSize);
        hpprintf("Denoiser Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSize);
        hpprintf("Compute Normalizer Scratch Buffer: %llu bytes\n", denoiserSizes.scratchSizeForComputeNormalizer);
        denoiserStateBuffer.resize(denoiserSizes.stateSize, 1);
        denoiserScratchBuffer.resize(
            std::max(denoiserSizes.scratchSize, denoiserSizes.scratchSizeForComputeNormalizer), 1);

        denoisingTasks.resize(numTasks);
        denoiser.getTasks(denoisingTasks.data());

        denoiser.setupState(stream, denoiserStateBuffer, denoiserScratchBuffer);

        if (denoiserSizes.internalGuideLayerPixelSize > 0) {
            for (int bufIdx = 0; bufIdx < 2; ++bufIdx) {
                if (internalGuideLayers[bufIdx].isInitialized())
                    internalGuideLayers[bufIdx].resize(
                        denoisedWidth * denoisedHeight, denoiserSizes.internalGuideLayerPixelSize);
                else
                    internalGuideLayers[bufIdx].initialize(
                        cuContext, cudau::BufferType::Device,
                        denoisedWidth * denoisedHeight, denoiserSizes.internalGuideLayerPixelSize);
            }
        }

        for (int bufIdx = 0; bufIdx < 2; ++bufIdx)
            linearDenoisedBeautyBuffers[bufIdx].resize(denoisedWidth * denoisedHeight);

        hdrNormalizer.resize(denoiserSizes.normalizerSize, 1);
    };

    const auto finalizeRenderingResResources = [&]
    () {
        outputArray.finalize();

        rngBuffer.finalize();

        linearFlowBuffer.finalize();
        linearNormalBuffer.finalize();
        linearAlbedoBuffer.finalize();
        linearBeautyBuffer.finalize();

        normalAccumBuffer.finalize();
        albedoAccumBuffer.finalize();
        beautyAccumBuffer.finalize();
    };

    const auto finalizeDenoiser = [&]
    () {
        for (int bufIdx = 1; bufIdx >= 0; --bufIdx) {
            linearDenoisedBeautyBuffers[bufIdx].finalize();
            internalGuideLayers[bufIdx].finalize();
        }

        hdrNormalizer.finalize();

        denoiserScratchBuffer.finalize();
        denoiserStateBuffer.finalize();
        denoiser.destroy();
    };

    initializeRenderingResResources(renderWidth, renderHeight);
    initializeDenoiser(renderWidth, renderHeight, performUpscale);

    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer render;
        cudau::Timer denoise;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            render.initialize(context);
            denoise.initialize(context);
        }
        void finalize() {
            denoise.finalize();
            render.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    GPUTimer gpuTimers[2];
    gpuTimers[0].initialize(cuContext);
    gpuTimers[1].initialize(cuContext);

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;
        GPUTimer &curGPUTimer = gpuTimers[frameIndex % 2];

        plp.prevCamera = plp.camera;

        // Camera Window
        bool cameraIsActuallyMoving = args.cameraIsActuallyMoving;
        {
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            if (ImGui::InputFloat3("Position", reinterpret_cast<float*>(&args.cameraPosition)))
                cameraIsActuallyMoving = true;
            static float rollPitchYaw[3];
            args.tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw)) {
                args.cameraOrientation = qFromEulerAngles(
                    rollPitchYaw[0] * pi_v<float> / 180,
                    rollPitchYaw[1] * pi_v<float> / 180,
                    rollPitchYaw[2] * pi_v<float> / 180);
                cameraIsActuallyMoving = true;
            }
            ImGui::Text("Pos. Speed (T/G): %g", args.cameraPositionalMovingSpeed);

            ImGui::End();
        }

        plp.camera.position = args.cameraPosition;
        plp.camera.orientation = args.tempCameraOrientation.toMatrix3x3();



        static bool enableJittering = false;
        static bool animate = true;
        static bool useTemporalDenosier = true;
        static int32_t log2NumSamplesPerFrame = 0;
        static Shared::BufferToDisplay bufferTypeToDisplay = Shared::BufferToDisplay::DenoisedBeauty;
        static float motionVectorScale = -1.0f;
        bool lastFrameWasAnimated = false;
        bool oldUseLowResRendering = useLowResRendering;
        bool denoiserModelChanged = false;
        {
            ImGui::SetNextWindowPos(ImVec2(712, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            if (ImGui::Button(animate ? "Stop" : "Play")) {
                if (animate)
                    lastFrameWasAnimated = true;
                animate = !animate;
            }

            ImGui::Text("Resolution:");
            ImGui::Text("Rendering in: %4d x %4d", renderWidth, renderHeight);
            const int32_t denoisedScale = performUpscale ? 2 : 1;
            ImGui::Text("Denoised output: %4d x %4d",
                        denoisedScale * renderWidth, denoisedScale * renderHeight);
            if (ImGui::Checkbox("Low Res Rendering", &useLowResRendering)) {
                if (!useLowResRendering)
                    performUpscale = false;
                denoiserModelChanged = true;
            }

            if (ImGui::Checkbox("Temporal Denoiser", &useTemporalDenosier))
                denoiserModelChanged = true;

            ImGui::BeginDisabled(!useLowResRendering);
            if (ImGui::Checkbox("with Upscaling", &performUpscale))
                denoiserModelChanged = true;
            ImGui::EndDisabled();

            if (denoiserModelChanged) {
                glFinish();
                CUDADRV_CHECK(cuStreamSynchronize(curStream));

                Assert(!performUpscale || useLowResRendering, "Invalid configuration.");

                if (performUpscale) {
                    denoiserModel = useTemporalDenosier ?
                        OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X :
                        OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
                }
                else {
                    denoiserModel = useTemporalDenosier ?
                        OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV :
                        OPTIX_DENOISER_MODEL_KIND_AOV;
                }

                renderWidth = args.windowContentRenderWidth;
                renderHeight = args.windowContentRenderHeight;
                if (useLowResRendering) {
                    renderWidth = (renderWidth + 1) / 2;
                    renderHeight = (renderHeight + 1) / 2;
                }

                resizeRenderingResResources(renderWidth, renderHeight);
                finalizeDenoiser();
                initializeDenoiser(renderWidth, renderHeight, performUpscale);
            }

            //ImGui::Checkbox("Jittering", &enableJittering);

            ImGui::InputLog2Int("spp per frame", &log2NumSamplesPerFrame, 6);

            ImGui::Text("Buffer to Display");
            ImGui::RadioButtonE("Noisy Beauty", &bufferTypeToDisplay, Shared::BufferToDisplay::NoisyBeauty);
            ImGui::RadioButtonE("Albedo", &bufferTypeToDisplay, Shared::BufferToDisplay::Albedo);
            ImGui::RadioButtonE("Normal", &bufferTypeToDisplay, Shared::BufferToDisplay::Normal);
            ImGui::RadioButtonE("Flow", &bufferTypeToDisplay, Shared::BufferToDisplay::Flow);
            ImGui::RadioButtonE("Denoised Beauty", &bufferTypeToDisplay, Shared::BufferToDisplay::DenoisedBeauty);

            ImGui::SliderFloat("Flow Scale", &motionVectorScale, -2.0f, 2.0f);

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 144), ImGuiCond_FirstUseEver);
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float cudaFrameTime = curGPUTimer.frame.report();
            float updateTime = curGPUTimer.update.report();
            float renderTime = curGPUTimer.render.report();
            float denoiseTime = curGPUTimer.denoise.report();
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime);
            ImGui::Text("  Update: %.3f [ms]", updateTime);
            ImGui::Text("  Render: %.3f [ms]", renderTime);
            ImGui::Text("  Denoise: %.3f [ms]", denoiseTime);

            ImGui::End();
        }



        curGPUTimer.frame.start(curStream);

        // JP: 各インスタンスのトランスフォームを更新する。
        // EN: Update the transform of each instance.
        if (animate || lastFrameWasAnimated) {
            for (int i = 0; i < bunnyInsts.size(); ++i) {
                MovingInstance &bunnyInst = bunnyInsts[i];
                bunnyInst.update(animate ? 1.0f / 60.0f : 0.0f);
                // TODO: まとめて送る。
                CUDADRV_CHECK(cuMemcpyHtoDAsync(
                    instDataBuffer.getCUdeviceptrAt(bunnyInst.ID),
                    &bunnyInst.instData, sizeof(bunnyInsts[i].instData), curStream));
            }
        }

        /*
        JP: IASのリビルドを行う。
            アップデートの代用としてのリビルドでは、インスタンスの追加・削除や
            ASビルド設定の変更を行っていないのでmarkDirty()やprepareForBuild()は必要無い。
        EN: Rebuild the IAS.
            Rebuild as the alternative for update doesn't involves
            add/remove of instances and changes of AS build settings
            so neither of markDirty() nor prepareForBuild() is required.
        */
        curGPUTimer.update.start(curStream);
        if (animate)
            plp.travHandle = ias.rebuild(curStream, instanceBuffer, iasMem, asBuildScratchMem);
        curGPUTimer.update.stop(curStream);

        bool firstAccumFrame =
            animate ||
            cameraIsActuallyMoving ||
            args.resized ||
            frameIndex == 0 ||
            (oldUseLowResRendering != useLowResRendering);
        bool isNewSequence = args.resized || frameIndex == 0 || denoiserModelChanged;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;
        plp.enableJittering = enableJittering;
        plp.resetFlowBuffer = isNewSequence;

        // Render
        curGPUTimer.render.start(curStream);
        uint32_t sppPerFrame = 1 << log2NumSamplesPerFrame;
        for (int smpIdx = 0; smpIdx < sppPerFrame; ++smpIdx) {
            plp.numAccumFrames = numAccumFrames;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
            pipeline.launch(curStream, plpOnDevice, renderWidth, renderHeight, 1);
            ++numAccumFrames;
            plp.resetFlowBuffer = false;
        }
        curGPUTimer.render.stop(curStream);

        curGPUTimer.denoise.start(curStream);

        // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
        // EN: Copy the results to the linear buffers (and normalize normals).
        kernelCopyToLinearBuffers.launchWithThreadDim(
            curStream, cudau::dim3(renderWidth, renderHeight),
            beautyAccumBuffer.getSurfaceObject(0),
            albedoAccumBuffer.getSurfaceObject(0),
            normalAccumBuffer.getSurfaceObject(0),
            linearBeautyBuffer,
            linearAlbedoBuffer,
            linearNormalBuffer,
            uint2(renderWidth, renderHeight));

        cudau::TypedBuffer<float4> &linearDenoisedBeautyBuffer = linearDenoisedBeautyBuffers[frameIndex % 2];
        inputBuffers.previousDenoisedBeauty = isNewSequence ?
            linearBeautyBuffer :
            linearDenoisedBeautyBuffers[(frameIndex + 1) % 2];
        optixu::BufferView internalGuideLayerForNextFrame;
        if (useTemporalDenosier) {
            inputBuffers.previousInternalGuideLayer = internalGuideLayers[(frameIndex + 1) % 2];
            internalGuideLayerForNextFrame = internalGuideLayers[frameIndex % 2];

            if (isNewSequence)
                CUDADRV_CHECK(cuMemsetD8Async(
                    inputBuffers.previousInternalGuideLayer.getCUdeviceptr(), 0,
                    inputBuffers.previousInternalGuideLayer.sizeInBytes(), curStream));
        }

        /*
        JP: パストレーシング結果のデノイズ。
            毎フレーム呼ぶ必要があるのはcomputeNormalizer()とinvoke()。
            サイズが足りていればcomputeNormalizer()のスクラッチバッファーとしてデノイザーのものが再利用できる。
        EN: Denoise the path tracing result.
            computeNormalizer() and invoke() should be called every frame.
            Reusing the scratch buffer for denoising for computeNormalizer() is possible if its size is enough.
        */
        denoiser.computeNormalizer(
            curStream,
            linearBeautyBuffer, OPTIX_PIXEL_FORMAT_FLOAT4,
            denoiserScratchBuffer, hdrNormalizer.getCUdeviceptr());
        for (int i = 0; i < denoisingTasks.size(); ++i)
            denoiser.invoke(
                curStream, denoisingTasks[i],
                inputBuffers, optixu::IsFirstFrame(isNewSequence),
                hdrNormalizer.getCUdeviceptr(), 0.0f,
                linearDenoisedBeautyBuffer,
                nullptr, // no AOV outputs
                internalGuideLayerForNextFrame);

        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        // JP: デノイズ結果や中間バッファーの可視化。
        // EN: Visualize the denosed result or intermediate buffers.
        void* bufferToDisplay = nullptr;
        switch (bufferTypeToDisplay) {
        case Shared::BufferToDisplay::NoisyBeauty:
            bufferToDisplay = linearBeautyBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Albedo:
            bufferToDisplay = linearAlbedoBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Normal:
            bufferToDisplay = linearNormalBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::Flow:
            bufferToDisplay = linearFlowBuffer.getDevicePointer();
            break;
        case Shared::BufferToDisplay::DenoisedBeauty:
            bufferToDisplay = linearDenoisedBeautyBuffer.getDevicePointer();
            break;
        default:
            Assert_ShouldNotBeCalled();
            break;
        }
        const int2 srcImageSize = int2(renderWidth, renderHeight)
            * ((performUpscale && bufferTypeToDisplay == Shared::BufferToDisplay::DenoisedBeauty) ? 2 : 1);
        kernelVisualizeToOutputBuffer.launchWithThreadDim(
            curStream, cudau::dim3(args.windowContentRenderWidth, args.windowContentRenderHeight),
            bufferToDisplay,
            bufferTypeToDisplay,
            0.5f, std::pow(10.0f, motionVectorScale),
            outputBufferSurfaceHolder.getNext(),
            int2(args.windowContentRenderWidth, args.windowContentRenderHeight), srcImageSize,
            performUpscale, useLowResRendering);

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);

        curGPUTimer.denoise.stop(curStream);

        curGPUTimer.frame.stop(curStream);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB =
            bufferTypeToDisplay == Shared::BufferToDisplay::NoisyBeauty ||
            bufferTypeToDisplay == Shared::BufferToDisplay::DenoisedBeauty;
        ret.finish = false;

        if (takeScreenShot && frameIndex + 1 == 60) {
            CUDADRV_CHECK(cuStreamSynchronize(curStream));
            const uint32_t numPixels = args.windowContentRenderWidth * args.windowContentRenderHeight;
            auto rawImage = new float4[numPixels];
            glGetTextureSubImage(
                args.outputTexture->getHandle(), 0,
                0, 0, 0, args.windowContentRenderWidth, args.windowContentRenderHeight, 1,
                GL_RGBA, GL_FLOAT, sizeof(float4) * numPixels, rawImage);
            saveImage("output.png", args.windowContentRenderWidth, args.windowContentRenderHeight, rawImage,
                      false, true);
            delete[] rawImage;
            ret.finish = true;
        }

        return ret;
    };

    const auto onResolutionChange = [&]
    (CUstream curStream, uint64_t frameIndex,
     int32_t windowContentWidth, int32_t windowContentHeight) {
         outputArray.finalize();
         outputArray.initializeFromGLTexture2D(
             cuContext, framework.getOutputTexture().getHandle(),
             cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

         renderWidth = windowContentWidth;
         renderHeight = windowContentHeight;
         if (useLowResRendering) {
             renderWidth = (renderWidth + 1) / 2;
             renderHeight = (renderHeight + 1) / 2;
         }

         resizeRenderingResResources(renderWidth, renderHeight);
         resizeDenoiser(renderWidth, renderHeight, performUpscale, curStream);

         // EN: update the pipeline parameters.
         plp.imageSize = int2(renderWidth, renderHeight);
         plp.camera.aspect = (float)renderWidth / renderHeight;
    };

    framework.run(onRenderLoop, onResolutionChange);

    gpuTimers[1].finalize();
    gpuTimers[0].finalize();

    finalizeDenoiser();
    finalizeRenderingResResources();

    outputBufferSurfaceHolder.finalize();

    framework.finalize();

    // END: Display the window.
    // ----------------------------------------------------------------



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    for (int i = bunnyInsts.size() - 1; i >= 0; --i)
        bunnyInsts[i].inst.destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    instDataBuffer.finalize();

    bunny.finalize();    
    areaLight.finalize();
    room.finalize();

    scene.destroy();

    for (int i = NumBunnies - 1; i >= 0; --i)
        bunnyMats[i].destroy();
    areaLightMat.destroy();
    CUDADRV_CHECK(cuTexObjectDestroy(floorMatData.texture));
    floorArray.finalize();
    floorMat.destroy();
    rightWallMat.destroy();
    leftWallMat.destroy();
    CUDADRV_CHECK(cuTexObjectDestroy(farSideWallMatData.texture));
    farSideWallArray.finalize();
    farSideWallMat.destroy();
    ceilingMat.destroy();



    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    pathTracingRayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(stream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
