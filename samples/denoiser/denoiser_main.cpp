#include "denoiser_shared.h"

#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"
#include "../../ext/tiny_obj_loader.h"

static void loadObjFile(const std::filesystem::path &filepath,
                        std::vector<Shared::Vertex>* vertices, std::vector<Shared::Triangle>* triangles);

int32_t main(int32_t argc, const char* argv[]) try {
    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    int32_t cuDeviceCount;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
    // EN: This sample uses two-level AS (single-level instancing).
    pipeline.setPipelineOptions(8, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "denoiser/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup pathTracingRayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("pathTracing"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::ProgramGroup emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::ProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr,
        emptyModule, nullptr);
    optixu::ProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroup(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"),
        emptyModule, nullptr);

    pipeline.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(pathTracingRayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    pipeline.setShaderBindingTable(&shaderBindingTable);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

#define USE_BLOCK_COMPRESSED_TEXTURE

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
        texSampler.setFilterMode(cudau::TextureFilterMode::Linear,
                                 cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
            int32_t width, height, mipCount;
            size_t* sizes;
            dds::Format format;
            uint8_t** ddsData = dds::load("../../data/TexturesCom_FabricPlain0077_1_seamless_S.DDS",
                                          &width, &height, &mipCount, &sizes, &format);

            farSideWallArray.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                                          cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                          width, height, 1/*mipCount*/);
            for (int i = 0; i < farSideWallArray.getNumMipmapLevels(); ++i)
                farSideWallArray.transfer<uint8_t>(ddsData[i], sizes[i], i);

            dds::free(ddsData, mipCount, sizes);
#else
            int32_t width, height, n;
            uint8_t* linearImageData = stbi_load("../../data/TexturesCom_FabricPlain0077_1_seamless_S.jpg",
                                                 &width, &height, &n, 4);
            farSideWallArray.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                                          cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                          width, height, 1);
            farSideWallArray.transfer<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
#endif
        }
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
        texSampler.setFilterMode(cudau::TextureFilterMode::Linear,
                                 cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
            int32_t width, height, mipCount;
            size_t* sizes;
            dds::Format format;
            uint8_t** ddsData = dds::load("../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.DDS",
                                          &width, &height, &mipCount, &sizes, &format);

            floorArray.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                    width, height, 1/*mipCount*/);
            for (int i = 0; i < floorArray.getNumMipmapLevels(); ++i)
                floorArray.transfer<uint8_t>(ddsData[i], sizes[i], i);

            dds::free(ddsData, mipCount, sizes);
#else
            int32_t width, height, n;
            uint8_t* linearImageData = stbi_load("../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.jpg",
                                                 &width, &height, &n, 4);
            floorArray.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                    width, height, 1);
            floorArray.transfer<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
#endif
        }
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

    optixu::GeometryInstance roomGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> roomVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> roomTriangleBuffer;
    cudau::TypedBuffer<uint8_t> roomMatIndexBuffer;
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
            { 4, 5, 6 }, { 4, 6, 7 },
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

        roomVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        roomTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
        roomMatIndexBuffer.initialize(cuContext, cudau::BufferType::Device, matIndices, lengthof(matIndices));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = roomVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = roomTriangleBuffer.getDevicePointer();

        roomGeomInst.setVertexBuffer(&roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(&roomTriangleBuffer);
        roomGeomInst.setNumMaterials(5, &roomMatIndexBuffer, sizeof(uint8_t));
        roomGeomInst.setMaterial(0, 0, floorMat);
        roomGeomInst.setMaterial(0, 1, farSideWallMat);
        roomGeomInst.setMaterial(0, 2, ceilingMat);
        roomGeomInst.setMaterial(0, 3, leftWallMat);
        roomGeomInst.setMaterial(0, 4, rightWallMat);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setGeometryFlags(1, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setGeometryFlags(2, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setGeometryFlags(3, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setGeometryFlags(4, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setUserData(geomData);
    }

    optixu::GeometryInstance areaLightGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> areaLightVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> areaLightTriangleBuffer;
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

        areaLightVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLightTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLightVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = areaLightTriangleBuffer.getDevicePointer();

        areaLightGeomInst.setVertexBuffer(&areaLightVertexBuffer);
        areaLightGeomInst.setTriangleBuffer(&areaLightTriangleBuffer);
        areaLightGeomInst.setNumMaterials(1, nullptr);
        areaLightGeomInst.setMaterial(0, 0, areaLightMat);
        areaLightGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLightGeomInst.setUserData(geomData);
    }

    optixu::GeometryInstance bunnyGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> bunnyVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> bunnyTriangleBuffer;
    {
        std::vector<Shared::Vertex> vertices;
        std::vector<Shared::Triangle> triangles;
        loadObjFile("../../data/stanford_bunny_309_faces.obj", &vertices, &triangles);

        bunnyVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunnyTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunnyVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = bunnyTriangleBuffer.getDevicePointer();

        bunnyGeomInst.setVertexBuffer(&bunnyVertexBuffer);
        bunnyGeomInst.setTriangleBuffer(&bunnyTriangleBuffer);
        bunnyGeomInst.setNumMaterials(1, nullptr);
        for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
            bunnyGeomInst.setMaterial(matSetIdx, 0, bunnyMats[matSetIdx]);
        bunnyGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunnyGeomInst.setUserData(geomData);
    }



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure roomGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer roomGasMem;
    cudau::Buffer roomGasCompactedMem;
    roomGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    roomGas.setNumMaterialSets(1);
    roomGas.setNumRayTypes(0, Shared::NumRayTypes);
    roomGas.addChild(roomGeomInst);
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure areaLightGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer areaLightGasMem;
    cudau::Buffer areaLightGasCompactedMem;
    areaLightGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    areaLightGas.setNumMaterialSets(1);
    areaLightGas.setNumRayTypes(0, Shared::NumRayTypes);
    areaLightGas.addChild(areaLightGeomInst);
    areaLightGas.prepareForBuild(&asMemReqs);
    areaLightGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure bunnyGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer bunnyGasMem;
    cudau::Buffer bunnyGasCompactedMem;
    bunnyGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    bunnyGas.setNumMaterialSets(NumBunnies);
    for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
        bunnyGas.setNumRayTypes(matSetIdx, Shared::NumRayTypes);
    bunnyGas.addChild(bunnyGeomInst);
    bunnyGas.prepareForBuild(&asMemReqs);
    bunnyGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    roomGas.rebuild(cuStream, roomGasMem, asBuildScratchMem);
    areaLightGas.rebuild(cuStream, areaLightGasMem, asBuildScratchMem);
    bunnyGas.rebuild(cuStream, bunnyGasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    // EN: Perform compaction for static meshes.
    size_t roomGasCompactedSizze;
    roomGas.prepareForCompact(&roomGasCompactedSizze);
    roomGasCompactedMem.initialize(cuContext, cudau::BufferType::Device, roomGasCompactedSizze, 1);
    size_t areaLightGasCompactedSize;
    areaLightGas.prepareForCompact(&areaLightGasCompactedSize);
    areaLightGasCompactedMem.initialize(cuContext, cudau::BufferType::Device, areaLightGasCompactedSize, 1);
    size_t bunnyGasCompactedSize;
    bunnyGas.prepareForCompact(&bunnyGasCompactedSize);
    bunnyGasCompactedMem.initialize(cuContext, cudau::BufferType::Device, bunnyGasCompactedSize, 1);

    roomGas.compact(cuStream, roomGasCompactedMem);
    roomGas.removeUncompacted();
    areaLightGas.compact(cuStream, areaLightGasCompactedMem);
    areaLightGas.removeUncompacted();
    bunnyGas.compact(cuStream, bunnyGasCompactedMem);
    bunnyGas.removeUncompacted();



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGas);

    float areaLightInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance areaLightInst = scene.createInstance();
    areaLightInst.setChild(areaLightGas);
    areaLightInst.setTransform(areaLightInstXfm);

    std::vector<optixu::Instance> bunnyInsts;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * M_PI / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        Shared::MaterialData matData;
        matData.albedo = sRGB_degamma(HSVtoRGB(std::fmod((GoldenAngle * i) / (2 * M_PI), 1.0f),
                                               std::sqrt(r / 0.9f),
                                               1.0f));
        bunnyMats[i].setUserData(matData);

        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.003f + tt * 0.0006f;
        float bunnyInstXfm[] = {
            scale, 0, 0, x,
            0, scale, 0, -1 + (1 - tt),
            0, 0, scale, z
        };
        optixu::Instance bunnyInst = scene.createInstance();
        bunnyInst.setChild(bunnyGas, i);
        bunnyInst.setTransform(bunnyInstXfm);
        bunnyInsts.push_back(bunnyInst);
    }



    // JP: IAS作成時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when creating an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    for (int i = 0; i < bunnyInsts.size(); ++i)
        ias.addChild(bunnyInsts[i]);
    ias.prepareForBuild(&asMemReqs, &numInstances);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, numInstances);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    if (maxSizeOfScratchBuffer > asBuildScratchMem.sizeInBytes())
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr uint32_t renderTargetSizeX = 1024;
    constexpr uint32_t renderTargetSizeY = 1024;
    cudau::Array colorAccumBuffer;
    cudau::Array albedoAccumBuffer;
    cudau::Array normalAccumBuffer;
    colorAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                  renderTargetSizeX, renderTargetSizeY, 1);
    albedoAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    normalAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    cudau::TypedBuffer<float4> linearColorBuffer;
    cudau::TypedBuffer<float4> linearAlbedoBuffer;
    cudau::TypedBuffer<float4> linearNormalBuffer;
    cudau::TypedBuffer<float4> linearOutputBuffer;
    linearColorBuffer.initialize(cuContext, cudau::BufferType::Device,
                                 renderTargetSizeX * renderTargetSizeY);
    linearAlbedoBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);
    linearNormalBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);
    linearOutputBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);

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



    // JP: デノイザーのセットアップ。
    // EN: Setup a denoiser.
    constexpr bool useTiledDenoising = false; // Change this to true to use tiled denoising.
    constexpr uint32_t tileWidth = useTiledDenoising ? 256 : 0;
    constexpr uint32_t tileHeight = useTiledDenoising ? 256 : 0;
    optixu::Denoiser denoiser = optixContext.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL);
    denoiser.setModel(OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0);
    size_t stateSize;
    size_t scratchSize;
    size_t scratchSizeForComputeIntensity;
    uint32_t numTasks;
    denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                     &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                     &numTasks);
    hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
    hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
    hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    denoiserStateBuffer.initialize(cuContext, cudau::BufferType::Device, stateSize, 1);
    denoiserScratchBuffer.initialize(cuContext, cudau::BufferType::Device,
                                     std::max(scratchSize, scratchSizeForComputeIntensity), 1);

    std::vector<optixu::DenoisingTask> denoisingTasks(numTasks);
    denoiser.getTasks(denoisingTasks.data());

    denoiser.setLayers(&linearColorBuffer, &linearAlbedoBuffer, &linearNormalBuffer, &linearOutputBuffer,
                       OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4);
    denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(&moduleCopyBuffers, (getExecutableDirectory() / "denoiser/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyBuffers(moduleCopyBuffers, "copyBuffers", cudau::dim3(8, 8), 0);

    CUdeviceptr hdrIntensity;
    CUDADRV_CHECK(cuMemAlloc(&hdrIntensity, sizeof(float)));



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.colorAccumBuffer = colorAccumBuffer.getSurfaceObject(0);
    plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
    plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.16f);
    plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(&hitGroupSBT);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    cudau::Timer timerRender;
    cudau::Timer timerDenoise;
    timerRender.initialize(cuContext);
    timerDenoise.initialize(cuContext);
    
    // JP: レンダリング
    // EN: Render
    constexpr uint32_t numSamples = 8;
    timerRender.start(cuStream);
    for (int frameIndex = 0; frameIndex < numSamples; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }

    // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
    // EN: Copy the results to the linear buffers (and normalize normals).
    cudau::dim3 dimCopyBuffers = kernelCopyBuffers.calcGridDim(renderTargetSizeX, renderTargetSizeY);
    kernelCopyBuffers(cuStream, dimCopyBuffers,
                      colorAccumBuffer.getSurfaceObject(0),
                      albedoAccumBuffer.getSurfaceObject(0),
                      normalAccumBuffer.getSurfaceObject(0),
                      linearColorBuffer.getDevicePointer(),
                      linearAlbedoBuffer.getDevicePointer(),
                      linearNormalBuffer.getDevicePointer(),
                      uint2(renderTargetSizeX, renderTargetSizeY));
    timerRender.stop(cuStream);

    // JP: パストレーシング結果のデノイズ。
    //     毎フレーム呼ぶ必要があるのはcomputeIntensity()とinvoke()。
    //     computeIntensity()は自作することもできる。
    //     サイズが足りていればcomputeIntensity()のスクラッチバッファーとしてデノイザーのものが再利用できる。
    // EN: Denoise the path tracing result.
    //     computeIntensity() and invoke() should be calld every frame.
    //     You can also create a custom computeIntensity().
    //     Reusing the scratch buffer for denoising for computeIntensity() is possible if its size is enough.
    timerDenoise.start(cuStream);
    denoiser.computeIntensity(cuStream, denoiserScratchBuffer, hdrIntensity);
    for (int i = 0; i < denoisingTasks.size(); ++i)
        denoiser.invoke(cuStream, false, hdrIntensity, 0.0f, denoisingTasks[i]);
    timerDenoise.stop(cuStream);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    hpprintf("Render %u [spp]: %.3f[ms]\n", numSamples, timerRender.report());
    hpprintf("Denoise: %.3f[ms]\n", timerDenoise.report());

    timerDenoise.finalize();
    timerRender.finalize();



    // JP: 結果とデノイズ用付随バッファーの画像出力。
    // EN: Output the result and buffers associated to the denoiser as images.
    auto colorPixels = colorAccumBuffer.map<float4>();
    auto albedoPixels = albedoAccumBuffer.map<float4>();
    auto normalPixels = normalAccumBuffer.map<float4>();
    auto outputPixels = linearOutputBuffer.map();
    std::vector<uint32_t> colorImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> albedoImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> normalImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> outputImageData(renderTargetSizeX * renderTargetSizeY);
    for (int y = 0; y < renderTargetSizeY; ++y) {
        for (int x = 0; x < renderTargetSizeX; ++x) {
            uint32_t linearIndex = renderTargetSizeX * y + x;

            float4 color = colorPixels[linearIndex];
            color.x = sRGB_gamma_s(1 - std::exp(-color.x));
            color.y = sRGB_gamma_s(1 - std::exp(-color.y));
            color.z = sRGB_gamma_s(1 - std::exp(-color.z));
            uint32_t &dstColor = colorImageData[linearIndex];
            dstColor = (std::min<uint32_t>(255, 255 * color.x) << 0) |
                       (std::min<uint32_t>(255, 255 * color.y) << 8) |
                       (std::min<uint32_t>(255, 255 * color.z) << 16) |
                       (std::min<uint32_t>(255, 255 * color.w) << 24);

            float4 albedo = albedoPixels[linearIndex];
            uint32_t &dstAlbedo = albedoImageData[linearIndex];
            dstAlbedo = (std::min<uint32_t>(255, 255 * albedo.x) << 0) |
                        (std::min<uint32_t>(255, 255 * albedo.y) << 8) |
                        (std::min<uint32_t>(255, 255 * albedo.z) << 16) |
                        (std::min<uint32_t>(255, 255 * albedo.w) << 24);

            float4 normal = normalPixels[linearIndex];
            uint32_t &dstNormal = normalImageData[linearIndex];
            dstNormal = (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.x)) << 0) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.y)) << 8) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.z)) << 16) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.w)) << 24);

            float4 output = outputPixels[linearIndex];
            output.x = sRGB_gamma_s(1 - std::exp(-output.x));
            output.y = sRGB_gamma_s(1 - std::exp(-output.y));
            output.z = sRGB_gamma_s(1 - std::exp(-output.z));
            uint32_t &dstOutput = outputImageData[linearIndex];
            dstOutput = (std::min<uint32_t>(255, 255 * output.x) << 0) |
                        (std::min<uint32_t>(255, 255 * output.y) << 8) |
                        (std::min<uint32_t>(255, 255 * output.z) << 16) |
                        (std::min<uint32_t>(255, 255 * output.w) << 24);
        }
    }
    linearOutputBuffer.unmap();
    normalAccumBuffer.unmap();
    albedoAccumBuffer.unmap();
    colorAccumBuffer.unmap();

    stbi_write_bmp("color.bmp", renderTargetSizeX, renderTargetSizeY, 4, colorImageData.data());
    stbi_write_bmp("albedo.bmp", renderTargetSizeX, renderTargetSizeY, 4, albedoImageData.data());
    stbi_write_bmp("normal.bmp", renderTargetSizeX, renderTargetSizeY, 4, normalImageData.data());
    stbi_write_bmp("color_denoised.bmp", renderTargetSizeX, renderTargetSizeY, 4, outputImageData.data());



    CUDADRV_CHECK(cuMemFree(plpOnDevice));


    
    CUDADRV_CHECK(cuMemFree(hdrIntensity));

    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));
    
    denoiserScratchBuffer.finalize();
    denoiserStateBuffer.finalize();
    
    denoiser.destroy();
    
    rngBuffer.finalize();

    linearOutputBuffer.finalize();
    linearNormalBuffer.finalize();
    linearAlbedoBuffer.finalize();
    linearColorBuffer.finalize();

    normalAccumBuffer.finalize();
    albedoAccumBuffer.finalize();
    colorAccumBuffer.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    hitGroupSBT.finalize();

    for (int i = bunnyInsts.size() - 1; i >= 0; --i)
        bunnyInsts[i].destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    bunnyGasCompactedMem.finalize();
    areaLightGasCompactedMem.finalize();
    roomGasCompactedMem.finalize();
    bunnyGasMem.finalize();
    bunnyGas.destroy();
    areaLightGasMem.finalize();
    areaLightGas.destroy();
    roomGasMem.finalize();
    roomGas.destroy();

    bunnyTriangleBuffer.finalize();
    bunnyVertexBuffer.finalize();
    bunnyGeomInst.destroy();
    
    areaLightTriangleBuffer.finalize();
    areaLightVertexBuffer.finalize();
    areaLightGeomInst.destroy();

    roomMatIndexBuffer.finalize();
    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

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

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}

void loadObjFile(const std::filesystem::path &filepath,
                 std::vector<Shared::Vertex>* vertices, std::vector<Shared::Triangle>* triangles) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                                filepath.string().c_str());

    // Record unified unique vertices.
    std::map<std::tuple<int32_t, int32_t>, Shared::Vertex> unifiedVertexMap;
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
                auto key = std::make_tuple(idx.vertex_index, idx.normal_index);
                unifiedVertexMap[key] = Shared::Vertex{
                    make_float3(attrib.vertices[3 * idx.vertex_index + 0],
                                attrib.vertices[3 * idx.vertex_index + 1],
                                attrib.vertices[3 * idx.vertex_index + 2]),
                    make_float3(0, 0, 0),
                    make_float2(0, 0)
                };
            }

            idxOffset += numFaceVertices;
        }
    }

    // Assign a vertex index to each of unified unique unifiedVertexMap.
    std::map<std::tuple<int32_t, int32_t>, uint32_t> vertexIndices;
    vertices->resize(unifiedVertexMap.size());
    uint32_t vertexIndex = 0;
    for (const auto &kv : unifiedVertexMap) {
        vertices->at(vertexIndex) = kv.second;
        vertexIndices[kv.first] = vertexIndex++;
    }
    unifiedVertexMap.clear();

    // Calculate triangle index buffer.
    triangles->clear();
    for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
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
            auto key0 = std::make_tuple(idx0.vertex_index, idx0.normal_index);
            auto key1 = std::make_tuple(idx1.vertex_index, idx1.normal_index);
            auto key2 = std::make_tuple(idx2.vertex_index, idx2.normal_index);

            triangles->push_back(Shared::Triangle{
                vertexIndices.at(key0),
                vertexIndices.at(key1),
                vertexIndices.at(key2) });

            idxOffset += numFaceVertices;
        }
    }
    vertexIndices.clear();

    for (int tIdx = 0; tIdx < triangles->size(); ++tIdx) {
        const Shared::Triangle &tri = triangles->at(tIdx);
        Shared::Vertex &v0 = vertices->at(tri.index0);
        Shared::Vertex &v1 = vertices->at(tri.index1);
        Shared::Vertex &v2 = vertices->at(tri.index2);
        float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        v0.normal += gn;
        v1.normal += gn;
        v2.normal += gn;
    }
    for (int vIdx = 0; vIdx < vertices->size(); ++vIdx) {
        Shared::Vertex &v = vertices->at(vIdx);
        v.normal = normalize(v.normal);
    }
}
