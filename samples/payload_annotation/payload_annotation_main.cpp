﻿/*

JP: このサンプルはペイロードの使用方法を明示的に指定することでパイプラインの最適化を助ける方法について示します。
    ペイロードのパラメターごとの各シェーダーにおけるアクセスフラグを記述したデータを作成し、
    モジュールやプログラムグループ作成時に指定、併せてカーネル内でも使用するペイロードタイプを指定することで
    ペイロード使用方法のアノテーションを行うことができます。
    アノテーションを適切に行うことによって複雑なパイプラインにおけるレジスター使用量を削減し、
    性能向上につながる可能性があります。
    各ペイロードがどのようにアノテートされているかは payload_annotation_shared.h を参照してください。

EN: This sample shows how to explicitly annotate payload usages to help pipeline optimization.
    Create data describing access flags in each shader per payload parameter to pass to module and 
    program group creation along with specifying a used payload type in kernel to annotate payload usages.
    Appropriate annotations allow to reduce register consumption in complex pipelines and possibly
    improve performance.
    Refer to payload_annotation_shared.h for details on how each payload is annotated.
*/

#include "payload_annotation_shared.h"

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

    // JP: アノテーション機能を使用する場合は、パイプラインオプションにおけるペイロードサイズにゼロを指定する。
    // EN: Specify zero for the payload size in the pipeline option in the case where using the annotation feature.
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::usePayloadAnnotation ? 0 :
        std::max(Shared::SearchRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords);
    pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline.setPipelineOptions(pipelineOptions);

    optixu::PayloadType payloadTypes[2];
    if constexpr (Shared::usePayloadAnnotation) {
        // JP: 2つのレイタイプに関わるペイロードタイプを作成する。
        //     各ペイロードがどのようにアノテートされているかは
        //     payload_annotation_shared.hを参照。
        // EN: Create payload types for two ray types.
        //     Refer to payload_annotation_shared.h
        //     for details on how each payload is annotated.
        payloadTypes[0] = Shared::SearchRayPayloadSignature::getPayloadType();
        payloadTypes[1] = Shared::VisibilityRayPayloadSignature::getPayloadType();
    }
    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "payload_annotation/ptxes/optix_path_tracing.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE),
        nullptr, 0,
        payloadTypes, Shared::usePayloadAnnotation ? lengthof(payloadTypes) : 0);

    optixu::Module emptyModule;

    optixu::Program pathTracingRayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("pathTracing"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

    // JP: Miss Programで使用されているペイロードタイプ情報を渡す。
    // EN: Pass the payload type used in the miss program.
    optixu::Program missProgram = pipeline.createMissProgram(
        moduleOptiX, RT_MS_NAME_STR("miss"),
        Shared::usePayloadAnnotation ? payloadTypes[0] : optixu::PayloadType());
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    // JP: それぞれのヒットグループで使われているペイロードタイプ情報を渡す。
    // EN: Pass the payload type for each hit group.
    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr,
        Shared::usePayloadAnnotation ? payloadTypes[0] : optixu::PayloadType());
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"),
        Shared::usePayloadAnnotation ? payloadTypes[1] : optixu::PayloadType());

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

    std::vector<optixu::Instance> bunnyInsts;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        Shared::MaterialData matData;
        matData.albedo = sRGB_degamma(HSVtoRGB(
            std::fmod((GoldenAngle * i) / (2 * pi_v<float>), 1.0f),
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
        bunnyInst.setChild(bunny.optixGas, i);
        bunnyInst.setTransform(bunnyInstXfm);
        bunnyInsts.push_back(bunnyInst);
    }



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    for (int i = 0; i < bunnyInsts.size(); ++i)
        ias.addChild(bunnyInsts[i]);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
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
        info.geom->optixGas.compact(
            cuStream,
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

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr uint32_t renderTargetSizeX = 1024;
    constexpr uint32_t renderTargetSizeY = 1024;
    cudau::Array colorAccumBuffer;
    colorAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        renderTargetSizeX, renderTargetSizeY, 1);

    cudau::Array rngBuffer;
    rngBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::UInt32, sizeof(Shared::PCG32RNG) / sizeof(uint32_t),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        renderTargetSizeX, renderTargetSizeY, 1);
    {
        std::mt19937_64 rng(591842031321323413);

        auto rngs = rngBuffer.map<Shared::PCG32RNG>();
        for (int y = 0; y < renderTargetSizeY; ++y)
            for (int x = 0; x < renderTargetSizeX; ++x)
                rngs[y * renderTargetSizeX + x].setState(rng());
        rngBuffer.unmap();
    };



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.rngBuffer = rngBuffer.getSurfaceObject(0);
    plp.colorAccumBuffer = colorAccumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.16f);
    plp.camera.orientation = rotateY3x3(pi_v<float>);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    cudau::Timer timerRender;
    timerRender.initialize(cuContext);
    
    // JP: レンダリング
    // EN: Render
    constexpr uint32_t numSamples = 64;
    timerRender.start(cuStream);
    for (int frameIndex = 0; frameIndex < numSamples; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
    timerRender.stop(cuStream);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    float renderTime = timerRender.report();
    hpprintf("Render %u [spp]: %.3f[ms]\n", numSamples, renderTime);

    timerRender.finalize();

    // JP: 結果の画像出力。
    // EN: Output the result as an image.
    saveImage("output.png", colorAccumBuffer, true, true);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));


    
    rngBuffer.finalize();

    colorAccumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    for (int i = bunnyInsts.size() - 1; i >= 0; --i)
        bunnyInsts[i].destroy();
    areaLightInst.destroy();
    roomInst.destroy();

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

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
