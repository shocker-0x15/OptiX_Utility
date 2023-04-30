/*

JP: 

    --no-dmm: DMMを無効化する。
    --no-index-buffer: OMM用のインデックスバッファーを使用しない。

EN: 

    [1] Displacement Micro-Map SDK: https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/

    --no-dmm: Disable DMM.
    --no-index-buffer: Specify not to use index buffers for OMM.

*/

#include "displacement_micro_map_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

int32_t main(int32_t argc, const char* argv[]) try {
    bool useDMM = true;
    bool useDmmIndexBuffer = true;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--no-dmm") {
            useDMM = false;
        }
        else if (arg == "--no-index-buffer") {
            useDmmIndexBuffer = false;
        }
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

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

    // JP: Displacement Micro-Mapを使う場合、プリミティブ種別のフラグを適切に設定する必要がある。
    // EN: Appropriately setting primitive type flags is required when using displacement micro-map.
    pipeline.setPipelineOptions(
        std::max(Shared::PrimaryRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_NONE/*DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE)*/,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE,
        optixu::UseMotionBlur::No);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "displacement_micro_map/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT/*DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT)*/,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE/*DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE)*/);

    optixu::Module emptyModule;

    optixu::Program rayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

    optixu::Program missProgram = pipeline.createMissProgram(
        moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));

    pipeline.link(2);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);
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

    constexpr bool useBlockCompressedTexture = true;

    optixu::Material defaultMat = optixContext.createMaterial();
    defaultMat.setHitGroup(Shared::RayType_Primary, shadingHitProgramGroup);
    defaultMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer dmmBuildScratchMem;
    cudau::Buffer asBuildScratchMem;

    constexpr optixu::IndexSize dmmIndexSize = optixu::IndexSize::k2Bytes;

    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        struct MaterialGroup {
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            optixu::GeometryInstance optixGeomInst;
            cudau::Array heightTexArray;
            CUtexObject heightTexObj;

            optixu::OpacityMicroMapArray optixDmmArray;
            cudau::Buffer rawDmmArray;
            cudau::TypedBuffer<OptixOpacityMicromapDesc> dmmDescs;
            cudau::Buffer dmmIndexBuffer;
            cudau::Buffer dmmArrayMem;
        };
        std::vector<MaterialGroup> matGroups;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            for (auto it = matGroups.rbegin(); it != matGroups.rend(); ++it) {
                it->dmmArrayMem.finalize();
                it->dmmIndexBuffer.finalize();
                it->dmmDescs.finalize();
                it->rawDmmArray.finalize();
                it->optixDmmArray.destroy();

                if (it->heightTexObj) {
                    CUDADRV_CHECK(cuTexObjectDestroy(it->heightTexObj));
                    it->heightTexArray.finalize();
                }
                it->triangleBuffer.finalize();
                it->optixGeomInst.destroy();
            }
            vertexBuffer.finalize();
        }
    };

    Geometry floor;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floor.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        floor.optixGas = scene.createGeometryAccelerationStructure();
        floor.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        floor.optixGas.setNumMaterialSets(1);
        floor.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        Geometry::MaterialGroup group;
        {
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = floor.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.texture = 0;
            geomData.albedo = float3(0.8f, 0.8f, 0.8f);

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(floor.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            floor.optixGas.addChild(group.optixGeomInst);
            floor.matGroups.push_back(std::move(group));
        }

        floor.optixGas.prepareForBuild(&asMemReqs);
        floor.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry displacedMesh;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-1.0f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(1.0f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(1.0f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        displacedMesh.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        // JP: DMMを適用するジオメトリやそれを含むGASは通常の三角形用のもので良い。
        // EN: 
        displacedMesh.optixGas = scene.createGeometryAccelerationStructure();
        displacedMesh.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        displacedMesh.optixGas.setNumMaterialSets(1);
        displacedMesh.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        Geometry::MaterialGroup group;
        {
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = floor.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.texture = 0;
            geomData.albedo = float3(0.8f, 0.8f, 0.8f);

            std::filesystem::path heightMapPath = R"(../../data/mountain_heightmap.png)";
            {
                int32_t width, height, n;
                uint8_t* linearImageData = stbi_load(
                    heightMapPath.string().c_str(),
                    &width, &height, &n, 1);
                group.heightTexArray.initialize2D(
                    cuContext, cudau::ArrayElementType::UInt8, 1,
                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                    width, height, 1);
                group.heightTexArray.write<uint8_t>(linearImageData, width * height);
                stbi_image_free(linearImageData);

                cudau::TextureSampler texSampler;
                texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
                texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
                texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);
                texSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
                texSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
                group.heightTexObj = texSampler.createTextureObject(group.heightTexArray);
                //geomData.heightTexture = group.heightTexObj;
            }

            size_t scratchMemSizeForDMM = getScratchMemSizeForDMMGenerator(lengthof(triangles));
            cudau::Buffer scratchMemForDMM;
            scratchMemForDMM.initialize(cuContext, cudau::BufferType::Device, scratchMemSizeForDMM, 1);

            DMMGeneratorContext dmmContext;
            uint32_t histInDMMArray[shared::NumDMMFormats];
            uint32_t histInMesh[shared::NumDMMFormats];
            uint64_t rawDmmArraySize = 0;
            if (useDMM) {
                initializeDMMGeneratorContext(
                    getExecutableDirectory() / "displacement_micro_map/ptxes",
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, position),
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, texCoord),
                    sizeof(Shared::Vertex),
                    group.triangleBuffer.getCUdeviceptr(), sizeof(Shared::Triangle), lengthof(triangles),
                    group.heightTexObj,
                    make_uint2(group.heightTexArray.getWidth(), group.heightTexArray.getHeight()), 1, 0,
                    shared::DMMFormat_Level0, shared::DMMFormat_Level5, 0,
                    useDmmIndexBuffer, 1 << static_cast<uint32_t>(dmmIndexSize),
                    scratchMemForDMM.getCUdeviceptr(), scratchMemForDMM.sizeInBytes(),
                    &dmmContext);

                countDMMFormats(dmmContext, histInDMMArray, histInMesh, &rawDmmArraySize);
            }

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(floor.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            floor.optixGas.addChild(group.optixGeomInst);
            floor.matGroups.push_back(std::move(group));
        }

        floor.optixGas.prepareForBuild(&asMemReqs);
        floor.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを基にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floor.optixGas);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(floorInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    floor.optixGas.rebuild(cuStream, floor.gasMem, asBuildScratchMem);

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
        { &floor, 0, 0 },
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



    constexpr uint32_t renderTargetSizeX = 1280;
    constexpr uint32_t renderTargetSizeY = 720;
    cudau::Array colorAccumBuffer;
    colorAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        renderTargetSizeX, renderTargetSizeY, 1);


    
    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.colorAccumBuffer = colorAccumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 6.0f, 6.0f);
    plp.camera.orientation = rotateY3x3(pi_v<float>) * rotateX3x3(pi_v<float> / 4.0f);
    plp.lightDirection = normalize(float3(1, 5, 2));
    plp.lightRadiance = float3(7.5f, 7.5f, 7.5f);
    plp.envRadiance = float3(0.10f, 0.13f, 0.9f);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    cudau::Timer timerRender;
    timerRender.initialize(cuContext);
    
    // JP: レンダリング
    // EN: Render
    timerRender.start(cuStream);
    const uint32_t superSampleSize = 8;
    plp.superSampleSizeMinus1 = superSampleSize - 1;
    for (int frameIndex = 0; frameIndex < superSampleSize * superSampleSize; ++frameIndex) {
        plp.sampleIndex = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
    timerRender.stop(cuStream);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    float renderTime = timerRender.report();
    hpprintf("Render: %.3f[ms]\n", renderTime);

    timerRender.finalize();

    // JP: 結果の画像出力。
    // EN: Output the result as an image.
    saveImage("output.png", colorAccumBuffer, true, true);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));


    
    colorAccumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    floorInst.destroy();

    displacedMesh.finalize();
    floor.finalize();

    scene.destroy();

    defaultMat.destroy();



    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

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
