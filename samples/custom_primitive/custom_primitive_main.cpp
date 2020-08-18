#include "custom_primitive_shared.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"

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

    // JP: カスタムプリミティブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    // EN: Appropriately setting primitive type flags is required since this sample uses custom primitive intersection.
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "custom_primitive/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: このヒットグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: This hit group is for ray-triangle intersection, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        emptyModule, nullptr);

    // JP: このヒットグループはレイと球の交叉判定用なのでカスタムのIntersectionプログラムを渡す。
    // EN: This is for ray-sphere intersection, so pass a custom intersection program.
    optixu::ProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        moduleOptiX, RT_IS_NAME_STR("intersectSphere"));

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.setMaxTraceDepth(1);
    pipeline.link(DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material matForTriangles = optixContext.createMaterial();
    matForTriangles.setHitGroup(Shared::RayType_Primary, hitProgramGroupForTriangles);
    optixu::Material matForSpheres = optixContext.createMaterial();
    matForSpheres.setHitGroup(Shared::RayType_Primary, hitProgramGroupForSpheres);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    cudau::TypedBuffer<Shared::GeometryData> geomDataBuffer;
    geomDataBuffer.initialize(cuContext, cudau::BufferType::Device, 3);
    Shared::GeometryData* geomData = geomDataBuffer.map();

    uint32_t geomInstIndex = 0;

    optixu::GeometryInstance geomInstRoom = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferRoom;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferRoom;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
            // back wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 0) },
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
            // back wall
            { 4, 5, 6 }, { 4, 6, 7 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        vertexBufferRoom.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferRoom.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInstRoom.setVertexBuffer(&vertexBufferRoom);
        geomInstRoom.setTriangleBuffer(&triangleBufferRoom);
        geomInstRoom.setNumMaterials(1, nullptr);
        geomInstRoom.setMaterial(0, 0, matForTriangles);
        geomInstRoom.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstRoom.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferRoom.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferRoom.getDevicePointer();

        ++geomInstIndex;
    }

    optixu::GeometryInstance geomInstAreaLight = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferAreaLight;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferAreaLight;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.75f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.75f, 0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.25f, 0.75f, 0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.25f, 0.75f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        vertexBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInstAreaLight.setVertexBuffer(&vertexBufferAreaLight);
        geomInstAreaLight.setTriangleBuffer(&triangleBufferAreaLight);
        geomInstAreaLight.setNumMaterials(1, nullptr);
        geomInstAreaLight.setMaterial(0, 0, matForTriangles);
        geomInstAreaLight.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstAreaLight.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferAreaLight.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferAreaLight.getDevicePointer();

        ++geomInstIndex;
    }

    // JP: カスタムプリミティブ用GeometryInstanceは生成時に指定する必要がある。
    // EN: GeometryInstance for custom primitives requires to be specified at the creation.
    optixu::GeometryInstance geomInstSpheres = scene.createGeometryInstance(true);
    cudau::TypedBuffer<AABB> aabbBufferSpheres;
    cudau::TypedBuffer<Shared::SphereParameter> paramBufferSpheres;
    {
        constexpr uint32_t numPrimitives = 25;
        aabbBufferSpheres.initialize(cuContext, cudau::BufferType::Device, numPrimitives);
        paramBufferSpheres.initialize(cuContext, cudau::BufferType::Device, numPrimitives);

        // JP: 各球のパラメターとAABBを設定する。
        //     これらはもちろんCUDAカーネルで設定することも可能。
        // EN: Set the parameters and AABB for each sphere.
        //     Of course, these can be set using a CUDA kernel.
        AABB* aabbs = aabbBufferSpheres.map();
        Shared::SphereParameter* params = paramBufferSpheres.map();
        std::mt19937 rng(1290527201);
        std::uniform_real_distribution u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::SphereParameter &param = params[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = -0.8f + 1.6f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param.center = make_float3(x, y, z);
            param.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);
            param.texCoordMultiplier = 10;

            AABB &aabb = aabbs[i];
            aabb = AABB();
            aabb.unify(param.center - make_float3(param.radius));
            aabb.unify(param.center + make_float3(param.radius));
        }
        paramBufferSpheres.unmap();
        aabbBufferSpheres.unmap();

        geomInstSpheres.setCustomPrimitiveAABBBuffer(&aabbBufferSpheres);
        geomInstSpheres.setNumMaterials(1, nullptr);
        geomInstSpheres.setMaterial(0, 0, matForSpheres);
        geomInstSpheres.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstSpheres.setUserData(geomInstIndex);

        geomData[geomInstIndex].aabbBuffer = aabbBufferSpheres.getDevicePointer();
        geomData[geomInstIndex].paramBuffer = paramBufferSpheres.getDevicePointer();

        ++geomInstIndex;
    }

    geomDataBuffer.unmap();



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure gasRoom = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasRoomMem;
    cudau::Buffer gasRoomCompactedMem;
    gasRoom.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    gasRoom.setNumMaterialSets(1);
    gasRoom.setNumRayTypes(0, Shared::NumRayTypes);
    gasRoom.addChild(geomInstRoom);
    gasRoom.addChild(geomInstAreaLight);
    gasRoom.prepareForBuild(&asMemReqs);
    gasRoomMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: カスタムプリミティブ用のGASは三角形用のGASとは別にする必要がある。
    //     GAS生成時にカスタムプリミティブ用であることを指定する。
    // EN: GAS for custom primitives must be created separately with GAS for triangles.
    //     Specify that the GAS is for custom primitives at the creation.
    optixu::GeometryAccelerationStructure gasCustomPrimitives = scene.createGeometryAccelerationStructure(true);
    cudau::Buffer gasCustomPrimitivesMem;
    cudau::Buffer gasCustomPrimitivesCompactedMem;
    gasCustomPrimitives.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    gasCustomPrimitives.setNumMaterialSets(1);
    gasCustomPrimitives.setNumRayTypes(0, Shared::NumRayTypes);
    gasCustomPrimitives.addChild(geomInstSpheres);
    gasCustomPrimitives.prepareForBuild(&asMemReqs);
    gasCustomPrimitivesMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    gasRoom.rebuild(cuStream, gasRoomMem, asBuildScratchMem);
    gasCustomPrimitives.rebuild(cuStream, gasCustomPrimitivesMem, asBuildScratchMem);

    // JP: 静的なGASはコンパクションもしておく。
    // EN: Perform compaction for static GAS.
    size_t compactedASSize;
    gasRoom.prepareForCompact(&compactedASSize);
    gasRoomCompactedMem.initialize(cuContext, cudau::BufferType::Device, compactedASSize, 1);
    gasRoom.compact(cuStream, gasRoomCompactedMem);
    gasRoom.removeUncompacted();

    gasCustomPrimitives.prepareForCompact(&compactedASSize);
    gasCustomPrimitivesCompactedMem.initialize(cuContext, cudau::BufferType::Device, compactedASSize, 1);
    gasCustomPrimitives.compact(cuStream, gasCustomPrimitivesCompactedMem);
    gasCustomPrimitives.removeUncompacted();



    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    scene.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance instRoom = scene.createInstance();
    instRoom.setChild(gasRoom);

    optixu::Instance instCustomPrimitives = scene.createInstance();
    instCustomPrimitives.setChild(gasCustomPrimitives);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    ias.addChild(instRoom);
    ias.addChild(instCustomPrimitives);
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
    optixu::HostBlockBuffer2D<float4, 1> accumBuffer;
    accumBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.geomInstData = geomDataBuffer.getDevicePointer();
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.resultBuffer = accumBuffer.getBlockBuffer2D();
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5);
    plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(&shaderBindingTable);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
    pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    accumBuffer.map();
    std::vector<uint32_t> imageData(renderTargetSizeX * renderTargetSizeY);
    for (int y = 0; y < renderTargetSizeY; ++y) {
        for (int x = 0; x < renderTargetSizeX; ++x) {
            const float4 &srcPix = accumBuffer(x, y);
            uint32_t &dstPix = imageData[y * renderTargetSizeX + x];
            dstPix = (std::min<uint32_t>(255, 255 * srcPix.x) <<  0) |
                     (std::min<uint32_t>(255, 255 * srcPix.y) <<  8) |
                     (std::min<uint32_t>(255, 255 * srcPix.z) << 16) |
                     (std::min<uint32_t>(255, 255 * srcPix.w) << 24);
        }
    }
    accumBuffer.unmap();

    stbi_write_bmp("output.bmp", renderTargetSizeX, renderTargetSizeY, 4, imageData.data());



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    instCustomPrimitives.destroy();
    instRoom.destroy();

    shaderBindingTable.finalize();

    asBuildScratchMem.finalize();
    gasCustomPrimitivesCompactedMem.finalize();
    gasCustomPrimitivesMem.finalize();
    gasCustomPrimitives.destroy();
    gasRoomCompactedMem.finalize();
    gasRoomMem.finalize();
    gasRoom.destroy();

    paramBufferSpheres.finalize();
    aabbBufferSpheres.finalize();
    geomInstSpheres.destroy();

    triangleBufferAreaLight.finalize();
    vertexBufferAreaLight.finalize();
    geomInstAreaLight.destroy();

    triangleBufferRoom.finalize();
    vertexBufferRoom.finalize();
    geomInstRoom.destroy();

    geomDataBuffer.finalize();

    scene.destroy();

    matForSpheres.destroy();
    matForTriangles.destroy();

    hitProgramGroupForSpheres.destroy();
    hitProgramGroupForTriangles.destroy();

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
