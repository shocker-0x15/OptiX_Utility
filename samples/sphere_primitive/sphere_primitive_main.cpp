/*

JP: このサンプルは球を扱う方法を示します。
    球とレイの交叉判定はOptiXによって内部的に扱われます。
    球は級の中心の頂点バッファーと球ごとの半径を表すバッファーから構成されます。

EN: This sample shows how to handle spheres.
    Intersection test between a ray and a sphere is handled internally by OptiX.
    Spheres consist of a vertex buffer of sphere centers and a buffer for the radius of each sphere.

*/

#include "sphere_primitive_shared.h"

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

    // JP: 球との衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    //     球のアトリビュートサイズは1Dword(float)。
    // EN: Appropriately setting primitive type flags is required since this sample uses sphere intersection.
    //     The attribute size of spheres is 1 Dword (float).
    pipeline.setPipelineOptions(
        Shared::MyPayloadSignature::numDwords,
        std::max(optixu::calcSumDwords<float2>(),
                 optixu::calcSumDwords<float>()),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        optixu::UseMotionBlur::No, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "sphere_primitive/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::ProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    /*
    JP: 球用のヒットグループを作成する。
        球には三角形と同様、ビルトインのIntersection Programが使われるのでユーザーが指定する必要はない。
        球を含むことになるASと同じビルド設定を予め指定しておく必要がある。
    EN: Create a hit group for spheres.
        Sphere uses a built-in intersection program similar to triangle,
        so the user doesn't need to specify it.
        The same build configuration as an AS having the sphere is required.
    */
    constexpr optixu::ASTradeoff sphereASTradeOff = optixu::ASTradeoff::PreferFastTrace;
    constexpr optixu::AllowUpdate sphereASUpdatable = optixu::AllowUpdate::No;
    constexpr optixu::AllowCompaction sphereASCompactable = optixu::AllowCompaction::Yes;
    constexpr auto useEmbeddedVertexData = optixu::AllowRandomVertexAccess(Shared::useEmbeddedVertexData);
    optixu::ProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroupForSphereIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        sphereASTradeOff, sphereASUpdatable, sphereASCompactable, useEmbeddedVertexData);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

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

    optixu::GeometryInstance roomGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> roomVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> roomTriangleBuffer;
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
            { 4, 7, 6 }, { 4, 6, 5 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        roomVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        roomTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = roomVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = roomTriangleBuffer.getDevicePointer();

        roomGeomInst.setVertexBuffer(roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(roomTriangleBuffer);
        roomGeomInst.setNumMaterials(1, optixu::BufferView());
        roomGeomInst.setMaterial(0, 0, matForTriangles);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setUserData(geomData);
    }

    // JP: 球用GeometryInstanceは生成時に指定する必要がある。
    // EN: GeometryInstance for spheres requires to be specified at the creation.

    // Spheres
    optixu::GeometryInstance sphereGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::Spheres);
    cudau::TypedBuffer<Shared::SphereParameter> sphereParamBuffer;
    {
        std::vector<Shared::SphereParameter> params;
        {
            std::mt19937 rng(390318410);
            std::uniform_int_distribution<uint32_t> uSeg(3, 5);
            std::uniform_real_distribution<float> u01;

            constexpr uint32_t numSpheres = 100;
            params.resize(numSpheres);
            for (int sphIdx = 0; sphIdx < numSpheres; ++sphIdx) {
                Shared::SphereParameter &param = params[sphIdx];
                param.center = float3(
                    -0.85f + 1.7f * u01(rng),
                    -0.85f + 1.7f * u01(rng),
                    -0.85f + 1.7f * u01(rng));
                param.radius = 0.025f + 0.1f * u01(rng);
            }

            Shared::SphereParameter param;
            param.center = float3(0.0f);
            param.radius = 5.0f;
            params.push_back(param);
        }

        sphereParamBuffer.initialize(cuContext, cudau::BufferType::Device, params);

        Shared::GeometryData geomData = {};
        geomData.sphereParamBuffer = sphereParamBuffer.getDevicePointer();

        sphereGeomInst.setVertexBuffer(optixu::BufferView(
            sphereParamBuffer.getCUdeviceptr() + offsetof(Shared::SphereParameter, center),
            sphereParamBuffer.numElements(), sphereParamBuffer.stride()));
        sphereGeomInst.setRadiusBuffer(optixu::BufferView(
            sphereParamBuffer.getCUdeviceptr() + offsetof(Shared::SphereParameter, radius),
            sphereParamBuffer.numElements(), sphereParamBuffer.stride()));
        sphereGeomInst.setMaterial(0, 0, matForSpheres);
        sphereGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        sphereGeomInst.setUserData(geomData);
    }



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure roomGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer roomGasMem;
    roomGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes,
        optixu::AllowRandomVertexAccess::No);
    roomGas.setNumMaterialSets(1);
    roomGas.setNumRayTypes(0, Shared::NumRayTypes);
    roomGas.addChild(roomGeomInst);
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: 球用のGASは三角形用のGASとは別にする必要がある。
    //     GAS生成時に球用であることを指定する。
    // EN: GAS for spheres must be created separately with GAS for triangles.
    //     Specify that the GAS is for spheres at the creation.
    optixu::GeometryAccelerationStructure spheresGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::Spheres);
    cudau::Buffer spheresGasMem;
    spheresGas.setConfiguration(
        sphereASTradeOff, sphereASUpdatable, sphereASCompactable,
        useEmbeddedVertexData);
    spheresGas.setNumMaterialSets(1);
    spheresGas.setNumRayTypes(0, Shared::NumRayTypes);
    spheresGas.addChild(sphereGeomInst);
    spheresGas.prepareForBuild(&asMemReqs);
    spheresGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGas);

    optixu::Instance spheresInst = scene.createInstance();
    spheresInst.setChild(spheresGas);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::No,
        optixu::AllowRandomInstanceAccess::No);
    ias.addChild(roomInst);
    ias.addChild(spheresInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    roomGas.rebuild(cuStream, roomGasMem, asBuildScratchMem);
    spheresGas.rebuild(cuStream, spheresGasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        optixu::GeometryAccelerationStructure gas;
        cudau::Buffer* mem;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { roomGas, &roomGasMem, 0, 0 },
        { spheresGas, &spheresGasMem, 0, 0 },
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.gas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.gas.compact(
            cuStream,
            optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].gas.removeUncompacted();
        gasList[i].mem->finalize();
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
    optixu::HostBlockBuffer2D<float4, 1> accumBuffer;
    accumBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.resultBuffer = accumBuffer.getBlockBuffer2D();
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5f);
    plp.camera.orientation = rotateY3x3(pi_v<float>);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
    pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    saveImage("output.png", accumBuffer, false, false);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    spheresInst.destroy();
    roomInst.destroy();

    asBuildScratchMem.finalize();
    spheresGas.destroy();
    roomGas.destroy();

    sphereParamBuffer.finalize();
    sphereGeomInst.destroy();

    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

    scene.destroy();

    matForSpheres.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

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
