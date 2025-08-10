/*

JP: このサンプルは三角形とカーブ以外のカスタムプリミティブを扱う方法を示します。
    三角形とレイの交叉判定はOptiXによって内部的に扱われますが、カスタムプリミティブの場合は
    ユーザーが独自の交叉判定プログラムを記述します。
    Geometry Acceleration Structureは各プリミティブを囲むAABBの配列に対して構築します。

EN: This sample shows how to handle custom primitives other than triangles or curves.
    Intersection test between a ray and a triangle is handled internally by OptiX
    but in the case of custom primitives the user writes own program to test intersection.
    A geometry acceleration structure is built over an array of AABBs, where each encloses a primitive.

*/

#include "custom_primitive_shared.h"

int32_t main(int32_t argc, const char* argv[]) try {
    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
#if CUDA_VERSION < 13000
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
#else
    CUctxCreateParams cuCtxCreateParams = {};
    CUDADRV_CHECK(cuCtxCreate(&cuContext, &cuCtxCreateParams, 0, 0));
#endif
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: カスタムプリミティブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    // EN: Appropriately setting primitive type flags is required since this sample uses custom primitive intersection.
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.attributeCountInDwords = std::max<uint32_t>(
        optixu::calcSumDwords<float2>(),
        Shared::PartialSphereAttributeSignature::numDwords);
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "custom_primitive/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::HitProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    // JP: このヒットグループはレイと(部分)球の交叉判定用なのでカスタムのIntersectionプログラムを渡す。
    // EN: This is for ray-(partial-)sphere intersection, so pass a custom intersection program.
    optixu::HitProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroupForCustomIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        moduleOptiX, RT_IS_NAME_STR("partialSphere"));

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setMissRayTypeCount(Shared::NumRayTypes);
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
        geomData.vertexBuffer = roomVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = roomTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        roomGeomInst.setVertexBuffer(roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(roomTriangleBuffer);
        roomGeomInst.setMaterialCount(1, optixu::BufferView());
        roomGeomInst.setMaterial(0, 0, matForTriangles);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        roomGeomInst.setUserData(geomData);
    }

    optixu::GeometryInstance areaLightGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> areaLightVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> areaLightTriangleBuffer;
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

        areaLightVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLightTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLightVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = areaLightTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        areaLightGeomInst.setVertexBuffer(areaLightVertexBuffer);
        areaLightGeomInst.setTriangleBuffer(areaLightTriangleBuffer);
        areaLightGeomInst.setMaterialCount(1, optixu::BufferView());
        areaLightGeomInst.setMaterial(0, 0, matForTriangles);
        areaLightGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLightGeomInst.setUserData(geomData);
    }

    // JP: カスタムプリミティブ用GeometryInstanceは生成時に指定する必要がある。
    // EN: GeometryInstance for custom primitives requires to be specified at the creation.
    optixu::GeometryInstance customPrimsGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);
    cudau::TypedBuffer<AABB> customPrimsAabbBuffer;
    cudau::TypedBuffer<Shared::PartialSphereParameter> spheresParamBuffer;
    {
        constexpr uint32_t numPrimitives = 25;
        customPrimsAabbBuffer.initialize(cuContext, cudau::BufferType::Device, numPrimitives);
        spheresParamBuffer.initialize(cuContext, cudau::BufferType::Device, numPrimitives);

        // JP: 各球のパラメターとAABBを設定する。
        //     これらはもちろんCUDAカーネルで設定することも可能。
        // EN: Set the parameters and AABB for each sphere.
        //     Of course, these can be set using a CUDA kernel.
        AABB* aabbs = customPrimsAabbBuffer.map();
        Shared::PartialSphereParameter* params = spheresParamBuffer.map();
        std::mt19937 rng(1290527201);
        std::uniform_real_distribution<float> u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::PartialSphereParameter &param = params[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = -0.8f + 1.6f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param.center = make_float3(x, y, z);
            param.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);
            //// Full Sphere
            //param.minTheta = 0;
            //param.maxTheta = pi_v<float>;
            //param.minPhi = 0;
            //param.maxPhi = 2 * pi_v<float>;
            // Partial Sphere
            param.minTheta = pi_v<float> * (0.15f * u01(rng));
            param.maxTheta = pi_v<float> * (0.85f + 0.15f * u01(rng));
            param.minPhi = 2 * pi_v<float> * (0.3f * u01(rng));
            param.maxPhi = 2 * pi_v<float> * (0.7f + 0.3f * u01(rng));

            // JP: 簡単のため完全な球に対するAABBを計算しているが、
            //     可能であれば実際のジオメトリ(ここでは部分的な球)に対して計算したほうが良い。
            // EN: Compute AABB for the full sphere for simplicity, but computing
            //     AABB for the actual geometry (the partial sphere here) is better if possible.
            AABB &aabb = aabbs[i];
            aabb = AABB();
            aabb.unify(param.center - make_float3(param.radius));
            aabb.unify(param.center + make_float3(param.radius));
        }
        spheresParamBuffer.unmap();
        customPrimsAabbBuffer.unmap();

        Shared::GeometryData geomData = {};
        geomData.aabbBuffer = customPrimsAabbBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.paramBuffer = spheresParamBuffer.getROBuffer<enableBufferOobCheck>();

        customPrimsGeomInst.setCustomPrimitiveAABBBuffer(customPrimsAabbBuffer);
        customPrimsGeomInst.setMaterialCount(1, optixu::BufferView());
        customPrimsGeomInst.setMaterial(0, 0, matForSpheres);
        customPrimsGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        customPrimsGeomInst.setUserData(geomData);
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
        optixu::AllowCompaction::Yes);
    roomGas.setMaterialSetCount(1);
    roomGas.setRayTypeCount(0, Shared::NumRayTypes);
    roomGas.addChild(roomGeomInst);
    roomGas.addChild(areaLightGeomInst);
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: カスタムプリミティブ用のGASは三角形用のGASとは別にする必要がある。
    //     GAS生成時にカスタムプリミティブ用であることを指定する。
    // EN: GAS for custom primitives must be created separately with GAS for triangles.
    //     Specify that the GAS is for custom primitives at the creation.
    optixu::GeometryAccelerationStructure customPrimitivesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::CustomPrimitives);
    cudau::Buffer customPrimitivesGasMem;
    customPrimitivesGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
    customPrimitivesGas.setMaterialSetCount(1);
    customPrimitivesGas.setRayTypeCount(0, Shared::NumRayTypes);
    customPrimitivesGas.addChild(customPrimsGeomInst);
    customPrimitivesGas.prepareForBuild(&asMemReqs);
    customPrimitivesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGas);

    optixu::Instance customPrimitivesInst = scene.createInstance();
    customPrimitivesInst.setChild(customPrimitivesGas);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(roomInst);
    ias.addChild(customPrimitivesInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    roomGas.rebuild(cuStream, roomGasMem, asBuildScratchMem);
    customPrimitivesGas.rebuild(cuStream, customPrimitivesGasMem, asBuildScratchMem);

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
        { customPrimitivesGas, &customPrimitivesGasMem, 0, 0 },
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

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    customPrimitivesInst.destroy();
    roomInst.destroy();

    customPrimitivesGasMem.finalize();
    customPrimitivesGas.destroy();
    roomGasMem.finalize();
    roomGas.destroy();

    spheresParamBuffer.finalize();
    customPrimsAabbBuffer.finalize();
    customPrimsGeomInst.destroy();

    areaLightTriangleBuffer.finalize();
    areaLightVertexBuffer.finalize();
    areaLightGeomInst.destroy();

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
