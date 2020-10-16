/*

JP: このサンプルはデフォーメーション(変形)ブラーを扱うGASを構築する方法を示します。
    複数のモーションステップに対応する頂点(もしくはAABB)バッファーをGeometryInstanceに設定し
    GASに適切なモーション設定を行うことでデフォーメーションブラーに対応するGASを構築できます。
EN: This sample shows how to build a GAS to handle deformation blur.
    Set a vertex (or AABB) buffer to each of multiple motion steps of GeometryInstance and
    set appropriate motion configuration to a GAS to build a GAS capable of deformation blur.
*/

// TODO: Demonstrate custom primitive deformation blur.

#include "deformation_blur_shared.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"
#include "../../ext/tiny_obj_loader.h"

static void loadObj(const std::string &filepath,
                    std::vector<Shared::Vertex>* vertices,
                    std::vector<Shared::Triangle>* triangles);

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
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                true, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "deformation_blur/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: このグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: This group is for ray-triangle intersection, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroup = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        emptyModule, nullptr);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(getView(shaderBindingTable), shaderBindingTable.getMappedPointer());

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material mat0 = optixContext.createMaterial();
    mat0.setHitGroup(Shared::RayType_Primary, hitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    // JP: このサンプルではデフォーメーションブラーに焦点を当て、
    //     ほかをシンプルにするために1つのGASあたり1つのGeometryInstanceとする。
    // EN: Use one GeometryInstance per GAS for simplicty and
    //     to focus on deformation blur in this sample.
    struct Geometry {
        std::vector<cudau::TypedBuffer<Shared::Vertex>> vertexBuffers;
        cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        optixu::GeometryInstance optixGeomInst;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            triangleBuffer.finalize();
            for (int i = vertexBuffers.size() - 1; i >= 0; --i)
                vertexBuffers[i].finalize();
            optixGeomInst.destroy();
        }
    };

    Geometry room;
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

        room.vertexBuffers.resize(1);
        room.vertexBuffers[0].initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        room.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffers[0] = room.vertexBuffers[0].getDevicePointer();
        geomData.triangleBuffer = room.triangleBuffer.getDevicePointer();
        geomData.numMotionSteps = 1;

        room.optixGeomInst = scene.createGeometryInstance();
        room.optixGeomInst.setVertexBuffer(getView(room.vertexBuffers[0]));
        room.optixGeomInst.setTriangleBuffer(getView(room.triangleBuffer));
        room.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        room.optixGeomInst.setMaterial(0, 0, mat0);
        room.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setUserData(geomData);

        room.optixGas = scene.createGeometryAccelerationStructure();
        room.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        room.optixGas.setNumMaterialSets(1);
        room.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
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

        areaLight.vertexBuffers.resize(1);
        areaLight.vertexBuffers[0].initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLight.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffers[0] = areaLight.vertexBuffers[0].getDevicePointer();
        geomData.triangleBuffer = areaLight.triangleBuffer.getDevicePointer();
        geomData.numMotionSteps = 1;

        areaLight.optixGeomInst = scene.createGeometryInstance();
        areaLight.optixGeomInst.setVertexBuffer(getView(areaLight.vertexBuffers[0]));
        areaLight.optixGeomInst.setTriangleBuffer(getView(areaLight.triangleBuffer));
        areaLight.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        areaLight.optixGeomInst.setMaterial(0, 0, mat0);
        areaLight.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLight.optixGeomInst.setUserData(geomData);

        areaLight.optixGas = scene.createGeometryAccelerationStructure();
        areaLight.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        areaLight.optixGas.setNumMaterialSets(1);
        areaLight.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        areaLight.optixGas.addChild(areaLight.optixGeomInst);
        areaLight.optixGas.prepareForBuild(&asMemReqs);
        areaLight.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry bunny;
    {
        std::vector<Shared::Vertex> vertices;
        std::vector<Shared::Triangle> triangles;
        loadObj("../../data/stanford_bunny_309_faces.obj",
                &vertices, &triangles);

        // JP: 頂点バッファーを2ステップ分作る。
        //     2ステップ目は頂点位置を爆発するようにずらす。
        // EN: Create vertex buffer for two steps.
        //     The second step displaces the positions of vertices like explosion.
        uint32_t numMotionSteps = 2;
        bunny.vertexBuffers.resize(numMotionSteps);
        bunny.vertexBuffers[0].initialize(cuContext, cudau::BufferType::Device, vertices);
        for (int i = 0; i < vertices.size(); ++i) {
            Shared::Vertex &v = vertices[i];
            v.position = v.position + v.normal * length(v.position - float3(0, 0, 42)) * 0.25f;
        }
        bunny.vertexBuffers[1].initialize(cuContext, cudau::BufferType::Device, vertices);
        bunny.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        for (int i = 0; i < numMotionSteps; ++i)
            geomData.vertexBuffers[i] = bunny.vertexBuffers[i].getDevicePointer();
        geomData.triangleBuffer = bunny.triangleBuffer.getDevicePointer();
        geomData.numMotionSteps = numMotionSteps;

        bunny.optixGeomInst = scene.createGeometryInstance();
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        bunny.optixGeomInst.setNumMotionSteps(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            bunny.optixGeomInst.setVertexBuffer(getView(bunny.vertexBuffers[i]), i);
        bunny.optixGeomInst.setTriangleBuffer(getView(bunny.triangleBuffer));
        bunny.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        bunny.optixGeomInst.setMaterial(0, 0, mat0);
        bunny.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunny.optixGeomInst.setUserData(geomData);

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        bunny.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        bunny.optixGas.setNumMaterialSets(1);
        bunny.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        bunny.optixGas.addChild(bunny.optixGeomInst);
        bunny.optixGas.prepareForBuild(&asMemReqs);
        bunny.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    room.optixGas.rebuild(cuStream, getView(room.gasMem), getView(asBuildScratchMem));
    areaLight.optixGas.rebuild(cuStream, getView(areaLight.gasMem), getView(asBuildScratchMem));
    bunny.optixGas.rebuild(cuStream, getView(bunny.gasMem), getView(asBuildScratchMem));

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        optixu::GeometryAccelerationStructure gas;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { room.optixGas, 0, 0 },
        { areaLight.optixGas, 0, 0 },
        { bunny.optixGas, 0, 0 },
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
        info.gas.compact(cuStream, optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset,
                                                      info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i)
        gasList[i].gas.removeUncompacted();



    // JP: IAS作成前には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     before creating an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);



    // JP: インスタンスを作成する。
    // EN: Create instances.
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

    Matrix3x3 bunnyMatSR = rotateY3x3(M_PI / 4) * scale3x3(0.012f);
    float bunnyInstXfm[] = {
        bunnyMatSR.m00, bunnyMatSR.m01, bunnyMatSR.m02, 0,
        bunnyMatSR.m10, bunnyMatSR.m11, bunnyMatSR.m12, -0.5f,
        bunnyMatSR.m20, bunnyMatSR.m21, bunnyMatSR.m22, 0
    };
    optixu::Instance bunnyInst = scene.createInstance();
    bunnyInst.setChild(bunny.optixGas);
    bunnyInst.setTransform(bunnyInstXfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    constexpr uint32_t numMotionKeys = 3;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    ias.addChild(bunnyInst);
    ias.prepareForBuild(&asMemReqs, &numInstances);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, numInstances);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    if (maxSizeOfScratchBuffer > asBuildScratchMem.sizeInBytes())
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, getView(instanceBuffer), getView(iasMem), getView(asBuildScratchMem));

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr uint32_t renderTargetSizeX = 1024;
    constexpr uint32_t renderTargetSizeY = 1024;
    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 4> rngBuffer;
    rngBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);
    {
        std::mt19937 rng(50932423);

        rngBuffer.map();
        for (int y = 0; y < renderTargetSizeY; ++y) {
            for (int x = 0; x < renderTargetSizeX; ++x) {
                rngBuffer(x, y).setState(rng());
            }
        }
        rngBuffer.unmap();
    }

    optixu::HostBlockBuffer2D<float4, 1> accumBuffer;
    accumBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.accumBuffer = accumBuffer.getBlockBuffer2D();
    plp.timeBegin = 0.0f;
    plp.timeEnd = 1.0f;
    plp.numAccumFrames = 0;
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5f);
    plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(getView(hitGroupSBT), hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));

    for (int frameIndex = 0; frameIndex < 1024; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
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



    rngBuffer.finalize();
    accumBuffer.finalize();



    compactedASMem.finalize();
    
    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    hitGroupSBT.finalize();

    bunnyInst.destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    bunny.finalize();
    areaLight.finalize();
    room.finalize();

    scene.destroy();

    mat0.destroy();



    shaderBindingTable.finalize();

    hitProgramGroup.destroy();

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

void loadObj(const std::string &filepath,
             std::vector<Shared::Vertex>* vertices,
             std::vector<Shared::Triangle>* triangles) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filepath.c_str());

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
        (*vertices)[vertexIndex] = kv.second;
        vertexIndices[kv.first] = vertexIndex++;
    }
    unifiedVertexMap.clear();

    // Calculate triangle index buffer.
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
        const Shared::Triangle &tri = (*triangles)[tIdx];
        Shared::Vertex &v0 = (*vertices)[tri.index0];
        Shared::Vertex &v1 = (*vertices)[tri.index1];
        Shared::Vertex &v2 = (*vertices)[tri.index2];
        float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
        v0.normal += gn;
        v1.normal += gn;
        v2.normal += gn;
    }
    for (int vIdx = 0; vIdx < vertices->size(); ++vIdx) {
        Shared::Vertex &v = (*vertices)[vIdx];
        v.normal = normalize(v.normal);
    }
}
