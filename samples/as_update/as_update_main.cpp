/*

JP: このサンプルはGAS, IASのアップデート方法を示します。
    一般的にAS(特にGAS)を作り直すことはそれなりに高コストであるため、頂点やインスタンスの
    変位量が小さなときはリビルドの代わりにアップデートで済ますほうが好ましい場合があります。

EN: This sample shows how to update GAS and IAS.
    Building an acceleration structure from scratch is costly in general so only updating is sometime
    preferrable instead of rebuild when the displacement of each vertex or instance is small.

*/

#include "as_update_shared.h"

#include "../common/obj_loader.h"

#include "../common/gui_common.h"



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "as_update";

    bool takeScreenShot = false;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot")
            takeScreenShot = true;
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }



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
    pipelineOptions.numPayloadValuesInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.numAttributeValuesInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(resourceDir / "ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::HitProgramGroup hitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1);

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



    // JP: 頂点変異・法線計算カーネルを準備する。
    // EN: Prepare kernels for vertex displacement and recalculate normals.
    CUmodule moduleDeform;
    CUDADRV_CHECK(cuModuleLoad(
        &moduleDeform, (resourceDir / "ptxes/deform.ptx").string().c_str()));
    cudau::Kernel deform(moduleDeform, "deform", cudau::dim3(32), 0);
    cudau::Kernel accumulateVertexNormals(moduleDeform, "accumulateVertexNormals", cudau::dim3(32), 0);
    cudau::Kernel normalizeVertexNormals(moduleDeform, "normalizeVertexNormals", cudau::dim3(32), 0);

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
        roomGeomInst.setNumMaterials(1, optixu::BufferView());
        roomGeomInst.setMaterial(0, 0, mat0);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
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
        geomData.vertexBuffer = areaLightVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = areaLightTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        areaLightGeomInst.setVertexBuffer(areaLightVertexBuffer);
        areaLightGeomInst.setTriangleBuffer(areaLightTriangleBuffer);
        areaLightGeomInst.setNumMaterials(1, optixu::BufferView());
        areaLightGeomInst.setMaterial(0, 0, mat0);
        areaLightGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLightGeomInst.setUserData(geomData);
    }

    optixu::GeometryInstance bunnyGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> bunnyVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> bunnyTriangleBuffer;
    cudau::TypedBuffer<Shared::Vertex> deformedBunnyVertexBuffer;
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

        bunnyVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunnyTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);
        deformedBunnyVertexBuffer = bunnyVertexBuffer.copy();

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = deformedBunnyVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = bunnyTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        bunnyGeomInst.setVertexBuffer(deformedBunnyVertexBuffer);
        bunnyGeomInst.setTriangleBuffer(bunnyTriangleBuffer);
        bunnyGeomInst.setNumMaterials(1, optixu::BufferView());
        bunnyGeomInst.setMaterial(0, 0, mat0);
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
    roomGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
    roomGas.setNumMaterialSets(1);
    roomGas.setNumRayTypes(0, Shared::NumRayTypes);
    roomGas.addChild(roomGeomInst);
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure areaLightGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer areaLightGasMem;
    areaLightGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
    areaLightGas.setNumMaterialSets(1);
    areaLightGas.setNumRayTypes(0, Shared::NumRayTypes);
    areaLightGas.addChild(areaLightGeomInst);
    areaLightGas.prepareForBuild(&asMemReqs);
    areaLightGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure bunnyGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer bunnyGasMem;
    // JP: update()を使用するためにアップデート可能に設定しておく。
    // EN: Make the AS updatable to use update().
    bunnyGas.setConfiguration(
        optixu::ASTradeoff::PreferFastBuild,
        optixu::AllowUpdate::Yes,
        optixu::AllowCompaction::Yes);
    bunnyGas.setNumMaterialSets(1);
    bunnyGas.setNumRayTypes(0, Shared::NumRayTypes);
    bunnyGas.addChild(bunnyGeomInst);
    bunnyGas.prepareForBuild(&asMemReqs);
    bunnyGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer =
        std::max(maxSizeOfScratchBuffer,
                 std::max(asMemReqs.tempSizeInBytes, asMemReqs.tempUpdateSizeInBytes));



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGas);

    float areaLightInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.75f,
        0, 0, 1, 0
    };
    optixu::Instance areaLightInst = scene.createInstance();
    areaLightInst.setChild(areaLightGas);
    areaLightInst.setTransform(areaLightInstXfm);

    struct Bunny {
        optixu::Instance inst;
        float scale_t;
        float scaleBase;
        float scaleFreq;
        float scaleAmp;
        float anglularPos_t;
        float angularPosFreq;
        float y_t;
        float yBase;
        float yFreq;
        float yAmp;

        void setTransform() {
            float scale = scaleBase + scaleAmp * std::sin(2 * pi_v<float> * (scale_t / scaleFreq));

            float angle = 2 * pi_v<float> * (anglularPos_t / angularPosFreq);
            float x = 0.7f * std::cos(angle);
            float z = 0.7f * std::sin(angle);

            float y = yBase + yAmp * std::sin(2 * pi_v<float> * (y_t / yFreq));

            float bunnyXfm[] = {
                scale, 0, 0, x,
                0, scale, 0, y,
                0, 0, scale, z,
            };
            inst.setTransform(bunnyXfm);
        }

        void initializeState(
            float initScale_t, float _scaleBase, float _scaleFreq, float _scaleAmp,
            float initAnglularPos_t, float _angularPosFreq,
            float initY_t, float _yBase, float _yFreq, float _yAmp) {
            scale_t = initScale_t;
            scaleBase = _scaleBase;
            scaleFreq = _scaleFreq;
            scaleAmp = _scaleAmp;
            anglularPos_t = initAnglularPos_t;
            angularPosFreq = _angularPosFreq;
            y_t = initY_t;
            yBase = _yBase;
            yFreq = _yFreq;
            yAmp = _yAmp;

            scale_t = std::fmod(scale_t, scaleFreq);
            anglularPos_t = std::fmod(anglularPos_t, angularPosFreq);
            y_t = std::fmod(y_t, yFreq);
            setTransform();
        }

        void update(float dt) {
            scale_t = std::fmod(scale_t + dt, scaleFreq);
            anglularPos_t = std::fmod(anglularPos_t + dt, angularPosFreq);
            y_t = std::fmod(y_t + dt, yFreq);
            setTransform();
        }
    };

    constexpr uint32_t NumBunnies = 3;
    std::vector<Bunny> bunnies(NumBunnies);
    for (int i = 0; i < NumBunnies; ++i) {
        Bunny bunny;
        bunny.inst = scene.createInstance();
        bunny.inst.setChild(bunnyGas);
        bunny.initializeState(
            i * 3.0f / NumBunnies, 0.005f, 3.0f, 0.001f,
            i * 5.0f / NumBunnies, 5.0f,
            i * 7.0f / NumBunnies, -0.25f, 7.0f, 0.25f);
        bunnies[i] = bunny;
    }



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    // JP: update()を使用するためにアップデート可能に設定しておく。
    // EN: Make the AS updatable to use update().
    ias.setConfiguration(
        optixu::ASTradeoff::PreferFastBuild,
        optixu::AllowUpdate::Yes);
    ias.addChild(roomInst);
    ias.addChild(areaLightInst);
    for (int i = 0; i < bunnies.size(); ++i)
        ias.addChild(bunnies[i].inst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer =
        std::max(maxSizeOfScratchBuffer,
                 std::max(asMemReqs.tempSizeInBytes, asMemReqs.tempUpdateSizeInBytes));



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    roomGas.rebuild(stream, roomGasMem, asBuildScratchMem);
    areaLightGas.rebuild(stream, areaLightGasMem, asBuildScratchMem);
    bunnyGas.rebuild(stream, bunnyGasMem, asBuildScratchMem);

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
        { areaLightGas, &areaLightGasMem, 0, 0 },
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
            stream,
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

    OptixTraversableHandle travHandle = ias.rebuild(stream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr int32_t initWindowContentWidth = 640;
    constexpr int32_t initWindowContentHeight = 640;

    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = (float)initWindowContentWidth / initWindowContentHeight;

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - AS Update";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = initWindowContentWidth;
    initConfig.windowContentRenderHeight = initWindowContentHeight;
    initConfig.cameraPosition = make_float3(0, 0, 3.2f);
    initConfig.cameraOrientation = qRotateY(pi_v<float>);
    initConfig.cameraMovingSpeed = 0.01f;
    initConfig.cuContext = cuContext;

    GUIFramework framework;
    framework.initialize(initConfig);

    cudau::Array outputArray;
    outputArray.initializeFromGLTexture2D(
        cuContext, framework.getOutputTexture().getHandle(),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputBufferSurfaceHolder.initialize({ &outputArray });

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;

        // Camera Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            ImGui::InputFloat3("Position", reinterpret_cast<float*>(&args.cameraPosition));
            static float rollPitchYaw[3];
            args.tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw))
                args.cameraOrientation = qFromEulerAngles(
                    rollPitchYaw[0] * pi_v<float> / 180,
                    rollPitchYaw[1] * pi_v<float> / 180,
                    rollPitchYaw[2] * pi_v<float> / 180);
            ImGui::Text("Pos. Speed (T/G): %g", args.cameraPositionalMovingSpeed);

            ImGui::End();
        }

        plp.camera.position = args.cameraPosition;
        plp.camera.orientation = args.tempCameraOrientation.toMatrix3x3();



        // JP: Bunnyの頂点に変異を加えてGASをアップデートする。
        //     法線ベクトルも修正する。
        // EN: Deform bunnys' vertices and update its GAS.
        //     Modify normal vectors as well.
        {
            float t = 0.5f + 0.5f * std::sin(2 * pi_v<float> *static_cast<float>(frameIndex % 180) / 180);
            deform.launchWithThreadDim(
                curStream, cudau::dim3(bunnyVertexBuffer.numElements()),
                bunnyVertexBuffer, deformedBunnyVertexBuffer,
                bunnyVertexBuffer.numElements(), 20.0f, t);
            accumulateVertexNormals.launchWithThreadDim(
                curStream, cudau::dim3(bunnyTriangleBuffer.numElements()),
                deformedBunnyVertexBuffer, bunnyTriangleBuffer,
                bunnyTriangleBuffer.numElements());
            normalizeVertexNormals.launchWithThreadDim(
                curStream, cudau::dim3(bunnyVertexBuffer.numElements()),
                deformedBunnyVertexBuffer,
                bunnyVertexBuffer.numElements());
            bunnyGas.update(curStream, asBuildScratchMem);
        }

        // JP: 各インスタンスのトランスフォームを更新する。
        // EN: Update the transform of each instance.
        for (int i = 0; i < bunnies.size(); ++i)
            bunnies[i].update(1.0f / 60.0f);

        /*
        JP: IASのアップデートを行う。
            品質を維持するためにたまにはリビルドする。
            アップデートの代用としてのリビルドでは、インスタンスの追加・削除や
            ASビルド設定の変更を行っていないのでmarkDirty()やprepareForBuild()は必要無い。
        EN: Update the IAS.
            Sometimes perform rebuild to maintain AS quality.
            Rebuild as the alternative for update doesn't involves
            add/remove of instances and changes of AS build settings
            so neither of markDirty() nor prepareForBuild() is required.
        */
        if (frameIndex % 10 == 0)
            plp.travHandle = ias.rebuild(curStream, instanceBuffer, iasMem, asBuildScratchMem);
        else
            ias.update(curStream, asBuildScratchMem);

        // Render
        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        plp.resultBuffer = outputBufferSurfaceHolder.getNext();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
        pipeline.launch(
            curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = false;
        ret.finish = false;

        if (takeScreenShot && frameIndex + 1 == 60) {
            CUDADRV_CHECK(cuStreamSynchronize(curStream));
            const uint32_t numPixels = args.windowContentRenderWidth * args.windowContentRenderHeight;
            auto rawImage = new float4[numPixels];
            glGetTextureSubImage(
                args.outputTexture->getHandle(), 0,
                0, 0, 0, args.windowContentRenderWidth, args.windowContentRenderHeight, 1,
                GL_RGBA, GL_FLOAT,
                sizeof(float4) * numPixels, rawImage);
            saveImage("output.png", args.windowContentRenderWidth, args.windowContentRenderHeight, rawImage,
                      false, false);
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

         // EN: update the pipeline parameters.
         plp.imageSize = int2(windowContentWidth, windowContentHeight);
         plp.camera.aspect = (float)windowContentWidth / windowContentHeight;
    };

    framework.run(onRenderLoop, onResolutionChange);

    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();

    framework.finalize();

    // END: Display the window.
    // ----------------------------------------------------------------



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    for (int i = bunnies.size() - 1; i >= 0; --i)
        bunnies[i].inst.destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    bunnyGasMem.finalize();
    bunnyGas.destroy();
    areaLightGasMem.finalize();
    areaLightGas.destroy();
    roomGasMem.finalize();
    roomGas.destroy();

    deformedBunnyVertexBuffer.finalize();
    bunnyTriangleBuffer.finalize();
    bunnyVertexBuffer.finalize();
    bunnyGeomInst.destroy();

    areaLightTriangleBuffer.finalize();
    areaLightVertexBuffer.finalize();
    areaLightGeomInst.destroy();

    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

    scene.destroy();

    mat0.destroy();



    CUDADRV_CHECK(cuModuleUnload(moduleDeform));

    shaderBindingTable.finalize();

    hitProgramGroup.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

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
