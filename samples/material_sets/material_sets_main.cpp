/*

JP: 前の「single-level instancingサンプル」は複製されたGAS全てに同じマテリアルを使用していましたが、
    このサンプルでは同じGASを参照するそれぞれのインスタンスで異なるマテリアルを使用する方法を示します。
    GASの配下には複数のジオメトリインスタンスが所属し、そしてそれぞれに複数のマテリアルが設定されうるため
    これら複数マテリアルはひとつのGASのなかで「マテリアルセット」の概念をなします。
    そしてGASには複数のマテリアルセットを設定することができ、GASを参照するインスタンス単位でいずれかの
    マテリアルセットを指定します。ジオメトリインスタンスにマテリアルを設定する際、
    マテリアルのインデックスだけでなくもう一つのインデックス、「マテリアルセットインデックス」を使用します。
    これまでのサンプルではマテリアルセットインデックスに常にゼロを設定していましたが、
    このインデックスを変えることで同じマテリアルのインデックスに複数のマテリアルを設定することができます。

EN: The previous "single-level instancing" sample used the same material for all the replicated GASs,
    but this sample shows how to use different materials for each of instances referring the same GAS.
    Multiple geometry instances belong to a GAS and each of them can be set multiple materials,
    these multiple materials inside a GAS forms the concept of "material set".
    A GAS can have multiple material sets and every instance referring a GAS specifies any of material sets.
    We use a material index as well as another index, "material set index" when setting a material to
    a GeometryInstance. Previous samples always use zero as a material set index,
    multiple materials can be set to the same material index by changing the material set index.

*/

#include "material_sets_shared.h"

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

    // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
    // EN: This sample uses two-level AS (single-level instancing).
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "material_sets/ptxes/optix_kernels.optixir");
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

    optixu::Material ceilingMat = optixContext.createMaterial();
    ceilingMat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
    Shared::MaterialData ceilingMatData = {};
    ceilingMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    ceilingMat.setUserData(ceilingMatData);

    optixu::Material farSideWallMat = optixContext.createMaterial();
    farSideWallMat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
    Shared::MaterialData farSideWallMatData = {};
    farSideWallMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    farSideWallMat.setUserData(farSideWallMatData);

    optixu::Material leftWallMat = optixContext.createMaterial();
    leftWallMat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
    Shared::MaterialData leftWallMatData = {};
    leftWallMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    leftWallMat.setUserData(leftWallMatData);

    optixu::Material rightWallMat = optixContext.createMaterial();
    rightWallMat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
    Shared::MaterialData rightWallMatData = {};
    rightWallMatData.color = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    rightWallMat.setUserData(rightWallMatData);

    optixu::Material floorMat = optixContext.createMaterial();
    floorMat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
    Shared::MaterialData floorMatData = {};
    floorMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    floorMat.setUserData(floorMatData);

    constexpr uint32_t Ngon = 6;
    constexpr uint32_t NumPolygonInstances = 100;
    std::vector<std::array<optixu::Material, Ngon>> polygonMaterials(NumPolygonInstances);
    for (int i = 0; i < NumPolygonInstances; ++i) {
        for (int j = 0; j < Ngon; ++j) {
            optixu::Material mat = optixContext.createMaterial();
            mat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);
            polygonMaterials[i][j] = mat;
        }
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
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
            // far side wall
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

        roomVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        roomTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
        roomMatIndexBuffer.initialize(cuContext, cudau::BufferType::Device, matIndices, lengthof(matIndices));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = roomVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = roomTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        roomGeomInst.setVertexBuffer(roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(roomTriangleBuffer);
        roomGeomInst.setMaterialCount(5, roomMatIndexBuffer, optixu::IndexSize::k1Byte);
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

    optixu::GeometryInstance multiMatPolygonGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> multiMatPolygonVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> multiMatPolygonTriangleBuffer;
    cudau::TypedBuffer<uint8_t> multiMatPolygonMaterialIndexBuffer;
    {
        std::vector<Shared::Vertex> vertices(Ngon + 1);
        std::vector<Shared::Triangle> triangles(Ngon);
        std::vector<uint8_t> matIndices(Ngon);
        vertices[0] = Shared::Vertex{ float3(0, 0, 0), float3(0, 1, 0), float2(0, 0) };
        for (int i = 0; i < Ngon; ++i) {
            float angle = 2 * pi_v<float> * static_cast<float>(i) / Ngon;
            vertices[1 + i] = Shared::Vertex{
                float3(std::cos(angle), std::sin(angle), 0),
                float3(0, 1, 0),
                float2(0.5f + 0.5f * std::sin(angle), 0.5f + 0.5f * std::cos(angle))
            };
            triangles[i] = Shared::Triangle{ 0, static_cast<uint32_t>(1 + i), static_cast<uint32_t>(1 + (i + 1) % Ngon) };
            matIndices[i] = i;
        }

        multiMatPolygonVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        multiMatPolygonTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);
        multiMatPolygonMaterialIndexBuffer.initialize(cuContext, cudau::BufferType::Device, matIndices);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = multiMatPolygonVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = multiMatPolygonTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        multiMatPolygonGeomInst.setVertexBuffer(multiMatPolygonVertexBuffer);
        multiMatPolygonGeomInst.setTriangleBuffer(multiMatPolygonTriangleBuffer);
        multiMatPolygonGeomInst.setMaterialCount(Ngon, multiMatPolygonMaterialIndexBuffer, optixu::IndexSize::k1Byte);
        // JP: GASのインスタンスごとに異なるマテリアルを使用できるように
        //     各マテリアルセットのスロットにマテリアルをセットする。
        // EN: Set a material to each slot of material sets
        //     so that each GAS instance uses different material than others.
        for (int matSetIdx = 0; matSetIdx < NumPolygonInstances; ++matSetIdx) {
            // JP: 0-49のインスタンスは完全に独自のマテリアルを使用する。
            // EN: Each of instances 0-49 uses completely unique materials.
            if (matSetIdx < 50) {
                for (int i = 0; i < Ngon; ++i)
                    multiMatPolygonGeomInst.setMaterial(matSetIdx, i, polygonMaterials[matSetIdx][i]);
            }
            // JP: 50-74のインスタンスはマテリアルをセットしていないため、それぞれのマテリアルは
            //     マテリアルセット0のものにフォールバックされる。
            // EN: Each of instances 50-74 doesn't set any materials, every material falls back to
            //     the one of material set 0.
            else if (matSetIdx < 75) {
                //for (int i = 0; i < Ngon; ++i)
                //    multiMatPolygonGeomInst.setMaterial(matSetIdx, i, optixu::Material());
            }
            // JP: 75-99のインスタンスは半分のマテリアルが独自、もう半分がマテリアルセット0のものにフォールバックされる。
            // EN: Each of instances 75-99 uses unique materials for half of them,
            //     the others fall back to the one of material set 0.
            else {
                for (int i = 0; i < Ngon; i += 2)
                    multiMatPolygonGeomInst.setMaterial(matSetIdx, i, polygonMaterials[matSetIdx][i]);
            }
        }
        for (int i = 0; i < Ngon; ++i)
            multiMatPolygonGeomInst.setGeometryFlags(i, OPTIX_GEOMETRY_FLAG_NONE);
        multiMatPolygonGeomInst.setUserData(geomData);
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
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: GASのインスタンスごとに異なるマテリアルを使用できるようにマテリアルセットの数を設定する。
    // EN: Set a material set value so that each GAS uses different material than others.
    optixu::GeometryAccelerationStructure polygonGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer polygonGasMem;
    polygonGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
    polygonGas.setMaterialSetCount(NumPolygonInstances);
    polygonGas.addChild(multiMatPolygonGeomInst);
    for (int matSetIdx = 0; matSetIdx < NumPolygonInstances; ++matSetIdx)
        polygonGas.setRayTypeCount(matSetIdx, Shared::NumRayTypes);
    polygonGas.prepareForBuild(&asMemReqs);
    polygonGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGas);

    std::vector<optixu::Instance> polygonInsts;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumPolygonInstances; ++i) {
        float t = static_cast<float>(i) / (NumPolygonInstances - 1);
        float x = -0.9f + 1.8f * static_cast<float>(i % 10) / 9;
        float y = 0.9f - 1.8f * static_cast<float>(i / 10) / 9;

        for (int j = 0; j < Ngon; ++j) {
            Shared::MaterialData matData;
            matData.color = HSVtoRGB(
                static_cast<float>(j) / Ngon + (GoldenAngle * i) / (2 * pi_v<float>),
                1.0f, 1 - 0.9f * t);
            polygonMaterials[i][j].setUserData(matData);
        }

        float scale = 0.08f;
        float polygonInstXfm[] = {
            scale, 0, 0, x,
            0, scale, 0, y,
            0, 0, scale, 0
        };
        optixu::Instance polygonInst = scene.createInstance();
        // JP: インスタンスごとに異なるマテリアルセットを使用する。
        // EN: Use different material set per instance.
        polygonInst.setChild(polygonGas, i);
        polygonInst.setTransform(polygonInstXfm);
        polygonInsts.push_back(polygonInst);
    }



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(roomInst);
    for (int i = 0; i < polygonInsts.size(); ++i)
        ias.addChild(polygonInsts[i]);
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
    polygonGas.rebuild(cuStream, polygonGasMem, asBuildScratchMem);

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
        { polygonGas, &polygonGasMem, 0, 0 },
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
    plp.camera.position = make_float3(0, 0, 3.5);
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

    for (int i = polygonInsts.size() - 1; i >= 0; --i)
        polygonInsts[i].destroy();
    roomInst.destroy();

    polygonGasMem.finalize();
    polygonGas.destroy();
    roomGasMem.finalize();
    roomGas.destroy();

    multiMatPolygonMaterialIndexBuffer.finalize();
    multiMatPolygonTriangleBuffer.finalize();
    multiMatPolygonVertexBuffer.finalize();
    multiMatPolygonGeomInst.destroy();

    roomMatIndexBuffer.finalize();
    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

    scene.destroy();

    floorMat.destroy();
    rightWallMat.destroy();
    leftWallMat.destroy();
    farSideWallMat.destroy();
    ceilingMat.destroy();



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
