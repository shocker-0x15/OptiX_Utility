/*

JP: このサンプルはGASを参照する複数のインスタンスからInstance Acceleration Structure (IAS)
    を構築する方法を示します。それぞれのインスタンスにはトランスフォームを設定することが可能で、
    複数のインスタンスから同一のGASを参照することができるため、
    シーン中にジオメトリを省メモリかつ大量に複製することができます。
    また、インスタンスのトランスフォームを更新した場合もIASの再構築またはアップデートだけ行えば良いので
    動的なシーンを低負荷に扱うことができます。

EN: This sample shows how to build an instance acceleration structure (IAS) from multiple instances where
    each of them refers a GAS. Each instance can set a transform and it is possible to refer the same GAS from
    multiple instances, allowing to replicate a lot of geometry in a scene without much memory.
    Additionally, only rebuilding or updating an IAS is required when updating an instance's transform so
    you can handle dynamic scene with low cost.

*/

#include "single_level_instancing_shared.h"

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
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "single_level_instancing/ptxes/optix_kernels.ptx");
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
            { 4, 5, 6 }, { 4, 6, 7 },
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

        roomGeomInst.setVertexBuffer(getView(roomVertexBuffer));
        roomGeomInst.setTriangleBuffer(getView(roomTriangleBuffer));
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
        geomData.vertexBuffer = areaLightVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = areaLightTriangleBuffer.getDevicePointer();

        areaLightGeomInst.setVertexBuffer(getView(areaLightVertexBuffer));
        areaLightGeomInst.setTriangleBuffer(getView(areaLightTriangleBuffer));
        areaLightGeomInst.setNumMaterials(1, optixu::BufferView());
        areaLightGeomInst.setMaterial(0, 0, mat0);
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

        bunnyGeomInst.setVertexBuffer(getView(bunnyVertexBuffer));
        bunnyGeomInst.setTriangleBuffer(getView(bunnyTriangleBuffer));
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
    roomGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    roomGas.setNumMaterialSets(1);
    roomGas.setNumRayTypes(0, Shared::NumRayTypes);
    roomGas.addChild(roomGeomInst);
    roomGas.prepareForBuild(&asMemReqs);
    roomGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure areaLightGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer areaLightGasMem;
    areaLightGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    areaLightGas.setNumMaterialSets(1);
    areaLightGas.setNumRayTypes(0, Shared::NumRayTypes);
    areaLightGas.addChild(areaLightGeomInst);
    areaLightGas.prepareForBuild(&asMemReqs);
    areaLightGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure bunnyGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer bunnyGasMem;
    bunnyGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    bunnyGas.setNumMaterialSets(1);
    bunnyGas.setNumRayTypes(0, Shared::NumRayTypes);
    bunnyGas.addChild(bunnyGeomInst);
    bunnyGas.prepareForBuild(&asMemReqs);
    bunnyGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    roomGas.rebuild(cuStream, getView(roomGasMem), getView(asBuildScratchMem));
    areaLightGas.rebuild(cuStream, getView(areaLightGasMem), getView(asBuildScratchMem));
    bunnyGas.rebuild(cuStream, getView(bunnyGasMem), getView(asBuildScratchMem));

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
        { roomGas, 0, 0 },
        { areaLightGas, 0, 0 },
        { bunnyGas, 0, 0 }
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

    constexpr uint32_t NumBunnies = 100;
    std::vector<optixu::Instance> bunnyInsts(NumBunnies);
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * M_PI / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.003f + tt * 0.0006f;
        float bunnyInstXfm[] = {
            scale, 0, 0, x,
            0, scale, 0, -0.999f + (1 - tt),
            0, 0, scale, z
        };
        optixu::Instance bunnyInst = scene.createInstance();
        bunnyInst.setChild(bunnyGas);
        bunnyInst.setTransform(bunnyInstXfm);
        bunnyInsts[i] = bunnyInst;
    }



    // JP: IAS作成時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when creating an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);



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

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, getView(instanceBuffer), getView(iasMem), getView(asBuildScratchMem));

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
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5);
    plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(getView(hitGroupSBT), hitGroupSBT.getMappedPointer());

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



    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    hitGroupSBT.finalize();

    for (int i = bunnyInsts.size() - 1; i >= 0; --i)
        bunnyInsts[i].destroy();
    areaLightInst.destroy();
    roomInst.destroy();

    compactedASMem.finalize();
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

    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

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
