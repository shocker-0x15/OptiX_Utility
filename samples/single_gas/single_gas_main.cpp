/*

JP: このサンプルはOptiXがサポートするグラフのなかで最もシンプルな構成である
    単一のGeometry Acceleration Strucutre (GAS)の構築方法を示します。
    ひとつのGASは複数のジオメトリ(とそれぞれの静的なトランスフォーム)から構築されます。
    
EN: This sample shows how to build a single geometry acceleration structure (GAS)
    which is the simplest graph configuration the OptiX supports.
    A GAS builds from multiple geometries (and their static transforms).

*/

#include "single_gas_shared.h"

#include "../common/obj_loader.h"

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

    // JP: このサンプルでは単一のGASのみを使用する。
    //     カーネル中で使用しているアトリビュートサイズは2Dwords(三角形の重心座標 float2)。
    // EN: This sample uses only a single GAS.
    //     The attribute size used by the kernel is 2 Dwords (triangle barycentrics float2).
    pipeline.setPipelineOptions(optixu::calcSumDwords<PayloadSignature>(),
                                optixu::calcSumDwords<float2>(),
                                "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "single_gas/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen0"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss0"));

    optixu::ProgramGroup hitProgramGroup = pipeline.createHitProgramGroupForBuiltinIS(
        OPTIX_PRIMITIVE_TYPE_TRIANGLE,
        moduleOptiX, RT_CH_NAME_STR("closesthit0"),
        emptyModule, nullptr);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    // JP: シーンに依存しないシェーダーバインディングテーブルの確保。
    //     OptiX UtilityのAPIにCUDA UtilityのBufferを直接渡しているように見えるが、
    //     実際には暗黙的な変換がかかっていることに注意(optixu_on_cudau.h 参照)。
    //     OptiX UtilityはCUDA Utilityには直接依存しない。
    // EN: Allocate the shader binding table which doesn't depend on a scene.
    //     It appears directly passing Buffer of CUDA Utility to OptiX Utility API
    //     but note that there is actually implicit conversion (see optixu_on_cudau.h).
    //     OptiX Utility is not directly dependent on CUDA Utility.
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

    optixu::Material mat0 = optixContext.createMaterial();
    mat0.setHitGroup(Shared::RayType_Primary, hitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    cudau::TypedBuffer<Shared::GeometryPreTransform> preTransformBuffer;
    preTransformBuffer.initialize(cuContext, cudau::BufferType::Device, 3);
    Shared::GeometryPreTransform* preTransforms = preTransformBuffer.map();

    uint32_t geomInstIndex = 0;

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

        Matrix3x3 matSR = Matrix3x3();

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = roomVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = roomTriangleBuffer.getDevicePointer();
        geomData.matSR_N = transpose(inverse(matSR));

        // JP: GeometryInstanceに頂点バッファーと三角形(インデックス)バッファーを渡す。
        //     GeometryInstanceはバッファーの参照を持つだけなので一時変数のバッファーを渡したり
        //     Acceleration Structureのビルド時に解放されていないように注意。
        //     OptiX UtilityのAPIにCUDA UtilityのBufferを直接渡しているように見えるが、
        //     実際には暗黙的な変換がかかっていることに注意(optixu_on_cudau.h 参照)。
        // EN: Pass the vertex buffer and triangle (index) buffer to the GeometryInstance.
        //     Note that GeometryInstance just takes a reference to a buffer and doesn't hold it,
        //     so do not pass a buffer of temporary variable or be careful so that the buffer is not
        //     released when building an acceleration structure.
        //     It appears directly passing Buffer of CUDA Utility to OptiX Utility API
        //     but note that there is actually implicit conversion (see optixu_on_cudau.h).
        roomGeomInst.setVertexBuffer(roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(roomTriangleBuffer);
        roomGeomInst.setNumMaterials(1, optixu::BufferView());
        roomGeomInst.setMaterial(0, 0, mat0);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        // JP: GeometryInstanceに設定したユーザーデータはGPUカーネル内で参照できる。
        // EN: The user data set to GeometryInstance can be accessed in a GPU kernel.
        roomGeomInst.setUserData(geomData);

        preTransforms[geomInstIndex] = Shared::GeometryPreTransform(matSR, make_float3(0.0f, 0.0f, 0.0f));

        ++geomInstIndex;
    }

    optixu::GeometryInstance areaLightGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> areaLightVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> areaLightTriangleBuffer;
    {
#define USE_TRIANGLE_SROUP_FOR_AREA_LIGHT

#if defined(USE_TRIANGLE_SROUP_FOR_AREA_LIGHT)
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.0f,  0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3( 0.25f, 0.0f,  0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3( 0.25f, 0.0f,  0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3( 0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        areaLightVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
#else
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.0f,  0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3( 0.25f, 0.0f,  0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3( 0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        areaLightVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLightTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
#endif

        Matrix3x3 matSR = Matrix3x3();

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLightVertexBuffer.getDevicePointer();
#if !defined(USE_TRIANGLE_SROUP_FOR_AREA_LIGHT)
        geomData.triangleBuffer = areaLightTriangleBuffer.getDevicePointer();
#endif
        geomData.matSR_N = transpose(inverse(matSR));

        // JP: インデックスバッファーを設定しない場合はトライアングルスープとして取り扱われる。
        // EN: It will be interpreted as triangle soup if not setting an index buffer.
        areaLightGeomInst.setVertexBuffer(areaLightVertexBuffer);
#if !defined(USE_TRIANGLE_SROUP_FOR_AREA_LIGHT)
        areaLightGeomInst.setTriangleBuffer(&areaLightTriangleBuffer);
#endif
        areaLightGeomInst.setNumMaterials(1, optixu::BufferView());
        areaLightGeomInst.setMaterial(0, 0, mat0);
        areaLightGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLightGeomInst.setUserData(geomData);

        preTransforms[geomInstIndex] = Shared::GeometryPreTransform(matSR, make_float3(0.0f, 0.75f, 0.0f));

        ++geomInstIndex;
    }

    optixu::GeometryInstance bunnyGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> bunnyVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> bunnyTriangleBuffer;
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
                        triangles.size(),
                        triangles.data());
        }

        bunnyVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunnyTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Matrix3x3 matSR = rotateY3x3(M_PI / 4) * scale3x3(0.012f);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunnyVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = bunnyTriangleBuffer.getDevicePointer();
        geomData.matSR_N = transpose(inverse(matSR));

        bunnyGeomInst.setVertexBuffer(bunnyVertexBuffer);
        bunnyGeomInst.setTriangleBuffer(bunnyTriangleBuffer);
        bunnyGeomInst.setNumMaterials(1, optixu::BufferView());
        bunnyGeomInst.setMaterial(0, 0, mat0);
        bunnyGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunnyGeomInst.setUserData(geomData);

        preTransforms[geomInstIndex] = Shared::GeometryPreTransform(matSR, make_float3(0.0f, -1.0f, 0.0f));

        ++geomInstIndex;
    }

    preTransformBuffer.unmap();



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create an geometry acceleration structure.
    optixu::GeometryAccelerationStructure gas = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasMem;
    cudau::Buffer gasCompactedMem;
    gas.setConfiguration(optixu::ASTradeoff::Default, false, true, false);
    gas.setNumMaterialSets(1);
    gas.setNumRayTypes(0, Shared::NumRayTypes);
    gas.addChild(roomGeomInst/*, preTransformBuffer.getCUdeviceptrAt(0)*/); // Identity transform can be ommited.
    // JP: GASにGeometryInstanceを追加するときに追加の静的Transformを指定できる。
    //     指定されたTransformを用いてAcceleration Structureが作られる。
    //     ただしカーネル内でユーザー自身が与えるジオメトリ情報には変換がかかっていないことには注意する必要がある。
    // EN: It is possible to specify an additional static transform when adding a GeometryInstance to a GAS.
    //     Acceleration structure is built using the specified transform.
    //     Note that geometry that given by the user in a kernel is not transformed.
    gas.addChild(areaLightGeomInst, preTransformBuffer.getCUdeviceptrAt(1));
    gas.addChild(bunnyGeomInst, preTransformBuffer.getCUdeviceptrAt(2));
    gas.prepareForBuild(&asMemReqs);
    gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    OptixTraversableHandle travHandle = gas.rebuild(cuStream, gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    // EN: Perform compaction for static meshes.
    size_t compactedASSize;
    gas.prepareForCompact(&compactedASSize);
    gasCompactedMem.initialize(cuContext, cudau::BufferType::Device, compactedASSize, 1);
    travHandle = gas.compact(cuStream, gasCompactedMem);
    gas.removeUncompacted();
    gasMem.finalize();



    // JP: シーンのシェーダーバインディングテーブルの確保。
    // EN: Allocate the shader binding table for the scene.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

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

    gasCompactedMem.finalize();
    asBuildScratchMem.finalize();
    gas.destroy();

    bunnyTriangleBuffer.finalize();
    bunnyVertexBuffer.finalize();
    bunnyGeomInst.destroy();
    
    areaLightTriangleBuffer.finalize();
    areaLightVertexBuffer.finalize();
    areaLightGeomInst.destroy();

    roomTriangleBuffer.finalize();
    roomVertexBuffer.finalize();
    roomGeomInst.destroy();

    preTransformBuffer.finalize();

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
