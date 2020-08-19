#include "single_gas_shared.h"

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

    // JP: このサンプルでは単一のGASのみを使用する。
    // EN: This sample uses only a single GAS.
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
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

    // JP: このヒットグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: This hit group is for ray-triangle intersection, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroup = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit0"),
        emptyModule, nullptr,
        emptyModule, nullptr);

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

        Matrix3x3 matSR = Matrix3x3();

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = roomVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = roomTriangleBuffer.getDevicePointer();
        geomData.matSR_N = transpose(inverse(matSR));

        roomGeomInst.setVertexBuffer(&roomVertexBuffer);
        roomGeomInst.setTriangleBuffer(&roomTriangleBuffer);
        roomGeomInst.setNumMaterials(1, nullptr);
        roomGeomInst.setMaterial(0, 0, mat0);
        roomGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
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
        areaLightGeomInst.setVertexBuffer(&areaLightVertexBuffer);
#if !defined(USE_TRIANGLE_SROUP_FOR_AREA_LIGHT)
        areaLightGeomInst.setTriangleBuffer(&areaLightTriangleBuffer);
#endif
        areaLightGeomInst.setNumMaterials(1, nullptr);
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
        loadObjFile("../../data/stanford_bunny_309_faces.obj", &vertices, &triangles);

        bunnyVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunnyTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Matrix3x3 matSR = rotateY3x3(M_PI / 4) * scale3x3(0.012f);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunnyVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = bunnyTriangleBuffer.getDevicePointer();
        geomData.matSR_N = transpose(inverse(matSR));

        bunnyGeomInst.setVertexBuffer(&bunnyVertexBuffer);
        bunnyGeomInst.setTriangleBuffer(&bunnyTriangleBuffer);
        bunnyGeomInst.setNumMaterials(1, nullptr);
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



    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    scene.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);

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

    shaderBindingTable.finalize();

    gasCompactedMem.finalize();
    asBuildScratchMem.finalize();
    gasMem.finalize();
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
