#include "multi_level_instancing_shared.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../ext/stb_image_write.h"
#include "../ext/tiny_obj_loader.h"

int32_t mainFunc(int32_t argc, const char* argv[]) {
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

    // JP: このサンプルでは多段階のAS、トランスフォームを使用する。
    // EN: This sample uses two-level AS (single-level instancing).
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                true, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "multi_level_instancing/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: これらのグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: These are for ray-triangle hit groups, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroup0 = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit0"),
        emptyModule, nullptr,
        emptyModule, nullptr);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.setMaxTraceDepth(1);
    pipeline.link(DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE),
                  false);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);

    OptixStackSizes stackSizes;

    rayGenProgram.getStackSize(&stackSizes);
    uint32_t cssRG = stackSizes.cssRG;

    missProgram.getStackSize(&stackSizes);
    uint32_t cssMS = stackSizes.cssMS;

    hitProgramGroup0.getStackSize(&stackSizes);
    uint32_t cssCH = stackSizes.cssCH;

    uint32_t dcStackSizeFromTrav = 0; // This sample doesn't call a direct callable during traversal.
    uint32_t dcStackSizeFromState = 0;
    // Possible Program Paths:
    // RG - CH
    // RG - MS
    uint32_t ccStackSize = cssRG + std::max(cssCH, cssMS);
    pipeline.setStackSize(dcStackSizeFromTrav, dcStackSizeFromState, ccStackSize, 3);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material mat0 = optixContext.createMaterial();
    mat0.setHitGroup(Shared::RayType_Primary, hitProgramGroup0);

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

    optixu::GeometryInstance geomInstBox = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferBox;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferBox;
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

        vertexBufferBox.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferBox.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInstBox.setVertexBuffer(&vertexBufferBox);
        geomInstBox.setTriangleBuffer(&triangleBufferBox);
        geomInstBox.setNumMaterials(1, nullptr);
        geomInstBox.setMaterial(0, 0, mat0);
        geomInstBox.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstBox.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferBox.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferBox.getDevicePointer();

        ++geomInstIndex;
    }

    optixu::GeometryInstance geomInstAreaLight = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferAreaLight;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferAreaLight;
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

        vertexBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInstAreaLight.setVertexBuffer(&vertexBufferAreaLight);
        geomInstAreaLight.setTriangleBuffer(&triangleBufferAreaLight);
        geomInstAreaLight.setNumMaterials(1, nullptr);
        geomInstAreaLight.setMaterial(0, 0, mat0);
        geomInstAreaLight.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstAreaLight.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferAreaLight.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferAreaLight.getDevicePointer();

        ++geomInstIndex;
    }

    optixu::GeometryInstance geomInstBunny = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferBunny;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferBunny;
    float3 bboxMinBunny = make_float3(INFINITY);
    float3 bboxMaxBunny  = make_float3(-INFINITY);
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "../data/stanford_bunny_309_faces.obj");

        constexpr float scale = 1.0f;

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
                        make_float3(scale * attrib.vertices[3 * idx.vertex_index + 0],
                        scale * attrib.vertices[3 * idx.vertex_index + 1],
                        scale * attrib.vertices[3 * idx.vertex_index + 2]),
                        make_float3(0, 0, 0),
                        make_float2(0, 0)
                    };
                }

                idxOffset += numFaceVertices;
            }
        }

        // Assign a vertex index to each of unified unique unifiedVertexMap.
        std::map<std::tuple<int32_t, int32_t>, uint32_t> vertexIndices;
        std::vector<Shared::Vertex> vertices(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            vertices[vertexIndex] = kv.second;
            vertexIndices[kv.first] = vertexIndex++;
            float3 p = kv.second.position;
            bboxMinBunny = min(bboxMinBunny, p);
            bboxMaxBunny = max(bboxMaxBunny, p);
        }
        unifiedVertexMap.clear();

        // Calculate triangle index buffer.
        std::vector<Shared::Triangle> triangles;
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

                triangles.push_back(Shared::Triangle{
                    vertexIndices.at(key0),
                    vertexIndices.at(key1),
                    vertexIndices.at(key2) });

                idxOffset += numFaceVertices;
            }
        }
        vertexIndices.clear();

        for (int tIdx = 0; tIdx < triangles.size(); ++tIdx) {
            const Shared::Triangle &tri = triangles[tIdx];
            Shared::Vertex &v0 = vertices[tri.index0];
            Shared::Vertex &v1 = vertices[tri.index1];
            Shared::Vertex &v2 = vertices[tri.index2];
            float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
            v0.normal += gn;
            v1.normal += gn;
            v2.normal += gn;
        }
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            Shared::Vertex &v = vertices[vIdx];
            v.normal = normalize(v.normal);
        }

        vertexBufferBunny.initialize(cuContext, cudau::BufferType::Device, vertices);
        triangleBufferBunny.initialize(cuContext, cudau::BufferType::Device, triangles);

        geomInstBunny.setVertexBuffer(&vertexBufferBunny);
        geomInstBunny.setTriangleBuffer(&triangleBufferBunny);
        geomInstBunny.setNumMaterials(1, nullptr);
        geomInstBunny.setMaterial(0, 0, mat0);
        geomInstBunny.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstBunny.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferBunny.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferBunny.getDevicePointer();

        ++geomInstIndex;
    }

    geomDataBuffer.unmap();



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure gasBox = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasBoxMem;
    cudau::Buffer gasBoxCompactedMem;
    gasBox.setConfiguration(true, false, true, false);
    gasBox.setNumMaterialSets(1);
    gasBox.setNumRayTypes(0, Shared::NumRayTypes);
    gasBox.addChild(geomInstBox);
    gasBox.prepareForBuild(&asMemReqs);
    gasBoxMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure gasAreaLight = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasAreaLightMem;
    cudau::Buffer gasAreaLightCompactedMem;
    gasAreaLight.setConfiguration(true, false, true, false);
    gasAreaLight.setNumMaterialSets(1);
    gasAreaLight.setNumRayTypes(0, Shared::NumRayTypes);
    gasAreaLight.addChild(geomInstAreaLight);
    gasAreaLight.prepareForBuild(&asMemReqs);
    gasAreaLightMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure gasBunny = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasBunnyMem;
    cudau::Buffer gasBunnyCompactedMem;
    gasBunny.setConfiguration(true, false, true, false);
    gasBunny.setNumMaterialSets(1);
    gasBunny.setNumRayTypes(0, Shared::NumRayTypes);
    gasBunny.addChild(geomInstBunny);
    gasBunny.prepareForBuild(&asMemReqs);
    gasBunnyMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    gasBox.rebuild(cuStream, gasBoxMem, asBuildScratchMem);
    gasAreaLight.rebuild(cuStream, gasAreaLightMem, asBuildScratchMem);
    gasBunny.rebuild(cuStream, gasBunnyMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    // EN: Perform compaction for static meshes.
    size_t gasBoxCompactedSize;
    gasBox.prepareForCompact(&gasBoxCompactedSize);
    gasBoxCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasBoxCompactedSize, 1);
    size_t gasAreaLightCompactedSize;
    gasAreaLight.prepareForCompact(&gasAreaLightCompactedSize);
    gasAreaLightCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasAreaLightCompactedSize, 1);
    size_t gasBunnyCompactedSize;
    gasBunny.prepareForCompact(&gasBunnyCompactedSize);
    gasBunnyCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasBunnyCompactedSize, 1);

    gasBox.compact(cuStream, gasBoxCompactedMem);
    gasBox.removeUncompacted();
    gasAreaLight.compact(cuStream, gasAreaLightCompactedMem);
    gasAreaLight.removeUncompacted();
    gasBunny.compact(cuStream, gasBunnyCompactedMem);
    gasBunny.removeUncompacted();



    // JP: IAS作成前には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     before creating an IAS.
    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    scene.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);



    // JP: インスタンスを作成する。
    // EN: Create instances.
    optixu::Instance instBox = scene.createInstance();
    instBox.setChild(gasBox);

    float instAreaLightTr[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance instAreaLight = scene.createInstance();
    instAreaLight.setChild(gasAreaLight);
    instAreaLight.setTransform(instAreaLightTr);

    constexpr uint32_t NumBunnies = 100;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * M_PI / (GoldenRatio * GoldenRatio);
    struct Transform {
        float3 scale[2];
        Quaternion orientation[2];
        float3 translation[2];
        optixu::Transform optixTransform;
    };
    std::vector<Transform> transformsBunny;
    cudau::TypedBuffer<optixu::TransformMemory> transformMem;
    std::vector<optixu::Instance> instsBunny;
    transformMem.initialize(cuContext, cudau::BufferType::Device, NumBunnies);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.003f + tt * 0.0006f;
        float y = -1 + (1 - tt);

        Transform tr;
        tr.scale[0] = make_float3(scale);
        tr.orientation[0] = Quaternion(0, 0, 0, 1);
        tr.translation[0] = make_float3(x, y, z);
        tr.scale[1] = make_float3(scale);
        tr.orientation[1] = Quaternion(0, 0, 0, 1);
        tr.translation[1] = make_float3(x, y + 0.1f, z);

        tr.optixTransform = scene.createTransform();
        tr.optixTransform.setChild(gasBunny);
        tr.optixTransform.setSRTMotion(reinterpret_cast<float*>(&tr.scale[0]),
                                       reinterpret_cast<float*>(&tr.orientation[0]),
                                       reinterpret_cast<float*>(&tr.translation[0]),
                                       reinterpret_cast<float*>(&tr.scale[1]),
                                       reinterpret_cast<float*>(&tr.orientation[1]),
                                       reinterpret_cast<float*>(&tr.translation[1]));
        tr.optixTransform.setMotionOptions(2, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        tr.optixTransform.rebuild(cuStream, transformMem.getDevicePointerAt(i));
        transformsBunny.push_back(tr);
        
        optixu::Instance instBunny = scene.createInstance();
        instBunny.setChild(tr.optixTransform);
        instsBunny.push_back(instBunny);
    }



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    uint32_t numAABBs;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    cudau::TypedBuffer<OptixAabb> aabbBuffer;
    constexpr uint32_t numMotionKeys = 2;
    ias.setConfiguration(true, false, false);
    ias.setMotionOptions(numMotionKeys, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
    ias.addChild(instBox);
    ias.addChild(instAreaLight);
    for (int i = 0; i < instsBunny.size(); ++i)
        ias.addChild(instsBunny[i]);
    ias.prepareForBuild(&asMemReqs, &numInstances, &numAABBs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, numInstances);
    aabbBuffer.initialize(cuContext, cudau::BufferType::Device, numAABBs);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    if (maxSizeOfScratchBuffer > asBuildScratchMem.sizeInBytes())
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);

    {
        OptixAabb* aabbs = aabbBuffer.map();
        // First two instances don't require AABBs and its values will be ignored.
        {
            aabbs[0].minX = -1.0f;
            aabbs[0].minY = -1.0f;
            aabbs[0].minZ = -1.0f;
            aabbs[0].maxX = 1.0f;
            aabbs[0].maxY = 1.0f;
            aabbs[0].maxZ = 1.0f;
            aabbs[1].minX = -1.0f;
            aabbs[1].minY = -1.0f;
            aabbs[1].minZ = -1.0f;
            aabbs[1].maxX = 1.0f;
            aabbs[1].maxY = 1.0f;
            aabbs[1].maxZ = 1.0f;
        }
        {
            aabbs[2].minX = -0.25f;
            aabbs[2].minY = 0.0f;
            aabbs[2].minZ = -0.25f;
            aabbs[2].maxX = 0.25f;
            aabbs[2].maxY = 0.0f;
            aabbs[2].maxZ = 0.25f;
            aabbs[3].minX = -0.25f;
            aabbs[3].minY = 0.0f;
            aabbs[3].minZ = -0.25f;
            aabbs[3].maxX = 0.25f;
            aabbs[3].maxY = 0.0f;
            aabbs[3].maxZ = 0.25f;
        }
        for (int instIdx = 2; instIdx < (2 + instsBunny.size()); ++instIdx) {
            const Transform &trBunny = transformsBunny[instIdx - 2];
            for (int keyIdx = 0; keyIdx < numMotionKeys; ++keyIdx) {
                OptixAabb &aabb = aabbs[instIdx * numMotionKeys + keyIdx];

                Matrix3x3 sr =
                    trBunny.orientation[keyIdx].toMatrix3x3() *
                    scale3x3(trBunny.scale[keyIdx]);
                float3 trans = trBunny.translation[keyIdx];

                float3 c;
                float3 minP = make_float3(INFINITY);
                float3 maxP = make_float3(-INFINITY);

                c = sr * float3(bboxMinBunny.x, bboxMinBunny.y, bboxMinBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMaxBunny.x, bboxMinBunny.y, bboxMinBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMinBunny.x, bboxMaxBunny.y, bboxMinBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMaxBunny.x, bboxMaxBunny.y, bboxMinBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMinBunny.x, bboxMinBunny.y, bboxMaxBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMaxBunny.x, bboxMinBunny.y, bboxMaxBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMinBunny.x, bboxMaxBunny.y, bboxMaxBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);
                c = sr * float3(bboxMaxBunny.x, bboxMaxBunny.y, bboxMaxBunny.z) + trans;
                minP = min(minP, c);
                maxP = max(maxP, c);

                aabb.minX = minP.x;
                aabb.minY = minP.y;
                aabb.minZ = minP.z;
                aabb.maxX = maxP.x;
                aabb.maxY = maxP.y;
                aabb.maxZ = maxP.z;
            }
        }
        aabbBuffer.unmap();
    }

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, instanceBuffer, aabbBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    //{
    //    const OptixAabb* aabbs = aabbBuffer.map();
    //    for (int i = 0; i < aabbBuffer.numElements(); ++i) {
    //        const OptixAabb &aabb = aabbs[i];
    //        hpprintf("%3u: (%9.6f, %9.6f, %9.6f) - (%9.6f, %9.6f, %9.6f)\n", i,
    //                 aabb.minX, aabb.minY, aabb.minZ,
    //                 aabb.maxX, aabb.maxY, aabb.maxZ);
    //    }
    //    aabbBuffer.unmap();
    //}

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
    plp.geomInstData = geomDataBuffer.getDevicePointer();
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.accumBuffer = accumBuffer.getBlockBuffer2D();
    plp.timeBegin = 0.0f;
    plp.timeEnd = 1.0f;
    plp.numAccumFrames = 0;
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5);
    plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(&shaderBindingTable);

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

    asBuildScratchMem.finalize();

    aabbBuffer.finalize();
    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    shaderBindingTable.finalize();

    for (int i = instsBunny.size() - 1; i >= 0; --i) {
        instsBunny[i].destroy();
        transformsBunny[i].optixTransform.destroy();
    }
    transformMem.finalize();
    instAreaLight.destroy();
    instBox.destroy();

    gasBunnyCompactedMem.finalize();
    gasAreaLightCompactedMem.finalize();
    gasBoxCompactedMem.finalize();
    gasBunnyMem.finalize();
    gasBunny.destroy();
    gasAreaLightMem.finalize();
    gasAreaLight.destroy();
    gasBoxMem.finalize();
    gasBox.destroy();

    triangleBufferBunny.finalize();
    vertexBufferBunny.finalize();
    geomInstBunny.destroy();
    
    triangleBufferAreaLight.finalize();
    vertexBufferAreaLight.finalize();
    geomInstAreaLight.destroy();

    triangleBufferBox.finalize();
    vertexBufferBox.finalize();
    geomInstBox.destroy();

    geomDataBuffer.finalize();

    scene.destroy();

    mat0.destroy();

    hitProgramGroup0.destroy();

    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
    }

    return 0;
}
