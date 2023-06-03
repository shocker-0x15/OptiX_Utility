/*

JP: このサンプルはインタラクティブなアプリケーション上におけるオブジェクトピック実装の一例を示します。
    APIの使い方として新しいものを示すサンプルではありません。
    例として透視投影と正距円筒図法による投影を使ったカメラを示します。

EN: This sample demonstrates an example of implementing object picking in an interactive application.
    This sample has nothing new for API usage.
    As examples, cameras using perspective projection and equirectangular projection are provided.

*/

#include "pick_shared.h"

#include "../common/obj_loader.h"

#include "../common/gui_common.h"



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "pick";

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

    optixu::Context optixContext = optixu::Context::create(cuContext);

    struct Pipeline {
        optixu::Pipeline pipeline;
        optixu::Module module;
        std::map<std::string, optixu::Program> rayGenPrograms;
        optixu::Program missProgram;
        optixu::HitProgramGroup hitProgramGroup;
        cudau::Buffer shaderBindingTable;
        cudau::Buffer hitGroupShaderBindingTable;

        void finalize() {
            hitGroupShaderBindingTable.finalize();
            shaderBindingTable.finalize();
            hitProgramGroup.destroy();
            missProgram.destroy();
            for (auto it = rayGenPrograms.begin(); it != rayGenPrograms.end(); ++it)
                it->second.destroy();
            module.destroy();
            pipeline.destroy();
        }
    };

    optixu::Module emptyModule;

    Pipeline pickPipeline;
    {
        Pipeline &p = pickPipeline;
        optixu::Pipeline &optixPipeline = p.pipeline;

        optixPipeline = optixContext.createPipeline();
        optixPipeline.setPipelineOptions(
            Shared::PickPayloadSignature::numDwords,
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(Shared::PickPipelineLaunchParameters),
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::vector<char> optixIr =
            readBinaryFile(resourceDir / "ptxes/pick_kernels.optixir");
        p.module = optixPipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        p.rayGenPrograms["perspective"] =
            optixPipeline.createRayGenProgram(p.module, RT_RG_NAME_STR("perspectiveRaygen"));
        p.rayGenPrograms["equirectangular"] =
            optixPipeline.createRayGenProgram(p.module, RT_RG_NAME_STR("equirectangularRaygen"));

        p.missProgram = optixPipeline.createMissProgram(p.module, RT_MS_NAME_STR("miss"));

        p.hitProgramGroup = optixPipeline.createHitProgramGroupForTriangleIS(
            p.module, RT_CH_NAME_STR("closesthit"),
            emptyModule, nullptr);

        optixPipeline.link(1);

        optixPipeline.setNumMissRayTypes(Shared::NumPickRayTypes);
        optixPipeline.setMissProgram(Shared::PickRayType_Primary, p.missProgram);

        size_t sbtSize;
        optixPipeline.generateShaderBindingTableLayout(&sbtSize);
        p.shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
        p.shaderBindingTable.setMappedMemoryPersistent(true);
        optixPipeline.setShaderBindingTable(p.shaderBindingTable, p.shaderBindingTable.getMappedPointer());
    }

    Pipeline renderPipeline;
    {
        Pipeline &p = renderPipeline;
        optixu::Pipeline &optixPipeline = p.pipeline;

        optixPipeline = optixContext.createPipeline();
        optixPipeline.setPipelineOptions(
            Shared::RenderPayloadSignature::numDwords,
            optixu::calcSumDwords<float2>(),
            "plp", sizeof(Shared::RenderPipelineLaunchParameters),
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
            OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
            DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
            OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

        const std::vector<char> optixIr =
            readBinaryFile(resourceDir / "ptxes/render_kernels.optixir");
        p.module = optixPipeline.createModuleFromOptixIR(
            optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
            DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
            DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

        p.rayGenPrograms["perspective"] =
            optixPipeline.createRayGenProgram(p.module, RT_RG_NAME_STR("perspectiveRaygen"));
        p.rayGenPrograms["equirectangular"] =
            optixPipeline.createRayGenProgram(p.module, RT_RG_NAME_STR("equirectangularRaygen"));

        p.missProgram = optixPipeline.createMissProgram(p.module, RT_MS_NAME_STR("miss"));

        p.hitProgramGroup = optixPipeline.createHitProgramGroupForTriangleIS(
            p.module, RT_CH_NAME_STR("closesthit"),
            emptyModule, nullptr);

        optixPipeline.link(1);

        optixPipeline.setNumMissRayTypes(Shared::NumRayTypes);
        optixPipeline.setMissProgram(Shared::RayType_Primary, p.missProgram);

        size_t sbtSize;
        optixPipeline.generateShaderBindingTableLayout(&sbtSize);
        p.shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
        p.shaderBindingTable.setMappedMemoryPersistent(true);
        optixPipeline.setShaderBindingTable(p.shaderBindingTable, p.shaderBindingTable.getMappedPointer());
    }

    uint32_t maxNumRayTypes = std::max<uint32_t>(Shared::NumPickRayTypes, Shared::NumRayTypes);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    uint32_t matID = 0;
    std::vector<std::string> matInfos;

    optixu::Material ceilingMat = optixContext.createMaterial();
    ceilingMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    ceilingMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Ceiling");
    Shared::MaterialData ceilingMatData = {};
    ceilingMatData.matID = matID++;
    ceilingMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    ceilingMat.setUserData(ceilingMatData);

    optixu::Material farSideWallMat = optixContext.createMaterial();
    farSideWallMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    farSideWallMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Far side Wall");
    Shared::MaterialData farSideWallMatData = {};
    farSideWallMatData.matID = matID++;
    farSideWallMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    farSideWallMat.setUserData(farSideWallMatData);

    optixu::Material leftWallMat = optixContext.createMaterial();
    leftWallMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    leftWallMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Left Wall");
    Shared::MaterialData leftWallMatData = {};
    leftWallMatData.matID = matID++;
    leftWallMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    leftWallMat.setUserData(leftWallMatData);

    optixu::Material rightWallMat = optixContext.createMaterial();
    rightWallMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    rightWallMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Right Wall");
    Shared::MaterialData rightWallMatData = {};
    rightWallMatData.matID = matID++;
    rightWallMatData.color = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    rightWallMat.setUserData(rightWallMatData);

    optixu::Material floorMat = optixContext.createMaterial();
    floorMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    floorMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Floor");
    Shared::MaterialData floorMatData = {};
    floorMatData.matID = matID++;
    floorMatData.color = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    floorMat.setUserData(floorMatData);

    optixu::Material areaLightMat = optixContext.createMaterial();
    areaLightMat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
    areaLightMat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
    matInfos.push_back("Area Light");
    Shared::MaterialData areaLightMatData = {};
    areaLightMatData.matID = matID++;
    areaLightMatData.color = make_float3(sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f));
    areaLightMat.setUserData(areaLightMatData);

    std::mt19937 rngColor(2493572);
    std::uniform_real_distribution<float> uColor;

    constexpr uint32_t NumBunnies = 10;
    std::vector<optixu::Material> bunnyMats(NumBunnies);
    for (int i = 0; i < NumBunnies; ++i) {
        optixu::Material &mat = bunnyMats[i];
        mat = optixContext.createMaterial();
        mat.setHitGroup(Shared::PickRayType_Primary, pickPipeline.hitProgramGroup);
        mat.setHitGroup(Shared::RayType_Primary, renderPipeline.hitProgramGroup);
        char str[256];
        snprintf(str, sizeof(str), "Bunny %u", i);
        matInfos.push_back(str);
        Shared::MaterialData matData = {};
        matData.matID = matID++;
        matData.color = HSVtoRGB(uColor(rngColor),
                                 uColor(rngColor),
                                 sRGB_degamma_s(0.75f + 0.25f * uColor(rngColor)));
        mat.setUserData(matData);
    }

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    uint32_t geomID = 0;
    uint32_t gasID = 0;
    std::vector<std::string> geomInfos;
    std::vector<std::string> gasInfos;

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    struct Mesh {
        struct Group {
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            cudau::TypedBuffer<uint8_t> matIndexBuffer;
            optixu::GeometryInstance optixGeomInst;
        };

        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        std::vector<Group> groups;

        void finalize() {
            for (auto it = groups.rbegin(); it != groups.rend(); ++it) {
                it->optixGeomInst.destroy();
                it->matIndexBuffer.finalize();
                it->triangleBuffer.finalize();
            }
            vertexBuffer.finalize();
        }
    };

    struct GeometryGroup {
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
        }
    };

    Mesh room;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(1, 0) },
            // far side wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 2) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 0) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 2) },
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

        room.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        Mesh::Group group;
        group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
        group.matIndexBuffer.initialize(cuContext, cudau::BufferType::Device, matIndices, lengthof(matIndices));

        geomInfos.push_back("Room");
        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = room.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
        geomData.geomID = geomID++;

        group.optixGeomInst = scene.createGeometryInstance();
        group.optixGeomInst.setVertexBuffer(room.vertexBuffer);
        group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
        group.optixGeomInst.setNumMaterials(5, group.matIndexBuffer, optixu::IndexSize::k1Byte);
        group.optixGeomInst.setMaterial(0, 0, floorMat);
        group.optixGeomInst.setMaterial(0, 1, farSideWallMat);
        group.optixGeomInst.setMaterial(0, 2, ceilingMat);
        group.optixGeomInst.setMaterial(0, 3, leftWallMat);
        group.optixGeomInst.setMaterial(0, 4, rightWallMat);
        group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setGeometryFlags(1, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setGeometryFlags(2, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setGeometryFlags(3, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setGeometryFlags(4, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setUserData(geomData);

        room.groups.push_back(std::move(group));
    }

    Mesh areaLight;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.9f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.9f, 0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.25f, 0.9f, 0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.25f, 0.9f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        areaLight.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        Mesh::Group group;
        group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInfos.push_back("Area Light");
        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLight.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
        geomData.geomID = geomID++;

        group.optixGeomInst = scene.createGeometryInstance();
        group.optixGeomInst.setVertexBuffer(areaLight.vertexBuffer);
        group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
        group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        group.optixGeomInst.setMaterial(0, 0, areaLightMat);
        group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setUserData(geomData);

        areaLight.groups.push_back(std::move(group));
    }

    Mesh bunny;
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

        bunny.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);

        Mesh::Group group;
        group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        geomInfos.push_back("Bunny");
        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunny.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
        geomData.geomID = geomID++;

        group.optixGeomInst = scene.createGeometryInstance();
        group.optixGeomInst.setVertexBuffer(bunny.vertexBuffer);
        group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
        group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
            group.optixGeomInst.setMaterial(matSetIdx, 0, bunnyMats[matSetIdx]);
        group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        group.optixGeomInst.setUserData(geomData);

        bunny.groups.push_back(std::move(group));
    }



    GeometryGroup roomGroup;
    {
        gasInfos.push_back("Room");
        Shared::GASData gasData = {};
        gasData.gasID = gasID++;

        roomGroup.optixGas = scene.createGeometryAccelerationStructure();
        roomGroup.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        roomGroup.optixGas.setNumMaterialSets(1);
        roomGroup.optixGas.setNumRayTypes(0, maxNumRayTypes);
        for (auto it = room.groups.cbegin(); it != room.groups.cend(); ++it)
            roomGroup.optixGas.addChild(it->optixGeomInst);
        for (auto it = areaLight.groups.cbegin(); it != areaLight.groups.cend(); ++it)
            roomGroup.optixGas.addChild(it->optixGeomInst);
        Shared::GASChildData childData = {};
        childData.gasChildID = 0;
        for (int i = 0; i < roomGroup.optixGas.getNumChildren(); ++i) {
            roomGroup.optixGas.setChildUserData(childData.gasChildID, childData);
            ++childData.gasChildID;
        }
        roomGroup.optixGas.prepareForBuild(&asMemReqs);
        roomGroup.optixGas.setUserData(gasData);
        roomGroup.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    GeometryGroup bunnyGroup;
    {
        gasInfos.push_back("Bunny");
        Shared::GASData gasData = {};
        gasData.gasID = gasID++;

        bunnyGroup.optixGas = scene.createGeometryAccelerationStructure();
        bunnyGroup.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        bunnyGroup.optixGas.setNumMaterialSets(NumBunnies);
        for (int matSetIdx = 0; matSetIdx < NumBunnies; ++matSetIdx)
            bunnyGroup.optixGas.setNumRayTypes(matSetIdx, maxNumRayTypes);
        for (auto it = bunny.groups.cbegin(); it != bunny.groups.cend(); ++it)
            bunnyGroup.optixGas.addChild(it->optixGeomInst);
        Shared::GASChildData childData = {};
        childData.gasChildID = 0;
        for (int i = 0; i < bunnyGroup.optixGas.getNumChildren(); ++i) {
            bunnyGroup.optixGas.setChildUserData(childData.gasChildID, childData);
            ++childData.gasChildID;
        }
        bunnyGroup.optixGas.prepareForBuild(&asMemReqs);
        bunnyGroup.optixGas.setUserData(gasData);
        bunnyGroup.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    std::vector<std::string> instInfos;
    uint32_t instID = 0;
    
    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    instInfos.push_back("Room");
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(roomGroup.optixGas);
    roomInst.setID(instID++);

    std::vector<optixu::Instance> bunnyInsts;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * pi_v<float> / (GoldenRatio * GoldenRatio);
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        char str[256];
        snprintf(str, sizeof(str), "Bunny %u", i);
        instInfos.push_back(str);
        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.006f + tt * 0.003f;
        float bunnyInstXfm[] = {
            scale, 0, 0, x,
            0, scale, 0, -1 + (1 - tt),
            0, 0, scale, z
        };
        optixu::Instance bunnyInst = scene.createInstance();
        bunnyInst.setChild(bunnyGroup.optixGas, i);
        bunnyInst.setTransform(bunnyInstXfm);
        bunnyInst.setID(instID++);
        bunnyInsts.push_back(bunnyInst);
    }



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(roomInst);
    for (int i = 0; i < bunnyInsts.size(); ++i)
        ias.addChild(bunnyInsts[i]);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Mesh Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    roomGroup.optixGas.rebuild(stream, roomGroup.gasMem, asBuildScratchMem);
    bunnyGroup.optixGas.rebuild(stream, bunnyGroup.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        GeometryGroup* geomGroup;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &roomGroup, 0, 0 },
        { &bunnyGroup, 0, 0 }
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.geomGroup->optixGas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.geomGroup->optixGas.compact(
            stream, optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].geomGroup->optixGas.removeUncompacted();
        gasList[i].geomGroup->gasMem.finalize();
    }



    // JP: IASビルド時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when building an IAS.
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);

    pickPipeline.hitGroupShaderBindingTable.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    pickPipeline.hitGroupShaderBindingTable.setMappedMemoryPersistent(true);
    renderPipeline.hitGroupShaderBindingTable.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    renderPipeline.hitGroupShaderBindingTable.setMappedMemoryPersistent(true);

    OptixTraversableHandle travHandle = ias.rebuild(stream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(stream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr int32_t initWindowContentWidth = 800;
    constexpr int32_t initWindowContentHeight = 800;

    Shared::PerspectiveCamera perspCamera;
    perspCamera.fovY = 50 * pi_v<float> / 180;
    perspCamera.aspect = (float)initWindowContentWidth / initWindowContentHeight;

    Shared::EquirectangularCamera equirecCamera;
    equirecCamera.horizentalExtent = pi_v<float> * (float)initWindowContentWidth / initWindowContentHeight;
    equirecCamera.verticalExtent = pi_v<float>;
    
    Shared::PickPipelineLaunchParameters pickPlp;
    pickPlp.travHandle = travHandle;
    pickPlp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    pickPlp.perspCamera = perspCamera;
    pickPlp.equirecCamera = equirecCamera;

    Shared::RenderPipelineLaunchParameters renderPlp;
    renderPlp.travHandle = travHandle;
    renderPlp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    renderPlp.perspCamera = perspCamera;
    renderPlp.equirecCamera = equirecCamera;
    renderPlp.colorInterp = 0.1f;

    cudau::TypedBuffer<Shared::PickInfo> pickInfos[2];
    for (int i = 0; i < lengthof(pickInfos); ++i) {
        pickInfos[i].initialize(cuContext, cudau::BufferType::Device, 1);
        pickInfos[i].setMappedMemoryPersistent(true);
        auto value = pickInfos[i].map();
        *value = {};
        pickInfos[i].unmap();
    }

    pickPipeline.pipeline.setScene(scene);
    pickPipeline.pipeline.setHitGroupShaderBindingTable(
        pickPipeline.hitGroupShaderBindingTable,
        pickPipeline.hitGroupShaderBindingTable.getMappedPointer());
    renderPipeline.pipeline.setScene(scene);
    renderPipeline.pipeline.setHitGroupShaderBindingTable(
        renderPipeline.hitGroupShaderBindingTable,
        renderPipeline.hitGroupShaderBindingTable.getMappedPointer());

    CUdeviceptr pickPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&pickPlpOnDevice, sizeof(pickPlp)));

    CUdeviceptr renderPlpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&renderPlpOnDevice, sizeof(renderPlp)));

    pickPipeline.pipeline.setRayGenerationProgram(pickPipeline.rayGenPrograms.at("perspective"));
    renderPipeline.pipeline.setRayGenerationProgram(renderPipeline.rayGenPrograms.at("perspective"));

    enum class CameraType {
        Perspective = 0,
        Equirectangular,
    };
    CameraType cameraType = CameraType::Perspective;



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - Pick";
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
        cudau::TypedBuffer<Shared::PickInfo> &curPickInfo = pickInfos[args.frameIndex % 2];

        if (takeScreenShot) { // Fix mouse position input for the test.
            args.mouseX = 400;
            args.mouseY = 360;
        }

        // Camera Window
        {
            ImGui::Begin("Camera & Rendering", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

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

            if (ImGui::RadioButtonE("Perspective", &cameraType, CameraType::Perspective)) {
                pickPipeline.pipeline.setRayGenerationProgram(
                    pickPipeline.rayGenPrograms.at("perspective"));
                renderPipeline.pipeline.setRayGenerationProgram(
                    renderPipeline.rayGenPrograms.at("perspective"));
            }
            if (ImGui::RadioButtonE("Equirectangular", &cameraType, CameraType::Equirectangular)) {
                pickPipeline.pipeline.setRayGenerationProgram(
                    pickPipeline.rayGenPrograms.at("equirectangular"));
                renderPipeline.pipeline.setRayGenerationProgram(
                    renderPipeline.rayGenPrograms.at("equirectangular"));
            }

            ImGui::Separator();

            ImGui::SliderFloat("Material <-> Normal", &renderPlp.colorInterp, 0.0f, 1.0f);

            ImGui::End();
        }


        // Pick Information Window
        {
            const auto helpMarker = [](const char* desc) {
                ImGui::TextDisabled("(?)");
                if (ImGui::IsItemHovered()) {
                    ImGui::BeginTooltip();
                    ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
                    ImGui::TextUnformatted(desc);
                    ImGui::PopTextWrapPos();
                    ImGui::EndTooltip();
                }
            };

            Shared::PickInfo pickInfo = curPickInfo.map()[0];
            curPickInfo.unmap();
            char instIndexStr[32], matIndexStr[32], primIndexStr[32];
            char instIDStr[32], gasIDStr[32], gasChildIDStr[32], geomIDStr[32], matIDStr[32];
            char instNameStr[64], gasNameStr[64], geomNameStr[64], matNameStr[64];
            if (pickInfo.hit) {
                snprintf(instIndexStr, sizeof(instIndexStr), "%3u", pickInfo.instanceIndex);
                snprintf(matIndexStr, sizeof(matIndexStr), "%3u", pickInfo.matIndex);
                snprintf(primIndexStr, sizeof(primIndexStr), "%3u", pickInfo.primIndex);
                snprintf(instIDStr, sizeof(instIDStr), "%3u", pickInfo.instanceID);
                snprintf(gasIDStr, sizeof(gasIDStr), "%3u", pickInfo.gasID);
                snprintf(gasChildIDStr, sizeof(gasChildIDStr), "%3u", pickInfo.gasChildID);
                snprintf(geomIDStr, sizeof(geomIDStr), "%3u", pickInfo.geomID);
                snprintf(matIDStr, sizeof(matIDStr), "%3u", pickInfo.matID);

                snprintf(instNameStr, sizeof(instNameStr), "%s", instInfos[pickInfo.instanceID].c_str());
                snprintf(gasNameStr, sizeof(gasNameStr), "%s", gasInfos[pickInfo.gasID].c_str());
                snprintf(geomNameStr, sizeof(geomNameStr), "%s", geomInfos[pickInfo.geomID].c_str());
                snprintf(matNameStr, sizeof(matNameStr), "%s", matInfos[pickInfo.matID].c_str());
            }
            else {
                snprintf(instIndexStr, sizeof(instIndexStr), "N/A");
                snprintf(matIndexStr, sizeof(matIndexStr), "N/A");
                snprintf(primIndexStr, sizeof(primIndexStr), "N/A");
                snprintf(instIDStr, sizeof(instIDStr), "N/A");
                snprintf(gasIDStr, sizeof(gasIDStr), "N/A");
                snprintf(gasChildIDStr, sizeof(gasChildIDStr), "N/A");
                snprintf(geomIDStr, sizeof(geomIDStr), "N/A");
                snprintf(matIDStr, sizeof(matIDStr), "N/A");

                snprintf(instNameStr, sizeof(instNameStr), "N/A");
                snprintf(gasNameStr, sizeof(gasNameStr), "N/A");
                snprintf(geomNameStr, sizeof(geomNameStr), "N/A");
                snprintf(matNameStr, sizeof(matNameStr), "N/A");
            }

            ImGui::Begin("Pick Info", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("Mouse: %d, %d", static_cast<int32_t>(args.mouseX), static_cast<int32_t>(args.mouseY));

            ImGui::Separator();
            ImGui::Text("System Values:");
            helpMarker("Instance index in an IAS"); ImGui::SameLine();
            ImGui::Text(" Instance Index: %s", instIndexStr);
            helpMarker("Material index in a GAS"); ImGui::SameLine();
            ImGui::Text(" Material Index: %s", matIndexStr);
            helpMarker("Primitive index in a geometry"); ImGui::SameLine();
            ImGui::Text("Primitive Index: %s", primIndexStr);

            ImGui::Separator();
            ImGui::Text("User Data:");
            helpMarker("via Instance::setID()"); ImGui::SameLine();
            ImGui::Text(" Instance: %s, Name: %s", instIDStr, instNameStr);
            helpMarker("via GeometryAccelerationStructure::setUserData()"); ImGui::SameLine();
            ImGui::Text("      GAS: %s, Name: %s", gasIDStr, gasNameStr);
            helpMarker("via GeometryAccelerationStructure::setChildUserData()"); ImGui::SameLine();
            ImGui::Text("GAS Child: %s", gasChildIDStr);
            helpMarker("via GeometryInstance::setUserData()"); ImGui::SameLine();
            ImGui::Text(" Geometry: %s, Name: %s", geomIDStr, geomNameStr);
            helpMarker("via Material::setUserData()"); ImGui::SameLine();
            ImGui::Text(" Material: %s, Name: %s", matIDStr, matNameStr);

            ImGui::End();
        }



        Matrix3x3 oriMat = args.tempCameraOrientation.toMatrix3x3();

        // Pick
        pickPlp.position = args.cameraPosition;
        pickPlp.orientation = oriMat;
        pickPlp.mousePosition = int2(static_cast<int32_t>(args.mouseX),
                                     static_cast<int32_t>(args.mouseY));
        pickPlp.pickInfo = curPickInfo.getDevicePointer();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(pickPlpOnDevice, &pickPlp, sizeof(pickPlp), curStream));
        pickPipeline.pipeline.launch(curStream, pickPlpOnDevice, 1, 1, 1);

        
        // Render
        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        renderPlp.position = args.cameraPosition;
        renderPlp.orientation = oriMat;
        renderPlp.pickInfo = curPickInfo.getDevicePointer();
        renderPlp.resultBuffer = outputBufferSurfaceHolder.getNext();

        CUDADRV_CHECK(cuMemcpyHtoDAsync(renderPlpOnDevice, &renderPlp, sizeof(renderPlp), curStream));
        renderPipeline.pipeline.launch(
            curStream, renderPlpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);

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
                GL_RGBA, GL_FLOAT, sizeof(float4) * numPixels, rawImage);
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
         perspCamera.aspect = (float)windowContentWidth / windowContentHeight;
         equirecCamera.horizentalExtent = pi_v<float> * (float)windowContentWidth / windowContentHeight;
         equirecCamera.verticalExtent = pi_v<float>;

         pickPlp.imageSize = int2(windowContentWidth, windowContentHeight);
         pickPlp.perspCamera = perspCamera;
         pickPlp.equirecCamera = equirecCamera;
         renderPlp.imageSize = int2(windowContentWidth, windowContentHeight);
         renderPlp.perspCamera = perspCamera;
         renderPlp.equirecCamera = equirecCamera;
    };

    framework.run(onRenderLoop, onResolutionChange);

    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();

    framework.finalize();

    // END: Display the window.
    // ----------------------------------------------------------------



    CUDADRV_CHECK(cuMemFree(renderPlpOnDevice));
    CUDADRV_CHECK(cuMemFree(pickPlpOnDevice));

    for (int i = lengthof(pickInfos) - 1; i >= 0; --i)
        pickInfos[i].finalize();



    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    for (int i = bunnyInsts.size() - 1; i >= 0; --i)
        bunnyInsts[i].destroy();
    roomInst.destroy();

    bunnyGroup.finalize();
    roomGroup.finalize();

    bunny.finalize();
    areaLight.finalize();
    room.finalize();

    scene.destroy();

    for (int i = NumBunnies - 1; i >= 0; --i)
        bunnyMats[i].destroy();
    areaLightMat.destroy();
    floorMat.destroy();
    rightWallMat.destroy();
    leftWallMat.destroy();
    farSideWallMat.destroy();
    ceilingMat.destroy();



    renderPipeline.finalize();
    pickPipeline.finalize();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(stream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
