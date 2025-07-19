/*

JP: 

EN: 

*/

#include "clusters_shared.h"

#include "../common/gui_common.h"

class ExIfStream : public std::ifstream {
public:
    using std::ifstream::ifstream;

    template <typename T>
    ExIfStream &read(T* const ptr, uint32_t count = 1) {
        std::ifstream::read(reinterpret_cast<char*>(ptr), sizeof(T) * count);
        return *this;
    }

    template <typename T>
    ExIfStream &read(std::vector<T> &vec) {
        uint32_t count = static_cast<uint32_t>(vec.size());
        std::ifstream::read(reinterpret_cast<char*>(vec.data()), sizeof(T) * count);
        return *this;
    }
};



struct HierarchicalMesh {
    struct LevelInfo {
        uint32_t startClusterIndex;
        uint32_t clusterCount;
    };

    OptixClusterAccelBuildMode static constexpr clasAccelBuildMode =
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;

    cudau::TypedBuffer<Shared::Vertex> vertexPool;
    cudau::TypedBuffer<Shared::LocalTriangle> trianglePool;
    cudau::TypedBuffer<uint32_t> childIndexPool;
    cudau::TypedBuffer<Shared::Cluster> clusters;

    cudau::TypedBuffer<OptixClusterAccelBuildInputTrianglesArgs> argsArray;
    cudau::TypedBuffer<OptixClusterAccelBuildInputTrianglesArgs> argsArrayToBuild;
    cudau::TypedBuffer<CUdeviceptr> clasHandles;
    cudau::TypedBuffer<uint32_t> usedFlags;
    cudau::TypedBuffer<uint32_t> indexMapClusterToClasBuild;
    cudau::TypedBuffer<Shared::ClusterSetInfo> clusterSetInfo;
    cudau::Buffer clasSetMem;
    optixu::ClusterAccelerationStructureSet clasSet;
    OptixAccelBufferSizes asMemReqs;

    std::vector<LevelInfo> levelInfos;
    uint32_t maxVertCountPerCluster;
    uint32_t maxTriCountPerCluster;

    bool read(
        const CUcontext cuContext, const optixu::Scene scene, const optixu::Material mat,
        const std::filesystem::path &filePath)
    {
        ExIfStream ifs(filePath, std::ios::binary);
        if (!ifs)
            return false;

        ifs
            .read(&maxVertCountPerCluster)
            .read(&maxTriCountPerCluster);

        uint32_t vertexCount, triangleCount, childIndexCount, clusterCount, levelCount;
        ifs
            .read(&vertexCount)
            .read(&triangleCount)
            .read(&childIndexCount)
            .read(&clusterCount)
            .read(&levelCount);

        size_t const triBufSize = triangleCount * sizeof(Shared::LocalTriangle);
        size_t const paddingByteCount = ((triBufSize + 3) / 4 * 4) - triBufSize;

        std::vector<Shared::Vertex> verticesOnHost(vertexCount);
        std::vector<Shared::LocalTriangle> trianglesOnHost(triangleCount);
        std::vector<uint8_t> padding(paddingByteCount, 0);
        std::vector<uint32_t> childIndicesOnHost(childIndexCount);
        std::vector<Shared::Cluster> clustersOnHost(clusterCount);
        std::vector<uint32_t> levelStartClusterIndicesOnHost(levelCount);
        ifs
            .read(verticesOnHost)
            .read(trianglesOnHost)
            .read(padding)
            .read(childIndicesOnHost)
            .read(clustersOnHost)
            .read(levelStartClusterIndicesOnHost);

        levelInfos.resize(levelCount);
        for (uint32_t level = 0; level < levelCount; ++level) {
            LevelInfo &levelInfo = levelInfos[level];
            levelInfo.startClusterIndex = levelStartClusterIndicesOnHost[level];
            if (level + 1 < levelCount) {
                levelInfo.clusterCount =
                  levelStartClusterIndicesOnHost[level + 1] - levelStartClusterIndicesOnHost[level];
            }
            else {
                levelInfo.clusterCount = clusterCount - levelStartClusterIndicesOnHost[level];
            }
        }

        vertexPool.initialize(
            cuContext, cudau::BufferType::Device, verticesOnHost);
        trianglePool.initialize(
            cuContext, cudau::BufferType::Device, trianglesOnHost);
        childIndexPool.initialize(
            cuContext, cudau::BufferType::Device, childIndicesOnHost);
        clusters.initialize(
            cuContext, cudau::BufferType::Device, clustersOnHost);



        std::vector<OptixClusterAccelBuildInputTrianglesArgs> argsArrayOnHost(clusterCount);
        for (uint32_t cIdx = 0; cIdx < clusterCount; ++cIdx) {
            const Shared::Cluster &cluster = clustersOnHost[cIdx];

            OptixClusterAccelBuildInputTrianglesArgs args = {};
            args.clusterId = cIdx;
            args.clusterFlags = OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE;
            args.triangleCount = cluster.triangleCount;
            args.vertexCount = cluster.vertexCount;
            args.positionTruncateBitCount = 0;
            args.indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_8BIT;
            args.opacityMicromapIndexFormat = 0; // not used in this sample
            args.basePrimitiveInfo.sbtIndex = 0;
            args.basePrimitiveInfo.primitiveFlags = OPTIX_CLUSTER_ACCEL_PRIMITIVE_FLAG_NONE;
            args.indexBufferStrideInBytes = sizeof(uint8_t);
            args.vertexBufferStrideInBytes = sizeof(Shared::Vertex);
            args.primitiveInfoBufferStrideInBytes = 0; // not used in this sample
            args.opacityMicromapIndexBufferStrideInBytes = 0; // not used in this sample
            args.indexBuffer = trianglePool.getCUdeviceptrAt(cluster.triPoolStartIndex);
            args.vertexBuffer = vertexPool.getCUdeviceptrAt(cluster.vertPoolStartIndex);
            args.primitiveInfoBuffer = 0; // not used in this sample
            args.opacityMicromapArray = 0; // not used in this sample
            args.opacityMicromapIndexBuffer = 0; // not used in this sample
            args.instantiationBoundingBoxLimit = 0; // ignored for this arg type

            argsArrayOnHost[cIdx] = args;
        }
        argsArray.initialize(cuContext, cudau::BufferType::Device, argsArrayOnHost);

        argsArrayToBuild.initialize(cuContext, cudau::BufferType::Device, clusterCount);
        clasHandles.initialize(cuContext, cudau::BufferType::Device, clusterCount);
        const uint32_t usedFlagsBinCount = (clusterCount + 31) / 32;
        usedFlags.initialize(cuContext, cudau::BufferType::Device, usedFlagsBinCount);
        indexMapClusterToClasBuild.initialize(cuContext, cudau::BufferType::Device, clusterCount);

        Shared::ClusterSetInfo clusterSetInfoOnHost = {};
        clusterSetInfoOnHost.argsArray = argsArrayToBuild.getDevicePointer();
        clusterSetInfoOnHost.clasHandles = clasHandles.getDevicePointer();
        clusterSetInfoOnHost.usedFlags = usedFlags.getDevicePointer();
        clusterSetInfoOnHost.indexMapClusterToClasBuild = indexMapClusterToClasBuild.getDevicePointer();
        clusterSetInfoOnHost.argsCountToBuild = 0;
        clusterSetInfo.initialize(cuContext, cudau::BufferType::Device, 1, clusterSetInfoOnHost);

        clasSet = scene.createClusterAccelerationStructureSet();
        clasSet.setBuildInput(
            OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
            clusterCount, OPTIX_VERTEX_FORMAT_FLOAT3,
            1, maxTriCountPerCluster, maxVertCountPerCluster,
            0, 0, 0);
        clasSet.setMaterialCount(1);
        clasSet.setMaterial(0, mat);

        clasSet.prepareForBuild(&asMemReqs);

        Shared::HierarchicalMeshData hiMeshData = {};
        hiMeshData.vertexPool = vertexPool.getROBuffer<enableBufferOobCheck>();
        hiMeshData.trianglePool = trianglePool.getROBuffer<enableBufferOobCheck>();
        hiMeshData.clusters = clusters.getROBuffer<enableBufferOobCheck>();
        clasSet.setUserData(hiMeshData);

        clasSetMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);

        return true;
    }

    void finalize() {
        vertexPool.finalize();
        trianglePool.finalize();
        childIndexPool.finalize();
        clusters.finalize();

        argsArray.finalize();
        argsArrayToBuild.finalize();
        clasHandles.finalize();
        usedFlags.finalize();
        indexMapClusterToClasBuild.finalize();
        clusterSetInfo.finalize();
        clasSet.destroy();
        clasSetMem.finalize();
    }
};

struct HiMeshInstanceSet {
    struct Instance {
        optixu::Instance optixInst;
        cudau::TypedBuffer<uint32_t> indexMapClasHandleToCluster;
        cudau::TypedBuffer<CUdeviceptr> clasHandles;
        Shared::InstanceTransform transform;
    };

    const HierarchicalMesh* hiMesh;

    uint32_t maxClasCountPerCgas;
    optixu::ClusterGeometryAccelerationStructureSet cgasSet;
    cudau::Buffer cgasSetMem;
    cudau::TypedBuffer<uint32_t> cgasCount;
    cudau::TypedBuffer<OptixClusterAccelBuildInputClustersArgs> cgasArgsArray;
    cudau::TypedBuffer<OptixTraversableHandle> cgasHandles;
    OptixAccelBufferSizes asMemReqs;

    std::vector<Instance> instances;
    uint32_t startInstanceId;

    cudau::TypedBuffer<uint32_t> clasBuildCounts[2];

    void initialize(
        const CUcontext cuContext, const optixu::Scene scene, const HierarchicalMesh* _hiMesh,
        const Shared::InstanceTransform* const transforms, const uint32_t instCount,
        Shared::InstanceStaticInfo* const instStaticInfos, const uint32_t startInstId)
    {
        hiMesh = _hiMesh;

        maxClasCountPerCgas = hiMesh->levelInfos[0].clusterCount;

        cgasSet = scene.createClusterGeometryAccelerationStructureSet();
        cgasSet.setRayTypeCount(Shared::NumRayTypes);
        cgasSet.setBuildInput(
            OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
            instCount, maxClasCountPerCgas, maxClasCountPerCgas * instCount,
            &asMemReqs);
        cgasSet.setChild(hiMesh->clasSet);

        cgasSetMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);

        cgasCount.initialize(cuContext, cudau::BufferType::Device, 1, instCount);
        cgasArgsArray.initialize(cuContext, cudau::BufferType::Device, instCount);
        cgasHandles.initialize(cuContext, cudau::BufferType::Device, instCount);

        instances.resize(instCount);
        startInstanceId = startInstId;
        for (uint32_t cgasIdx = 0; cgasIdx < instCount; ++cgasIdx) {
            Instance &inst = instances[cgasIdx];
            const uint32_t instId = startInstId + cgasIdx;
            inst.optixInst = scene.createInstance();
            inst.optixInst.setChild(cgasSet, cgasIdx);
            inst.optixInst.setID(instId);

            inst.indexMapClasHandleToCluster.initialize(
                cuContext, cudau::BufferType::Device, maxClasCountPerCgas);
            inst.clasHandles.initialize(
                cuContext, cudau::BufferType::Device, maxClasCountPerCgas);

            inst.transform = transforms[cgasIdx];

            Shared::InstanceStaticInfo &instStaticInfo = instStaticInfos[instId];
            instStaticInfo.cgas.indexMapClasHandleToCluster =
                inst.indexMapClasHandleToCluster.getDevicePointer();
            instStaticInfo.cgas.clasHandles = inst.clasHandles.getDevicePointer();
        }

        for (uint32_t i = 0; i < 2; ++i)
            clasBuildCounts[i].initialize(cuContext, cudau::BufferType::Device, instCount);
    }

    void finalize() {
        cgasSet.destroy();
        cgasSetMem.finalize();
        cgasCount.finalize();
        cgasArgsArray.finalize();
        cgasHandles.finalize();
        for (uint32_t cgasIdx = 0; cgasIdx < instances.size(); ++cgasIdx) {
            Instance &inst = instances[cgasIdx];
            inst.optixInst.destroy();
            inst.indexMapClasHandleToCluster.finalize();
            inst.clasHandles.finalize();
        }

        for (uint32_t i = 0; i < 2; ++i)
            clasBuildCounts[i].finalize();
    }

    void setInstanceInfos(Shared::InstanceDynamicInfo* instDynamicInfos) const {
        for (uint32_t cgasIdx = 0; cgasIdx < instances.size(); ++cgasIdx) {
            const Instance &inst = instances[cgasIdx];
            Shared::InstanceDynamicInfo &instDynamicInfo = instDynamicInfos[startInstanceId + cgasIdx];
            instDynamicInfo.cgas.clasHandleCount = 0;
            instDynamicInfo.transform = inst.transform;

            const Shared::InstanceTransform &xfm = inst.transform;
            const Matrix3x3 matSR = xfm.scale * xfm.orientation.toMatrix3x3();
            const float xfm_values[] = {
                matSR.m00, matSR.m01, matSR.m02, xfm.position.x,
                matSR.m10, matSR.m11, matSR.m12, xfm.position.y,
                matSR.m20, matSR.m21, matSR.m22, xfm.position.z,
            };
            inst.optixInst.setTransform(xfm_values);
        }
    }
};



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "clusters";

    bool takeScreenShot = false;
    auto visualizationMode = Shared::VisualizationMode_ShadingNormal;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot") {
            takeScreenShot = true;
        }
        else if (arg == "--visualize") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --visualize is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "shading-normal")
                visualizationMode = Shared::VisualizationMode_ShadingNormal;
            else if (visType == "geom-normal")
                visualizationMode = Shared::VisualizationMode_GeometricNormal;
            else if (visType == "cluster")
                visualizationMode = Shared::VisualizationMode_Cluster;
            else
                throw std::runtime_error("Argument for --visualize is invalid.");
            argIdx += 1;
        }
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    /*
    JP: クラスタージオメトリを使う場合はパイプラインで許可する必要がある。
    EN: When using clustered geometry, it must be allowed in the pipeline.
    */
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.attributeCountInDwords = optixu::calcSumDwords<float2>();
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipelineOptions.allowClusteredGeometry = optixu::AllowClusteredGeometry::Yes;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "clusters/ptxes/optix_kernels.optixir");
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

    // JP: 
    // EN: 
    CUmodule lodDecisionModule;
    CUDADRV_CHECK(cuModuleLoad(
        &lodDecisionModule, (resourceDir / "ptxes/lod_decision_kernels.ptx").string().c_str()));
    cudau::Kernel emitClasArgsArray(lodDecisionModule, "emitClasArgsArray", cudau::dim3(32), 0);
    cudau::Kernel copyClasHandles(lodDecisionModule, "copyClasHandles", cudau::dim3(32), 0);
    cudau::Kernel emitClusterGasArgsArray(lodDecisionModule, "emitClusterGasArgsArray", cudau::dim3(32), 0);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material mat = optixContext.createMaterial();
    mat.setHitGroup(Shared::RayType_Primary, hitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    OptixAccelBufferSizes asMemReqs;
    size_t maxSizeOfScratchBuffer = 0;
    cudau::Buffer asBuildScratchMem;

    constexpr uint32_t maxInstCount = 1024;
    cudau::TypedBuffer<Shared::InstanceStaticInfo> instStaticInfoBuffer(
        cuContext, cudau::BufferType::Device, maxInstCount);

    uint32_t instId = 0;
    Shared::InstanceStaticInfo* instStaticInfos = instStaticInfoBuffer.map();



    HierarchicalMesh bunnyHiMesh;
    bunnyHiMesh.read(
        cuContext, scene, mat,
        R"(E:\assets\McguireCGArchive\bunny\bunny_000.himesh)");
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, bunnyHiMesh.asMemReqs.tempSizeInBytes);

    const Shared::InstanceTransform bunnyInstXfms[] = {
        { 1.0f, Quaternion(), float3(0.0f, 0.0f, 0.0f) },
        { 1.0f, Quaternion(), float3(-3.0f, 0.0f, -5.0f) },
        { 1.0f, Quaternion(), float3(15.0f, 0.0f, -50.0f) },
    };
    constexpr uint32_t bunnyCount = lengthof(bunnyInstXfms);
    HiMeshInstanceSet bunnyHiMeshInstSet;
    bunnyHiMeshInstSet.initialize(
        cuContext, scene, &bunnyHiMesh,
        bunnyInstXfms, bunnyCount, instStaticInfos, instId);
    instId += bunnyCount;
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, bunnyHiMeshInstSet.asMemReqs.tempSizeInBytes);



    HierarchicalMesh dragonHiMesh;
    dragonHiMesh.read(
        cuContext, scene, mat,
        R"(E:\assets\McguireCGArchive\dragon\dragon_000.himesh)");
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dragonHiMesh.asMemReqs.tempSizeInBytes);

    const Shared::InstanceTransform dragonInstXfms[] = {
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(1.0f, 0.7f, -3.0f) },
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(5.0f, 0.7f, -10.0f) },
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(-20.0f, 0.7f, -100.0f) },
    };
    constexpr uint32_t dragonCount = lengthof(dragonInstXfms);
    HiMeshInstanceSet dragonHiMeshInstSet;
    dragonHiMeshInstSet.initialize(
        cuContext, scene, &dragonHiMesh,
        dragonInstXfms, dragonCount, instStaticInfos, instId);
    instId += dragonCount;
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dragonHiMeshInstSet.asMemReqs.tempSizeInBytes);



    optixu::GeometryInstance floorGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> floorVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> floorTriangleBuffer;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-100.0f, 0.0f, -100.0f), make_float3(0, 1, 0) },
            { make_float3(-100.0f, 0.0f, 100.0f), make_float3(0, 1, 0) },
            { make_float3(100.0f, 0.0f, 100.0f), make_float3(0, 1, 0) },
            { make_float3(100.0f, 0.0f, -100.0f), make_float3(0, 1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floorVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        floorTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::NormalMeshData meshData = {};
        meshData.vertices = floorVertexBuffer.getROBuffer<enableBufferOobCheck>();
        meshData.triangles = floorTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        floorGeomInst.setVertexBuffer(floorVertexBuffer);
        floorGeomInst.setTriangleBuffer(floorTriangleBuffer);
        floorGeomInst.setMaterialCount(1, optixu::BufferView());
        floorGeomInst.setMaterial(0, 0, mat);
        floorGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        floorGeomInst.setUserData(meshData);
    }

    optixu::GeometryAccelerationStructure floorGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer floorGasMem;
    floorGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
    floorGas.setMaterialSetCount(1);
    floorGas.setRayTypeCount(0, Shared::NumRayTypes);
    floorGas.addChild(floorGeomInst);
    floorGas.prepareForBuild(&asMemReqs);
    floorGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floorGas);
    floorInst.setID(instId++);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    for (uint32_t bunnyIdx = 0; bunnyIdx < bunnyCount; ++bunnyIdx)
        ias.addChild(bunnyHiMeshInstSet.instances[bunnyIdx].optixInst);
    for (uint32_t dragonIdx = 0; dragonIdx < dragonCount; ++dragonIdx)
        ias.addChild(dragonHiMeshInstSet.instances[dragonIdx].optixInst);
    ias.addChild(floorInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: IASビルド時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when building an IAS.
    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    floorGas.rebuild(0, floorGasMem, asBuildScratchMem);

    instStaticInfoBuffer.unmap();

    HiMeshInstanceSet* const hiMeshInstSets[] = {
        &bunnyHiMeshInstSet,
        &dragonHiMeshInstSet,
    };
    const char* hiMeshInstSetNames[] = {
        "Bunny",
        "Dragon",
    };

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr int32_t initWindowContentWidth = 1280;
    constexpr int32_t initWindowContentHeight = 720;

    const auto computeHaltonSequence = []
    (uint32_t base, uint32_t idx) {
        const float recBase = 1.0f / base;
        float ret = 0.0f;
        float scale = 1.0f;
        while (idx) {
            scale *= recBase;
            ret += (idx % base) * scale;
            idx /= base;
        }
        return ret;
    };
    float2 subPixelOffsets[64];
    for (int i = 0; i < lengthof(subPixelOffsets); ++i)
        subPixelOffsets[i] = float2(computeHaltonSequence(2, i), computeHaltonSequence(3, i));

    Shared::PipelineLaunchParameters plp;
    plp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(initWindowContentWidth) / initWindowContentHeight;
    plp.instStaticInfoBuffer = instStaticInfoBuffer.getROBuffer<enableBufferOobCheck>();

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - Clusters";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = initWindowContentWidth;
    initConfig.windowContentRenderHeight = initWindowContentHeight;
    initConfig.cameraPosition = make_float3(0, 1.0f, 3.16f);
    initConfig.cameraOrientation = qRotateY(pi_v<float>);
    initConfig.cameraMovingSpeed = 0.05f;
    initConfig.cuContext = cuContext;

    GUIFramework framework;
    framework.initialize(initConfig);

    cudau::Array outputArray;
    outputArray.initializeFromGLTexture2D(
        cuContext, framework.getOutputTexture().getHandle(),
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable);

    cudau::InteropSurfaceObjectHolder<2> outputBufferSurfaceHolder;
    outputBufferSurfaceHolder.initialize({ &outputArray });

    constexpr Shared::PickInfo initPickInfo = {
        0xFFFF'FFFF,
        OPTIX_CLUSTER_ID_INVALID,
        0xFFFF'FFFF,
        float2(0.0f, 0.0f)
    };

    struct GPUTimer {
        cudau::Timer frame;
        cudau::Timer update;
        cudau::Timer render;

        void initialize(CUcontext context) {
            frame.initialize(context);
            update.initialize(context);
            render.initialize(context);
        }
        void finalize() {
            render.finalize();
            update.finalize();
            frame.finalize();
        }
    };

    cudau::TypedBuffer<Shared::InstanceDynamicInfo> instDynamicInfoBuffers[2];
    cudau::TypedBuffer<Shared::PickInfo> pickInfos[2];
    GPUTimer gpuTimers[2];
    for (uint32_t i = 0; i < 2; ++i) {
        instDynamicInfoBuffers[i].initialize(cuContext, cudau::BufferType::Device, maxInstCount);
        pickInfos[i].initialize(cuContext, cudau::BufferType::Device, 1, initPickInfo);
        gpuTimers[i].initialize(cuContext);
    }

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;
        const uint32_t bufIdx = frameIndex % 2;
        GPUTimer &curGPUTimer = gpuTimers[bufIdx];
        cudau::TypedBuffer<Shared::PickInfo> &curPickInfo = pickInfos[bufIdx];
        cudau::TypedBuffer<Shared::InstanceDynamicInfo> &curInstDynamicInfoBuffer =
            instDynamicInfoBuffers[bufIdx];

        // Camera Window
        bool cameraIsActuallyMoving = args.cameraIsActuallyMoving;
        {
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera & Rendering", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            ImGui::Text("W/A/S/D/R/F: Move, Q/E: Tilt");
            ImGui::Text("Mouse Middle Drag: Rotate");

            if (ImGui::InputFloat3("Position", reinterpret_cast<float*>(&args.cameraPosition)))
                cameraIsActuallyMoving = true;
            static float rollPitchYaw[3];
            args.tempCameraOrientation.toEulerAngles(&rollPitchYaw[0], &rollPitchYaw[1], &rollPitchYaw[2]);
            rollPitchYaw[0] *= 180 / pi_v<float>;
            rollPitchYaw[1] *= 180 / pi_v<float>;
            rollPitchYaw[2] *= 180 / pi_v<float>;
            if (ImGui::InputFloat3("Roll/Pitch/Yaw", rollPitchYaw)) {
                args.cameraOrientation = qFromEulerAngles(
                    rollPitchYaw[0] * pi_v<float> / 180,
                    rollPitchYaw[1] * pi_v<float> / 180,
                    rollPitchYaw[2] * pi_v<float> / 180);
                cameraIsActuallyMoving = true;
            }
            ImGui::Text("Pos. Speed (T/G): %g", args.cameraPositionalMovingSpeed);

            ImGui::End();
        }

        plp.camera.position = args.cameraPosition;
        plp.camera.orientation = args.tempCameraOrientation.toMatrix3x3();



        // Debug Window
        static bool enableJittering = false;
        static Shared::LoDMode lodMode = Shared::LoDMode_ViewAdaptive;
        static int32_t lodLevel = 0;
        static bool lockLod = false;
        bool enableJitteringChanged = false;
        bool lodModeChanged = false;
        bool lodLevelChanged = false;
        bool visModeChanged = false;
        bool lockLodChanged = false;
        {
            ImGui::SetNextWindowPos(ImVec2(712, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            const bool oldEnableJittering = enableJittering;
            ImGui::Checkbox("Jittering", &enableJittering);
            enableJitteringChanged = enableJittering != oldEnableJittering;

            const Shared::LoDMode oldLodMode = lodMode;
            const uint32_t oldLodLevel = lodLevel;
            const bool oldLockLod = lockLod;
            ImGui::CollapsingHeader("LoD", ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::RadioButtonE("View Adaptive", &lodMode, Shared::LoDMode_ViewAdaptive);
            ImGui::RadioButtonE("Manual Uniform", &lodMode, Shared::LoDMode_ManualUniform);
            if (ImGui::SliderInt("Level", &lodLevel, 0, 15))
                lodMode = Shared::LoDMode_ManualUniform;
            ImGui::Checkbox("Lock LoD", &lockLod);
            lodModeChanged = lodMode != oldLodMode;
            lodLevelChanged = lodLevel != oldLodLevel;
            lockLodChanged = lockLod != oldLockLod;

            ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PushID("visMode");
            visModeChanged |= ImGui::RadioButtonE(
                "Shading Normal", &visualizationMode, Shared::VisualizationMode_ShadingNormal);
            visModeChanged |= ImGui::RadioButtonE(
                "Geometric Normal", &visualizationMode, Shared::VisualizationMode_GeometricNormal);
            visModeChanged |= ImGui::RadioButtonE(
                "Cluster", &visualizationMode, Shared::VisualizationMode_Cluster);
            visModeChanged |= ImGui::RadioButtonE(
                "Level", &visualizationMode, Shared::VisualizationMode_Level);
            visModeChanged |= ImGui::RadioButtonE(
                "Triangle", &visualizationMode, Shared::VisualizationMode_Triangle);
            ImGui::PopID();

            const Shared::PickInfo pickInfo = curPickInfo.map(curStream)[0];
            curPickInfo.unmap(curStream);

            ImGui::CollapsingHeader("Cursor Info", ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::Text(
                "Cursor: %u, %u",
                uint32_t(args.mouseX), uint32_t(args.mouseY));
            ImGui::Text("Instance Index: %u", pickInfo.instanceIndex);
            ImGui::Text("Cluster ID: %u", pickInfo.clusterId);
            ImGui::Text("Primitive Index: %u", pickInfo.primitiveIndex);
            ImGui::Text("Barycentrics: %.3f, %.3f", pickInfo.barycentrics.x, pickInfo.barycentrics.y);
            ImGui::Text("Cluster Info");
            ImGui::Text("  Level: %u", pickInfo.cluster.level);
            ImGui::Text("  Vertex Count: %u", pickInfo.cluster.vertexCount);
            ImGui::Text("  Triangle Count: %u", pickInfo.cluster.triangleCount);

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 144), ImGuiCond_FirstUseEver);
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float cudaFrameTime = curGPUTimer.frame.report();
            float updateTime = curGPUTimer.update.report();
            float renderTime = curGPUTimer.render.report();
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime);
            ImGui::Text("  Update: %.3f [ms]", updateTime);
            ImGui::Text("  Render: %.3f [ms]", renderTime);

            ImGui::Separator();

            std::vector<Shared::InstanceDynamicInfo> instDynInfos(maxInstCount);
            curInstDynamicInfoBuffer.read(instDynInfos, curStream);

            for (uint32_t setIdx = 0; setIdx < lengthof(hiMeshInstSets); ++setIdx) {
                const HiMeshInstanceSet &hiMeshInstSet = *hiMeshInstSets[setIdx];
                const char* const setName = hiMeshInstSetNames[setIdx];

                uint32_t clasCountToBuild = 0;
                CUDADRV_CHECK(cuMemcpyDtoHAsync(
                    &clasCountToBuild, hiMeshInstSet.clasBuildCounts[bufIdx].getCUdeviceptr(),
                    sizeof(uint32_t), curStream));
                ImGui::Text("%s", setName);
                ImGui::Text("  Total CLAS Count: %u", clasCountToBuild);
                for (uint32_t cgasIdx = 0; cgasIdx < hiMeshInstSet.instances.size(); ++cgasIdx) {
                    const Shared::InstanceDynamicInfo &instDynInfo =
                        instDynInfos[hiMeshInstSet.startInstanceId + cgasIdx];
                    ImGui::Text("    Inst %u CLAS Count: %u", cgasIdx, instDynInfo.cgas.clasHandleCount);
                }
            }

            ImGui::End();
        }



        curGPUTimer.frame.start(curStream);

        if ((lodMode == Shared::LoDMode_ViewAdaptive ||
             lodModeChanged ||
             lodLevelChanged) && !lockLod ||
            frameIndex == 0)
        {
            const auto executeInBatch = [&]
            (std::function<void(const HiMeshInstanceSet &)> f) {
                for (uint32_t setIdx = 0; setIdx < lengthof(hiMeshInstSets); ++setIdx)
                    f(*hiMeshInstSets[setIdx]);
            };

            // JP: 
            // EN: 
            Shared::InstanceDynamicInfo* const instDynInfos = curInstDynamicInfoBuffer.map(curStream);
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                hiMeshInstSet.setInstanceInfos(instDynInfos);
            });
            curInstDynamicInfoBuffer.unmap(curStream);

            // JP: ビルドするCLAS数カウンターとArgs設定済みを表すフラグ列をリセットする。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const HierarchicalMesh &hiMesh = *hiMeshInstSet.hiMesh;

                const CUdeviceptr clasCountToBuildPtr =
                    hiMesh.clusterSetInfo.getCUdeviceptr() +
                    offsetof(Shared::ClusterSetInfo, argsCountToBuild);
                CUDADRV_CHECK(cuMemsetD32Async(clasCountToBuildPtr, 0, 1, curStream));
                CUDADRV_CHECK(cuMemsetD32Async(
                    hiMesh.usedFlags.getCUdeviceptr(), 0, hiMesh.usedFlags.sizeInBytes() / 4, curStream));
            });

            // JP: メッシュの各インスタンス・各クラスターをテストして、ビルドの必要があるクラスターを特定する。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const HierarchicalMesh &hiMesh = *hiMeshInstSet.hiMesh;
                const uint32_t hiMeshTotalClusterCount = hiMesh.clusters.numElements();
                const uint32_t hiMeshInstCount = hiMeshInstSet.instances.size();

                const uint32_t instClusterCountStride = (hiMeshTotalClusterCount + 31) / 32 * 32;
                emitClasArgsArray.launchWithThreadDim(
                    curStream, cudau::dim3(instClusterCountStride * hiMeshInstCount),
                    lodMode, lodLevel,
                    plp.camera.position, plp.camera.orientation,
                    plp.camera.fovY, args.windowContentRenderHeight,
                    hiMesh.clusters, hiMesh.argsArray,
                    hiMeshTotalClusterCount, hiMesh.levelInfos.size(),
                    hiMesh.clusterSetInfo,
                    instStaticInfoBuffer.getDevicePointerAt(hiMeshInstSet.startInstanceId),
                    curInstDynamicInfoBuffer.getDevicePointerAt(hiMeshInstSet.startInstanceId),
                    hiMeshInstCount);
            });

            // JP: 今回のフレームで使用するCLAS集合をビルドする。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const HierarchicalMesh &hiMesh = *hiMeshInstSet.hiMesh;

                const CUdeviceptr clasCountToBuildPtr =
                    hiMesh.clusterSetInfo.getCUdeviceptr() +
                    offsetof(Shared::ClusterSetInfo, argsCountToBuild);
                hiMesh.clasSet.rebuild(
                    curStream,
                    hiMesh.argsArrayToBuild, clasCountToBuildPtr, hiMesh.clasSetMem, asBuildScratchMem,
                    hiMesh.clasHandles);
            });

            // JP: 各インスタンスのCluster GAS構築のため、それぞれのCLASハンドルバッファーに
            //     対応するクラスターのハンドルをコピーする。
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const HierarchicalMesh &hiMesh = *hiMeshInstSet.hiMesh;
                const uint32_t hiMeshInstCount = hiMeshInstSet.instances.size();

                copyClasHandles.launchWithThreadDim(
                    curStream, cudau::dim3(hiMeshInstSet.maxClasCountPerCgas * hiMeshInstCount),
                    hiMeshInstSet.maxClasCountPerCgas, hiMesh.clusterSetInfo,
                    instStaticInfoBuffer.getDevicePointerAt(hiMeshInstSet.startInstanceId),
                    hiMeshInstCount);
            });

            // JP: 各インスタンスに対応するCGAS入力を生成する。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const uint32_t hiMeshInstCount = hiMeshInstSet.instances.size();

                emitClusterGasArgsArray.launchWithThreadDim(
                    curStream, cudau::dim3(hiMeshInstCount),
                    instStaticInfoBuffer.getDevicePointerAt(hiMeshInstSet.startInstanceId),
                    curInstDynamicInfoBuffer.getDevicePointerAt(hiMeshInstSet.startInstanceId),
                    hiMeshInstCount,
                    hiMeshInstSet.cgasArgsArray);
            });

            // JP: CGAS集合をビルドする。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                hiMeshInstSet.cgasSet.rebuild(
                    curStream,
                    hiMeshInstSet.cgasArgsArray, hiMeshInstSet.cgasCount.getCUdeviceptr(),
                    hiMeshInstSet.cgasSetMem, asBuildScratchMem,
                    hiMeshInstSet.cgasHandles);
            });

            // JP: 統計情報をCPU側で処理するためにコピーする。
            // EN: 
            executeInBatch([&](const HiMeshInstanceSet &hiMeshInstSet)
            {
                const HierarchicalMesh &hiMesh = *hiMeshInstSet.hiMesh;

                const cudau::TypedBuffer<uint32_t> &curClasBuildCount = hiMeshInstSet.clasBuildCounts[bufIdx];
                const CUdeviceptr clasCountToBuildPtr =
                    hiMesh.clusterSetInfo.getCUdeviceptr() +
                    offsetof(Shared::ClusterSetInfo, argsCountToBuild);

                CUDADRV_CHECK(cuMemcpyDtoDAsync(
                    curClasBuildCount.getCUdeviceptr(), clasCountToBuildPtr,
                    sizeof(uint32_t), curStream));
            });
        }

        /*
        JP: IASのリビルドを行う。
            アップデートの代用としてのリビルドでは、インスタンスの追加・削除や
            ASビルド設定の変更を行っていないのでmarkDirty()やprepareForBuild()は必要無い。
        EN: Rebuild the IAS.
            Rebuild as the alternative for update doesn't involves
            add/remove of instances and changes of AS build settings
            so neither of markDirty() nor prepareForBuild() is required.
        */
        curGPUTimer.update.start(curStream);
        //if (animate)
        plp.travHandle = ias.rebuild(curStream, instanceBuffer, iasMem, asBuildScratchMem);
        curGPUTimer.update.stop(curStream);

        bool firstAccumFrame =
            //animate ||
            cameraIsActuallyMoving ||
            args.resized ||
            frameIndex == 0 ||
            enableJitteringChanged || lodModeChanged || lodLevelChanged || lockLodChanged ||
            visModeChanged;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;

        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        // Render
        {
            curGPUTimer.render.start(curStream);

            plp.colorAccumBuffer = outputBufferSurfaceHolder.getNext();
            plp.pickInfo = curPickInfo.getDevicePointer();
            plp.instDynamicInfoBuffer = curInstDynamicInfoBuffer.getROBuffer<enableBufferOobCheck>();
            plp.mousePosition = uint2(uint32_t(args.mouseX), uint32_t(args.mouseY));
            plp.subPixelOffset = enableJittering ?
                subPixelOffsets[numAccumFrames % static_cast<uint32_t>(lengthof(subPixelOffsets))] :
                float2(0.5f, 0.5f);
            plp.sampleIndex = std::min(numAccumFrames, static_cast<uint32_t>(lengthof(subPixelOffsets)) - 1);
            plp.visMode = visualizationMode;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));

            pipeline.launch(
                curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);

            ++numAccumFrames;

            curGPUTimer.render.stop(curStream);
        }

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);

        curGPUTimer.frame.stop(curStream);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = false;
        ret.finish = false;

        if (takeScreenShot && frameIndex + 1 == lengthof(subPixelOffsets)) {
            CUDADRV_CHECK(cuStreamSynchronize(curStream));
            const uint32_t numPixels = args.windowContentRenderWidth * args.windowContentRenderHeight;
            auto rawImage = new float4[numPixels];
            glGetTextureSubImage(
                args.outputTexture->getHandle(), 0,
                0, 0, 0, args.windowContentRenderWidth, args.windowContentRenderHeight, 1,
                GL_RGBA, GL_FLOAT, sizeof(float4) * numPixels, rawImage);
            saveImage("output.png", args.windowContentRenderWidth, args.windowContentRenderHeight, rawImage,
                      false, ret.enable_sRGB);
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
        plp.camera.aspect = static_cast<float>(windowContentWidth) / windowContentHeight;
    };

    framework.run(onRenderLoop, onResolutionChange);

    for (uint32_t i = 0; i < 2; ++i) {
        instDynamicInfoBuffers[i].finalize();
        pickInfos[i].finalize();
        gpuTimers[i].finalize();
    }

    outputBufferSurfaceHolder.finalize();
    outputArray.finalize();

    framework.finalize();

    // END: Display the window.
    // ----------------------------------------------------------------



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    hitGroupSBT.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    floorInst.destroy();
    floorGasMem.finalize();
    floorGas.destroy();
    floorTriangleBuffer.finalize();
    floorVertexBuffer.finalize();
    floorGeomInst.destroy();

    dragonHiMeshInstSet.finalize();
    dragonHiMesh.finalize();

    bunnyHiMeshInstSet.finalize();
    bunnyHiMesh.finalize();

    instStaticInfoBuffer.finalize();

    scene.destroy();



    mat.destroy();



    CUDADRV_CHECK(cuModuleUnload(lodDecisionModule));

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
