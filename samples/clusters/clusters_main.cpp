/*

JP: このサンプルではクラスターAPIの使用方法を示す。
    従来のGASは単一のメッシュが持つ頂点・三角形列から直接構築されていたが、クラスターAPIの下では
    巨大な単一のGASを構築する代わりに、メッシュを複数のクラスターに(ユーザー側で)分割して管理し、
    CLAS列の構築とCLAS集合の上になるGASの構築でメッシュに対応するASを行う。
    CLASそれぞれは比較的小さなASであるため、メッシュ全体に対するASビルド時のGPU利用効率が向上し、
    メッシュの部分ごとに容易にクラスターを差し替えられることで柔軟な高密度ジオメトリ表現が可能になる。
    このサンプルではUnreal Engine 5 [1]で導入されたNaniteのように、DAGを使用したクラスター化メッシュの
    AS構築とレンダリングを行う。

EN: This sample demonstrates the usage of the cluster API.
    Unlike the traditional GAS, which was built directly from the vertex and triangle arrays of a single mesh,
    under the cluster API, the mesh is split into multiple clusters (managed by the user),
    and the AS corresponding to the mesh is constructed by building a set of CLAS and
    then building a GAS on top of the CLAS set.
    Each CLAS is a relatively small AS, which improves GPU utilization efficiency during AS build
    for the entire mesh, and allows for easy replacement of clusters for different parts of the mesh,
    enabling flexible high-density geometry representation.
    This sample performs AS construction and rendering of clustered meshes using a DAG,
    similar to Nanite introduced in Unreal Engine 5 [1].

[1] Nanite: A Deep Dive

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



static constexpr uint32_t minPosTruncBitCount = 0;

struct ClusteredMesh {
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

    cudau::TypedBuffer<uint32_t> outputSizesArray[2];

    std::vector<LevelInfo> levelInfos;
    uint32_t maxVertCountPerCluster;
    uint32_t maxTriCountPerCluster;

    bool read(
        const CUcontext cuContext, const optixu::Scene scene, const optixu::Material mat,
        const std::filesystem::path &filePath, const float errorScaleCoeff)
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



        // JP: このサンプルコードにおいては、各クラスターの設定はほとんど静的なのでホスト側で構築しておく。
        // EN: In this sample code, the settings for each cluster are mostly static,
        //     so we build them on the host side.
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
        clusterSetInfoOnHost.meshErrorScale = errorScaleCoeff * clustersOnHost.back().bounds.radius;
        clusterSetInfo.initialize(cuContext, cudau::BufferType::Device, 1, clusterSetInfoOnHost);

        clasSet = scene.createClusterAccelerationStructureSet();
        clasSet.setBuildInput(
            OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
            clusterCount, OPTIX_VERTEX_FORMAT_FLOAT3,
            1, maxTriCountPerCluster, maxVertCountPerCluster,
            0, 0, minPosTruncBitCount);
        clasSet.setMaterialCount(1);
        clasSet.setMaterial(0, mat);

        clasSet.prepareForBuild(&asMemReqs);

        AABB bbox;
        for (uint32_t vIdx = 0; vIdx < vertexCount; ++vIdx) {
            const Shared::Vertex &v = verticesOnHost[vIdx];
            bbox.unify(v.position);
        }

        Shared::ClusteredMeshData cMeshData = {};
        cMeshData.vertexPool = vertexPool.getROBuffer<enableBufferOobCheck>();
        cMeshData.trianglePool = trianglePool.getROBuffer<enableBufferOobCheck>();
        cMeshData.clusters = clusters.getROBuffer<enableBufferOobCheck>();
        cMeshData.bbox = bbox;
        clasSet.setUserData(cMeshData);

        clasSetMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);

        for (uint32_t i = 0; i < 2; ++i)
            outputSizesArray[i].initialize(cuContext, cudau::BufferType::Device, clusterCount);

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

        for (uint32_t i = 0; i < 2; ++i)
            outputSizesArray[i].finalize();
    }
};

struct ClusteredMeshInstanceSet {
    struct Instance {
        optixu::Instance optixInst;
        cudau::TypedBuffer<uint32_t> indexMapClasHandleToCluster;
        cudau::TypedBuffer<CUdeviceptr> clasHandles;
        Shared::InstanceTransform transform;
    };

    const ClusteredMesh* cMesh;

    uint32_t maxClasCountPerCgas;
    optixu::ClusterGeometryAccelerationStructureSet cgasSet;
    cudau::Buffer cgasSetMem;
    cudau::TypedBuffer<uint32_t> cgasCount;
    cudau::TypedBuffer<OptixClusterAccelBuildInputClustersArgs> cgasArgsArray;
    cudau::TypedBuffer<OptixTraversableHandle> cgasHandles;
    OptixAccelBufferSizes asMemReqs;

    std::vector<Instance> instances;
    uint32_t startInstanceId;

    cudau::TypedBuffer<uint32_t> clasBuildCountsArray[2];

    void initialize(
        const CUcontext cuContext, const optixu::Scene scene, const ClusteredMesh* _cMesh,
        const Shared::InstanceTransform* const transforms, const uint32_t instCount,
        Shared::InstanceStaticInfo* const instStaticInfos, const uint32_t startInstId)
    {
        cMesh = _cMesh;

        maxClasCountPerCgas = cMesh->levelInfos[0].clusterCount;

        cgasSet = scene.createClusterGeometryAccelerationStructureSet();
        cgasSet.setRayTypeCount(Shared::NumRayTypes);
        cgasSet.setBuildInput(
            OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE,
            instCount, maxClasCountPerCgas, maxClasCountPerCgas * instCount,
            &asMemReqs);
        cgasSet.setChild(cMesh->clasSet);

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
            clasBuildCountsArray[i].initialize(cuContext, cudau::BufferType::Device, 1 + instCount);
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
            clasBuildCountsArray[i].finalize();
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
    const std::filesystem::path dataDir = R"(../../data)";

    bool takeScreenShot = false;
    auto visualizationMode = Shared::VisualizationMode_Final;
    auto lodMode = Shared::LoDMode_ViewAdaptive;
    int32_t lodLevel = 0;

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
            if (visType == "final")
                visualizationMode = Shared::VisualizationMode_Final;
            else if (visType == "shading-normal")
                visualizationMode = Shared::VisualizationMode_ShadingNormal;
            else if (visType == "geom-normal")
                visualizationMode = Shared::VisualizationMode_GeometricNormal;
            else if (visType == "cluster")
                visualizationMode = Shared::VisualizationMode_Cluster;
            else if (visType == "level")
                visualizationMode = Shared::VisualizationMode_Level;
            else if (visType == "triangle")
                visualizationMode = Shared::VisualizationMode_Triangle;
            else
                throw std::runtime_error("Argument for --visualize is invalid.");
            argIdx += 1;
        }
        else if (arg == "--lod-mode") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --lod-mode is not complete.");
            std::string_view lodModeStr = argv[argIdx + 1];
            if (lodModeStr == "view-adaptive") {
                lodMode = Shared::LoDMode_ViewAdaptive;
                argIdx += 1;
            }
            else if (lodModeStr == "manual-uniform") {
                if (argIdx + 2 >= argc)
                    throw std::runtime_error("Argument for --lod-mode manual-uniform is not complete.");
                lodMode = Shared::LoDMode_ManualUniform;
                lodLevel = std::stoi(argv[argIdx + 2]);
                argIdx += 2;
            }
            else
                throw std::runtime_error("Argument for --lod-mode is invalid.");
        }
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

    // ----------------------------------------------------------------
    // JP: OptiXのコンテキストとパイプラインの設定。
    // EN: Settings for OptiX context and pipeline.

    CUcontext cuContext;
    CUDADRV_CHECK(cuInit(0));
#if CUDA_VERSION < 13000
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
#else
    CUctxCreateParams cuCtxCreateParams = {};
    CUDADRV_CHECK(cuCtxCreate(&cuContext, &cuCtxCreateParams, 0, 0));
#endif
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: クラスタージオメトリを使う場合はパイプラインで許可する必要がある。
    // EN: When using clustered geometry, it must be allowed in the pipeline.
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
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup primaryHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("visibility"),
        emptyModule, nullptr);

    // JP: このサンプルではClosest-Hit ProgramからシャドウレイをトレースするためTrace Depthは2になる。
    // EN: In this sample, the trace depth is 2 because shadow rays are traced from the Closest-Hit Program.
    pipeline.link(2);

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setMissRayTypeCount(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    pipeline.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);
    shaderBindingTable.setMappedMemoryPersistent(true);
    pipeline.setShaderBindingTable(shaderBindingTable, shaderBindingTable.getMappedPointer());

    // JP: 描画するクラスターを決定するカーネルなどをロードする。
    // EN: Load kernels to determine which clusters to render.
    CUmodule lodDecisionModule;
    CUDADRV_CHECK(cuModuleLoad(
        &lodDecisionModule, (resourceDir / "ptxes/lod_decision_kernels.ptx").string().c_str()));
    cudau::Kernel emitClasArgsArray(lodDecisionModule, "emitClasArgsArray", cudau::dim3(32), 0);
    cudau::Kernel copyClasHandles(lodDecisionModule, "copyClasHandles", cudau::dim3(32), 0);
    cudau::Kernel emitClusterGasArgsArray(lodDecisionModule, "emitClusterGasArgsArray", cudau::dim3(32), 0);
    cudau::Kernel copyDataForCpu(lodDecisionModule, "copyDataForCpu", cudau::dim3(32), 0);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material mat = optixContext.createMaterial();
    mat.setHitGroup(Shared::RayType_Primary, primaryHitProgramGroup);
    mat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);

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



    /*
    JP: クラスター化メッシュを読み込む。
        cmeshファイルは広く使われている形式ではなく著者オリジナル。データ構造はファイル読み込みコード参照。
        meshoptimizerとMETISを使用してメッシュのクラスター化、グルーピング、単純化を繰り返し行い、
        DAGで表現される階層構造と各クラスターごとのBounding Sphereやエラー値などを持っている。
    EN: Load clustered meshes.
        The cmesh file is not a widely used format and is original to the author.
        For the data structure, refer to the file reading code.
        The mesh is clustered, grouped, and simplified repeatedly using meshoptimizer and METIS,
        and it has a hierarchical structure represented by a DAG,
        along with bounding spheres and error values for each cluster.
    */

    ClusteredMesh bunnyCMesh;
    bunnyCMesh.read(cuContext, scene, mat, dataDir / "bunny_big.cmesh", 1.0f);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, bunnyCMesh.asMemReqs.tempSizeInBytes);
    printf(
        "Bunny: %u clusters, %u levels\n",
        bunnyCMesh.clusters.sizeInBytes(), bunnyCMesh.levelInfos.size());

    const Shared::InstanceTransform bunnyInstXfms[] = {
        { 1.0f, Quaternion(), float3(0.0f, 0.0f, 0.0f) },
        { 1.0f, Quaternion(), float3(-3.0f, 0.0f, -5.0f) },
        { 1.0f, Quaternion(), float3(15.0f, 0.0f, -50.0f) },
    };
    constexpr uint32_t bunnyCount = lengthof(bunnyInstXfms);
    ClusteredMeshInstanceSet bunnyCMeshInstSet;
    bunnyCMeshInstSet.initialize(
        cuContext, scene, &bunnyCMesh,
        bunnyInstXfms, bunnyCount, instStaticInfos, instId);
    instId += bunnyCount;
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, bunnyCMeshInstSet.asMemReqs.tempSizeInBytes);



    ClusteredMesh dragonCMesh;
    dragonCMesh.read(cuContext, scene, mat, dataDir / "dragon_big.cmesh", 0.2f);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dragonCMesh.asMemReqs.tempSizeInBytes);
    printf(
        "Dragon: %u clusters, %u levels\n",
        dragonCMesh.clusters.sizeInBytes(), dragonCMesh.levelInfos.size());

    const Shared::InstanceTransform dragonInstXfms[] = {
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(1.0f, 0.7f, -3.0f) },
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(5.0f, 0.7f, -10.0f) },
        { 2.5f, qRotateY(0.5f * pi_v<float>), float3(-20.0f, 0.7f, -100.0f) },
    };
    constexpr uint32_t dragonCount = lengthof(dragonInstXfms);
    ClusteredMeshInstanceSet dragonCMeshInstSet;
    dragonCMeshInstSet.initialize(
        cuContext, scene, &dragonCMesh,
        dragonInstXfms, dragonCount, instStaticInfos, instId);
    instId += dragonCount;
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dragonCMeshInstSet.asMemReqs.tempSizeInBytes);



    // JP: 床は通常メッシュ。
    // EN: The floor is a normal mesh.
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

        AABB bbox;
        for (const Shared::Vertex &v : vertices)
            bbox.unify(v.position);

        Shared::NormalMeshData meshData = {};
        meshData.vertices = floorVertexBuffer.getROBuffer<enableBufferOobCheck>();
        meshData.triangles = floorTriangleBuffer.getROBuffer<enableBufferOobCheck>();
        meshData.bbox = bbox;

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
        ias.addChild(bunnyCMeshInstSet.instances[bunnyIdx].optixInst);
    for (uint32_t dragonIdx = 0; dragonIdx < dragonCount; ++dragonIdx)
        ias.addChild(dragonCMeshInstSet.instances[dragonIdx].optixInst);
    ias.addChild(floorInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    cudau::Buffer hitGroupSBT;
    size_t hitGroupSbtSize;
    scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
    hitGroupSBT.initialize(cuContext, cudau::BufferType::Device, hitGroupSbtSize, 1);
    hitGroupSBT.setMappedMemoryPersistent(true);

    floorGas.rebuild(0, floorGasMem, asBuildScratchMem);

    instStaticInfoBuffer.unmap();

    ClusteredMeshInstanceSet* const cMeshInstSets[] = {
        &bunnyCMeshInstSet,
        &dragonCMeshInstSet,
    };
    const char* cMeshInstSetNames[] = {
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

    float lightDirPhi = -16;
    float lightDirTheta = 60;
    float lightStrengthInLog10 = 0.8f;

    Shared::PipelineLaunchParameters plp;
    plp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(initWindowContentWidth) / initWindowContentHeight;
    plp.instStaticInfoBuffer = instStaticInfoBuffer.getROBuffer<enableBufferOobCheck>();
    plp.envRadiance = float3(0.10f, 0.13f, 0.9f);

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
        cudau::Timer clasBuilds;
        cudau::Timer iasBuild;
        cudau::Timer render;

        void initialize(CUcontext context) {
            frame.initialize(context);
            clasBuilds.initialize(context);
            iasBuild.initialize(context);
            render.initialize(context);
        }
        void finalize() {
            render.finalize();
            iasBuild.finalize();
            clasBuilds.finalize();
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
        const uint32_t prevBufIdx = (frameIndex + 1) % 2;
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
        static bool lockLod = false;
        static int32_t posTruncBitWidth = 0;
        bool enableJitteringChanged = false;
        bool lightParamChanged = false;
        bool lodModeChanged = false;
        bool lodLevelChanged = false;
        bool visModeChanged = false;
        bool lockLodChanged = false;
        bool posTruncBitWidthChanged = false;
        {
            ImGui::SetNextWindowPos(ImVec2(712, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            const bool oldEnableJittering = enableJittering;
            ImGui::Checkbox("Jittering", &enableJittering);
            enableJitteringChanged = enableJittering != oldEnableJittering;

            if (ImGui::CollapsingHeader("Light", ImGuiTreeNodeFlags_DefaultOpen)) {
                const float oldStrength = lightStrengthInLog10;
                const float oldPhi = lightDirPhi;
                const float oldTheta = lightDirTheta;
                ImGui::SliderFloat("Strength", &lightStrengthInLog10, -2, 2);
                ImGui::SliderFloat("Phi", &lightDirPhi, -180, 180);
                ImGui::SliderFloat("Theta", &lightDirTheta, 0, 90);
                lightParamChanged =
                    lightStrengthInLog10 != oldStrength
                    || lightDirPhi != oldPhi || lightDirTheta != oldTheta;
            }

            if (ImGui::CollapsingHeader("Geometry", ImGuiTreeNodeFlags_DefaultOpen)) {
                const Shared::LoDMode oldLodMode = lodMode;
                const uint32_t oldLodLevel = lodLevel;
                const bool oldLockLod = lockLod;
                ImGui::Text("LoD");
                ImGui::RadioButtonE("View Adaptive", &lodMode, Shared::LoDMode_ViewAdaptive);
                ImGui::RadioButtonE("Manual Uniform", &lodMode, Shared::LoDMode_ManualUniform);
                if (ImGui::SliderInt("Level", &lodLevel, 0, 15))
                    lodMode = Shared::LoDMode_ManualUniform;
                ImGui::Checkbox("Lock", &lockLod);
                lodModeChanged = lodMode != oldLodMode;
                lodLevelChanged = lodLevel != oldLodLevel;
                lockLodChanged = lockLod != oldLockLod;

                ImGui::Separator();

                const int32_t oldPosTruncBitWidth = posTruncBitWidth;
                ImGui::SliderInt("Pos Truncation", &posTruncBitWidth, minPosTruncBitCount, 20);
                posTruncBitWidthChanged = posTruncBitWidth != oldPosTruncBitWidth;
            }

            if (ImGui::CollapsingHeader("Visualization", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushID("visMode");
                visModeChanged |= ImGui::RadioButtonE(
                    "Final", &visualizationMode, Shared::VisualizationMode_Final);
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
            }

            const Shared::PickInfo pickInfo = curPickInfo.map(curStream)[0];
            curPickInfo.unmap(curStream);

            if (ImGui::CollapsingHeader("Cursor Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text(
                    "Cursor: %u, %u",
                    uint32_t(args.mouseX), uint32_t(args.mouseY));
                ImGui::Text("Instance Index: %d", static_cast<int32_t>(pickInfo.instanceIndex));
                ImGui::Text("Cluster ID: %d", static_cast<int32_t>(pickInfo.clusterId));
                ImGui::Text("Primitive Index: %d", static_cast<int32_t>(pickInfo.primitiveIndex));
                ImGui::Text("Barycentrics: %.3f, %.3f", pickInfo.barycentrics.x, pickInfo.barycentrics.y);
                ImGui::Text("Cluster Info");
                ImGui::Text("  Level: %u", pickInfo.cluster.level);
                ImGui::Text("  Vertex Count: %u", pickInfo.cluster.vertexCount);
                ImGui::Text("  Triangle Count: %u", pickInfo.cluster.triangleCount);
            }

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 144), ImGuiCond_FirstUseEver);
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            float cudaFrameTime = curGPUTimer.frame.report();
            float clasBuildsTime = curGPUTimer.clasBuilds.report();
            float iasBuildTime = curGPUTimer.iasBuild.report();
            float renderTime = curGPUTimer.render.report();
            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("CUDA/OptiX GPU %.3f [ms]:", cudaFrameTime);
            ImGui::Text("  CLAS Builds: %.3f [ms]", clasBuildsTime);
            ImGui::Text("  IAS Build: %.3f [ms]", iasBuildTime);
            ImGui::Text("  Render: %.3f [ms]", renderTime);

            if (ImGui::CollapsingHeader("CLAS Stats", ImGuiTreeNodeFlags_DefaultOpen)) {
                for (uint32_t setIdx = 0; setIdx < lengthof(cMeshInstSets); ++setIdx) {
                    const ClusteredMeshInstanceSet &cMeshInstSet = *cMeshInstSets[setIdx];
                    const cudau::TypedBuffer<uint32_t> &curClasBuildCounts =
                        cMeshInstSet.clasBuildCountsArray[bufIdx];
                    const cudau::TypedBuffer<uint32_t> &curOutputSizes =
                        cMeshInstSet.cMesh->outputSizesArray[bufIdx];

                    std::vector<uint32_t> clasBuildCountsOnHost(curClasBuildCounts.numElements());
                    std::vector<uint32_t> outputSizesOnHost(curOutputSizes.numElements());
                    curClasBuildCounts.read(clasBuildCountsOnHost, curStream);
                    curOutputSizes.read(outputSizesOnHost, curStream);

                    const uint32_t buildCount = clasBuildCountsOnHost[0];

                    uint32_t minSize = UINT32_MAX, maxSize = 0;
                    uint32_t sumSizes = 0;
                    for (uint32_t i = 0; i < buildCount; ++i) {
                        minSize = std::min(minSize, outputSizesOnHost[i]);
                        maxSize = std::max(maxSize, outputSizesOnHost[i]);
                        sumSizes += outputSizesOnHost[i];
                    }

                    ImGui::Text("%s", cMeshInstSetNames[setIdx]);
                    ImGui::Text("  Total Count: %u", buildCount);
                    for (uint32_t cgasIdx = 0; cgasIdx < cMeshInstSet.instances.size(); ++cgasIdx) {
                        ImGui::Text("    Inst %u: %u", cgasIdx, clasBuildCountsOnHost[1 + cgasIdx]);
                    }
                    ImGui::Text("  Size: %u - %u [bytes]: ", minSize, maxSize);
                    ImGui::Text("    Avg: %u", buildCount > 0 ? sumSizes / buildCount : 0);
                }
            }

            ImGui::End();
        }



        curGPUTimer.frame.start(curStream);

        const auto executeInBatch = [&]
        (std::function<void(const ClusteredMeshInstanceSet &)> f) {
            for (uint32_t setIdx = 0; setIdx < lengthof(cMeshInstSets); ++setIdx)
                f(*cMeshInstSets[setIdx]);
        };

        // JP: クラスター化メッシュの各インスタンスのトランスフォームなどをセットし、
        //     Cluster GASに紐づくCLASカウンターをリセットする。
        // EN: Set transforms and other information for each instance of clustered meshes,
        //     and reset the CLAS counter associated with the Cluster GAS.
        Shared::InstanceDynamicInfo* const instDynInfos = curInstDynamicInfoBuffer.map(curStream);
        executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
        {
            cMeshInstSet.setInstanceInfos(instDynInfos);
        });
        curInstDynamicInfoBuffer.unmap(curStream);

        curGPUTimer.clasBuilds.start(curStream);

        if ((lodMode == Shared::LoDMode_ViewAdaptive ||
             lodModeChanged || lodLevelChanged || posTruncBitWidthChanged) && !lockLod ||
            frameIndex == 0)
        {
            // JP: 各クラスター化メッシュに関して、ビルドするCLASカウンターとArgs設定済みを表すフラグ列を
            //     リセットする。
            // EN: For each clustered mesh, reset the CLAS counter and flags indicating that the args are set.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;

                const CUdeviceptr clasCountToBuildPtr =
                    cMesh.clusterSetInfo.getCUdeviceptr() +
                    offsetof(Shared::ClusterSetInfo, argsCountToBuild);
                CUDADRV_CHECK(cuMemsetD32Async(clasCountToBuildPtr, 0, 1, curStream));
                CUDADRV_CHECK(cuMemsetD32Async(
                    cMesh.usedFlags.getCUdeviceptr(), 0, cMesh.usedFlags.sizeInBytes() / 4, curStream));
            });

            // JP: メッシュの各インスタンス・各クラスターをテストして、ビルドの必要があるクラスターを特定する。
            // EN: Test each instance and each cluster of the mesh to identify clusters that need to be built.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;
                const uint32_t cMeshTotalClusterCount = cMesh.clusters.numElements();
                const uint32_t cMeshInstCount = cMeshInstSet.instances.size();

                Shared::GeometryConfig geomConfig = {};
                geomConfig.lodMode = lodMode;
                geomConfig.manualUniformLevel = lodLevel;
                geomConfig.positionTruncateBitWidth = posTruncBitWidth;

                const uint32_t instClusterCountStride = (cMeshTotalClusterCount + 31) / 32 * 32;
                emitClasArgsArray.launchWithThreadDim(
                    curStream, cudau::dim3(instClusterCountStride * cMeshInstCount),
                    geomConfig,
                    plp.camera.position, plp.camera.orientation,
                    plp.camera.fovY, args.windowContentRenderHeight,
                    cMesh.clusters, cMesh.argsArray,
                    cMeshTotalClusterCount, cMesh.levelInfos.size(),
                    cMesh.clusterSetInfo,
                    instStaticInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId),
                    curInstDynamicInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId),
                    cMeshInstCount);
            });

            // JP: 今回のフレームで使用するCLAS集合をビルドする。
            // EN: Build the CLAS set to be used in this frame.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;
                const cudau::TypedBuffer<uint32_t> &curOutputSizes = cMesh.outputSizesArray[bufIdx];

                const CUdeviceptr clasCountToBuildPtr =
                    cMesh.clusterSetInfo.getCUdeviceptr() +
                    offsetof(Shared::ClusterSetInfo, argsCountToBuild);
                cMesh.clasSet.rebuild(
                    curStream,
                    cMesh.argsArrayToBuild, clasCountToBuildPtr, cMesh.clasSetMem, asBuildScratchMem,
                    cMesh.clasHandles, curOutputSizes);
            });

            // JP: 各インスタンスのCluster GAS構築のため、それぞれのCLASハンドルバッファーに
            //     対応するクラスターのハンドルをコピーする。
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;
                const uint32_t cMeshInstCount = cMeshInstSet.instances.size();

                copyClasHandles.launchWithThreadDim(
                    curStream, cudau::dim3(cMeshInstSet.maxClasCountPerCgas * cMeshInstCount),
                    cMeshInstSet.maxClasCountPerCgas, cMesh.clusterSetInfo,
                    instStaticInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId),
                    cMeshInstCount);
            });

            // JP: 各インスタンスに対応するCGAS入力を生成する。
            // EN: Generate CGAS inputs corresponding to each instance.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const uint32_t cMeshInstCount = cMeshInstSet.instances.size();

                emitClusterGasArgsArray.launchWithThreadDim(
                    curStream, cudau::dim3(cMeshInstCount),
                    instStaticInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId),
                    curInstDynamicInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId),
                    cMeshInstCount,
                    cMeshInstSet.cgasArgsArray);
            });

            // JP: 各クラスターメッシュのCGAS集合をビルドする。
            // EN: Build the CGAS set for each clustered mesh.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                cMeshInstSet.cgasSet.rebuild(
                    curStream,
                    cMeshInstSet.cgasArgsArray, cMeshInstSet.cgasCount.getCUdeviceptr(),
                    cMeshInstSet.cgasSetMem, asBuildScratchMem,
                    cMeshInstSet.cgasHandles);
            });

            // JP: 統計情報をCPU側で処理するためにコピーする。
            // EN: Copy statistics information to the CPU side.
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;
                const uint32_t cMeshInstCount = cMeshInstSet.instances.size();

                const cudau::TypedBuffer<uint32_t> &curClasBuildCounts =
                    cMeshInstSet.clasBuildCountsArray[bufIdx];

                copyDataForCpu.launchWithThreadDim(
                    curStream, cudau::dim3(cMeshInstCount),
                    cMesh.clusterSetInfo,
                    curInstDynamicInfoBuffer.getDevicePointerAt(cMeshInstSet.startInstanceId), cMeshInstCount,
                    curClasBuildCounts);
            });
        }
        else {
            executeInBatch([&](const ClusteredMeshInstanceSet &cMeshInstSet)
            {
                const ClusteredMesh &cMesh = *cMeshInstSet.cMesh;

                const cudau::TypedBuffer<uint32_t> &curClasBuildCounts =
                    cMeshInstSet.clasBuildCountsArray[bufIdx];
                const cudau::TypedBuffer<uint32_t> &prevClasBuildCounts =
                    cMeshInstSet.clasBuildCountsArray[prevBufIdx];
                const cudau::TypedBuffer<uint32_t> &curOutputSizes =
                    cMesh.outputSizesArray[bufIdx];
                const cudau::TypedBuffer<uint32_t> &prevOutputSizes =
                    cMesh.outputSizesArray[prevBufIdx];

                CUDADRV_CHECK(cuMemcpyDtoDAsync(
                    curClasBuildCounts.getCUdeviceptr(), prevClasBuildCounts.getCUdeviceptr(),
                    curClasBuildCounts.sizeInBytes(), curStream));
                CUDADRV_CHECK(cuMemcpyDtoDAsync(
                    curOutputSizes.getCUdeviceptr(), prevOutputSizes.getCUdeviceptr(),
                    curOutputSizes.sizeInBytes(), curStream));
            });
        }

        curGPUTimer.clasBuilds.stop(curStream);

        // JP: IASのリビルドを行う。
        // EN: Rebuild the IAS.
        curGPUTimer.iasBuild.start(curStream);
        //if (animate)
        plp.travHandle = ias.rebuild(curStream, instanceBuffer, iasMem, asBuildScratchMem);
        curGPUTimer.iasBuild.stop(curStream);

        bool firstAccumFrame =
            //animate ||
            cameraIsActuallyMoving ||
            args.resized ||
            frameIndex == 0 ||
            enableJitteringChanged || lightParamChanged ||
            lodModeChanged || lodLevelChanged || lockLodChanged || posTruncBitWidthChanged ||
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
            plp.lightDirection = fromPolarYUp(lightDirPhi * pi_v<float> / 180, lightDirTheta * pi_v<float> / 180);
            plp.lightRadiance = float3(std::pow(10.0f, lightStrengthInLog10));
            plp.subPixelOffset = enableJittering ?
                subPixelOffsets[numAccumFrames % static_cast<uint32_t>(lengthof(subPixelOffsets))] :
                float2(0.5f, 0.5f);
            plp.sampleIndex = std::min(numAccumFrames, static_cast<uint32_t>(lengthof(subPixelOffsets)) - 1);
            plp.visMode = visualizationMode;
            plp.posTruncateBitWidth = posTruncBitWidth;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));

            pipeline.launch(
                curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);

            ++numAccumFrames;

            curGPUTimer.render.stop(curStream);
        }

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);

        curGPUTimer.frame.stop(curStream);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = visualizationMode == Shared::VisualizationMode_Final;
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

    dragonCMeshInstSet.finalize();
    dragonCMesh.finalize();

    bunnyCMeshInstSet.finalize();
    bunnyCMesh.finalize();

    instStaticInfoBuffer.finalize();

    scene.destroy();



    mat.destroy();



    CUDADRV_CHECK(cuModuleUnload(lodDecisionModule));

    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
    primaryHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    rayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}
