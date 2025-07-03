/*

JP: 

EN: 

*/

#include "clusters_shared.h"
#define OPTIXU_ENABLE_PRIVATE_ACCESS 1
#include "../../optix_util_private.h"

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
    cudau::TypedBuffer<Shared::Vertex> vertexPool;
    cudau::TypedBuffer<Shared::LocalTriangle> trianglePool;
    cudau::TypedBuffer<uint32_t> childIndexPool;
    cudau::TypedBuffer<Shared::Cluster> clusters;
    cudau::TypedBuffer<uint32_t> levelStartClusterIndices;
    cudau::TypedBuffer<OptixClusterAccelBuildInputTrianglesArgs> argsArray;
    std::vector<uint32_t> levelStartClusterIndicesOnHost;
    uint32_t maxVertCountPerCluster;
    uint32_t maxTriCountPerCluster;

    bool read(CUcontext cuContext, const std::filesystem::path &filePath) {
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
        levelStartClusterIndicesOnHost.resize(levelCount);
        ifs
            .read(verticesOnHost)
            .read(trianglesOnHost)
            .read(padding)
            .read(childIndicesOnHost)
            .read(clustersOnHost)
            .read(levelStartClusterIndicesOnHost);

        vertexPool.initialize(
            cuContext, cudau::BufferType::Device, verticesOnHost);
        trianglePool.initialize(
            cuContext, cudau::BufferType::Device, trianglesOnHost);
        childIndexPool.initialize(
            cuContext, cudau::BufferType::Device, childIndicesOnHost);
        clusters.initialize(
            cuContext, cudau::BufferType::Device, clustersOnHost);
        levelStartClusterIndices.initialize(
            cuContext, cudau::BufferType::Device, levelStartClusterIndicesOnHost);

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
        argsArray.initialize(
            cuContext, cudau::BufferType::Device, argsArrayOnHost);

        return true;
    }

    void finalize() {
        vertexPool.finalize();
        trianglePool.finalize();
        childIndexPool.finalize();
        clusters.finalize();
        levelStartClusterIndices.finalize();
        argsArray.finalize();
    }
};

int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "clusters";

    bool takeScreenShot = false;
    auto visualizationMode = Shared::VisualizationMode_GeometricNormal;

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
            if (visType == "geom-normal")
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
    JP: 
    EN: 
    */
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.numPayloadValuesInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.numAttributeValuesInDwords = optixu::calcSumDwords<float2>();
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
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
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
    cudau::Kernel emitClusterArgsArray(lodDecisionModule, "emitClusterArgsArray", cudau::dim3(32), 0);
    cudau::Kernel emitCgasArgsArray(lodDecisionModule, "emitCgasArgsArray", cudau::dim3(32), 0);

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

#define OPTIX_CHECK(call) \
    do { \
        const OptixResult error = call; \
        if (error != OPTIX_SUCCESS) { \
            std::stringstream ss; \
            ss << "OptiX call (" << #call << ") failed: " \
               << "(" __FILE__ << ":" << __LINE__ << ")\n"; \
            throw std::runtime_error(ss.str().c_str()); \
        } \
    } while (0)

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    HierarchicalMesh himesh;
    himesh.read(
        cuContext, R"(E:\assets\McguireCGArchive\bunny\bunny_000.himesh)");

    Assert(
        himesh.maxVertCountPerCluster <= optixContext.getMaxVertexCountPerCluster(),
        "Too many vertices per cluster %u > %u.",
        himesh.maxVertCountPerCluster, optixContext.getMaxVertexCountPerCluster());
    Assert(
        himesh.maxTriCountPerCluster <= optixContext.getMaxTriangleCountPerCluster(),
        "Too many triangles per cluster %u > %u.",
        himesh.maxTriCountPerCluster, optixContext.getMaxTriangleCountPerCluster());

    const uint32_t maxInflightClusterCount =
        himesh.levelStartClusterIndicesOnHost[1] - himesh.levelStartClusterIndicesOnHost[0];

    cudau::TypedBuffer<uint32_t> clusterCount;
    cudau::TypedBuffer<OptixClusterAccelBuildInputTrianglesArgs> clusterArgs;
    OptixClusterAccelBuildInput clasBuildInput = {};
    OptixClusterAccelBuildMode constexpr clasAccelBuildMode =
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    cudau::Buffer clasSetMem;
    cudau::TypedBuffer<CUdeviceptr> clasHandles;
    {
        clusterCount.initialize(cuContext, cudau::BufferType::Device, 1);
        clusterArgs.initialize(cuContext, cudau::BufferType::Device, maxInflightClusterCount);

        clasBuildInput.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES;
        OptixClusterAccelBuildInputTriangles &triBuildInput = clasBuildInput.triangles;
        triBuildInput.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        triBuildInput.maxArgCount = maxInflightClusterCount;
        triBuildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triBuildInput.maxSbtIndexValue = 0;
        triBuildInput.maxUniqueSbtIndexCountPerArg = 1;
        triBuildInput.maxTriangleCountPerArg = himesh.maxTriCountPerCluster;
        triBuildInput.maxVertexCountPerArg = himesh.maxVertCountPerCluster;
        triBuildInput.maxTotalTriangleCount = 0; // optional
        triBuildInput.maxTotalVertexCount = 0; // optional
        triBuildInput.minPositionTruncateBitCount = 0;

        OPTIX_CHECK(optixClusterAccelComputeMemoryUsage(
            optixContext.getOptixDeviceContext(), clasAccelBuildMode,
            &clasBuildInput, &asMemReqs));
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

        clasSetMem.initialize(
            cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        clasHandles.initialize(
            cuContext, cudau::BufferType::Device, maxInflightClusterCount);
    }

    optixu::ClusterGeometryAccelerationStructure cgas = scene.createClusterGeometryAccelerationStructure();
    cudau::TypedBuffer<uint32_t> cgasCount;
    cudau::TypedBuffer<OptixClusterAccelBuildInputClustersArgs> cgasArgs;
    OptixClusterAccelBuildInput cgasBuildInput = {};
    OptixClusterAccelBuildMode constexpr cgasAccelBuildMode =
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    cudau::Buffer cgasSetMem;
    cudau::TypedBuffer<OptixTraversableHandle> cgasHandles;
    {
        cgasCount.initialize(cuContext, cudau::BufferType::Device, 1);
        cgasArgs.initialize(cuContext, cudau::BufferType::Device, 1);

        cgasBuildInput.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
        OptixClusterAccelBuildInputClusters &clusterBuildInput = cgasBuildInput.clusters;
        clusterBuildInput.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        clusterBuildInput.maxArgCount = 1;
        clusterBuildInput.maxTotalClusterCount = maxInflightClusterCount;
        clusterBuildInput.maxClusterCountPerArg = maxInflightClusterCount;

        OPTIX_CHECK(optixClusterAccelComputeMemoryUsage(
            optixContext.getOptixDeviceContext(), cgasAccelBuildMode,
            &cgasBuildInput, &asMemReqs));
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

        cgasSetMem.initialize(
            cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        cgasHandles.initialize(
            cuContext, cudau::BufferType::Device, 1);

        cgas.setHandleAddress(cgasHandles.getCUdeviceptrAt(0));
        cgas.setNumRayTypes(Shared::NumRayTypes);
        cgas.setSbtRequirements(
            sizeof(Shared::GeometryData), alignof(Shared::GeometryData),
            1);
    }


    
    float bunnyInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };
    optixu::Instance bunnyInst = scene.createInstance();
    bunnyInst.setChild(cgas);
    bunnyInst.setTransform(bunnyInstXfm);

    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(bunnyInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
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
    initConfig.cameraPosition = make_float3(0, 0, 3.16f);
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

    GPUTimer gpuTimers[2];
    gpuTimers[0].initialize(cuContext);
    gpuTimers[1].initialize(cuContext);

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;
        GPUTimer &curGPUTimer = gpuTimers[frameIndex % 2];

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
        static Shared::LoDMode lodMode = Shared::LoDMode_ViewAdaptive;
        static int32_t lodLevel = 0;
        static bool lockLod = false;
        bool lodModeChanged = false;
        bool lodLevelChanged = false;
        bool visModeChanged = false;
        bool lockLodChanged = false;
        {
            ImGui::SetNextWindowPos(ImVec2(712, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

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
            visModeChanged |= ImGui::RadioButtonE(
                "Geometric Normal", &visualizationMode, Shared::VisualizationMode_GeometricNormal);
            visModeChanged |= ImGui::RadioButtonE(
                "Cluster", &visualizationMode, Shared::VisualizationMode_Cluster);

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

            ImGui::End();
        }



        curGPUTimer.frame.start(curStream);

        if ((lodMode == Shared::LoDMode_ViewAdaptive ||
             lodModeChanged ||
             lodLevelChanged) && !lockLod ||
            frameIndex == 0)
        {
            clusterCount.fill(0, curStream);
            emitClusterArgsArray.launchWithThreadDim(
                curStream, cudau::dim3(himesh.clusters.numElements()),
                lodMode, lodLevel,
                plp.camera.position, plp.camera.orientation,
                plp.camera.fovY, args.windowContentRenderHeight,
                himesh.vertexPool, himesh.trianglePool,
                himesh.clusters, himesh.argsArray,
                himesh.clusters.numElements(),
                himesh.levelStartClusterIndices, himesh.levelStartClusterIndices.numElements(),
                clusterArgs, clusterCount);
            {
                OptixClusterAccelBuildModeDesc buildModeDesc = {};
                buildModeDesc.mode = clasAccelBuildMode;
                OptixClusterAccelBuildModeDescImplicitDest &buildModeDescImpDst =
                    buildModeDesc.implicitDest;
                buildModeDescImpDst.outputBuffer = clasSetMem.getCUdeviceptr();
                buildModeDescImpDst.outputBufferSizeInBytes = clasSetMem.sizeInBytes();
                buildModeDescImpDst.tempBuffer = asBuildScratchMem.getCUdeviceptr();
                buildModeDescImpDst.tempBufferSizeInBytes = asBuildScratchMem.sizeInBytes();
                buildModeDescImpDst.outputHandlesBuffer = clasHandles.getCUdeviceptr();
                buildModeDescImpDst.outputHandlesStrideInBytes = clasHandles.stride();
                buildModeDescImpDst.outputSizesBuffer = 0; // optional
                buildModeDescImpDst.outputSizesStrideInBytes = 0; // optional

                OPTIX_CHECK(optixClusterAccelBuild(
                    optixContext.getOptixDeviceContext(), curStream,
                    &buildModeDesc, &clasBuildInput,
                    clusterArgs.getCUdeviceptr(), clusterCount.getCUdeviceptr(), clusterArgs.stride()));
            }

            cgasCount.fill(0, curStream);
            emitCgasArgsArray.launchWithThreadDim(
                curStream, cudau::dim3(1),
                clasHandles, clasHandles.stride(), clusterCount,
                cgasArgs, cgasCount);
            {
                OptixClusterAccelBuildModeDesc buildModeDesc = {};
                buildModeDesc.mode = cgasAccelBuildMode;
                OptixClusterAccelBuildModeDescImplicitDest &buildModeDescImpDst =
                    buildModeDesc.implicitDest;
                buildModeDescImpDst.outputBuffer = cgasSetMem.getCUdeviceptr();
                buildModeDescImpDst.outputBufferSizeInBytes = cgasSetMem.sizeInBytes();
                buildModeDescImpDst.tempBuffer = asBuildScratchMem.getCUdeviceptr();
                buildModeDescImpDst.tempBufferSizeInBytes = asBuildScratchMem.sizeInBytes();
                buildModeDescImpDst.outputHandlesBuffer = cgasHandles.getCUdeviceptr();
                buildModeDescImpDst.outputHandlesStrideInBytes = cgasHandles.stride();
                buildModeDescImpDst.outputSizesBuffer = 0; // optional
                buildModeDescImpDst.outputSizesStrideInBytes = 0; // optional

                OPTIX_CHECK(optixClusterAccelBuild(
                    optixContext.getOptixDeviceContext(), curStream,
                    &buildModeDesc, &cgasBuildInput,
                    cgasArgs.getCUdeviceptr(), cgasCount.getCUdeviceptr(), cgasArgs.stride()));
            }
        }

        //// JP: 各インスタンスのトランスフォームを更新する。
        //// EN: Update the transform of each instance.
        //if (animate || lastFrameWasAnimated) {
        //    for (int i = 0; i < bunnyInsts.size(); ++i) {
        //        MovingInstance &bunnyInst = bunnyInsts[i];
        //        bunnyInst.update(animate ? 1.0f / 60.0f : 0.0f);
        //        // TODO: まとめて送る。
        //        CUDADRV_CHECK(cuMemcpyHtoDAsync(
        //            instDataBuffer.getCUdeviceptrAt(bunnyInst.ID),
        //            &bunnyInst.instData, sizeof(bunnyInsts[i].instData), curStream));
        //    }
        //}

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
            lodModeChanged || lodLevelChanged || lockLodChanged ||
            visModeChanged;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;

        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        // Render
        {
            curGPUTimer.render.start(curStream);

            plp.colorAccumBuffer = outputBufferSurfaceHolder.getNext();
            plp.subPixelOffset = subPixelOffsets[numAccumFrames % static_cast<uint32_t>(lengthof(subPixelOffsets))];
            plp.sampleIndex = std::min(numAccumFrames, static_cast<uint32_t>(lengthof(subPixelOffsets)) - 1);
            plp.visMode = visualizationMode;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
            {
                using namespace optixu;
                uint8_t* sbtHostMem = reinterpret_cast<uint8_t*>(hitGroupSBT.getMappedPointer());
                _Material* _mat = std::bit_cast<optixu::_Material*>(mat);
                SizeAlign curSizeAlign;
                _mat->setRecordHeader(
                    optixu::extract(pipeline),
                    Shared::RayType_Primary, sbtHostMem + 0,
                    &curSizeAlign);
                uint32_t offset;
                optixu::SizeAlign userDataSizeAlign(
                    sizeof(Shared::GeometryData),
                    alignof(Shared::GeometryData));
                curSizeAlign.add(userDataSizeAlign, &offset);
            }
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

    gpuTimers[1].finalize();
    gpuTimers[0].finalize();

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

    bunnyInst.destroy();

    cgasCount.finalize();
    cgasHandles.finalize();
    cgasSetMem.finalize();
    cgasArgs.finalize();
    cgas.destroy();

    clasHandles.finalize();
    clasSetMem.finalize();
    clusterArgs.finalize();
    clusterCount.finalize();

    himesh.finalize();

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
