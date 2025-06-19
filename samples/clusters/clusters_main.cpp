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
    struct Cluster {
        uint32_t vertPoolStartIndex;
        uint32_t triPoolStartIndex;
        uint32_t childIndexPoolStartIndex;
        uint32_t parentStartClusterIndex;
        uint32_t vertexCount : 12;
        uint32_t triangleCount : 12;
        uint32_t childCount : 4;
        uint32_t parentCount : 4;
    };

    cudau::TypedBuffer<Shared::Vertex> vertexPool;
    cudau::TypedBuffer<Shared::LocalTriangle> trianglePool;
    cudau::TypedBuffer<uint32_t> childIndexPool;
    cudau::TypedBuffer<Cluster> clusters;
    cudau::TypedBuffer<uint32_t> levelStartClusterIndices;
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

        std::vector<Shared::Vertex> verticesOnHost(vertexCount);
        std::vector<Shared::LocalTriangle> trianglesOnHost(triangleCount);
        std::vector<uint32_t> childIndicesOnHost(childIndexCount);
        std::vector<Cluster> clustersOnHost(clusterCount);
        levelStartClusterIndicesOnHost.resize(levelCount);
        ifs
            .read(verticesOnHost)
            .read(trianglesOnHost)
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

        return true;
    }
};

int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "clusters";

    bool takeScreenShot = false;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot") {
            takeScreenShot = true;
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
    JP: カーブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
        複数のカーブタイプがあり、このサンプルでは全て使用する。
        カーブのアトリビュートサイズは1Dword(float)。
    EN: Appropriately setting primitive type flags is required since this sample uses curve intersection.
        There are multiple curve types and the sample use all of them.
        The attribute size of curves is 1 Dword (float).
    */
    pipeline.setPipelineOptions(
        Shared::MyPayloadSignature::numDwords,
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

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

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    HierarchicalMesh himesh;
    himesh.read(
        cuContext, R"(E:\assets\McguireCGArchive\bunny\bunny_000.himesh)");

    constexpr uint32_t lodLevel = 0;
    const uint32_t levelStartClusterIndex = himesh.levelStartClusterIndicesOnHost[lodLevel];
    const uint32_t nextLevelStartClusterIndex =
        (lodLevel + 1) < himesh.levelStartClusterIndicesOnHost.size() ?
        himesh.levelStartClusterIndicesOnHost[lodLevel + 1] :
        himesh.clusters.numElements();
    const uint32_t levelClusterCount = nextLevelStartClusterIndex - levelStartClusterIndex;
    uint32_t maxClusterCount = levelClusterCount;

    cudau::TypedBuffer<OptixClusterAccelBuildInputTrianglesArgs> clusterArgs;
    OptixClusterAccelBuildInput clasBuildInput = {};
    OptixClusterAccelBuildMode constexpr clasAccelBuildMode =
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    cudau::Buffer clasSetMem;
    cudau::TypedBuffer<CUdeviceptr> clasHandles;
    cudau::TypedBuffer<uint32_t> clusterCount;
    {
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

        uint32_t maxVertexCountPerCluster;
        OPTIX_CHECK(optixDeviceContextGetProperty(
            optixContext.getOptixDeviceContext(),
            OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_VERTICES,
            &maxVertexCountPerCluster,
            sizeof(maxVertexCountPerCluster)));
        Assert(himesh.maxVertCountPerCluster < maxVertexCountPerCluster);

        uint32_t maxTriangleCountPerCluster;
        OPTIX_CHECK(optixDeviceContextGetProperty(
            optixContext.getOptixDeviceContext(),
            OPTIX_DEVICE_PROPERTY_LIMIT_MAX_CLUSTER_TRIANGLES,
            &maxTriangleCountPerCluster,
            sizeof(maxTriangleCountPerCluster)));
        Assert(himesh.maxTriCountPerCluster < maxTriangleCountPerCluster);

        std::vector<HierarchicalMesh::Cluster> clustersOnHost = himesh.clusters;
        std::vector<OptixClusterAccelBuildInputTrianglesArgs> clusterArgsOnHost(levelClusterCount);
        uint32_t maxSbtIndexValue = 0;
        for (uint32_t cIdx = 0; cIdx < levelClusterCount; ++cIdx) {
            const HierarchicalMesh::Cluster &cluster = clustersOnHost[levelStartClusterIndex + cIdx];
            OptixClusterAccelBuildInputTrianglesArgs &args = clusterArgsOnHost[cIdx];
            args = {};
            args.clusterId = levelStartClusterIndex + cIdx;
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
            args.indexBuffer = himesh.trianglePool.getCUdeviceptrAt(cluster.triPoolStartIndex);
            args.vertexBuffer = himesh.vertexPool.getCUdeviceptrAt(cluster.vertPoolStartIndex);
            args.primitiveInfoBuffer = 0; // not used in this sample
            args.opacityMicromapArray = 0; // not used in this sample
            args.opacityMicromapIndexBuffer = 0; // not used in this sample
            args.instantiationBoundingBoxLimit = 0; // ignored for this arg type

            maxSbtIndexValue = std::max(maxSbtIndexValue, args.basePrimitiveInfo.sbtIndex);
        }
        clusterArgs.initialize(cuContext, cudau::BufferType::Device, clusterArgsOnHost);

        clasBuildInput.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_CLUSTERS_FROM_TRIANGLES;
        OptixClusterAccelBuildInputTriangles &triBuildInput = clasBuildInput.triangles;
        triBuildInput.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        triBuildInput.maxArgCount = levelClusterCount;
        triBuildInput.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triBuildInput.maxSbtIndexValue = maxSbtIndexValue;
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
            cuContext, cudau::BufferType::Device, levelClusterCount, CUdeviceptr(0xFFFF'FFFF'FFFF'FFFF));
        clusterCount.initialize(
            cuContext, cudau::BufferType::Device, &levelClusterCount, 1);
    }

    optixu::ClusterGeometryAccelerationStructure cgas = scene.createClusterGeometryAccelerationStructure();
    cudau::TypedBuffer<OptixClusterAccelBuildInputClustersArgs> cgasArgs;
    OptixClusterAccelBuildInput cgasBuildInput = {};
    OptixClusterAccelBuildMode constexpr cgasAccelBuildMode =
        OPTIX_CLUSTER_ACCEL_BUILD_MODE_IMPLICIT_DESTINATIONS;
    cudau::Buffer cgasSetMem;
    cudau::TypedBuffer<OptixTraversableHandle> cgasHandles;
    cudau::TypedBuffer<uint32_t> cgasCount;
    {
        OptixClusterAccelBuildInputClustersArgs cgasArgOnHost = {};
        {
            cgasArgOnHost.clusterHandlesCount = levelClusterCount;
            cgasArgOnHost.clusterHandlesBufferStrideInBytes = clasHandles.stride();
            cgasArgOnHost.clusterHandlesBuffer = clasHandles.getCUdeviceptr();
        }
        cgasArgs.initialize(cuContext, cudau::BufferType::Device, &cgasArgOnHost, 1);

        cgasBuildInput.type = OPTIX_CLUSTER_ACCEL_BUILD_TYPE_GASES_FROM_CLUSTERS;
        OptixClusterAccelBuildInputClusters &clusterBuildInput = cgasBuildInput.clusters;
        clusterBuildInput.flags = OPTIX_CLUSTER_ACCEL_BUILD_FLAG_NONE;
        clusterBuildInput.maxArgCount = 1;
        clusterBuildInput.maxTotalClusterCount = maxClusterCount;
        clusterBuildInput.maxClusterCountPerArg = maxClusterCount;

        OPTIX_CHECK(optixClusterAccelComputeMemoryUsage(
            optixContext.getOptixDeviceContext(),
            cgasAccelBuildMode,
            &cgasBuildInput, &asMemReqs));
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

        cgasSetMem.initialize(
            cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        cgasHandles.initialize(
            cuContext, cudau::BufferType::Device, 1, CUdeviceptr(0xFFFF'FFFF'FFFF'FFFF));
        cgasCount.initialize(
            cuContext, cudau::BufferType::Device, 1);
        cgasCount.fill(1);

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
            optixContext.getOptixDeviceContext(), cuStream,
            &buildModeDesc, &clasBuildInput,
            clusterArgs.getCUdeviceptr(), clusterCount.getCUdeviceptr(), clusterArgs.stride()));
    }

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
            optixContext.getOptixDeviceContext(), cuStream,
            &buildModeDesc, &cgasBuildInput,
            cgasArgs.getCUdeviceptr(), cgasCount.getCUdeviceptr(), cgasArgs.stride()));
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
        {
            ImGui::SetNextWindowPos(ImVec2(712, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

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
            frameIndex == 0;
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
                optixu::SizeAlign userDatSizeAlign(
                    sizeof(Shared::GeometryData),
                    alignof(Shared::GeometryData));
                curSizeAlign.add(userDatSizeAlign, &offset);
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



    mat.destroy();



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
