/*

JP: このサンプルはAny-Hit Program呼び出しを削減することでアルファテストなどを高速化(*)する
    Opacity Micro-Map (OMM)の使用方法を示します。
    OMMは三角形メッシュにおけるテクスチャー等によるジオメトリの切り抜きに関する情報を事前計算したものです。
    GASの生成時に追加情報として渡すことで少量の追加メモリと引き換えにAny-Hit Programの呼び出し回数を削減し、
    アルファテストなどが有効なジオメトリに対するレイトレーシングを高速化することができます。
    OptiXのAPIにはOMM自体の生成処理は含まれていないため、何らかの手段を用いて生成する必要があります。
    OMM生成処理はテクスチャーとメッシュ間のマッピング、テクスチャー自体が静的な場合オフラインで予め行うことも可能です。
    このサンプルにはOMMの生成処理も含まれていますが、
    おそらくOpacity Micro-Map SDK [1]などのツールを使うほうが良いでしょう。

    *: このサンプル自体はOMMの使い方の説明目的なので、シーンが単純すぎて高速化が確認できない可能性があります。

    --no-omm: OMMを無効化する。
    --visualize ***: 可視化モードを切り替える。
      - final: 最終レンダリング。
      - barycentric: 重心座標の可視化。ベース三角形の形状を確認できる。
      - primary-any-hits: プライマリーレイをトレースする最中に生じたAny-Hit呼び出し回数の可視化。
      - shadow-any-hits: シャドウレイをトレースする最中に生じたAny-Hit呼び出し回数の可視化。
    --max-subdiv-level *: OMMの最大分割レベル。
    --subdiv-level-bias *: OMMの分割レベルへのバイアス。
    --no-index-buffer: OMM用のインデックスバッファーを使用しない。

EN: This sample shows how to use Opacity Micro-Map (OMM) which accelerates alpha tests (*), etc. by reducing
    any-hit program calls.
    OMM is precomputed information regarding geometry cutouts by textures or something for triangle mesh.
    Providing OMM as additional information when building a GAS costs a bit of additional memory but
    reduces any-hit program calls to accelerate ray tracing for geometries with alpha tests.
    OptiX API doesn't provide generation of OMM itself, so OMM generation by some means is required.
    OMM generation can be offline pre-computation if the mapping between a texture and a mesh and
    the texture itself are static.
    This sample provides OMM generation also, but you may want to use a tool like Opacity Micro-Map SDK [1].

    *: This sample itself is for demonstrating how to use OMM, therefore the scene is probably too simple
       to see the speedup.

    [1] Opacity Micro-Map SDK: https://github.com/NVIDIAGameWorks/Opacity-MicroMap-SDK

    --no-omm: Disable OMM.
    --visualize ***: You can change visualizing mode
      - final: Final rendering.
      - barycentric: Visualize barycentric coordinates, can be used to see the shapes of base triangles.
      - primary-any-hits: Visualize the number of any-hit calls during primary ray trace.
      - shadow-any-hits: Visualize the number of any-hit calls during shadow ray trace.
    --max-subdiv-level *: The maximum OMM subdivision level.
    --subdiv-level-bias *: The bias to OMM subdivision level.
    --no-index-buffer: Specify not to use index buffers for OMM.

*/

#include "opacity_micro_map_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

int32_t main(int32_t argc, const char* argv[]) try {
    bool useOMM = true;
    auto visualizationMode = Shared::VisualizationMode_Final;
    shared::OMMFormat maxOmmSubDivLevel = shared::OMMFormat_Level4;
    int32_t ommSubdivLevelBias = 0;
    bool useOmmIndexBuffer = true;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--no-omm") {
            useOMM = false;
        }
        else if (arg == "--visualize") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --visualize is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "final")
                visualizationMode = Shared::VisualizationMode_Final;
            else if (visType == "barycentric")
                visualizationMode = Shared::VisualizationMode_Barycentric;
            else if (visType == "primary-any-hits")
                visualizationMode = Shared::VisualizationMode_NumPrimaryAnyHits;
            else if (visType == "shadow-any-hits")
                visualizationMode = Shared::VisualizationMode_NumShadowAnyHits;
            else
                throw std::runtime_error("Argument for --visualize is invalid.");
            argIdx += 1;
        }
        else if (arg == "--max-subdiv-level") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --max-subdiv-level is not complete.");
            int32_t level = std::atoi(argv[argIdx + 1]);
            if (level < 0 || level > shared::OMMFormat_Level12)
                throw std::runtime_error("Invalid OMM subdivision level.");
            maxOmmSubDivLevel = static_cast<shared::OMMFormat>(level);
            argIdx += 1;
        }
        else if (arg == "--subdiv-level-bias") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --subdiv-level-bias is not complete.");
            ommSubdivLevelBias = std::atoi(argv[argIdx + 1]);
            argIdx += 1;
        }
        else if (arg == "--no-index-buffer") {
            useOmmIndexBuffer = false;
        }
        else
            throw std::runtime_error("Unknown command line argument.");
        ++argIdx;
    }

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

    // JP: Opacity Micro-Mapを使う場合、パイプラインオプションで使用を宣言する必要がある。
    // EN: Declaring the use of Opacity micro-map is required in the pipeline option when using it.
    pipeline.setPipelineOptions(
        std::max(Shared::PrimaryRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
        optixu::UseMotionBlur::No, optixu::UseOpacityMicroMaps(useOMM));

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "opacity_micro_map/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

    optixu::ProgramGroup missProgram = pipeline.createMissProgram(
        moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::ProgramGroup emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::ProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::ProgramGroup shadingWithAlphaHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        moduleOptiX, RT_AH_NAME_STR("primary"));
    optixu::ProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));
    optixu::ProgramGroup visibilityWithAlphaHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("visibilityWithAlpha"),
        moduleOptiX, RT_AH_NAME_STR("visibilityWithAlpha"));

    pipeline.link(2, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set,
    // the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Primary, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

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

    constexpr bool useBlockCompressedTexture = true;

    optixu::Material defaultMat = optixContext.createMaterial();
    defaultMat.setHitGroup(Shared::RayType_Primary, shadingHitProgramGroup);
    defaultMat.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);

    optixu::Material alphaTestMat = optixContext.createMaterial();
    alphaTestMat.setHitGroup(Shared::RayType_Primary, shadingWithAlphaHitProgramGroup);
    alphaTestMat.setHitGroup(Shared::RayType_Visibility, visibilityWithAlphaHitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer ommBuildScratchMem;
    cudau::Buffer asBuildScratchMem;

    constexpr optixu::IndexSize ommIndexSize = optixu::IndexSize::k2Bytes;

    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        struct MaterialGroup {
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            optixu::GeometryInstance optixGeomInst;
            cudau::Array texArray;
            CUtexObject texObj;

            optixu::OpacityMicroMapArray optixOmmArray;
            cudau::Buffer rawOmmArray;
            cudau::TypedBuffer<OptixOpacityMicromapDesc> ommDescs;
            cudau::Buffer ommIndexBuffer;
            cudau::Buffer ommArrayMem;
        };
        std::vector<MaterialGroup> matGroups;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            for (auto it = matGroups.rbegin(); it != matGroups.rend(); ++it) {
                it->ommArrayMem.finalize();
                it->ommIndexBuffer.finalize();
                it->ommDescs.finalize();
                it->rawOmmArray.finalize();
                it->optixOmmArray.destroy();

                if (it->texObj) {
                    CUDADRV_CHECK(cuTexObjectDestroy(it->texObj));
                    it->texArray.finalize();
                }
                it->triangleBuffer.finalize();
                it->optixGeomInst.destroy();
            }
            vertexBuffer.finalize();
        }
    };

    Geometry floor;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(100.0f, 0.0f, 100.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(100.0f, 0.0f, -100.0f), make_float3(0, 1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floor.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));

        floor.optixGas = scene.createGeometryAccelerationStructure();
        floor.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        floor.optixGas.setNumMaterialSets(1);
        floor.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        Geometry::MaterialGroup group;
        {
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = floor.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.texture = 0;
            geomData.albedo = float3(0.8f, 0.8f, 0.8f);

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(floor.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            floor.optixGas.addChild(group.optixGeomInst);
            floor.matGroups.push_back(std::move(group));
        }

        floor.optixGas.prepareForBuild(&asMemReqs);
        floor.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry alphaTestGeom;
    {
        std::filesystem::path filePath = R"(../../data/transparent_test.obj)";
        std::filesystem::path fileDir = filePath.parent_path();

        std::vector<obj::Vertex> vertices;
        std::vector<obj::MaterialGroup> matGroups;
        std::vector<obj::Material> materials;
        obj::load(filePath, &vertices, &matGroups, &materials);

        alphaTestGeom.vertexBuffer.initialize(
            cuContext, cudau::BufferType::Device,
            reinterpret_cast<Shared::Vertex*>(vertices.data()), vertices.size());

        // JP: ここではデフォルト値を使っているが、GASにはOpacity Micro-Mapに関連する設定がある。
        // EN: GAS has settings related opacity micro-map while the default values are used here.
        alphaTestGeom.optixGas = scene.createGeometryAccelerationStructure();
        alphaTestGeom.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes,
            optixu::AllowRandomVertexAccess::No,
            optixu::AllowOpacityMicroMapUpdate::No,
            optixu::AllowDisableOpacityMicroMaps::No);
        alphaTestGeom.optixGas.setNumMaterialSets(1);
        alphaTestGeom.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        uint32_t maxNumTrianglesPerGroup = 0;
        for (int groupIdx = 0; groupIdx < matGroups.size(); ++groupIdx) {
            const obj::MaterialGroup &srcGroup = matGroups[groupIdx];
            maxNumTrianglesPerGroup = std::max(
                maxNumTrianglesPerGroup,
                static_cast<uint32_t>(srcGroup.triangles.size()));
        }

        size_t scratchMemSizeForOMM = getScratchMemSizeForOMMGenerator(maxNumTrianglesPerGroup);
        cudau::Buffer scratchMemForOMM;
        scratchMemForOMM.initialize(cuContext, cudau::BufferType::Device, scratchMemSizeForOMM, 1);

        for (int groupIdx = 0; groupIdx < matGroups.size(); ++groupIdx) {
            const obj::MaterialGroup &srcGroup = matGroups[groupIdx];
            const obj::Material &srcMat = materials[srcGroup.materialIndex];
            const uint32_t numTriangles = srcGroup.triangles.size();

            Geometry::MaterialGroup group;
            group.triangleBuffer.initialize(
                cuContext, cudau::BufferType::Device,
                reinterpret_cast<const Shared::Triangle*>(srcGroup.triangles.data()),
                numTriangles);

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = alphaTestGeom.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.albedo = float3(srcMat.diffuse[0], srcMat.diffuse[1], srcMat.diffuse[2]);
            if (!srcMat.diffuseTexPath.empty()) {
                int32_t width, height, n;
                uint8_t* linearImageData = stbi_load(
                    srcMat.diffuseTexPath.string().c_str(),
                    &width, &height, &n, 4);
                group.texArray.initialize2D(
                    cuContext, cudau::ArrayElementType::UInt8, 4,
                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                    width, height, 1);
                group.texArray.write<uint8_t>(linearImageData, width * height * 4);
                stbi_image_free(linearImageData);

                cudau::TextureSampler texSampler;
                texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
                texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
                texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);
                texSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
                texSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
                group.texObj = texSampler.createTextureObject(group.texArray);
                geomData.texture = group.texObj;
            }

            // JP: まずは各三角形のOMMフォーマットを決定する。
            // EN: Fisrt, determine the OMM format of each triangle.
            OMMGeneratorContext ommContext;
            uint32_t histInOMMArray[shared::NumOMMFormats];
            uint32_t histInMesh[shared::NumOMMFormats];
            uint64_t rawOmmArraySize = 0;
            if (useOMM) {
                initializeOMMGeneratorContext(
                    alphaTestGeom.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, texCoord),
                    sizeof(Shared::Vertex),
                    group.triangleBuffer.getCUdeviceptr(), sizeof(Shared::Triangle), numTriangles,
                    group.texObj,
                    make_uint2(group.texArray.getWidth(), group.texArray.getHeight()), 4, 3,
                    shared::OMMFormat_Level0, maxOmmSubDivLevel, ommSubdivLevelBias,
                    useOmmIndexBuffer, 1 << static_cast<uint32_t>(ommIndexSize),
                    scratchMemForOMM.getCUdeviceptr(), scratchMemForOMM.sizeInBytes(),
                    &ommContext);

                countOMMFormats(ommContext, histInOMMArray, histInMesh, &rawOmmArraySize);
            }

            std::vector<OptixOpacityMicromapUsageCount> ommUsageCounts;
            hpprintf("Group %u (%u tris): OMM %s\n",
                     groupIdx, numTriangles, rawOmmArraySize > 0 ? "Enabled" : "Disabled");
            hpprintf("OMM Array Size: %llu [bytes]\n", rawOmmArraySize);
            if (rawOmmArraySize > 0) {
                uint32_t numOmms = 0;
                std::vector<OptixOpacityMicromapHistogramEntry> ommHistogramEntries;
                hpprintf("Histogram in OMM Array, Mesh\n");
                hpprintf("  None    : %5u, %5u\n",
                         histInOMMArray[shared::OMMFormat_None], histInMesh[shared::OMMFormat_None]);
                for (int i = shared::OMMFormat_Level0; i <= shared::OMMFormat_Level12; ++i) {
                    uint32_t countInOmmArray = histInOMMArray[i];
                    uint32_t countInMesh = histInMesh[i];
                    hpprintf("  Level %2u: %5u, %5u\n", i, countInOmmArray, countInMesh);

                    if (countInOmmArray > 0) {
                        OptixOpacityMicromapHistogramEntry histEntry;
                        histEntry.count = countInOmmArray;
                        histEntry.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
                        histEntry.subdivisionLevel = i;
                        ommHistogramEntries.push_back(histEntry);

                        numOmms += histInOMMArray[i];
                    }

                    if (countInMesh > 0) {
                        OptixOpacityMicromapUsageCount histEntry;
                        histEntry.count = countInMesh;
                        histEntry.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
                        histEntry.subdivisionLevel = i;
                        ommUsageCounts.push_back(histEntry);
                    }
                }
                hpprintf("\n");

                group.optixOmmArray = scene.createOpacityMicroMapArray();

                OptixMicromapBufferSizes ommArraySizes;
                group.optixOmmArray.setConfiguration(OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE);
                group.optixOmmArray.computeMemoryUsage(
                    ommHistogramEntries.data(), ommHistogramEntries.size(), &ommArraySizes);
                group.ommArrayMem.initialize(
                    cuContext, cudau::BufferType::Device, ommArraySizes.outputSizeInBytes, 1);

                // JP: このサンプルではASビルド用のスクラッチメモリをOMMビルドにも再利用する。
                // EN: This sample reuses the scratch memory for AS builds also for OMM builds.
                maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, ommArraySizes.tempSizeInBytes);



                group.rawOmmArray.initialize(cuContext, cudau::BufferType::Device, rawOmmArraySize, 1);
                group.ommDescs.initialize(cuContext, cudau::BufferType::Device, numOmms);
                if (useOmmIndexBuffer)
                    group.ommIndexBuffer.initialize(
                        cuContext, cudau::BufferType::Device,
                        numTriangles, 1 << static_cast<uint32_t>(ommIndexSize));
                group.optixOmmArray.setBuffers(group.rawOmmArray, group.ommDescs, group.ommArrayMem);

                // JP: 各三角形のOMMを生成する。
                // EN: Generate an OMM for each triangle.
                generateOMMArray(ommContext, group.rawOmmArray, group.ommDescs, group.ommIndexBuffer);
            }

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(alphaTestGeom.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            // JP: OMM ArrayをGeometryInstanceにセットする。
            // EN: Set the OMM array to the geometry instance.
            if (useOMM && group.optixOmmArray &&
                visualizationMode != Shared::VisualizationMode_Barycentric)
                group.optixGeomInst.setOpacityMicroMapArray(
                    group.optixOmmArray, ommUsageCounts.data(), ommUsageCounts.size(),
                    useOmmIndexBuffer ? group.ommIndexBuffer : optixu::BufferView(),
                    ommIndexSize);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, alphaTestMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            alphaTestGeom.optixGas.addChild(group.optixGeomInst);
            alphaTestGeom.matGroups.push_back(std::move(group));
        }

        scratchMemForOMM.finalize();

        alphaTestGeom.optixGas.prepareForBuild(&asMemReqs);
        alphaTestGeom.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを基にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floor.optixGas);

    optixu::Instance cutOutInst = scene.createInstance();
    cutOutInst.setChild(alphaTestGeom.optixGas);
    float xfm[] = {
        1.0f, 0.0f, 0.0f, 0,
        0.0f, 1.0f, 0.0f, 1.0f,
        0.0f, 0.0f, 1.0f, 0,
    };
    cutOutInst.setTransform(xfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(floorInst);
    ias.addChild(cutOutInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Opacity Micro-Map Arrayをビルドする。
    // EN: Build opacity micro-map arrays.
    for (int i = 0; i < alphaTestGeom.matGroups.size(); ++i) {
        const Geometry::MaterialGroup &group = alphaTestGeom.matGroups[i];
        if (!group.optixOmmArray)
            continue;

        group.optixOmmArray.rebuild(cuStream, asBuildScratchMem);
    }



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    floor.optixGas.rebuild(cuStream, floor.gasMem, asBuildScratchMem);
    alphaTestGeom.optixGas.rebuild(cuStream, alphaTestGeom.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        Geometry* geom;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &floor, 0, 0 },
        { &alphaTestGeom, 0, 0 },
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.geom->optixGas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.geom->optixGas.compact(
            cuStream,
            optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].geom->optixGas.removeUncompacted();
        gasList[i].geom->gasMem.finalize();
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



    constexpr uint32_t renderTargetSizeX = 1280;
    constexpr uint32_t renderTargetSizeY = 720;
    cudau::Array colorAccumBuffer;
    colorAccumBuffer.initialize2D(
        cuContext, cudau::ArrayElementType::Float32, 4,
        cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
        renderTargetSizeX, renderTargetSizeY, 1);


    
    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.colorAccumBuffer = colorAccumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 6.0f, 6.0f);
    plp.camera.orientation = rotateY3x3(pi_v<float>) * rotateX3x3(pi_v<float> / 4.0f);
    plp.lightDirection = normalize(float3(1, 5, 2));
    plp.lightRadiance = float3(7.5f, 7.5f, 7.5f);
    plp.envRadiance = float3(0.10f, 0.13f, 0.9f);
    plp.visualizationMode = visualizationMode;

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    cudau::Timer timerRender;
    timerRender.initialize(cuContext);
    
    // JP: レンダリング
    // EN: Render
    timerRender.start(cuStream);
    const uint32_t superSampleSize = 8;
    plp.superSampleSizeMinus1 = superSampleSize - 1;
    for (int frameIndex = 0; frameIndex < superSampleSize * superSampleSize; ++frameIndex) {
        plp.sampleIndex = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
    timerRender.stop(cuStream);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    float renderTime = timerRender.report();
    hpprintf("Render: %.3f[ms]\n", renderTime);

    timerRender.finalize();

    // JP: 結果の画像出力。
    // EN: Output the result as an image.
    saveImage("output.png", colorAccumBuffer, true, true);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));


    
    colorAccumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    cutOutInst.destroy();
    floorInst.destroy();

    alphaTestGeom.finalize();
    floor.finalize();

    scene.destroy();

    alphaTestMat.destroy();
    defaultMat.destroy();



    shaderBindingTable.finalize();

    visibilityWithAlphaHitProgramGroup.destroy();
    visibilityHitProgramGroup.destroy();
    shadingWithAlphaHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
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
