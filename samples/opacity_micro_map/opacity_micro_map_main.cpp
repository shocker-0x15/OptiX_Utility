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

    --no-omm: Disable OMM.
    --visualize ***: You can change visualizing mode
      - final: Final rendering.
      - barycentric: Visualize barycentric coordinates, can be used to see the shapes of base triangles.
      - primary-any-hits: Visualize the number of any-hit calls during primary ray trace.
      - shadow-any-hits: Visualize the number of any-hit calls during shadow ray trace.
    --max-subdiv-level *: The maximum OMM subdivision level.
    --subdiv-level-bias *: The bias to OMM subdivision level.
    --no-index-buffer: Specify not to use index buffers for OMM.

    [1] Opacity Micro-Map SDK: https://github.com/NVIDIAGameWorks/Opacity-MicroMap-SDK
        Opacity Micor-Map Samples: https://github.com/NVIDIAGameWorks/Opacity-MicroMap-Samples

*/

#include "opacity_micro_map_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

#include "../common/gui_common.h"

int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "opacity_micro_map";

    bool takeScreenShot = false;
    bool useOMM = true;
    auto visualizationMode = Shared::VisualizationMode_Final;
    shared::OMMFormat maxOmmSubDivLevel = shared::OMMFormat_Level4;
    int32_t ommSubdivLevelBias = 0;
    bool useOmmIndexBuffer = true;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--screen-shot") {
            takeScreenShot = true;
        }
        else if (arg == "--no-omm") {
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
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(
        cuContext, 4,
        optixu::EnableValidation::DEBUG_SELECT(Yes, No));

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: Opacity Micro-Mapを使う場合、パイプラインオプションで使用を宣言する必要がある。
    // EN: Declaring the use of Opacity micro-map is required in the pipeline option when using it.
    pipeline.setPipelineOptions(
        std::max(Shared::PrimaryRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE,
        optixu::UseMotionBlur::No, optixu::UseOpacityMicroMaps(useOMM));

    const std::vector<char> optixIr =
        readBinaryFile(resourceDir / "ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program rayGenProgram =
        pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");

    optixu::Program missProgram = pipeline.createMissProgram(
        moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::Program emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::HitProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr);
    optixu::HitProgramGroup shadingWithAlphaHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        moduleOptiX, RT_AH_NAME_STR("primary"));
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));
    optixu::HitProgramGroup visibilityWithAlphaHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("visibilityWithAlpha"),
        moduleOptiX, RT_AH_NAME_STR("visibilityWithAlpha"));

    pipeline.link(2);

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
            cudau::Buffer ommArrayMem;

            // JP: これらはOMM Arrayがビルドされた時点で不要になる。
            // EN: These are disposable once the OMM array is built.
            cudau::Buffer rawOmmArray;
            cudau::TypedBuffer<OptixOpacityMicromapDesc> ommDescs;

            // JP: これはOMM Arrayが関連づくGASがビルドされた時点で不要になる。
            // EN: This is disposable once the GAS to which the OMM array associated is built.
            cudau::Buffer ommIndexBuffer;
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
            geomData.vertexBuffer = floor.vertexBuffer.getROBuffer<enableBufferOobCheck>();
            geomData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
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
            geomData.vertexBuffer = alphaTestGeom.vertexBuffer.getROBuffer<enableBufferOobCheck>();
            geomData.triangleBuffer = group.triangleBuffer.getROBuffer<enableBufferOobCheck>();
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
                    resourceDir / "ptxes",
                    alphaTestGeom.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, texCoord),
                    sizeof(Shared::Vertex), vertices.size(),
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
    plp.travHandle = travHandle;
    plp.imageSize = int2(initWindowContentWidth, initWindowContentHeight);
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(initWindowContentWidth) / initWindowContentHeight;
    plp.envRadiance = float3(0.10f, 0.13f, 0.9f);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    // ----------------------------------------------------------------
    // JP: ウインドウの表示。
    // EN: Display the window.

    InitialConfig initConfig = {};
    initConfig.windowTitle = "OptiX Utility - Opacity Micro Map";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = initWindowContentWidth;
    initConfig.windowContentRenderHeight = initWindowContentHeight;
    initConfig.cameraPosition = make_float3(0, 6.0f, 6.0f);
    initConfig.cameraOrientation = qRotateY(pi_v<float>) * qRotateX(pi_v<float> / 4.0f);
    initConfig.cameraMovingSpeed = 0.1f;
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
        cudau::Timer render;

        void initialize(CUcontext context) {
            render.initialize(context);
        }
        void finalize() {
            render.finalize();
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
        static bool drawBaseEdges = false;
        bool visModeChanged = false;
        bool lightParamChanged = false;
        {
            ImGui::SetNextWindowPos(ImVec2(944, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            const float oldStrength = lightStrengthInLog10;
            const float oldPhi = lightDirPhi;
            const float oldTheta = lightDirTheta;
            ImGui::SliderFloat("Light Strength", &lightStrengthInLog10, -2, 2);
            ImGui::SliderFloat("Light Phi", &lightDirPhi, -180, 180);
            ImGui::SliderFloat("Light Theta", &lightDirTheta, 0, 90);
            lightParamChanged =
                lightStrengthInLog10 != oldStrength
                || lightDirPhi != oldPhi || lightDirTheta != oldTheta;

            ImGui::Separator();

            ImGui::Text("Buffer to Display");
            visModeChanged |= ImGui::RadioButtonE(
                "Final", &visualizationMode, Shared::VisualizationMode_Final);
            visModeChanged |= ImGui::RadioButtonE(
                "Barycentric", &visualizationMode, Shared::VisualizationMode_Barycentric);
            visModeChanged |= ImGui::RadioButtonE(
                "Primary Any Hits", &visualizationMode, Shared::VisualizationMode_NumPrimaryAnyHits);
            visModeChanged |= ImGui::RadioButtonE(
                "Shadow Any Hits", &visualizationMode, Shared::VisualizationMode_NumShadowAnyHits);

            visModeChanged |= ImGui::Checkbox("Draw Base Edges", &drawBaseEdges);

            ImGui::End();
        }

        // Stats Window
        {
            ImGui::SetNextWindowPos(ImVec2(8, 144), ImGuiCond_FirstUseEver);
            ImGui::Begin("Stats", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            static MovingAverageTime renderTime;

            renderTime.append(curGPUTimer.render.report());

            //ImGui::SetNextItemWidth(100.0f);
            ImGui::Text("render: %.3f [ms]", renderTime.getAverage());

            ImGui::End();
        }



        bool firstAccumFrame =
            cameraIsActuallyMoving ||
            args.resized ||
            frameIndex == 0 ||
            lightParamChanged ||
            visModeChanged;
        bool isNewSequence = args.resized || frameIndex == 0;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;

        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        // Render
        {
            curGPUTimer.render.start(curStream);

            plp.lightDirection = fromPolarYUp(lightDirPhi * pi_v<float> / 180, lightDirTheta * pi_v<float> / 180);
            plp.lightRadiance = float3(std::pow(10.0f, lightStrengthInLog10));
            plp.colorAccumBuffer = outputBufferSurfaceHolder.getNext();
            plp.visualizationMode = visualizationMode;
            plp.subPixelOffset = subPixelOffsets[numAccumFrames % static_cast<uint32_t>(lengthof(subPixelOffsets))];
            plp.sampleIndex = std::min(numAccumFrames, static_cast<uint32_t>(lengthof(subPixelOffsets)) - 1);
            plp.drawBaseEdges = drawBaseEdges;
            CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
            pipeline.launch(
                curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);
            ++numAccumFrames;

            curGPUTimer.render.stop(curStream);
        }

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = visualizationMode != Shared::VisualizationMode_Barycentric;
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
