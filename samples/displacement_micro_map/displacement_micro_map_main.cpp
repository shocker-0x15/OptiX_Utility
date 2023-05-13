/*

JP: このサンプルはディスプレイスメントマッピングによる高密度ジオメトリを効率的に表現するための
    Displacement Micro-Map (OMM)の使用方法を示します。
    DMMは三角形メッシュにおけるハイトマップなどによる凹凸情報を三角形ごとにコンパクトに格納したものです。
    GASの生成時に追加情報として渡すことで三角形メッシュに高密度な凹凸を較的省メモリに追加することができます。
    逆に粗いメッシュにDMMを付加することで、通常の三角形メッシュよりも遥かに省メモリなGASで同様のジオメトリを
    表現することができますしGASのビルドも高速になります。
    OptiXのAPIにはDMM自体の生成処理は含まれていないため、何らかの手段を用いて生成する必要があります。
    DMM生成処理はテクスチャーとメッシュ間のマッピング、テクスチャー自体が静的な場合オフラインで予め行うことも可能です。
    このサンプルにはDMMの生成処理も含まれていますが、
    おそらくDisplacement Micro-Map SDK [1]などのツールを使うほうが良いでしょう。

    --no-dmm: DMMを無効化する。
    --visualize ***: 可視化モードを切り替える。
      - final: 最終レンダリング。
      - barycentric: 重心座標の可視化。ベース三角形の形状を確認できる。
      - micro-barycentric: マイクロ三角形の重心座標の可視化。
      - normal: 法線ベクトルの可視化。
    --max-compressed-format ***: DMMのエンコードを強制的に指定する。
      - none: 強制しない。(自動的に決定される)
      - 64utris: DMMあたり64マイクロ三角形64Bのフォーマットを使う。
      - 256utris: DMMあたり256マイクロ三角形128Bのフォーマットを使う。
      - 1024utris: DMMあたり1024マイクロ三角形128Bのフォーマットを使う。
    --max-subdiv-level *: DMMの最大分割レベル。
    --subdiv-level-bias *: DMMの分割レベルへのバイアス。
    --displacement-bias *: ベース頂点のディスプレイスメントベクター方向への事前の移動量。
    --displacement-scale *: ベース頂点のディスプレイスメントベクター方向の変位スケール。
    --no-index-buffer: DMM用のインデックスバッファーを使用しない。

EN: This sample shows how to use Displacement Micro-Map (DMM) with which high-definition geometry by
    displacement mapping can be efficiently represented.
    DMM is a data structure which compactly stores per-triangle displacement information by height map or
    something else for a triangle mesh.
    Providing DMM as additional information when building a GAS adds high frequency geometric detail to
    a triangle mesh with relatively low additional memory.
    In other words, a geometry with similar complexity as a normal triangle mesh can be represented with
    a GAS with much less memory by adding DMM to a coarse mesh. This makes GAS build faster also.
    OptiX API doesn't provide generation of DMM itself, so DMM generation by some means is required.
    DMM generation can be offline pre-computation if the mapping between a texture and a mesh and
    the texture itself are static.
    This sample provides DMM generation also, but you may want to use a tool like Displacement Micro-Map SDK [1].

    [1] Displacement Micro-Map SDK: https://github.com/NVIDIAGameWorks/Displacement-MicroMap-SDK/

    --no-dmm: Disable DMM.
    --visualize ***: You can change visualizing mode.
      - final: Final rendering.
      - barycentric: Visualize barycentric coordinates, can be used to see the shapes of base triangles.
      - micro-barycentric: Visualize barycentric coordinates of micro-triangles.
      - normal: Visualize normal vectors.
    --max-compressed-format ***: Forcefully specify a DMM encoding.
      - none: Do not force (encodings are automatically determined)
      - 64utris: Use an encoding with 64 micro triangles, 64B per triangle.
      - 256utris: Use an encoding with 256 micro triangles, 128B per triangle.
      - 1024utris: Use an encoding with 1024 micro triangles, 128B per triangle.
    --max-subdiv-level *: The maximum DMM subdivision level.
    --subdiv-level-bias *: The bias to DMM subdivision level.
    --displacement-bias *: The amount of pre-movement of base vertices along displacement vectors.
    --displacement-scale *: The amount of displacement of base vertices along displacement vectors.
    --no-index-buffer: Specify not to use index buffers for DMM.

*/

#include "displacement_micro_map_shared.h"

#include "../common/obj_loader.h"
#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"

int32_t main(int32_t argc, const char* argv[]) try {
    bool useDMM = true;
    auto visualizationMode = Shared::VisualizationMode_Final;
    shared::DMMEncoding maxCompressedFormat = shared::DMMEncoding_None;
    shared::DMMSubdivLevel maxDmmSubDivLevel = shared::DMMSubdivLevel_5;
    int32_t dmmSubdivLevelBias = 0;
    bool useDmmIndexBuffer = true;
    float displacementBias = 0.0f;
    float displacementScale = 1.0f;

    uint32_t argIdx = 1;
    while (argIdx < argc) {
        std::string_view arg = argv[argIdx];
        if (arg == "--no-dmm") {
            useDMM = false;
        }
        else if (arg == "--visualize") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --visualize is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "final")
                visualizationMode = Shared::VisualizationMode_Final;
            else if (visType == "barycentric")
                visualizationMode = Shared::VisualizationMode_Barycentric;
            else if (visType == "micro-barycentric")
                visualizationMode = Shared::VisualizationMode_MicroBarycentric;
            else if (visType == "normal")
                visualizationMode = Shared::VisualizationMode_Normal;
            else
                throw std::runtime_error("Argument for --visualize is invalid.");
            argIdx += 1;
        }
        else if (arg == "--max-compressed-format") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --max-compressed-format is not complete.");
            std::string_view visType = argv[argIdx + 1];
            if (visType == "none")
                maxCompressedFormat = shared::DMMEncoding_None;
            else if (visType == "64utris")
                maxCompressedFormat = shared::DMMEncoding_64B_per_64MicroTris;
            else if (visType == "256utris")
                maxCompressedFormat = shared::DMMEncoding_128B_per_256MicroTris;
            else if (visType == "1024utris")
                maxCompressedFormat = shared::DMMEncoding_128B_per_1024MicroTris;
            else
                throw std::runtime_error("Argument for --max-compressed-format is invalid.");
            argIdx += 1;
        }
        else if (arg == "--max-subdiv-level") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --max-subdiv-level is not complete.");
            int32_t level = std::atoi(argv[argIdx + 1]);
            if (level < 0 || level > shared::DMMSubdivLevel_5)
                throw std::runtime_error("Invalid DMM subdivision level.");
            maxDmmSubDivLevel = static_cast<shared::DMMSubdivLevel>(level);
            argIdx += 1;
        }
        else if (arg == "--subdiv-level-bias") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --subdiv-level-bias is not complete.");
            dmmSubdivLevelBias = std::atoi(argv[argIdx + 1]);
            argIdx += 1;
        }
        else if (arg == "--no-index-buffer") {
            useDmmIndexBuffer = false;
        }
        else if (arg == "--displacement-bias") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --displacement-bias is not complete.");
            displacementBias = std::atof(argv[argIdx + 1]);
            argIdx += 1;
        }
        else if (arg == "--displacement-scale") {
            if (argIdx + 1 >= argc)
                throw std::runtime_error("Argument for --displacement-scale is not complete.");
            displacementScale = std::atof(argv[argIdx + 1]);
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
    int32_t cuDeviceCount;
    CUstream cuStream;
    CUDADRV_CHECK(cuInit(0));
    CUDADRV_CHECK(cuDeviceGetCount(&cuDeviceCount));
    CUDADRV_CHECK(cuCtxCreate(&cuContext, 0, 0));
    CUDADRV_CHECK(cuCtxSetCurrent(cuContext));
    CUDADRV_CHECK(cuStreamCreate(&cuStream, 0));

    optixu::Context optixContext = optixu::Context::create(cuContext);

    optixu::Pipeline pipeline = optixContext.createPipeline();

    // JP: Displacement Micro-Mapを使う場合、プリミティブ種別のフラグを適切に設定する必要がある。
    // EN: Appropriately setting primitive type flags is required when using displacement micro-map.
    pipeline.setPipelineOptions(
        std::max(Shared::PrimaryRayPayloadSignature::numDwords,
                 Shared::VisibilityRayPayloadSignature::numDwords),
        optixu::calcSumDwords<float2>(),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_NONE/*DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE)*/,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_DISPLACED_MICROMESH_TRIANGLE,
        optixu::UseMotionBlur::No);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "displacement_micro_map/ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT/*DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT)*/,
        OPTIX_COMPILE_DEBUG_LEVEL_NONE/*DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE)*/);

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
    optixu::HitProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"));

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

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer dmmBuildScratchMem;
    cudau::Buffer asBuildScratchMem;

    constexpr optixu::IndexSize dmmIndexSize = optixu::IndexSize::k2Bytes;

    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        struct MaterialGroup {
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
            optixu::GeometryInstance optixGeomInst;
            cudau::Array heightTexArray;
            CUtexObject heightTexObj;

            optixu::DisplacementMicroMapArray optixDmmArray;
            cudau::Buffer rawDmmArray;
            cudau::TypedBuffer<OptixDisplacementMicromapDesc> dmmDescs;
            cudau::Buffer dmmIndexBuffer;
            cudau::Buffer dmmArrayMem;
            cudau::TypedBuffer<float2> dmmVertexBiasAndScaleBuffer;
            cudau::TypedBuffer<OptixDisplacementMicromapFlags> dmmTriangleFlagsBuffer;
        };
        std::vector<MaterialGroup> matGroups;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            for (auto it = matGroups.rbegin(); it != matGroups.rend(); ++it) {
                it->dmmTriangleFlagsBuffer.finalize();
                it->dmmVertexBiasAndScaleBuffer.finalize();
                it->dmmArrayMem.finalize();
                it->dmmIndexBuffer.finalize();
                it->dmmDescs.finalize();
                it->rawDmmArray.finalize();
                it->optixDmmArray.destroy();

                if (it->heightTexObj) {
                    CUDADRV_CHECK(cuTexObjectDestroy(it->heightTexObj));
                    it->heightTexArray.finalize();
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

    Geometry displacedMesh;
    {
        std::vector<Shared::Vertex> vertices;
        std::vector<Shared::Triangle> triangles;
        constexpr uint32_t gridSize = 4;
        for (int iz = 0; iz <= gridSize; ++iz) {
            float pz = static_cast<float>(iz) / gridSize;
            float z = -2.0f + 4.0f * pz;
            for (int ix = 0; ix <= gridSize; ++ix) {
                float px = static_cast<float>(ix) / gridSize;
                float x = -2.0f + 4.0f * static_cast<float>(ix) / gridSize;
                Shared::Vertex vtx = {
                    make_float3(x, 0.0f, z), make_float3(0, 1, 0), make_float2(px, pz)
                };
                vertices.push_back(vtx);
            }
        }
        for (int iz = 0; iz < gridSize; ++iz) {
            for (int ix = 0; ix < gridSize; ++ix) {
                uint32_t baseIdx = iz * (gridSize + 1) + ix;
                Shared::Triangle triA = {
                    baseIdx,
                    baseIdx + (gridSize + 1),
                    baseIdx + (gridSize + 1) + 1,
                };
                triangles.push_back(triA);
                Shared::Triangle triB = {
                    baseIdx,
                    baseIdx + (gridSize + 1) + 1,
                    baseIdx + 1,
                };
                triangles.push_back(triB);
            }
        }

        //Shared::Vertex vertices[] = {
        //    { make_float3(0.75f, 3.9f, 0.0f), make_float3(0, 1, 0), make_float2(0.75f, 3.9f) },
        //    { make_float3(2.0f, 3.85f, 0.0f), make_float3(0, 1, 0), make_float2(2.0f, 3.85f) },
        //    { make_float3(3.3f, 4.55f, 0.0f), make_float3(0, 1, 0), make_float2(3.3f, 4.55f) },
        //    { make_float3(1.1f, 2.75f, 0.0f), make_float3(0, 1, 0), make_float2(1.1f, 2.75f) },
        //    { make_float3(3.4f, 3.2f, 0.0f), make_float3(0, 1, 0), make_float2(3.4f, 3.2f) },
        //    { make_float3(0.45f, 1.75f, 0.0f), make_float3(0, 1, 0), make_float2(0.45f, 1.75f) },
        //    { make_float3(1.8f, 1.85f, 0.0f), make_float3(0, 1, 0), make_float2(1.8f, 1.85f) },
        //    { make_float3(4.7f, 2.65f, 0.0f), make_float3(0, 1, 0), make_float2(4.7f, 2.65f) },
        //    { make_float3(3.5f, 1.75f, 0.0f), make_float3(0, 1, 0), make_float2(3.5f, 1.75f) },
        //    { make_float3(1.8f, 0.5f, 0.0f), make_float3(0, 1, 0), make_float2(1.8f, 0.5f) },
        //    { make_float3(3.5f, 0.7f, 0.0f), make_float3(0, 1, 0), make_float2(3.5f, 0.7f) },

        //    { make_float3(0.75f, 3.9f, -1.0f), make_float3(0, 1, 0), make_float2(0.75f, 3.9f) },
        //    { make_float3(2.0f, 3.85f, -1.0f), make_float3(0, 1, 0), make_float2(2.0f, 3.85f) },
        //    { make_float3(3.3f, 4.55f, -1.0f), make_float3(0, 1, 0), make_float2(3.3f, 4.55f) },
        //    { make_float3(1.1f, 2.75f, -1.0f), make_float3(0, 1, 0), make_float2(1.1f, 2.75f) },
        //    { make_float3(3.4f, 3.2f, -1.0f), make_float3(0, 1, 0), make_float2(3.4f, 3.2f) },
        //    { make_float3(0.45f, 1.75f, -1.0f), make_float3(0, 1, 0), make_float2(0.45f, 1.75f) },
        //    { make_float3(1.8f, 1.85f, -1.0f), make_float3(0, 1, 0), make_float2(1.8f, 1.85f) },
        //    { make_float3(4.7f, 2.65f, -1.0f), make_float3(0, 1, 0), make_float2(4.7f, 2.65f) },
        //    { make_float3(3.5f, 1.75f, -1.0f), make_float3(0, 1, 0), make_float2(3.5f, 1.75f) },
        //    { make_float3(1.8f, 0.5f, -1.0f), make_float3(0, 1, 0), make_float2(1.8f, 0.5f) },
        //    { make_float3(3.5f, 0.7f, -1.0f), make_float3(0, 1, 0), make_float2(3.5f, 0.7f) },
        //};

        //Shared::Triangle triangles[] = {
        //    { 0, 3, 1 },
        //    { 1, 4, 2 },
        //    { 1, 3, 4 },
        //    { 3, 5, 6 },
        //    { 3, 6, 4 },
        //    { 4, 6, 8 },
        //    { 4, 8, 7 },
        //    { 5, 9, 6 },
        //    { 6, 9, 8 },
        //    { 8, 9, 10 },

        //    { 11, 14, 12 },
        //    { 12, 15, 13 },
        //    { 12, 14, 15 },
        //    { 14, 16, 17 },
        //    { 14, 17, 15 },
        //    { 15, 17, 19 },
        //    { 15, 19, 18 },
        //    { 16, 20, 17 },
        //    { 17, 20, 19 },
        //    { 19, 20, 21 },
        //};

        displacedMesh.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);

        // JP: DMMを適用するジオメトリやそれを含むGASは通常の三角形用のもので良い。
        // EN: Geometry and GAS to which DMM applied are ones for ordinary triangle mesh.
        displacedMesh.optixGas = scene.createGeometryAccelerationStructure();
        displacedMesh.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        displacedMesh.optixGas.setNumMaterialSets(1);
        displacedMesh.optixGas.setNumRayTypes(0, Shared::NumRayTypes);

        Geometry::MaterialGroup group;
        {
            const uint32_t numTriangles = triangles.size();
            group.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

            Shared::GeometryInstanceData geomData = {};
            geomData.vertexBuffer = displacedMesh.vertexBuffer.getDevicePointer();
            geomData.triangleBuffer = group.triangleBuffer.getDevicePointer();
            geomData.texture = 0;
            geomData.albedo = float3(0.8f, 0.2f, 0.05f);

            std::filesystem::path heightMapPath = R"(../../data/mountain_heightmap.png)";
            {
                int32_t width, height, n;
                uint8_t* linearImageData = stbi_load(
                    heightMapPath.string().c_str(),
                    &width, &height, &n, 1);
                group.heightTexArray.initialize2D(
                    cuContext, cudau::ArrayElementType::UInt8, 1,
                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                    width, height, 1);
                group.heightTexArray.write<uint8_t>(linearImageData, width * height);
                stbi_image_free(linearImageData);

                cudau::TextureSampler texSampler;
                texSampler.setXyFilterMode(cudau::TextureFilterMode::Linear);
                texSampler.setMipMapFilterMode(cudau::TextureFilterMode::Point);
                texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat);
                texSampler.setWrapMode(0, cudau::TextureWrapMode::Repeat);
                texSampler.setWrapMode(1, cudau::TextureWrapMode::Repeat);
                group.heightTexObj = texSampler.createTextureObject(group.heightTexArray);
                //geomData.heightTexture = group.heightTexObj;
            }

            size_t scratchMemSizeForDMM = getScratchMemSizeForDMMGenerator(numTriangles);
            cudau::Buffer scratchMemForDMM;
            scratchMemForDMM.initialize(cuContext, cudau::BufferType::Device, scratchMemSizeForDMM, 1);

            // JP: まずは各三角形のDMMフォーマットを決定する。
            // EN: Fisrt, determine the DMM format of each triangle.
            DMMGeneratorContext dmmContext;
            uint32_t histInDMMArray[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels];
            uint32_t histInMesh[shared::NumDMMEncodingTypes][shared::NumDMMSubdivLevels];
            uint64_t rawDmmArraySize = 0;
            if (useDMM) {
                initializeDMMGeneratorContext(
                    getExecutableDirectory() / "displacement_micro_map/ptxes",
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, position),
                    displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, texCoord),
                    sizeof(Shared::Vertex), vertices.size(),
                    group.triangleBuffer.getCUdeviceptr(), sizeof(Shared::Triangle), numTriangles,
                    group.heightTexObj,
                    make_uint2(group.heightTexArray.getWidth(), group.heightTexArray.getHeight()), 1, 0,
                    maxCompressedFormat,
                    shared::DMMSubdivLevel_0, maxDmmSubDivLevel, dmmSubdivLevelBias,
                    useDmmIndexBuffer, 1 << static_cast<uint32_t>(dmmIndexSize),
                    scratchMemForDMM.getCUdeviceptr(), scratchMemForDMM.sizeInBytes(),
                    &dmmContext);

                countDMMFormats(dmmContext, histInDMMArray, histInMesh, &rawDmmArraySize);
            }

            std::vector<OptixDisplacementMicromapUsageCount> dmmUsageCounts;
            hpprintf("(%u tris): DMM %s\n",
                     numTriangles, rawDmmArraySize > 0 ? "Enabled" : "Disabled");
            hpprintf("DMM Array Size: %llu [bytes]\n", rawDmmArraySize);
            if (rawDmmArraySize > 0) {
                uint32_t numDmms = 0;
                std::vector<OptixDisplacementMicromapHistogramEntry> dmmHistogramEntries;
                hpprintf("Histogram in DMM Array, Mesh\n");
                hpprintf("         None    : %5u, %5u\n",
                         histInDMMArray[shared::DMMEncoding_None][0],
                         histInMesh[shared::DMMEncoding_None][0]);
                for (int enc = shared::DMMEncoding_64B_per_64MicroTris; enc <= shared::DMMEncoding_128B_per_1024MicroTris; ++enc) {
                    for (int level = shared::DMMSubdivLevel_0; level <= shared::DMMSubdivLevel_5; ++level) {
                        uint32_t countInDmmArray = histInDMMArray[enc][level];
                        uint32_t countInMesh = histInMesh[enc][level];
                        hpprintf("  Enc %u - Level %u: %5u, %5u\n", enc, level, countInDmmArray, countInMesh);

                        if (countInDmmArray > 0) {
                            OptixDisplacementMicromapHistogramEntry histEntry;
                            histEntry.count = countInDmmArray;
                            histEntry.format = static_cast<OptixDisplacementMicromapFormat>(enc);
                            histEntry.subdivisionLevel = level;
                            dmmHistogramEntries.push_back(histEntry);

                            numDmms += histInDMMArray[enc][level];
                        }

                        if (countInMesh > 0) {
                            OptixDisplacementMicromapUsageCount histEntry;
                            histEntry.count = countInMesh;
                            histEntry.format = static_cast<OptixDisplacementMicromapFormat>(enc);
                            histEntry.subdivisionLevel = level;
                            dmmUsageCounts.push_back(histEntry);
                        }
                    }
                }
                hpprintf("\n");

                group.optixDmmArray = scene.createDisplacementMicroMapArray();

                OptixMicromapBufferSizes dmmArraySizes;
                group.optixDmmArray.setConfiguration(OPTIX_DISPLACEMENT_MICROMAP_FLAG_PREFER_FAST_TRACE);
                group.optixDmmArray.computeMemoryUsage(
                    dmmHistogramEntries.data(), dmmHistogramEntries.size(), &dmmArraySizes);
                group.dmmArrayMem.initialize(
                    cuContext, cudau::BufferType::Device, dmmArraySizes.outputSizeInBytes, 1);

                // JP: このサンプルではASビルド用のスクラッチメモリをDMMビルドにも再利用する。
                // EN: This sample reuses the scratch memory for AS builds also for DMM builds.
                maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, dmmArraySizes.tempSizeInBytes);



                group.rawDmmArray.initialize(cuContext, cudau::BufferType::Device, rawDmmArraySize, 1);
                group.dmmDescs.initialize(cuContext, cudau::BufferType::Device, numDmms);
                if (useDmmIndexBuffer)
                    group.dmmIndexBuffer.initialize(
                        cuContext, cudau::BufferType::Device,
                        numTriangles, 1 << static_cast<uint32_t>(dmmIndexSize));
                group.optixDmmArray.setBuffers(group.rawDmmArray, group.dmmDescs, group.dmmArrayMem);

                group.dmmTriangleFlagsBuffer.initialize(cuContext, cudau::BufferType::Device, numTriangles);

                // JP: 各三角形のDMMを生成する。
                // EN: Generate an DMM for each triangle.
                generateDMMArray(
                    dmmContext,
                    group.rawDmmArray, group.dmmDescs, group.dmmIndexBuffer,
                    group.dmmTriangleFlagsBuffer);

                /*
                JP: 頂点ごとにディスプレイスメントのスケールと事前移動量を指定できる。
                    DMMに記録されているマイクロ頂点ごとの変位量と併せて、ディスプレイスメント適用後
                    のメッシュを最小限に含むように調節することでより高効率かつ高精度なレイトレースが可能となる。
                    が、このサンプルではシンプルにグローバルな値を指定する。
                EN: Specify displacement scale and the amount of pre-movement per vertex.
                    These amounts should be adjusted along with displacement amounts per micro-vertices in DMM
                    so that these tightly encapsulates the diplaced mesh for faster and more precise ray tracing.
                    However, this sample simply specifies globally uniform values.
                */
                std::vector<float2> vertexBiasAndScaleBuffer(
                    vertices.size(), float2(displacementBias, displacementScale));
                group.dmmVertexBiasAndScaleBuffer.initialize(
                    cuContext, cudau::BufferType::Device, vertexBiasAndScaleBuffer);
            }

            group.optixGeomInst = scene.createGeometryInstance();
            group.optixGeomInst.setVertexBuffer(displacedMesh.vertexBuffer);
            group.optixGeomInst.setTriangleBuffer(group.triangleBuffer);
            // JP: DMM ArrayをGeometryInstanceにセットする。
            // EN: Set the DMM array to the geometry instance.
            if (useDMM && group.optixDmmArray &&
                visualizationMode != Shared::VisualizationMode_Barycentric)
                group.optixGeomInst.setDisplacementMicroMapArray(
                    // JP: 頂点ごとのディスプレイスメント方向として法線ベクトルを再利用する。
                    // EN: Reuse the normal vectors as displacement directions per vertex.
                    optixu::BufferView(
                        displacedMesh.vertexBuffer.getCUdeviceptr() + offsetof(Shared::Vertex, normal),
                        displacedMesh.vertexBuffer.numElements(),
                        sizeof(Shared::Vertex)),
                    group.dmmVertexBiasAndScaleBuffer,
                    group.dmmTriangleFlagsBuffer,
                    group.optixDmmArray, dmmUsageCounts.data(), dmmUsageCounts.size(),
                    useDmmIndexBuffer ? group.dmmIndexBuffer : optixu::BufferView(),
                    dmmIndexSize, 0,
                    OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3,
                    OPTIX_DISPLACEMENT_MICROMAP_BIAS_AND_SCALE_FORMAT_FLOAT2);
            group.optixGeomInst.setNumMaterials(1, optixu::BufferView());
            group.optixGeomInst.setMaterial(0, 0, defaultMat);
            group.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
            group.optixGeomInst.setUserData(geomData);

            displacedMesh.optixGas.addChild(group.optixGeomInst);
            displacedMesh.matGroups.push_back(std::move(group));
        }

        displacedMesh.optixGas.prepareForBuild(&asMemReqs);
        displacedMesh.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GASを基にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floor.optixGas);

    optixu::Instance displacedMeshInst = scene.createInstance();
    displacedMeshInst.setChild(displacedMesh.optixGas);
    float xfm[] = {
        1.0f, 0.0f, 0.0f, 0,
        0.0f, 1.0f, 0.0f, 0.5f,
        0.0f, 0.0f, 1.0f, 0,
    };
    displacedMeshInst.setTransform(xfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(floorInst);
    ias.addChild(displacedMeshInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Displacement Micro-Map Arrayをビルドする。
    // EN: Build displacement micro-map arrays.
    for (int i = 0; i < displacedMesh.matGroups.size(); ++i) {
        const Geometry::MaterialGroup &group = displacedMesh.matGroups[i];
        if (!group.optixDmmArray)
            continue;

        group.optixDmmArray.rebuild(cuStream, asBuildScratchMem);
    }



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    floor.optixGas.rebuild(cuStream, floor.gasMem, asBuildScratchMem);
    displacedMesh.optixGas.rebuild(cuStream, displacedMesh.gasMem, asBuildScratchMem);

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
        { &displacedMesh, 0, 0 },
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
    plp.lightDirection = normalize(float3(-2, 5, 2));
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

    displacedMeshInst.destroy();
    floorInst.destroy();

    displacedMesh.finalize();
    floor.finalize();

    scene.destroy();

    defaultMat.destroy();



    shaderBindingTable.finalize();

    visibilityHitProgramGroup.destroy();
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
