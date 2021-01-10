/*

JP: このサンプルはカーブを扱う方法を示します。
    カーブとレイの交叉判定はOptiXによって内部的に扱われます。

EN: This sample shows how to handle curves.
    Intersection test between a ray and a curve is handled internally by OptiX.

*/

#include "curve_primitive_shared.h"

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

    // JP: カーブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    //     複数のカーブタイプがあり、このサンプルでは全て使用する。
    //     カーブのアトリビュートサイズは1Dword(float)。
    // EN: Appropriately setting primitive type flags is required since this sample uses curve intersection.
    //     There are multiple curve types and the sample use all of them.
    //     The attribute size of curves is 1 Dword (float).
    pipeline.setPipelineOptions(optixu::calcSumDwords<PayloadSignature>(),
                                std::max(optixu::calcSumDwords<float2>(),
                                         optixu::calcSumDwords<float>()),
                                "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
                                OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR |
                                OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE |
                                OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "curve_primitive/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::ProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroupForBuiltinIS(
        OPTIX_PRIMITIVE_TYPE_TRIANGLE,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);
    optixu::ProgramGroup hitProgramGroupForLinearCurves = pipeline.createHitProgramGroupForBuiltinIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);
    optixu::ProgramGroup hitProgramGroupForQuadraticCurves = pipeline.createHitProgramGroupForBuiltinIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);
    optixu::ProgramGroup hitProgramGroupForCubicCurves = pipeline.createHitProgramGroupForBuiltinIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    // JP: このサンプルはRay Generation Programからしかレイトレースを行わないのでTrace Depthは1になる。
    // EN: Trace depth is 1 because this sample trace rays only from the ray generation program.
    pipeline.link(1, DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(rayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
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

    optixu::Material matForTriangles = optixContext.createMaterial();
    matForTriangles.setHitGroup(Shared::RayType_Primary, hitProgramGroupForTriangles);
    optixu::Material matForLinearCurves = optixContext.createMaterial();
    matForLinearCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForLinearCurves);
    optixu::Material matForQuadraticCurves = optixContext.createMaterial();
    matForQuadraticCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForQuadraticCurves);
    optixu::Material matForCubicCurves = optixContext.createMaterial();
    matForCubicCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCubicCurves);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    optixu::GeometryInstance floorGeomInst = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> floorVertexBuffer;
    cudau::TypedBuffer<Shared::Triangle> floorTriangleBuffer;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-1.0f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.0f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.0f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floorVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        floorTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = floorVertexBuffer.getDevicePointer();
        geomData.triangleBuffer = floorTriangleBuffer.getDevicePointer();

        floorGeomInst.setVertexBuffer(floorVertexBuffer);
        floorGeomInst.setTriangleBuffer(floorTriangleBuffer);
        floorGeomInst.setNumMaterials(1, optixu::BufferView());
        floorGeomInst.setMaterial(0, 0, matForTriangles);
        floorGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        floorGeomInst.setUserData(geomData);
    }

    const auto generateCurves = [](std::vector<Shared::CurveVertex>* vertices,
                                   std::vector<uint32_t>* indices,
                                   float xStart, float xEnd, uint32_t numX,
                                   float zStart, float zEnd, uint32_t numZ,
                                   float baseWidth, uint32_t curveDegree)
    {
        std::mt19937 rng(390318410);
        std::uniform_int_distribution<uint32_t> uSeg(3, 5);
        std::uniform_real_distribution<float> u01;

        vertices->clear();
        indices->clear();

        const float deltaX = (xEnd - xStart) / numX;
        const float deltaZ = (zEnd - zStart) / numZ;
        for (int iz = 0; iz < numZ; ++iz) {
            float pz = (iz + 0.5f) / numZ;
            float z = (1 - pz) * zStart + pz * zEnd;
            for (int ix = 0; ix < numX; ++ix) {
                float px = (ix + 0.5f) / numX;
                float x = (1 - px) * xStart + px * xEnd;

                uint32_t numSegments = uSeg(rng);
                uint32_t indexStart = vertices->size();

                // Beginning phantom points
                if (curveDegree > 1) {
                    float3 pos = float3(0.0f, 0.0f, 0.0f);
                    float width = baseWidth;
                    vertices->push_back(Shared::CurveVertex{ pos, width });
                }

                // Base
                {
                    float3 pos = float3(x, 0.0f, z);
                    float width = baseWidth;
                    vertices->push_back(Shared::CurveVertex{ pos, width });
                }
                for (int s = 0; s < numSegments; ++s) {
                    float p = (float)(s + 1) / numSegments;
                    float3 pos = float3(x + 0.3f * deltaX * (u01(rng) - 0.5f),
                                        0.1f * (s + 1),
                                        z + 0.3f * deltaZ * (u01(rng) - 0.5f));
                    float width = baseWidth * (1 - p);
                    vertices->push_back(Shared::CurveVertex{ pos, width });
                }

                // Ending phantom points
                if (curveDegree > 1) {
                    float width = 0.0f;
                    float3 pm1 = (*vertices)[vertices->size() - 1].position;
                    float3 pm2 = (*vertices)[vertices->size() - 2].position;
                    float3 d = pm1 - pm2;
                    if (curveDegree == 2)
                        d *= 1e-3f;
                    float3 pos = pm1 + d;
                    vertices->push_back(Shared::CurveVertex{ pos, width });
                }

                // Modify the beginning phantom points
                if (curveDegree > 1) {
                    float3 p1 = (*vertices)[indexStart + 1].position;
                    float3 p2 = (*vertices)[indexStart + 2].position;
                    float3 d = p1 - p2;
                    if (curveDegree == 2)
                        d *= 1e-3f;
                    (*vertices)[indexStart].position = p1 + d;
                }

                for (int s = 0; s < vertices->size() - indexStart - curveDegree; ++s)
                    indices->push_back(indexStart + s);
            }
        }
    };

    uint32_t numX = 5;
    uint32_t numZ = 15;
    float baseWidth = 0.03f;

    // JP: カーブ用GeometryInstanceは生成時に指定する必要がある。
    // EN: GeometryInstance for curves requires to be specified at the creation.

    // Linear Segments
    optixu::GeometryInstance linearCurveGeomInst = scene.createGeometryInstance(optixu::GeometryType::LinearSegments);
    cudau::TypedBuffer<Shared::CurveVertex> linearCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> linearCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves(&vertices, &indices,
                       -1.0f / 3.0f + 0.05f, 1.0f / 3.0f - 0.05f, numX,
                       -1.0f + 0.05f, 1.0f - 0.05f, numZ,
                       baseWidth, 1);

        linearCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        linearCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = linearCurveVertexBuffer.getDevicePointer();
        geomData.segmentIndexBuffer = linearCurveSegmentIndexBuffer.getDevicePointer();

        linearCurveGeomInst.setVertexBuffer(optixu::BufferView(
            linearCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            linearCurveVertexBuffer.numElements(), linearCurveVertexBuffer.stride()));
        linearCurveGeomInst.setWidthBuffer(optixu::BufferView(
            linearCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            linearCurveVertexBuffer.numElements(), linearCurveVertexBuffer.stride()));
        linearCurveGeomInst.setSegmentIndexBuffer(linearCurveSegmentIndexBuffer);
        linearCurveGeomInst.setMaterial(0, 0, matForLinearCurves);
        linearCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        linearCurveGeomInst.setUserData(geomData);
    }

    // Quadratic B-Splines
    optixu::GeometryInstance quadraticCurveGeomInst = scene.createGeometryInstance(optixu::GeometryType::QuadraticBSplines);
    cudau::TypedBuffer<Shared::CurveVertex> quadraticCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> quadraticCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves(&vertices, &indices,
                       -1.0f / 3.0f + 0.05f, 1.0f / 3.0f - 0.05f, numX,
                       -1.0f + 0.05f, 1.0f - 0.05f, numZ,
                       baseWidth, 2);

        quadraticCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        quadraticCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = quadraticCurveVertexBuffer.getDevicePointer();
        geomData.segmentIndexBuffer = quadraticCurveSegmentIndexBuffer.getDevicePointer();

        quadraticCurveGeomInst.setVertexBuffer(optixu::BufferView(
            quadraticCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            quadraticCurveVertexBuffer.numElements(), quadraticCurveVertexBuffer.stride()));
        quadraticCurveGeomInst.setWidthBuffer(optixu::BufferView(
            quadraticCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            quadraticCurveVertexBuffer.numElements(), quadraticCurveVertexBuffer.stride()));
        quadraticCurveGeomInst.setSegmentIndexBuffer(quadraticCurveSegmentIndexBuffer);
        quadraticCurveGeomInst.setMaterial(0, 0, matForQuadraticCurves);
        quadraticCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        quadraticCurveGeomInst.setUserData(geomData);
    }

    // Cubic B-Splines
    optixu::GeometryInstance cubicCurveGeomInst = scene.createGeometryInstance(optixu::GeometryType::CubicBSplines);
    cudau::TypedBuffer<Shared::CurveVertex> cubicCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> cubicCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves(&vertices, &indices,
                       -1.0f / 3.0f + 0.05f, 1.0f / 3.0f - 0.05f, numX,
                       -1.0f + 0.05f, 1.0f - 0.05f, numZ,
                       baseWidth, 3);

        cubicCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        cubicCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = cubicCurveVertexBuffer.getDevicePointer();
        geomData.segmentIndexBuffer = cubicCurveSegmentIndexBuffer.getDevicePointer();

        cubicCurveGeomInst.setVertexBuffer(optixu::BufferView(
            cubicCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            cubicCurveVertexBuffer.numElements(), cubicCurveVertexBuffer.stride()));
        cubicCurveGeomInst.setWidthBuffer(optixu::BufferView(
            cubicCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            cubicCurveVertexBuffer.numElements(), cubicCurveVertexBuffer.stride()));
        cubicCurveGeomInst.setSegmentIndexBuffer(cubicCurveSegmentIndexBuffer);
        cubicCurveGeomInst.setMaterial(0, 0, matForCubicCurves);
        cubicCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        cubicCurveGeomInst.setUserData(geomData);
    }



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure floorGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer floorGasMem;
    floorGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
    floorGas.setNumMaterialSets(1);
    floorGas.setNumRayTypes(0, Shared::NumRayTypes);
    floorGas.addChild(floorGeomInst);
    floorGas.prepareForBuild(&asMemReqs);
    floorGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: カーブ用のGASは三角形用のGASとは別にする必要がある。また、次数の異なるカーブは混ぜることができない。
    //     GAS生成時にカーブ用であることを指定する。
    // EN: GAS for curves must be created separately with GAS for triangles.
    //     Also, curves with different degrees can't be mixed.
    //     Specify that the GAS is for curves at the creation.
#if defined(USE_EMBEDDED_DATA)
    constexpr bool useVertexData = true;
#else
    constexpr bool useVertexData = false;
#endif

    optixu::GeometryAccelerationStructure linearCurvesGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::LinearSegments);
    cudau::Buffer linearCurvesGasMem;
    linearCurvesGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, useVertexData);
    linearCurvesGas.setNumMaterialSets(1);
    linearCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    linearCurvesGas.addChild(linearCurveGeomInst);
    linearCurvesGas.prepareForBuild(&asMemReqs);
    linearCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure quadraticCurvesGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::QuadraticBSplines);
    cudau::Buffer quadraticCurvesGasMem;
    quadraticCurvesGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, useVertexData);
    quadraticCurvesGas.setNumMaterialSets(1);
    quadraticCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    quadraticCurvesGas.addChild(quadraticCurveGeomInst);
    quadraticCurvesGas.prepareForBuild(&asMemReqs);
    quadraticCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure cubicCurvesGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::CubicBSplines);
    cudau::Buffer cubicCurvesGasMem;
    cubicCurvesGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, useVertexData);
    cubicCurvesGas.setNumMaterialSets(1);
    cubicCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    cubicCurvesGas.addChild(cubicCurveGeomInst);
    cubicCurvesGas.prepareForBuild(&asMemReqs);
    cubicCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();

    floorInst.setChild(floorGas);
    float linearCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, -2.0f / 3.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance linearCurvesInst = scene.createInstance();
    linearCurvesInst.setChild(linearCurvesGas);
    linearCurvesInst.setTransform(linearCurvesInstXfm);

    float quadraticCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance quadraticCurvesInst = scene.createInstance();
    quadraticCurvesInst.setChild(quadraticCurvesGas);
    quadraticCurvesInst.setTransform(quadraticCurvesInstXfm);

    float cubicCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, 2.0f / 3.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance cubicCurvesInst = scene.createInstance();
    cubicCurvesInst.setChild(cubicCurvesGas);
    cubicCurvesInst.setTransform(cubicCurvesInstXfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    ias.addChild(floorInst);
    ias.addChild(linearCurvesInst);
    ias.addChild(quadraticCurvesInst);
    ias.addChild(cubicCurvesInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    floorGas.rebuild(cuStream, floorGasMem, asBuildScratchMem);
    linearCurvesGas.rebuild(cuStream, linearCurvesGasMem, asBuildScratchMem);
    quadraticCurvesGas.rebuild(cuStream, quadraticCurvesGasMem, asBuildScratchMem);
    cubicCurvesGas.rebuild(cuStream, cubicCurvesGasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        optixu::GeometryAccelerationStructure gas;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { floorGas, 0, 0 },
        { linearCurvesGas, 0, 0 },
        { quadraticCurvesGas, 0, 0 },
        { cubicCurvesGas, 0, 0 },
    };
    size_t compactedASMemOffset = 0;
    for (int i = 0; i < lengthof(gasList); ++i) {
        CompactedASInfo &info = gasList[i];
        compactedASMemOffset = alignUp(compactedASMemOffset, OPTIX_ACCEL_BUFFER_BYTE_ALIGNMENT);
        info.offset = compactedASMemOffset;
        info.gas.prepareForCompact(&info.size);
        compactedASMemOffset += info.size;
    }
    cudau::Buffer compactedASMem;
    compactedASMem.initialize(cuContext, cudau::BufferType::Device, compactedASMemOffset, 1);
    for (int i = 0; i < lengthof(gasList); ++i) {
        const CompactedASInfo &info = gasList[i];
        info.gas.compact(cuStream, optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset,
                                                      info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i)
        gasList[i].gas.removeUncompacted();

    cubicCurvesGasMem.finalize();
    quadraticCurvesGasMem.finalize();
    linearCurvesGasMem.finalize();
    floorGasMem.finalize();



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
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.accumBuffer = accumBuffer.getBlockBuffer2D();
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 1.0f, 2.5f);
    plp.camera.orientation = rotateX3x3(-M_PI / 8) * rotateY3x3(M_PI);
    //plp.camera.position = make_float3(0, 0.01f, 2.5f);
    //plp.camera.orientation = rotateY3x3(M_PI);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    for (int frameIndex = 0; frameIndex < 1024; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    saveImage("output.png", accumBuffer, false, false);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();
    rngBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    cubicCurvesInst.destroy();
    quadraticCurvesInst.destroy();
    linearCurvesInst.destroy();
    floorInst.destroy();

    asBuildScratchMem.finalize();
    cubicCurvesGas.destroy();
    quadraticCurvesGas.destroy();
    linearCurvesGas.destroy();
    floorGas.destroy();

    cubicCurveSegmentIndexBuffer.finalize();
    cubicCurveVertexBuffer.finalize();
    cubicCurveGeomInst.destroy();

    quadraticCurveSegmentIndexBuffer.finalize();
    quadraticCurveVertexBuffer.finalize();
    quadraticCurveGeomInst.destroy();

    linearCurveSegmentIndexBuffer.finalize();
    linearCurveVertexBuffer.finalize();
    linearCurveGeomInst.destroy();

    floorTriangleBuffer.finalize();
    floorVertexBuffer.finalize();
    floorGeomInst.destroy();

    scene.destroy();

    matForCubicCurves.destroy();
    matForQuadraticCurves.destroy();
    matForLinearCurves.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

    hitProgramGroupForCubicCurves.destroy();
    hitProgramGroupForQuadraticCurves.destroy();
    hitProgramGroupForLinearCurves.destroy();
    hitProgramGroupForTriangles.destroy();

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
