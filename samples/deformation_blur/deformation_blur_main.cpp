﻿/*

JP: このサンプルはデフォーメーション(変形)ブラーを扱うGASを構築する方法を示します。
    複数のモーションステップに対応する頂点(もしくはAABB)バッファーをGeometryInstanceに設定し
    GASに適切なモーション設定を行うことでデフォーメーションブラーに対応するGASを構築できます。
EN: This sample shows how to build a GAS to handle deformation blur.
    Set a vertex (or AABB) buffer to each of multiple motion steps of GeometryInstance and
    set appropriate motion configuration to a GAS to build a GAS capable of deformation blur.
*/

#include "deformation_blur_shared.h"

#include "../common/obj_loader.h"

#include "../common/gui_common.h"



int32_t main(int32_t argc, const char* argv[]) try {
    const std::filesystem::path resourceDir = getExecutableDirectory() / "deformation_blur";

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
    JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
        カーブ・球・カスタムプリミティブとの衝突判定を使うため
        プリミティブ種別のフラグを適切に設定する必要がある。
        変形モーションブラーを使用するのでUseMotionBlur::Yesを指定する。
    EN: This sample uses two-level AS (single-level instancing).
        Appropriately setting primitive type flags is required since this sample uses curve, sphere and
        custom primitive intersection.
        Specify UseMotionBlur::Yes since the sample uses deformation motion blur.
    */
    optixu::PipelineOptions pipelineOptions;
    pipelineOptions.payloadCountInDwords = Shared::MyPayloadSignature::numDwords;
    pipelineOptions.attributeCountInDwords = std::max<uint32_t>({
        optixu::calcSumDwords<float2>(),
        optixu::calcSumDwords<float>(),
        Shared::PartialSphereAttributeSignature::numDwords });
    pipelineOptions.launchParamsVariableName = "plp";
    pipelineOptions.sizeOfLaunchParams = sizeof(Shared::PipelineLaunchParameters);
    pipelineOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
    pipelineOptions.supportedPrimitiveTypeFlags =
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    pipelineOptions.useMotionBlur = optixu::UseMotionBlur::Yes;
    pipeline.setPipelineOptions(pipelineOptions);

    const std::vector<char> optixIr =
        readBinaryFile(resourceDir / "ptxes/optix_kernels.optixir");
    optixu::Module moduleOptiX = pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::Program rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::Program exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::Program missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::HitProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    constexpr auto useEmbeddedVertexData = optixu::AllowRandomVertexAccess(Shared::useEmbeddedVertexData);

    constexpr OptixCurveEndcapFlags curveEndcap = OPTIX_CURVE_ENDCAP_ON;
    constexpr optixu::ASTradeoff curveASTradeOff = optixu::ASTradeoff::PreferFastTrace;
    constexpr optixu::AllowUpdate curveASUpdatable = optixu::AllowUpdate::No;
    constexpr optixu::AllowCompaction curveASCompactable = optixu::AllowCompaction::Yes;
    optixu::HitProgramGroup hitProgramGroupForCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);

    constexpr optixu::ASTradeoff sphereASTradeOff = optixu::ASTradeoff::PreferFastTrace;
    constexpr optixu::AllowUpdate sphereASUpdatable = optixu::AllowUpdate::No;
    constexpr optixu::AllowCompaction sphereASCompactable = optixu::AllowCompaction::Yes;
    optixu::HitProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroupForSphereIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        sphereASTradeOff, sphereASUpdatable, sphereASCompactable, useEmbeddedVertexData);

    // JP: このヒットグループはレイと(部分)球の交叉判定用なのでカスタムのIntersectionプログラムを渡す。
    // EN: This is for ray-(partial-)sphere intersection, so pass a custom intersection program.
    optixu::HitProgramGroup hitProgramGroupForPartialSpheres = pipeline.createHitProgramGroupForCustomIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        moduleOptiX, RT_IS_NAME_STR("partialSphere"));

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

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    optixu::Material matForTriangles = optixContext.createMaterial();
    matForTriangles.setHitGroup(Shared::RayType_Primary, hitProgramGroupForTriangles);
    optixu::Material matForCurves = optixContext.createMaterial();
    matForCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCurves);
    optixu::Material matForSpheres = optixContext.createMaterial();
    matForSpheres.setHitGroup(Shared::RayType_Primary, hitProgramGroupForSpheres);
    optixu::Material matForPartialSpheres = optixContext.createMaterial();
    matForPartialSpheres.setHitGroup(Shared::RayType_Primary, hitProgramGroupForPartialSpheres);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: このサンプルではデフォーメーションブラーに焦点を当て、
    //     ほかをシンプルにするために1つのGASあたり1つのGeometryInstanceとする。
    // EN: Use one GeometryInstance per GAS for simplicty and
    //     to focus on deformation blur in this sample.
    struct Geometry {
        struct TriangleMesh {
            std::vector<cudau::TypedBuffer<Shared::Vertex>> vertexBuffers;
            cudau::TypedBuffer<ROBuffer<Shared::Vertex>> vertexBufferPointerBuffer;
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        };
        struct Curves {
            std::vector<cudau::TypedBuffer<Shared::CurveVertex>> vertexBuffers;
            cudau::TypedBuffer<ROBuffer<Shared::CurveVertex>> vertexBufferPointerBuffer;
            cudau::TypedBuffer<uint32_t> segmentIndexBuffer;
        };
        struct Spheres {
            std::vector<cudau::TypedBuffer<Shared::Sphere>> sphereBuffers;
            cudau::TypedBuffer<ROBuffer<Shared::Sphere>> sphereBufferPointerBuffer;
        };
        struct CustomPrimitives {
            std::vector<cudau::TypedBuffer<Shared::PartialSphereParameter>> partialSphereParamBuffers;
            cudau::TypedBuffer<ROBuffer<Shared::PartialSphereParameter>> partialSphereParamBufferPointerBuffer;
        };
        std::variant<TriangleMesh, Curves, Spheres, CustomPrimitives> shape;
        optixu::GeometryInstance optixGeomInst;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            if (std::holds_alternative<TriangleMesh>(shape)) {
                auto &geom = std::get<TriangleMesh>(shape);
                geom.triangleBuffer.finalize();
                geom.vertexBufferPointerBuffer.finalize();
                for (int i = geom.vertexBuffers.size() - 1; i >= 0; --i)
                    geom.vertexBuffers[i].finalize();
            }
            else if (std::holds_alternative<Curves>(shape)) {
                auto &geom = std::get<Curves>(shape);
                geom.segmentIndexBuffer.finalize();
                geom.vertexBufferPointerBuffer.finalize();
                for (int i = geom.vertexBuffers.size() - 1; i >= 0; --i)
                    geom.vertexBuffers[i].finalize();
            }
            else if (std::holds_alternative<Spheres>(shape)) {
                auto &geom = std::get<Spheres>(shape);
                geom.sphereBufferPointerBuffer.finalize();
                for (int i = geom.sphereBuffers.size() - 1; i >= 0; --i)
                    geom.sphereBuffers[i].finalize();
            }
            else {
                auto &geom = std::get<CustomPrimitives>(shape);
                geom.partialSphereParamBufferPointerBuffer.finalize();
                for (int i = geom.partialSphereParamBuffers.size() - 1; i >= 0; --i)
                    geom.partialSphereParamBuffers[i].finalize();
            }
            optixGeomInst.destroy();
        }
    };

    Geometry bunny;
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

        bunny.shape = Geometry::TriangleMesh();
        auto &shape = std::get<Geometry::TriangleMesh>(bunny.shape);

        // JP: 頂点バッファーを2ステップ分作る。
        //     2ステップ目は頂点位置を爆発するようにずらす。
        // EN: Create vertex buffers for two steps.
        //     The second step displaces the positions of vertices like explosion.
        uint32_t numMotionSteps = 2;
        shape.vertexBuffers.resize(numMotionSteps);
        shape.vertexBuffers[0].initialize(cuContext, cudau::BufferType::Device, vertices);
        for (int i = 0; i < vertices.size(); ++i) {
            Shared::Vertex &v = vertices[i];
            v.position = v.position + v.normal * length(v.position - float3(0, 0, 42)) * 0.25f;
        }
        shape.vertexBuffers[1].initialize(cuContext, cudau::BufferType::Device, vertices);
        shape.vertexBufferPointerBuffer.initialize(cuContext, cudau::BufferType::Device, numMotionSteps);
        shape.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        auto vertexBufferPointers = shape.vertexBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            vertexBufferPointers[i] = shape.vertexBuffers[i].getROBuffer<enableBufferOobCheck>();
        shape.vertexBufferPointerBuffer.unmap();
        geomData.vertexBuffers = shape.vertexBufferPointerBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = shape.triangleBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.numMotionSteps = numMotionSteps;

        bunny.optixGeomInst = scene.createGeometryInstance();
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        bunny.optixGeomInst.setMotionStepCount(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            bunny.optixGeomInst.setVertexBuffer(shape.vertexBuffers[i], i);
        bunny.optixGeomInst.setTriangleBuffer(shape.triangleBuffer);
        bunny.optixGeomInst.setMaterialCount(1, optixu::BufferView());
        bunny.optixGeomInst.setMaterial(0, 0, matForTriangles);
        bunny.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunny.optixGeomInst.setUserData(geomData);

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        bunny.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        bunny.optixGas.setMaterialSetCount(1);
        bunny.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        bunny.optixGas.addChild(bunny.optixGeomInst);
        bunny.optixGas.prepareForBuild(&asMemReqs);
        bunny.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry curves;
    {
        uint32_t numMotionSteps = 8;
        std::vector<std::vector<Shared::CurveVertex>> vertices(numMotionSteps);
        std::vector<uint32_t> indices;
        {
            const auto calcPosition = [](float p) {
                float r = (1 - p) * 0.1f + p * 0.2f;
                float y = p * 1.0f;
                float angle = 10 * pi_v<float> * p;
                Shared::CurveVertex v;
                v.position = float3(
                    r * std::cos(angle),
                    y,
                    r * std::sin(angle));
                return v;
            };

            constexpr uint32_t numSegments = 100;
            constexpr float begin0 = 0.0f;
            constexpr float end0 = 0.7f;
            constexpr float begin1 = 0.3f;
            constexpr float end1 = 1.0f;
            for (int i = 0; i < numSegments; ++i) {
                float posp = (float)i / (numSegments - 1);
                float width = 0.025f * std::sin(pi_v<float> * posp);
                for (int j = 0; j < numMotionSteps; ++j) {
                    float tp = (float)j / (numMotionSteps - 1);
                    float begin = (1 - tp) * begin0 + tp * begin1;
                    float end = (1 - tp) * end0 + tp * end1;
                    Shared::CurveVertex v = calcPosition((1 - posp) * begin + posp * end);
                    v.width = width;
                    vertices[j].push_back(v);
                }
            }

            for (int i = 0; i < numSegments - 3; ++i)
                indices.push_back(i);
        }

        curves.shape = Geometry::Curves();
        auto &shape = std::get<Geometry::Curves>(curves.shape);

        // JP: 頂点バッファーをモーションステップ分作る。
        // EN: Create vertex buffers for each motion step.
        shape.vertexBuffers.resize(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            shape.vertexBuffers[i].initialize(cuContext, cudau::BufferType::Device, vertices[i]);
        shape.vertexBufferPointerBuffer.initialize(cuContext, cudau::BufferType::Device, numMotionSteps);
        shape.segmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        auto vertexBufferPointers = shape.vertexBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            vertexBufferPointers[i] = shape.vertexBuffers[i].getROBuffer<enableBufferOobCheck>();
        shape.vertexBufferPointerBuffer.unmap();
        geomData.curveVertexBuffers = shape.vertexBufferPointerBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = shape.segmentIndexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.numMotionSteps = numMotionSteps;

        curves.optixGeomInst = scene.createGeometryInstance(optixu::GeometryType::CubicBSplines);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        curves.optixGeomInst.setMotionStepCount(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i) {
            cudau::TypedBuffer<Shared::CurveVertex> &buffer = shape.vertexBuffers[i];
            curves.optixGeomInst.setVertexBuffer(
                optixu::BufferView(buffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
                                   buffer.numElements(), sizeof(Shared::CurveVertex)), i);
            curves.optixGeomInst.setWidthBuffer(
                optixu::BufferView(buffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
                                   buffer.numElements(), sizeof(Shared::CurveVertex)), i);
        }
        curves.optixGeomInst.setSegmentIndexBuffer(shape.segmentIndexBuffer);
        curves.optixGeomInst.setCurveEndcapFlags(curveEndcap);
        curves.optixGeomInst.setMaterial(0, 0, matForCurves);
        curves.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        curves.optixGeomInst.setUserData(geomData);

        curves.optixGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::CubicBSplines);
        curves.optixGas.setConfiguration(
            curveASTradeOff, curveASUpdatable, curveASCompactable);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        curves.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        curves.optixGas.setMaterialSetCount(1);
        curves.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        curves.optixGas.addChild(curves.optixGeomInst);
        curves.optixGas.prepareForBuild(&asMemReqs);
        curves.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry spheres;
    {
        spheres.shape = Geometry::Spheres();
        auto &shape = std::get<Geometry::Spheres>(spheres.shape);

        constexpr uint32_t numPrimitives = 25;
        std::vector<Shared::Sphere> sphereParams0(numPrimitives);
        std::vector<Shared::Sphere> sphereParams1(numPrimitives);

        std::mt19937 rng(1290527201);
        std::uniform_real_distribution u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::Sphere &param0 = sphereParams0[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = -0.2f + 0.4f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param0.center = float3(x, y, z);
            param0.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);

            Shared::Sphere &param1 = sphereParams1[i];
            param1 = param0;
            param1.center += 0.4f * float3(u01(rng) - 0.5f,
                                           u01(rng) - 0.5f,
                                           u01(rng) - 0.5f);
            param1.radius *= 0.5f + 1.0f * u01(rng);
        }

        // JP: 球バッファーを2ステップ分作る。
        // EN: Create sphere buffers for two steps.
        uint32_t numMotionSteps = 2;
        shape.sphereBuffers.resize(numMotionSteps);
        shape.sphereBuffers[0].initialize(cuContext, cudau::BufferType::Device, sphereParams0);
        shape.sphereBuffers[1].initialize(cuContext, cudau::BufferType::Device, sphereParams1);
        shape.sphereBufferPointerBuffer.initialize(cuContext, cudau::BufferType::Device, numMotionSteps);

        Shared::GeometryData geomData = {};
        auto paramBufferPointers = shape.sphereBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            paramBufferPointers[i] = shape.sphereBuffers[i].getROBuffer<enableBufferOobCheck>();
        shape.sphereBufferPointerBuffer.unmap();
        geomData.sphereBuffers = shape.sphereBufferPointerBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.numMotionSteps = numMotionSteps;

        spheres.optixGeomInst = scene.createGeometryInstance(optixu::GeometryType::Spheres);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        spheres.optixGeomInst.setMotionStepCount(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i) {
            optixu::BufferView centerView(
                shape.sphereBuffers[i].getCUdeviceptr() + offsetof(Shared::Sphere, center),
                shape.sphereBuffers[i].numElements(),
                shape.sphereBuffers[i].stride());
            spheres.optixGeomInst.setVertexBuffer(centerView, i);
            optixu::BufferView radiusView(
                shape.sphereBuffers[i].getCUdeviceptr() + offsetof(Shared::Sphere, radius),
                shape.sphereBuffers[i].numElements(),
                shape.sphereBuffers[i].stride());
            spheres.optixGeomInst.setRadiusBuffer(radiusView, i);
        }
        spheres.optixGeomInst.setMaterialCount(1, optixu::BufferView());
        spheres.optixGeomInst.setMaterial(0, 0, matForSpheres);
        spheres.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        spheres.optixGeomInst.setUserData(geomData);

        spheres.optixGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::Spheres);
        spheres.optixGas.setConfiguration(
            sphereASTradeOff, sphereASUpdatable, sphereASCompactable);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        spheres.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        spheres.optixGas.setMaterialSetCount(1);
        spheres.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        spheres.optixGas.addChild(spheres.optixGeomInst);
        spheres.optixGas.prepareForBuild(&asMemReqs);
        spheres.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry partialSpheres;
    {
        partialSpheres.shape = Geometry::CustomPrimitives();
        auto &shape = std::get<Geometry::CustomPrimitives>(partialSpheres.shape);

        constexpr uint32_t numPrimitives = 25;
        std::vector<Shared::PartialSphereParameter> sphereParams0(numPrimitives);
        std::vector<Shared::PartialSphereParameter> sphereParams1(numPrimitives);

        std::mt19937 rng(1290527201);
        std::uniform_real_distribution u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::PartialSphereParameter &param0 = sphereParams0[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = -0.2f + 0.4f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param0.center = float3(x, y, z);
            param0.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);
            param0.minTheta = pi_v<float> *(0.15f * u01(rng));
            param0.maxTheta = pi_v<float> *(0.85f + 0.15f * u01(rng));
            param0.minPhi = 2 * pi_v<float> *(0.3f * u01(rng));
            param0.maxPhi = 2 * pi_v<float> *(0.7f + 0.3f * u01(rng));
            param0.aabb = AABB();
            param0.aabb.unify(param0.center - float3(param0.radius));
            param0.aabb.unify(param0.center + float3(param0.radius));

            Shared::PartialSphereParameter &param1 = sphereParams1[i];
            param1 = param0;
            param1.minPhi -= 0.5f * u01(rng);
            param1.maxPhi += 0.5f * u01(rng);
            param1.minTheta = pi_v<float> *(0.15f * u01(rng));
            param1.maxTheta = pi_v<float> *(0.85f + 0.15f * u01(rng));
            param1.aabb = AABB();
            param1.aabb.unify(param1.center - float3(param1.radius));
            param1.aabb.unify(param1.center + float3(param1.radius));
        }

        // JP: AABBバッファーを2ステップ分作る。
        // EN: Create AABB buffers for two steps.
        uint32_t numMotionSteps = 2;
        shape.partialSphereParamBuffers.resize(numMotionSteps);
        shape.partialSphereParamBuffers[0].initialize(cuContext, cudau::BufferType::Device, sphereParams0);
        shape.partialSphereParamBuffers[1].initialize(cuContext, cudau::BufferType::Device, sphereParams1);
        shape.partialSphereParamBufferPointerBuffer.initialize(
            cuContext, cudau::BufferType::Device, numMotionSteps);

        Shared::GeometryData geomData = {};
        auto paramBufferPointers = shape.partialSphereParamBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            paramBufferPointers[i] = shape.partialSphereParamBuffers[i].getROBuffer<enableBufferOobCheck>();
        shape.partialSphereParamBufferPointerBuffer.unmap();
        geomData.partialSphereParamBuffers =
            shape.partialSphereParamBufferPointerBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.numMotionSteps = numMotionSteps;

        partialSpheres.optixGeomInst = scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        partialSpheres.optixGeomInst.setMotionStepCount(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            partialSpheres.optixGeomInst.setCustomPrimitiveAABBBuffer(shape.partialSphereParamBuffers[i], i);
        partialSpheres.optixGeomInst.setMaterialCount(1, optixu::BufferView());
        partialSpheres.optixGeomInst.setMaterial(0, 0, matForPartialSpheres);
        partialSpheres.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        partialSpheres.optixGeomInst.setUserData(geomData);

        partialSpheres.optixGas =
            scene.createGeometryAccelerationStructure(optixu::GeometryType::CustomPrimitives);
        partialSpheres.optixGas.setConfiguration(
            optixu::ASTradeoff::PreferFastTrace,
            optixu::AllowUpdate::No,
            optixu::AllowCompaction::Yes);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        partialSpheres.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        partialSpheres.optixGas.setMaterialSetCount(1);
        partialSpheres.optixGas.setRayTypeCount(0, Shared::NumRayTypes);
        partialSpheres.optixGas.addChild(partialSpheres.optixGeomInst);
        partialSpheres.optixGas.prepareForBuild(&asMemReqs);
        partialSpheres.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: インスタンスを作成する。
    // EN: Create instances.
    Matrix3x3 bunnyMatSR = rotateY3x3(pi_v<float> / 4) * scale3x3(0.015f);
    float bunnyInstXfm[] = {
        bunnyMatSR.m00, bunnyMatSR.m01, bunnyMatSR.m02, 0,
        bunnyMatSR.m10, bunnyMatSR.m11, bunnyMatSR.m12, -0.6f,
        bunnyMatSR.m20, bunnyMatSR.m21, bunnyMatSR.m22, 0
    };
    optixu::Instance bunnyInst = scene.createInstance();
    bunnyInst.setChild(bunny.optixGas);
    bunnyInst.setTransform(bunnyInstXfm);

    float curveInstAXfm[] = {
        1, 0, 0, -1.2f,
        0, 1, 0, -0.4f,
        0, 0, 1, 0.0f
    };
    optixu::Instance curveInstA = scene.createInstance();
    curveInstA.setChild(curves.optixGas);
    curveInstA.setTransform(curveInstAXfm);

    float curveInstBXfm[] = {
        1, 0, 0, 1.2f,
        0, 1, 0, -0.4f,
        0, 0, 1, 0.0f
    };
    optixu::Instance curveInstB = scene.createInstance();
    curveInstB.setChild(curves.optixGas);
    curveInstB.setTransform(curveInstBXfm);

    float sphereInstXfm[] = {
        1, 0, 0, 0.0f,
        0, 1, 0, 1.0f,
        0, 0, 1, 0.0f
    };
    optixu::Instance spheresInst = scene.createInstance();
    spheresInst.setChild(spheres.optixGas);
    spheresInst.setTransform(sphereInstXfm);

    float partialSphereInstXfm[] = {
        1, 0, 0, 0.0f,
        0, 1, 0, -1.0f,
        0, 0, 1, 0.0f
    };
    optixu::Instance partialSpheresInst = scene.createInstance();
    partialSpheresInst.setChild(partialSpheres.optixGas);
    partialSpheresInst.setTransform(partialSphereInstXfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(bunnyInst);
    ias.addChild(curveInstA);
    ias.addChild(curveInstB);
    ias.addChild(spheresInst);
    ias.addChild(partialSpheresInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getChildCount());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    bunny.optixGas.rebuild(cuStream, bunny.gasMem, asBuildScratchMem);
    curves.optixGas.rebuild(cuStream, curves.gasMem, asBuildScratchMem);
    spheres.optixGas.rebuild(cuStream, spheres.gasMem, asBuildScratchMem);
    partialSpheres.optixGas.rebuild(cuStream, partialSpheres.gasMem, asBuildScratchMem);

    /*
    JP: 静的なメッシュはコンパクションもしておく。
        ここではモーションがあることが"動的"を意味しない。
        頻繁にASのリビルドが必要なものを"動的"、そうでないものを"静的"とする。
        複数のメッシュのASをひとつのバッファーに詰めて記録する。
    EN: Perform compaction for static meshes.
        The presence of motion does not mean "dynamic" here.
        Call things as "dynamic" for which we often need to rebuild the AS otherwise call them as "static".
        Record ASs of multiple meshes into single buffer back to back.
    */
    struct CompactedASInfo {
        Geometry* geom;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &bunny, 0, 0 },
        { &curves, 0, 0 },
        { &spheres, 0, 0 },
        { &partialSpheres, 0, 0 },
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



    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 4> rngBuffer;



    constexpr int32_t initWindowContentWidth = 1024;
    constexpr int32_t initWindowContentHeight = 1024;

    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize.x = initWindowContentWidth;
    plp.imageSize.y = initWindowContentHeight;
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
    initConfig.windowTitle = "OptiX Utility - Deformation Blur";
    initConfig.resourceDir = resourceDir;
    initConfig.windowContentRenderWidth = initWindowContentWidth;
    initConfig.windowContentRenderHeight = initWindowContentHeight;
    initConfig.cameraPosition = make_float3(0, 0, 3.8f);
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

    const auto initializeResDependentResources = [&]
    (int32_t width, int32_t height) {
        rngBuffer.initialize(cuContext, cudau::BufferType::Device, width, height);
        {
            std::mt19937 rng(50932423);

            rngBuffer.map();
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    rngBuffer(x, y).setState(rng());
                }
            }
            rngBuffer.unmap();
        }

        plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    };

    const auto resizeResDependentResources = [&]
    (int32_t width, int32_t height) {
        rngBuffer.resize(width, height);
        {
            std::mt19937 rng(50932423);

            rngBuffer.map();
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    rngBuffer(x, y).setState(rng());
                }
            }
            rngBuffer.unmap();
        }

        plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    };

    const auto finalizeResDependentResources = [&]
    () {
        rngBuffer.finalize();
    };

    initializeResDependentResources(initWindowContentWidth, initWindowContentHeight);

    const auto onRenderLoop = [&]
    (const RunArguments &args) {
        const uint64_t frameIndex = args.frameIndex;
        const CUstream curStream = args.curStream;

        // Camera Window
        bool cameraIsActuallyMoving = args.cameraIsActuallyMoving;
        {
            ImGui::SetNextWindowPos(ImVec2(8, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

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
        static float centerTime = 0.5f;
        static float exposureTime = 1.0f;
        static bool usePerPixelRNGs = true;
        bool resetAccum = false;
        {
            ImGui::SetNextWindowPos(ImVec2(690, 8), ImGuiCond_FirstUseEver);
            ImGui::Begin("Debug", nullptr, ImGuiWindowFlags_AlwaysAutoResize);

            resetAccum |= ImGui::SliderFloat("Center Time", &centerTime, 0.0f, 1.0f);
            resetAccum |= ImGui::SliderFloat("Exposure Time", &exposureTime, 0.0f, 1.0f);
            ImGui::Text(
                "Time: %.3f - %.3f",
                centerTime - 0.5f * exposureTime,
                centerTime + 0.5f * exposureTime);

            resetAccum |= ImGui::Checkbox("Per-pixel RNGs", &usePerPixelRNGs);

            ImGui::End();
        }



        bool firstAccumFrame =
            cameraIsActuallyMoving ||
            args.resized ||
            frameIndex == 0 ||
            resetAccum;
        static uint32_t numAccumFrames = 0;
        if (firstAccumFrame)
            numAccumFrames = 0;

        // Render
        outputBufferSurfaceHolder.beginCUDAAccess(curStream);

        static Shared::PCG32RNG globalRNG;
        if (numAccumFrames == 0)
            globalRNG.setState(419511321);

        plp.colorAccumBuffer = outputBufferSurfaceHolder.getNext();
        plp.timeBegin = centerTime - 0.5f * exposureTime;
        plp.timeEnd = centerTime + 0.5f * exposureTime;
        plp.numAccumFrames = numAccumFrames;
        plp.globalRNG = globalRNG;
        plp.usePerPixelRNGs = usePerPixelRNGs;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), curStream));
        pipeline.launch(
            curStream, plpOnDevice, args.windowContentRenderWidth, args.windowContentRenderHeight, 1);
        ++numAccumFrames;
        globalRNG();

        outputBufferSurfaceHolder.endCUDAAccess(curStream, true);



        ReturnValuesToRenderLoop ret = {};
        ret.enable_sRGB = false;
        ret.finish = false;

        if (takeScreenShot && frameIndex + 1 == 1024) {
            CUDADRV_CHECK(cuStreamSynchronize(curStream));
            const uint32_t numPixels = args.windowContentRenderWidth * args.windowContentRenderHeight;
            auto rawImage = new float4[numPixels];
            glGetTextureSubImage(
                args.outputTexture->getHandle(), 0,
                0, 0, 0, args.windowContentRenderWidth, args.windowContentRenderHeight, 1,
                GL_RGBA, GL_FLOAT,
                sizeof(float4) * numPixels, rawImage);
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

         resizeResDependentResources(windowContentWidth, windowContentHeight);

         // EN: update the pipeline parameters.
         plp.imageSize = int2(windowContentWidth, windowContentHeight);
         plp.camera.aspect = (float)windowContentWidth / windowContentHeight;
    };

    framework.run(onRenderLoop, onResolutionChange);

    finalizeResDependentResources();

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

    partialSpheresInst.destroy();
    spheresInst.destroy();
    curveInstB.destroy();
    curveInstA.destroy();
    bunnyInst.destroy();

    partialSpheres.finalize();
    spheres.finalize();
    curves.finalize();
    bunny.finalize();

    scene.destroy();

    matForPartialSpheres.destroy();
    matForSpheres.destroy();
    matForCurves.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

    hitProgramGroupForPartialSpheres.destroy();
    hitProgramGroupForSpheres.destroy();
    hitProgramGroupForCurves.destroy();
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
