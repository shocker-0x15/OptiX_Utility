/*

JP: このサンプルはデフォーメーション(変形)ブラーを扱うGASを構築する方法を示します。
    複数のモーションステップに対応する頂点(もしくはAABB)バッファーをGeometryInstanceに設定し
    GASに適切なモーション設定を行うことでデフォーメーションブラーに対応するGASを構築できます。
EN: This sample shows how to build a GAS to handle deformation blur.
    Set a vertex (or AABB) buffer to each of multiple motion steps of GeometryInstance and
    set appropriate motion configuration to a GAS to build a GAS capable of deformation blur.
*/

#include "deformation_blur_shared.h"

#include "../common/obj_loader.h"

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

    // JP: このサンプルでは2段階のAS(1段階のインスタンシング)を使用する。
    //     カーブ・カスタムプリミティブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    // EN: This sample uses two-level AS (single-level instancing).
    //     Appropriately setting primitive type flags is required since this sample uses curve and
    //     custom primitive intersection.
    pipeline.setPipelineOptions(Shared::PayloadSignature::numDwords,
                                std::max<uint32_t>({
                                    optixu::calcSumDwords<float2>(),
                                    optixu::calcSumDwords<float>(),
                                    Shared::SphereAttributeSignature::numDwords }),
                                "plp", sizeof(Shared::PipelineLaunchParameters),
                                true, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
                                OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE |
                                OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "deformation_blur/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::ProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroupForTriangleIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr);

    constexpr OptixCurveEndcapFlags curveEndcap = OPTIX_CURVE_ENDCAP_ON;
    constexpr optixu::ASTradeoff curveASTradeOff = optixu::ASTradeoff::PreferFastTrace;
    constexpr bool curveASUpdatable = false;
    constexpr bool curveASCompactable = true;
    optixu::ProgramGroup hitProgramGroupForCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, Shared::useEmbeddedVertexData);

    // JP: このヒットグループはレイと球の交叉判定用なのでカスタムのIntersectionプログラムを渡す。
    // EN: This is for ray-sphere intersection, so pass a custom intersection program.
    optixu::ProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroupForCustomIS(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        moduleOptiX, RT_IS_NAME_STR("intersectSphere"));

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
    optixu::Material matForCurves = optixContext.createMaterial();
    matForCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCurves);
    optixu::Material matForSpheres = optixContext.createMaterial();
    matForSpheres.setHitGroup(Shared::RayType_Primary, hitProgramGroupForSpheres);

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
            cudau::TypedBuffer<const Shared::Vertex*> vertexBufferPointerBuffer;
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        };
        struct Curves {
            std::vector<cudau::TypedBuffer<Shared::CurveVertex>> vertexBuffers;
            cudau::TypedBuffer<const Shared::CurveVertex*> vertexBufferPointerBuffer;
            cudau::TypedBuffer<uint32_t> segmentIndexBuffer;
        };
        struct CustomPrimitives {
            std::vector<cudau::TypedBuffer<Shared::SphereParameter>> paramBuffers;
            cudau::TypedBuffer<const Shared::SphereParameter*> paramBufferPointerBuffer;
        };
        std::variant<TriangleMesh, Curves, CustomPrimitives> shape;
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
            else {
                auto &geom = std::get<CustomPrimitives>(shape);
                geom.paramBufferPointerBuffer.finalize();
                for (int i = geom.paramBuffers.size() - 1; i >= 0; --i)
                    geom.paramBuffers[i].finalize();
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
                        triangles.size(),
                        triangles.data());
        }

        bunny.shape = Geometry::TriangleMesh();
        auto &shape = std::get<Geometry::TriangleMesh>(bunny.shape);

        // JP: 頂点バッファーを2ステップ分作る。
        //     2ステップ目は頂点位置を爆発するようにずらす。
        // EN: Create vertex buffer for two steps.
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
            vertexBufferPointers[i] = shape.vertexBuffers[i].getDevicePointer();
        shape.vertexBufferPointerBuffer.unmap();
        geomData.vertexBuffers = shape.vertexBufferPointerBuffer.getDevicePointer();
        geomData.triangleBuffer = shape.triangleBuffer.getDevicePointer();
        geomData.numMotionSteps = numMotionSteps;

        bunny.optixGeomInst = scene.createGeometryInstance();
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        bunny.optixGeomInst.setNumMotionSteps(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            bunny.optixGeomInst.setVertexBuffer(shape.vertexBuffers[i], i);
        bunny.optixGeomInst.setTriangleBuffer(shape.triangleBuffer);
        bunny.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        bunny.optixGeomInst.setMaterial(0, 0, matForTriangles);
        bunny.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunny.optixGeomInst.setUserData(geomData);

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        bunny.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        bunny.optixGas.setNumMaterialSets(1);
        bunny.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
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
                float angle = 10 * M_PI * p;
                Shared::CurveVertex v;
                v.position = float3(r * std::cos(angle),
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
                float width = 0.025f * std::sin(M_PI * posp);
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
        // EN: Create vertex buffer for each motion step.
        shape.vertexBuffers.resize(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            shape.vertexBuffers[i].initialize(cuContext, cudau::BufferType::Device, vertices[i]);
        shape.vertexBufferPointerBuffer.initialize(cuContext, cudau::BufferType::Device, numMotionSteps);
        shape.segmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        auto vertexBufferPointers = shape.vertexBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            vertexBufferPointers[i] = shape.vertexBuffers[i].getDevicePointer();
        shape.vertexBufferPointerBuffer.unmap();
        geomData.curveVertexBuffers = shape.vertexBufferPointerBuffer.getDevicePointer();
        geomData.segmentIndexBuffer = shape.segmentIndexBuffer.getDevicePointer();
        geomData.numMotionSteps = numMotionSteps;

        curves.optixGeomInst = scene.createGeometryInstance(optixu::GeometryType::CubicBSplines);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        curves.optixGeomInst.setNumMotionSteps(numMotionSteps);
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
        curves.optixGas.setConfiguration(curveASTradeOff, curveASUpdatable, curveASCompactable,
                                         Shared::useEmbeddedVertexData);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        curves.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        curves.optixGas.setNumMaterialSets(1);
        curves.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        curves.optixGas.addChild(curves.optixGeomInst);
        curves.optixGas.prepareForBuild(&asMemReqs);
        curves.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry spheres;
    {
        spheres.shape = Geometry::CustomPrimitives();
        auto &shape = std::get<Geometry::CustomPrimitives>(spheres.shape);

        constexpr uint32_t numPrimitives = 25;
        std::vector<Shared::SphereParameter> sphereParams0(numPrimitives);
        std::vector<Shared::SphereParameter> sphereParams1(numPrimitives);

        std::mt19937 rng(1290527201);
        std::uniform_real_distribution u01;
        for (int i = 0; i < numPrimitives; ++i) {
            Shared::SphereParameter &param0 = sphereParams0[i];
            float x = -0.8f + 1.6f * (i % 5) / 4.0f;
            float y = -1.0f + 0.4f * u01(rng);
            float z = -0.8f + 1.6f * (i / 5) / 4.0f;
            param0.center = float3(x, y, z);
            param0.radius = 0.1f + 0.1f * (u01(rng) - 0.5f);
            param0.aabb = AABB();
            param0.aabb.unify(param0.center - float3(param0.radius));
            param0.aabb.unify(param0.center + float3(param0.radius));

            Shared::SphereParameter &param1 = sphereParams1[i];
            param1 = param0;
            param1.center += 0.4f * float3(u01(rng) - 0.5f,
                                           u01(rng) - 0.5f,
                                           u01(rng) - 0.5f);
            param1.radius *= 0.5f + 1.0f * u01(rng);
            param1.aabb = AABB();
            param1.aabb.unify(param1.center - float3(param1.radius));
            param1.aabb.unify(param1.center + float3(param1.radius));
        }

        // JP: AABBバッファーを2ステップ分作る。
        // EN: Create AABB buffer for two steps.
        uint32_t numMotionSteps = 2;
        shape.paramBuffers.resize(numMotionSteps);
        shape.paramBuffers[0].initialize(cuContext, cudau::BufferType::Device, sphereParams0);
        shape.paramBuffers[1].initialize(cuContext, cudau::BufferType::Device, sphereParams1);
        shape.paramBufferPointerBuffer.initialize(cuContext, cudau::BufferType::Device, numMotionSteps);

        Shared::GeometryData geomData = {};
        auto paramBufferPointers = shape.paramBufferPointerBuffer.map();
        for (int i = 0; i < numMotionSteps; ++i)
            paramBufferPointers[i] = shape.paramBuffers[i].getDevicePointer();
        shape.paramBufferPointerBuffer.unmap();
        geomData.paramBuffers = shape.paramBufferPointerBuffer.getDevicePointer();
        geomData.numMotionSteps = numMotionSteps;

        spheres.optixGeomInst = scene.createGeometryInstance(optixu::GeometryType::CustomPrimitives);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        spheres.optixGeomInst.setNumMotionSteps(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            spheres.optixGeomInst.setCustomPrimitiveAABBBuffer(shape.paramBuffers[i], i);
        spheres.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        spheres.optixGeomInst.setMaterial(0, 0, matForSpheres);
        spheres.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        spheres.optixGeomInst.setUserData(geomData);

        spheres.optixGas = scene.createGeometryAccelerationStructure(optixu::GeometryType::CustomPrimitives);
        spheres.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        // JP: GASのモーション設定を行う。
        // EN: Set the GAS's motion configuration.
        spheres.optixGas.setMotionOptions(numMotionSteps, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        spheres.optixGas.setNumMaterialSets(1);
        spheres.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        spheres.optixGas.addChild(spheres.optixGeomInst);
        spheres.optixGas.prepareForBuild(&asMemReqs);
        spheres.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: インスタンスを作成する。
    // EN: Create instances.
    Matrix3x3 bunnyMatSR = rotateY3x3(M_PI / 4) * scale3x3(0.015f);
    float bunnyInstXfm[] = {
        bunnyMatSR.m00, bunnyMatSR.m01, bunnyMatSR.m02, 0,
        bunnyMatSR.m10, bunnyMatSR.m11, bunnyMatSR.m12, -0.2f,
        bunnyMatSR.m20, bunnyMatSR.m21, bunnyMatSR.m22, 0
    };
    optixu::Instance bunnyInst = scene.createInstance();
    bunnyInst.setChild(bunny.optixGas);
    bunnyInst.setTransform(bunnyInstXfm);

    float curveInstAXfm[] = {
        1, 0, 0, -1.0f,
        0, 1, 0, 0.0f,
        0, 0, 1, 0.0f
    };
    optixu::Instance curveInstA = scene.createInstance();
    curveInstA.setChild(curves.optixGas);
    curveInstA.setTransform(curveInstAXfm);

    float curveInstBXfm[] = {
        1, 0, 0, 1.0f,
        0, 1, 0, 0.0f,
        0, 0, 1, 0.0f
    };
    optixu::Instance curveInstB = scene.createInstance();
    curveInstB.setChild(curves.optixGas);
    curveInstB.setTransform(curveInstBXfm);

    optixu::Instance spheresInst = scene.createInstance();
    spheresInst.setChild(spheres.optixGas);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    ias.addChild(bunnyInst);
    ias.addChild(curveInstA);
    ias.addChild(curveInstB);
    ias.addChild(spheresInst);
    ias.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, ias.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    bunny.optixGas.rebuild(cuStream, bunny.gasMem, asBuildScratchMem);
    curves.optixGas.rebuild(cuStream, curves.gasMem, asBuildScratchMem);
    spheres.optixGas.rebuild(cuStream, spheres.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     ここではモーションがあることが"動的"を意味しない。頻繁にASのリビルドが必要なものを"動的"、そうでないものを"静的"とする。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     The existence of motion does not mean "dynamic" here.
    //     Call things as "dynamic" for which we often need to rebuild the AS otherwise call them as "static".
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        Geometry* geom;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { &bunny, 0, 0 },
        { &curves, 0, 0 },
        { &spheres, 0, 0 },
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
        info.geom->optixGas.compact(cuStream, optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset,
                                                      info.size, 1));
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
    plp.timeBegin = 0.0f;
    plp.timeEnd = 1.0f;
    plp.numAccumFrames = 0;
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5f);
    plp.camera.orientation = rotateY3x3(M_PI);

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



    rngBuffer.finalize();
    accumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    spheresInst.destroy();
    curveInstB.destroy();
    curveInstA.destroy();
    bunnyInst.destroy();

    spheres.finalize();
    curves.finalize();
    bunny.finalize();

    scene.destroy();

    matForSpheres.destroy();
    matForCurves.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

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
