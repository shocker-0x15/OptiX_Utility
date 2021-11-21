/*

JP: このサンプルはマルチレベルのインスタンシングとモーショントランスフォームの使用方法を示します。
    OptiXはシングルレベルのインスタンシングに加えて、マルチレベルインスタンシングをサポートします。
    マルチレベルインスタンシングではIASに所属するインスタンスがGASだけではなく、
    他のIASやトランスフォームを参照することができます。
    モーショントランスフォームにはひとつのASをセットし、ASのある時間中の動きを記述します。
    モーショントランスフォームを配下に持つIASは空間に加えて時間も考慮したレイトレース高速化機構を構築し、
    効率的なモーションブラーのレンダリングに使用できます。

EN: This sample shows how to use multi-level instancing as well as motion transform.
    OptiX supports multi-level instancing in addition to single-level instancing.
    In the case of multi-level instancing, an instance belonging to an IAS can refer not only GAS but also
    another IAS or a transform.
    Motion transform has an AS and describes the motion of the AS during a time interval.
    An IAS having a motion transform as a child builds an acceleration structure for raytracing which
    consider time as well as space, enabling efficient motion blur rendering.

*/

#include "multi_level_instancing_shared.h"

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

    // JP: このサンプルでは多段階のASとトランスフォームを使用する。
    // EN: This sample uses multi-level AS and transforms.
    pipeline.setPipelineOptions(optixu::calcSumDwords<PayloadSignature>(),
                                optixu::calcSumDwords<float2>(),
                                "plp", sizeof(Shared::PipelineLaunchParameters),
                                true, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                DEBUG_SELECT(OPTIX_EXCEPTION_FLAG_DEBUG, OPTIX_EXCEPTION_FLAG_NONE),
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "multi_level_instancing/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        DEBUG_SELECT(OPTIX_COMPILE_OPTIMIZATION_LEVEL_0, OPTIX_COMPILE_OPTIMIZATION_DEFAULT),
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    optixu::ProgramGroup hitProgramGroup = pipeline.createHitProgramGroupForTriangleIS(
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

    OptixStackSizes stackSizes;

    rayGenProgram.getStackSize(&stackSizes);
    uint32_t cssRG = stackSizes.cssRG;

    missProgram.getStackSize(&stackSizes);
    uint32_t cssMS = stackSizes.cssMS;

    hitProgramGroup.getStackSize(&stackSizes);
    uint32_t cssCH = stackSizes.cssCH;

    uint32_t dcStackSizeFromTrav = 0; // This sample doesn't call a direct callable during traversal.
    uint32_t dcStackSizeFromState = 0;
    // Possible Program Paths:
    // RG - CH
    // RG - MS
    uint32_t ccStackSize = cssRG + std::max(cssCH, cssMS);
    // The deepest path: IAS - IAS - SRTXfm - GAS
    uint32_t maxTraversableGraphDepth = 4;
    pipeline.setStackSize(dcStackSizeFromTrav, dcStackSizeFromState, ccStackSize, maxTraversableGraphDepth);

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

    optixu::Material mat0 = optixContext.createMaterial();
    mat0.setHitGroup(Shared::RayType_Primary, hitProgramGroup);

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: このサンプルではマルチレベルインスタンシングやトランスフォームに焦点を当て、
    //     ほかをシンプルにするために1つのGASあたり1つのGeometryInstanceとする。
    // EN: Use one GeometryInstance per GAS for simplicty and
    //     to focus on multi-level instancing and transforms in this sample.
    struct Geometry {
        cudau::TypedBuffer<Shared::Vertex> vertexBuffer;
        cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        optixu::GeometryInstance optixGeomInst;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            triangleBuffer.finalize();
            vertexBuffer.finalize();
            optixGeomInst.destroy();
        }
    };

    Geometry room;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
            // back wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(1, 0) },
            // ceiling
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, -1, 0), make_float2(1, 0) },
            // left wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(1, 0, 0), make_float2(0, 1) },
            { make_float3(-1.0f, 1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 1) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(1, 0, 0), make_float2(1, 0) },
            // right wall
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 0) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(-1, 0, 0), make_float2(0, 1) },
            { make_float3(1.0f, 1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(-1, 0, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            // floor
            { 0, 1, 2 }, { 0, 2, 3 },
            // back wall
            { 4, 7, 6 }, { 4, 6, 5 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        room.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        room.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = room.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = room.triangleBuffer.getDevicePointer();

        room.optixGeomInst = scene.createGeometryInstance();
        room.optixGeomInst.setVertexBuffer(room.vertexBuffer);
        room.optixGeomInst.setTriangleBuffer(room.triangleBuffer);
        room.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        room.optixGeomInst.setMaterial(0, 0, mat0);
        room.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        room.optixGeomInst.setUserData(geomData);

        room.optixGas = scene.createGeometryAccelerationStructure();
        room.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        room.optixGas.setNumMaterialSets(1);
        room.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        room.optixGas.addChild(room.optixGeomInst);
        room.optixGas.prepareForBuild(&asMemReqs);
        room.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry areaLight;
    {
        Shared::Vertex vertices[] = {
            { make_float3(-0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(0, 0) },
            { make_float3(-0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(0, 1) },
            { make_float3(0.25f, 0.0f, 0.25f), make_float3(0, -1, 0), make_float2(1, 1) },
            { make_float3(0.25f, 0.0f, -0.25f), make_float3(0, -1, 0), make_float2(1, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        areaLight.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        areaLight.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = areaLight.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = areaLight.triangleBuffer.getDevicePointer();

        areaLight.optixGeomInst = scene.createGeometryInstance();
        areaLight.optixGeomInst.setVertexBuffer(areaLight.vertexBuffer);
        areaLight.optixGeomInst.setTriangleBuffer(areaLight.triangleBuffer);
        areaLight.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        areaLight.optixGeomInst.setMaterial(0, 0, mat0);
        areaLight.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        areaLight.optixGeomInst.setUserData(geomData);

        areaLight.optixGas = scene.createGeometryAccelerationStructure();
        areaLight.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        areaLight.optixGas.setNumMaterialSets(1);
        areaLight.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        areaLight.optixGas.addChild(areaLight.optixGeomInst);
        areaLight.optixGas.prepareForBuild(&asMemReqs);
        areaLight.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

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

        bunny.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        bunny.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = bunny.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = bunny.triangleBuffer.getDevicePointer();

        bunny.optixGeomInst = scene.createGeometryInstance();
        bunny.optixGeomInst.setVertexBuffer(bunny.vertexBuffer);
        bunny.optixGeomInst.setTriangleBuffer(bunny.triangleBuffer);
        bunny.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        bunny.optixGeomInst.setMaterial(0, 0, mat0);
        bunny.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        bunny.optixGeomInst.setUserData(geomData);

        bunny.optixGas = scene.createGeometryAccelerationStructure();
        bunny.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        bunny.optixGas.setNumMaterialSets(1);
        bunny.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        bunny.optixGas.addChild(bunny.optixGeomInst);
        bunny.optixGas.prepareForBuild(&asMemReqs);
        bunny.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }

    Geometry cube;
    {
        std::vector<obj::Vertex> objVertices;
        std::vector<obj::MaterialGroup> objMatGroups;
        obj::load("../../data/subd_cube.obj", &objVertices, &objMatGroups, nullptr);

        // JP: このサンプルではobjのマテリアルを区別しないのでグループをひとつにまとめる。
        // EN: Combine groups into one because this sample doesn't distinguish obj materials.
        std::vector<Shared::Vertex> vertices;
        std::vector<Shared::Triangle> triangles;
        {
            vertices.resize(objVertices.size());
            for (int vIdx = 0; vIdx < objVertices.size(); ++vIdx) {
                const obj::Vertex &objVertex = objVertices[vIdx];
                vertices[vIdx] = Shared::Vertex{ objVertex.position, objVertex.normal, objVertex.texCoord };
            }
            for (int mIdx = 0; mIdx < objMatGroups.size(); ++mIdx) {
                const obj::MaterialGroup &matGroup = objMatGroups[mIdx];
                uint32_t baseIndex = triangles.size();
                triangles.resize(triangles.size() + matGroup.triangles.size());
                std::copy_n(reinterpret_cast<const Shared::Triangle*>(matGroup.triangles.data()),
                            matGroup.triangles.size(),
                            triangles.data() + baseIndex);
            }
        }

        cube.vertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        cube.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = cube.vertexBuffer.getDevicePointer();
        geomData.triangleBuffer = cube.triangleBuffer.getDevicePointer();

        cube.optixGeomInst = scene.createGeometryInstance();
        cube.optixGeomInst.setVertexBuffer(cube.vertexBuffer);
        cube.optixGeomInst.setTriangleBuffer(cube.triangleBuffer);
        cube.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        cube.optixGeomInst.setMaterial(0, 0, mat0);
        cube.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        cube.optixGeomInst.setUserData(geomData);

        cube.optixGas = scene.createGeometryAccelerationStructure();
        cube.optixGas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, true, false);
        cube.optixGas.setNumMaterialSets(1);
        cube.optixGas.setNumRayTypes(0, Shared::NumRayTypes);
        cube.optixGas.addChild(cube.optixGeomInst);
        cube.optixGas.prepareForBuild(&asMemReqs);
        cube.gasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
        maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);
    }



    // JP: GAS/Transformインスタンスを作成する。
    // EN: Create GAS/Transform instances.
    optixu::Instance roomInst = scene.createInstance();
    roomInst.setChild(room.optixGas);

    float areaLightInstXfm[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance areaLightInst = scene.createInstance();
    areaLightInst.setChild(areaLight.optixGas);
    areaLightInst.setTransform(areaLightInstXfm);

    struct Transform {
        struct SRT {
            float3 s;
            Quaternion o;
            float3 t;
        };
        std::vector<SRT> srts;
        optixu::Transform optixTransform;
        cudau::Buffer* deviceMem; // TODO?: define move constructor.
    };
    std::vector<Transform> objectTransforms;
    std::vector<optixu::Instance> objectInsts;

    // Bunny
    {
        Transform tr;
        Transform::SRT srt0;
        srt0.s = make_float3(0.005f);
        srt0.o = Quaternion(0, 0, 0, 1);
        srt0.t = make_float3(-0.5f, -1.0f, 0);
        tr.srts.push_back(srt0);
        Transform::SRT srt1;
        srt1.s = make_float3(0.005f);
        srt1.o = Quaternion(0, 0, 0, 1);
        srt1.t = make_float3(-0.5f, -1.0f + 0.2f, 0);
        tr.srts.push_back(srt1);
        Transform::SRT srt2;
        srt2.s = make_float3(0.005f);
        srt2.o = Quaternion(0, 0, 0, 1);
        srt2.t = make_float3(-0.5f + 0.2f, -1.0f + 0.4f, 0);
        tr.srts.push_back(srt2);

        size_t trMemSize;
        tr.optixTransform = scene.createTransform();
        tr.optixTransform.setConfiguration(optixu::TransformType::SRTMotion, tr.srts.size(), &trMemSize);
        tr.optixTransform.setMotionOptions(0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        tr.optixTransform.setChild(bunny.optixGas);
        for (int keyIdx = 0; keyIdx < tr.srts.size(); ++keyIdx)
            tr.optixTransform.setSRTMotionKey(keyIdx,
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].s),
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].o),
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].t));
        tr.deviceMem = new cudau::Buffer;
        tr.deviceMem->initialize(cuContext, cudau::BufferType::Device, trMemSize, 1);
        objectTransforms.push_back(tr);

        optixu::Instance inst = scene.createInstance();
        inst.setChild(tr.optixTransform);
        objectInsts.push_back(inst);
    }

    // Cube
    {
        Transform tr;
        Transform::SRT srt0;
        srt0.s = make_float3(0.25f);
        srt0.o = Quaternion(0, 0, 0, 1);
        srt0.t = make_float3(0, -0.5f, 0);
        tr.srts.push_back(srt0);
        Transform::SRT srt1;
        srt1.s = make_float3(0.25f);
        srt1.o = qRotateX(M_PI / 2);
        srt1.t = make_float3(0, -0.5f, 0);
        tr.srts.push_back(srt1);
        Transform::SRT srt2;
        srt2.s = make_float3(0.25f);
        srt2.o = qRotateX(M_PI);
        srt2.t = make_float3(0, -0.5f, 0);
        tr.srts.push_back(srt2);

        size_t trMemSize;
        tr.optixTransform = scene.createTransform();
        tr.optixTransform.setConfiguration(optixu::TransformType::SRTMotion, tr.srts.size(), &trMemSize);
        tr.optixTransform.setMotionOptions(0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
        tr.optixTransform.setChild(cube.optixGas);
        for (int keyIdx = 0; keyIdx < tr.srts.size(); ++keyIdx)
            tr.optixTransform.setSRTMotionKey(keyIdx,
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].s),
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].o),
                                              reinterpret_cast<float*>(&tr.srts[keyIdx].t));
        tr.deviceMem = new cudau::Buffer;
        tr.deviceMem->initialize(cuContext, cudau::BufferType::Device, trMemSize, 1);
        objectTransforms.push_back(tr);

        optixu::Instance inst = scene.createInstance();
        inst.setChild(tr.optixTransform);
        Matrix3x3 rotMat = rotate3x3(M_PI / 2, 0, 1, 0);
        float instXfm[] = {
            rotMat.m00, rotMat.m01, rotMat.m02, 0.5f,
            rotMat.m10, rotMat.m11, rotMat.m12, 0.0f,
            rotMat.m20, rotMat.m21, rotMat.m22, 0.0f,
        };
        inst.setTransform(instXfm);
        objectInsts.push_back(inst);
    }



    // JP: 下位のInstance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure of the lower layer.
    optixu::InstanceAccelerationStructure lowerIas = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    constexpr uint32_t numMotionKeys = 3;
    lowerIas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    lowerIas.setMotionOptions(numMotionKeys, 0.0f, 1.0f, OPTIX_MOTION_FLAG_NONE);
    lowerIas.addChild(roomInst);
    lowerIas.addChild(areaLightInst);
    for (int i = 0; i < objectInsts.size(); ++i)
        lowerIas.addChild(objectInsts[i]);
    lowerIas.prepareForBuild(&asMemReqs);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, lowerIas.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: IASインスタンスを作成する。
    // EN: Create IAS instances.
    optixu::Instance topInstA = scene.createInstance();
    topInstA.setChild(lowerIas);
    float instATr[] = {
        1, 0, 0, -1.25f,
        0, 1, 0, -1.25f,
        0, 0, 1, 0
    };
    topInstA.setTransform(instATr);

    optixu::Instance topInstB = scene.createInstance();
    topInstB.setChild(lowerIas);
    float instBTr[] = {
        1, 0, 0, 1.25f,
        0, 1, 0, -1.25f,
        0, 0, 1, 0
    };
    topInstB.setTransform(instBTr);

    optixu::Instance topInstC = scene.createInstance();
    topInstC.setChild(lowerIas);
    float instCTr[] = {
        1, 0, 0, -1.25f,
        0, 1, 0, 1.25f,
        0, 0, 1, 0
    };
    topInstC.setTransform(instCTr);

    optixu::Instance topInstD = scene.createInstance();
    topInstD.setChild(lowerIas);
    float instDTr[] = {
        1, 0, 0, 1.25f,
        0, 1, 0, 1.25f,
        0, 0, 1, 0
    };
    topInstD.setTransform(instDTr);



    // JP: 最上位のInstance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure of the top layer.
    optixu::InstanceAccelerationStructure topIas = scene.createInstanceAccelerationStructure();
    cudau::Buffer topIasMem;
    cudau::TypedBuffer<OptixInstance> topInstanceBuffer;
    topIas.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false, false);
    topIas.addChild(topInstA);
    topIas.addChild(topInstB);
    topIas.addChild(topInstC);
    topIas.addChild(topInstD);
    topIas.prepareForBuild(&asMemReqs);
    topIasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    topInstanceBuffer.initialize(cuContext, cudau::BufferType::Device, topIas.getNumChildren());
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    room.optixGas.rebuild(cuStream, room.gasMem, asBuildScratchMem);
    areaLight.optixGas.rebuild(cuStream, areaLight.gasMem, asBuildScratchMem);
    bunny.optixGas.rebuild(cuStream, bunny.gasMem, asBuildScratchMem);
    cube.optixGas.rebuild(cuStream, cube.gasMem, asBuildScratchMem);

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
        { &room, 0, 0 },
        { &areaLight, 0, 0 },
        { &bunny, 0, 0 },
        { &cube, 0, 0 },
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



    // JP: Transformビルド時には子のTraversable Handleが確定している必要がある。
    // EN: Traversable handle for the child must be fixed when building Transform.
    for (int tIdx = 0; tIdx < objectTransforms.size(); ++tIdx) {
        const Transform &tr = objectTransforms[tIdx];
        tr.optixTransform.rebuild(cuStream, *tr.deviceMem);
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

    lowerIas.rebuild(cuStream, instanceBuffer, iasMem, asBuildScratchMem);

    OptixTraversableHandle travHandle = topIas.rebuild(cuStream, topInstanceBuffer, topIasMem, asBuildScratchMem);

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
    plp.camera.position = make_float3(0, 0, 6.0);
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

    topInstanceBuffer.finalize();
    topIasMem.finalize();
    topIas.destroy();

    topInstD.destroy();
    topInstC.destroy();
    topInstB.destroy();
    topInstA.destroy();

    instanceBuffer.finalize();
    iasMem.finalize();
    lowerIas.destroy();

    for (int i = objectInsts.size() - 1; i >= 0; --i) {
        objectInsts[i].destroy();
        objectTransforms[i].deviceMem->finalize();
        delete objectTransforms[i].deviceMem;
        objectTransforms[i].optixTransform.destroy();
    }
    areaLightInst.destroy();
    roomInst.destroy();

    cube.finalize();
    bunny.finalize();
    areaLight.finalize();
    room.finalize();

    scene.destroy();

    mat0.destroy();



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
