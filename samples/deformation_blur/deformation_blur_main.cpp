/*

JP: このサンプルはデフォーメーション(変形)ブラーを扱うGASを構築する方法を示します。
    複数のモーションステップに対応する頂点(もしくはAABB)バッファーをGeometryInstanceに設定し
    GASに適切なモーション設定を行うことでデフォーメーションブラーに対応するGASを構築できます。
EN: This sample shows how to build a GAS to handle deformation blur.
    Set a vertex (or AABB) buffer to each of multiple motion steps of GeometryInstance and
    set appropriate motion configuration to a GAS to build a GAS capable of deformation blur.
*/

#include "deformation_blur_shared.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"

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
    //     カスタムプリミティブとの衝突判定を使うためプリミティブ種別のフラグを適切に設定する必要がある。
    // EN: This sample uses two-level AS (single-level instancing).
    //     Appropriately setting primitive type flags is required since this sample uses custom primitive intersection.
    pipeline.setPipelineOptions(3, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                true, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "deformation_blur/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup rayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("raygen"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));

    // JP: このグループはレイと三角形の交叉判定用なのでカスタムのIntersectionプログラムは不要。
    // EN: This group is for ray-triangle intersection, so we don't need custom intersection program.
    optixu::ProgramGroup hitProgramGroupForTriangles = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        emptyModule, nullptr);

    // JP: このヒットグループはレイと球の交叉判定用なのでカスタムのIntersectionプログラムを渡す。
    // EN: This is for ray-sphere intersection, so pass a custom intersection program.
    optixu::ProgramGroup hitProgramGroupForSpheres = pipeline.createHitProgramGroup(
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
            cudau::TypedBuffer<Shared::Triangle> triangleBuffer;
        };
        struct CustomPrimitives {
            std::vector<cudau::TypedBuffer<AABB>> aabbBuffers;
            std::vector<cudau::TypedBuffer<Shared::SphereParameter>> paramBuffers;
        };
        std::variant<TriangleMesh, CustomPrimitives> shape;
        optixu::GeometryInstance optixGeomInst;
        optixu::GeometryAccelerationStructure optixGas;
        cudau::Buffer gasMem;
        size_t compactedSize;

        void finalize() {
            gasMem.finalize();
            optixGas.destroy();
            if (std::holds_alternative<TriangleMesh>(shape)) {
                auto &triMesh = std::get<TriangleMesh>(shape);
                triMesh.triangleBuffer.finalize();
                for (int i = triMesh.vertexBuffers.size() - 1; i >= 0; --i)
                    triMesh.vertexBuffers[i].finalize();
            }
            else {
                auto &customPrims = std::get<CustomPrimitives>(shape);
                for (int i = customPrims.aabbBuffers.size() - 1; i >= 0; --i) {
                    customPrims.aabbBuffers[i].finalize();
                    customPrims.paramBuffers[i].finalize();
                }
            }
            optixGeomInst.destroy();
        }
    };

    Geometry bunny;
    {
        std::vector<obj::Vertex> objVertices;
        std::vector<obj::MaterialGroup> objMatGroups;
        obj::load("../../data/stanford_bunny_309_faces.obj", &objVertices, &objMatGroups, nullptr);

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
        shape.triangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles);

        Shared::GeometryData geomData = {};
        for (int i = 0; i < numMotionSteps; ++i)
            geomData.vertexBuffers[i] = shape.vertexBuffers[i].getDevicePointer();
        geomData.triangleBuffer = shape.triangleBuffer.getDevicePointer();

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

    Geometry spheres;
    {
        spheres.shape = Geometry::CustomPrimitives();
        auto &shape = std::get<Geometry::CustomPrimitives>(spheres.shape);

        constexpr uint32_t numPrimitives = 25;
        std::vector<AABB> aabbs0(numPrimitives);
        std::vector<AABB> aabbs1(numPrimitives);
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

            Shared::SphereParameter &param1 = sphereParams1[i];
            param1 = param0;
            param1.center += 0.4f * float3(u01(rng) - 0.5f,
                                           u01(rng) - 0.5f,
                                           u01(rng) - 0.5f);
            param1.radius *= 0.5f + 1.0f * u01(rng);

            AABB &aabb0 = aabbs0[i];
            aabb0 = AABB();
            aabb0.unify(param0.center - float3(param0.radius));
            aabb0.unify(param0.center + float3(param0.radius));

            AABB &aabb1 = aabbs1[i];
            aabb1 = AABB();
            aabb1.unify(param1.center - float3(param1.radius));
            aabb1.unify(param1.center + float3(param1.radius));
        }

        // JP: AABBバッファーを2ステップ分作る。
        // EN: Create AABB buffer for two steps.
        uint32_t numMotionSteps = 2;
        shape.aabbBuffers.resize(numMotionSteps);
        shape.aabbBuffers[0].initialize(cuContext, cudau::BufferType::Device, aabbs0);
        shape.aabbBuffers[1].initialize(cuContext, cudau::BufferType::Device, aabbs1);
        shape.paramBuffers.resize(numMotionSteps);
        shape.paramBuffers[0].initialize(cuContext, cudau::BufferType::Device, sphereParams0);
        shape.paramBuffers[1].initialize(cuContext, cudau::BufferType::Device, sphereParams1);

        Shared::GeometryData geomData = {};
        for (int i = 0; i < numMotionSteps; ++i) {
            geomData.aabbBuffers[i] = shape.aabbBuffers[i].getDevicePointer();
            geomData.paramBuffers[i] = shape.paramBuffers[i].getDevicePointer();
        }

        spheres.optixGeomInst = scene.createGeometryInstance(true);
        // JP: モーションステップ数を設定、各ステップに頂点バッファーを設定する。
        // EN: Set the number of motion steps then set the vertex buffer for each step.
        spheres.optixGeomInst.setNumMotionSteps(numMotionSteps);
        for (int i = 0; i < numMotionSteps; ++i)
            spheres.optixGeomInst.setCustomPrimitiveAABBBuffer(shape.aabbBuffers[i], i);
        spheres.optixGeomInst.setNumMaterials(1, optixu::BufferView());
        spheres.optixGeomInst.setMaterial(0, 0, matForSpheres);
        spheres.optixGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        spheres.optixGeomInst.setUserData(geomData);

        spheres.optixGas = scene.createGeometryAccelerationStructure(true);
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

    optixu::Instance spheresInst = scene.createInstance();
    spheresInst.setChild(spheres.optixGas);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace, false, false);
    ias.addChild(bunnyInst);
    ias.addChild(spheresInst);
    ias.prepareForBuild(&asMemReqs, &numInstances);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, numInstances);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: ASビルド用のスクラッチメモリを確保する。
    // EN: Allocate scratch memory for AS builds.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);



    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    bunny.optixGas.rebuild(cuStream, bunny.gasMem, asBuildScratchMem);
    spheres.optixGas.rebuild(cuStream, spheres.gasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     ここではモーションがあることが"動的"を意味しない。頻繁にASのリビルドが必要なものを"動的"、そうでないものを"静的"とする。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     The existence of motion does not mean "dynamic" here.
    //     Call things as "dynamic" for which we often need to rebuild the AS otherwise call them as "static".
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        optixu::GeometryAccelerationStructure gas;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { bunny.optixGas, 0, 0 },
        { spheres.optixGas, 0, 0 },
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

    accumBuffer.map();
    std::vector<uint32_t> imageData(renderTargetSizeX * renderTargetSizeY);
    for (int y = 0; y < renderTargetSizeY; ++y) {
        for (int x = 0; x < renderTargetSizeX; ++x) {
            const float4 &srcPix = accumBuffer(x, y);
            uint32_t &dstPix = imageData[y * renderTargetSizeX + x];
            dstPix = (std::min<uint32_t>(255, 255 * srcPix.x) <<  0) |
                     (std::min<uint32_t>(255, 255 * srcPix.y) <<  8) |
                     (std::min<uint32_t>(255, 255 * srcPix.z) << 16) |
                     (std::min<uint32_t>(255, 255 * srcPix.w) << 24);
        }
    }
    accumBuffer.unmap();

    stbi_write_bmp("output.bmp", renderTargetSizeX, renderTargetSizeY, 4, imageData.data());



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
    bunnyInst.destroy();

    spheres.finalize();
    bunny.finalize();

    scene.destroy();

    matForSpheres.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

    hitProgramGroupForSpheres.destroy();
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
