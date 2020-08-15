#include "denoiser_shared.h"

#include "../common/dds_loader.h"
#define STB_IMAGE_IMPLEMENTATION
#include "../../ext/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../../ext/stb_image_write.h"
#include "../../ext/tiny_obj_loader.h"

int32_t mainFunc(int32_t argc, const char* argv[]) {
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
    // EN: This sample uses two-level AS (single-level instancing).
    pipeline.setPipelineOptions(8, 2, "plp", sizeof(Shared::PipelineLaunchParameters),
                                false, OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
                                OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                                OPTIX_EXCEPTION_FLAG_DEBUG,
                                OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);

    const std::string ptx = readTxtFile(getExecutableDirectory() / "denoiser/ptxes/optix_kernels.ptx");
    optixu::Module moduleOptiX = pipeline.createModuleFromPTXString(
        ptx, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT,
        DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    optixu::Module emptyModule;

    optixu::ProgramGroup pathTracingRayGenProgram = pipeline.createRayGenProgram(moduleOptiX, RT_RG_NAME_STR("pathTracing"));
    //optixu::ProgramGroup exceptionProgram = pipeline.createExceptionProgram(moduleOptiX, "__exception__print");
    optixu::ProgramGroup missProgram = pipeline.createMissProgram(moduleOptiX, RT_MS_NAME_STR("miss"));
    optixu::ProgramGroup emptyMissProgram = pipeline.createMissProgram(emptyModule, nullptr);

    optixu::ProgramGroup shadingHitProgramGroup = pipeline.createHitProgramGroup(
        moduleOptiX, RT_CH_NAME_STR("shading"),
        emptyModule, nullptr,
        emptyModule, nullptr);
    optixu::ProgramGroup visibilityHitProgramGroup = pipeline.createHitProgramGroup(
        emptyModule, nullptr,
        moduleOptiX, RT_AH_NAME_STR("visibility"),
        emptyModule, nullptr);

    pipeline.setMaxTraceDepth(2);
    pipeline.link(DEBUG_SELECT(OPTIX_COMPILE_DEBUG_LEVEL_FULL, OPTIX_COMPILE_DEBUG_LEVEL_NONE));

    pipeline.setRayGenerationProgram(pathTracingRayGenProgram);
    // If an exception program is not set but exception flags are set, the default exception program will by provided by OptiX.
    //pipeline.setExceptionProgram(exceptionProgram);
    pipeline.setNumMissRayTypes(Shared::NumRayTypes);
    pipeline.setMissProgram(Shared::RayType_Search, missProgram);
    pipeline.setMissProgram(Shared::RayType_Visibility, emptyMissProgram);

    // END: Settings for OptiX context and pipeline.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: マテリアルのセットアップ。
    // EN: Setup materials.

    cudau::TypedBuffer<Shared::MaterialData> materialDataBuffer;
    materialDataBuffer.initialize(cuContext, cudau::BufferType::Device, 128);
    uint32_t materialID = 0;

    Shared::MaterialData* matData = materialDataBuffer.map();

//#define USE_BLOCK_COMPRESSED_TEXTURE

    uint32_t matCeilingIndex = materialID++;
    optixu::Material matCeiling = optixContext.createMaterial();
    matCeiling.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matCeiling.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matCeiling.setUserData(matCeilingIndex);
    Shared::MaterialData matCeilingData;
    matCeilingData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    matData[matCeilingIndex] = matCeilingData;

    uint32_t matFarSideWallIndex = materialID++;
    optixu::Material matFarSideWall = optixContext.createMaterial();
    matFarSideWall.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matFarSideWall.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matFarSideWall.setUserData(matFarSideWallIndex);
    Shared::MaterialData matFarSideWallData;
    //matFarSideWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.75), sRGB_degamma_s(0.75));
    cudau::Array arrayFarSideWall;
    {
        cudau::TextureSampler texSampler;
        texSampler.setFilterMode(cudau::TextureFilterMode::Linear,
                                 cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
            int32_t width, height, mipCount;
            size_t* sizes;
            dds::Format format;
            uint8_t** ddsData = dds::load("../../data/TexturesCom_FabricPlain0077_1_seamless_S.DDS", &width, &height, &mipCount, &sizes, &format);

            arrayCheckerBoard.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                                           cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                           width, height, 1/*mipCount*/);
            for (int i = 0; i < arrayCheckerBoard.getNumMipmapLevels(); ++i)
                arrayCheckerBoard.transfer<uint8_t>(ddsData[i], sizes[i], i);

            dds::free(ddsData, mipCount, sizes);
#else
            int32_t width, height, n;
            uint8_t* linearImageData = stbi_load("../../data/TexturesCom_FabricPlain0077_1_seamless_S.jpg", &width, &height, &n, 4);
            arrayFarSideWall.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                                    cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                    width, height, 1);
            arrayFarSideWall.transfer<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
#endif
        }
        matFarSideWallData.texture = texSampler.createTextureObject(arrayFarSideWall);
    }
    matData[matFarSideWallIndex] = matFarSideWallData;

    uint32_t matLeftWallIndex = materialID++;
    optixu::Material matLeftWall = optixContext.createMaterial();
    matLeftWall.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matLeftWall.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matLeftWall.setUserData(matLeftWallIndex);
    Shared::MaterialData matLeftWallData;
    matLeftWallData.albedo = make_float3(sRGB_degamma_s(0.75), sRGB_degamma_s(0.25), sRGB_degamma_s(0.25));
    matData[matLeftWallIndex] = matLeftWallData;

    uint32_t matRightWallIndex = materialID++;
    optixu::Material matRightWall = optixContext.createMaterial();
    matRightWall.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matRightWall.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matRightWall.setUserData(matRightWallIndex);
    Shared::MaterialData matRightWallData;
    matRightWallData.albedo = make_float3(sRGB_degamma_s(0.25), sRGB_degamma_s(0.25), sRGB_degamma_s(0.75));
    matData[matRightWallIndex] = matRightWallData;

    uint32_t matFloorIndex = materialID++;
    optixu::Material matFloor = optixContext.createMaterial();
    matFloor.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matFloor.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matFloor.setUserData(matFloorIndex);
    Shared::MaterialData matFloorData;
    cudau::Array arrayFloor;
    {
        cudau::TextureSampler texSampler;
        texSampler.setFilterMode(cudau::TextureFilterMode::Linear,
                                 cudau::TextureFilterMode::Linear);
        texSampler.setIndexingMode(cudau::TextureIndexingMode::NormalizedCoordinates);
        texSampler.setReadMode(cudau::TextureReadMode::NormalizedFloat_sRGB);

        {
#if defined(USE_BLOCK_COMPRESSED_TEXTURE)
            int32_t width, height, mipCount;
            size_t* sizes;
            dds::Format format;
            uint8_t** ddsData = dds::load("../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.DDS", &width, &height, &mipCount, &sizes, &format);

            arrayCheckerBoard.initialize2D(cuContext, cudau::ArrayElementType::BC1_UNorm, 1,
                                           cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                           width, height, 1/*mipCount*/);
            for (int i = 0; i < arrayCheckerBoard.getNumMipmapLevels(); ++i)
                arrayCheckerBoard.transfer<uint8_t>(ddsData[i], sizes[i], i);

            dds::free(ddsData, mipCount, sizes);
#else
            int32_t width, height, n;
            uint8_t* linearImageData = stbi_load("../../data/TexturesCom_FloorsCheckerboard0017_1_seamless_S.jpg", &width, &height, &n, 4);
            arrayFloor.initialize2D(cuContext, cudau::ArrayElementType::UInt8, 4,
                                           cudau::ArraySurface::Disable, cudau::ArrayTextureGather::Disable,
                                           width, height, 1);
            arrayFloor.transfer<uint8_t>(linearImageData, width * height * 4);
            stbi_image_free(linearImageData);
#endif
        }
        matFloorData.texture = texSampler.createTextureObject(arrayFloor);
    }
    matData[matFloorIndex] = matFloorData;

    uint32_t matAreaLightIndex = materialID++;
    optixu::Material matAreaLight = optixContext.createMaterial();
    matAreaLight.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matAreaLight.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matAreaLight.setUserData(matAreaLightIndex);
    Shared::MaterialData matAreaLightData;
    matAreaLightData.albedo = make_float3(sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f), sRGB_degamma_s(0.9f));
    matData[matAreaLightIndex] = matAreaLightData;

    uint32_t matBunnyIndex = materialID++;
    optixu::Material matBunny = optixContext.createMaterial();
    matBunny.setHitGroup(Shared::RayType_Search, shadingHitProgramGroup);
    matBunny.setHitGroup(Shared::RayType_Visibility, visibilityHitProgramGroup);
    matBunny.setUserData(matBunnyIndex);
    Shared::MaterialData matBunnyData;
    matBunnyData.albedo = make_float3(sRGB_degamma_s(0.25f), sRGB_degamma_s(0.75f), sRGB_degamma_s(0.25f));
    matData[matBunnyIndex] = matBunnyData;

    materialDataBuffer.unmap();

    // END: Setup materials.
    // ----------------------------------------------------------------



    // ----------------------------------------------------------------
    // JP: シーンのセットアップ。
    // EN: Setup a scene.

    optixu::Scene scene = optixContext.createScene();

    cudau::TypedBuffer<Shared::GeometryData> geomDataBuffer;
    geomDataBuffer.initialize(cuContext, cudau::BufferType::Device, 3);
    Shared::GeometryData* geomData = geomDataBuffer.map();

    uint32_t geomInstIndex = 0;

    optixu::GeometryInstance geomInstRoom = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferRoom;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferRoom;
    cudau::TypedBuffer<uint8_t> matIndexBufferRoom;
    {
        Shared::Vertex vertices[] = {
            // floor
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 1) },
            { make_float3(1.0f, -1.0f, 1.0f), make_float3(0, 1, 0), make_float2(1, 1) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 1, 0), make_float2(1, 0) },
            // far side wall
            { make_float3(-1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 0) },
            { make_float3(-1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(0, 2) },
            { make_float3(1.0f, 1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 2) },
            { make_float3(1.0f, -1.0f, -1.0f), make_float3(0, 0, 1), make_float2(2, 0) },
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
            // far side wall
            { 4, 5, 6 }, { 4, 6, 7 },
            // ceiling
            { 8, 11, 10 }, { 8, 10, 9 },
            // left wall
            { 15, 12, 13 }, { 15, 13, 14 },
            // right wall
            { 16, 19, 18 }, { 16, 18, 17 }
        };

        uint8_t matIndices[] = {
            0, 0,
            1, 1,
            2, 2,
            3, 3,
            4, 4,
        };

        vertexBufferRoom.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferRoom.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));
        matIndexBufferRoom.initialize(cuContext, cudau::BufferType::Device, matIndices, lengthof(matIndices));

        geomInstRoom.setVertexBuffer(&vertexBufferRoom);
        geomInstRoom.setTriangleBuffer(&triangleBufferRoom);
        geomInstRoom.setNumMaterials(5, &matIndexBufferRoom, sizeof(uint8_t));
        geomInstRoom.setMaterial(0, 0, matFloor);
        geomInstRoom.setMaterial(0, 1, matFarSideWall);
        geomInstRoom.setMaterial(0, 2, matCeiling);
        geomInstRoom.setMaterial(0, 3, matLeftWall);
        geomInstRoom.setMaterial(0, 4, matRightWall);
        geomInstRoom.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstRoom.setGeometryFlags(1, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstRoom.setGeometryFlags(2, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstRoom.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferRoom.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferRoom.getDevicePointer();

        ++geomInstIndex;
    }

    optixu::GeometryInstance geomInstAreaLight = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferAreaLight;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferAreaLight;
    uint32_t lightGeomInstIndex;
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

        vertexBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        triangleBufferAreaLight.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        geomInstAreaLight.setVertexBuffer(&vertexBufferAreaLight);
        geomInstAreaLight.setTriangleBuffer(&triangleBufferAreaLight);
        geomInstAreaLight.setNumMaterials(1, nullptr);
        geomInstAreaLight.setMaterial(0, 0, matAreaLight);
        geomInstAreaLight.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstAreaLight.setUserData(geomInstIndex);
        lightGeomInstIndex = geomInstIndex;

        geomData[geomInstIndex].vertexBuffer = vertexBufferAreaLight.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferAreaLight.getDevicePointer();

        ++geomInstIndex;
    }

    optixu::GeometryInstance geomInstBunny = scene.createGeometryInstance();
    cudau::TypedBuffer<Shared::Vertex> vertexBufferBunny;
    cudau::TypedBuffer<Shared::Triangle> triangleBufferBunny;
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn;
        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "../../data/stanford_bunny_309_faces.obj");

        // Record unified unique vertices.
        std::map<std::tuple<int32_t, int32_t>, Shared::Vertex> unifiedVertexMap;
        for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = shapes[sIdx];
            size_t idxOffset = 0;
            for (int fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                for (int vIdx = 0; vIdx < numFaceVertices; ++vIdx) {
                    tinyobj::index_t idx = shape.mesh.indices[idxOffset + vIdx];
                    auto key = std::make_tuple(idx.vertex_index, idx.normal_index);
                    unifiedVertexMap[key] = Shared::Vertex{
                        make_float3(attrib.vertices[3 * idx.vertex_index + 0],
                                    attrib.vertices[3 * idx.vertex_index + 1],
                                    attrib.vertices[3 * idx.vertex_index + 2]),
                        make_float3(0, 0, 0),
                        make_float2(0, 0)
                    };
                }

                idxOffset += numFaceVertices;
            }
        }

        // Assign a vertex index to each of unified unique unifiedVertexMap.
        std::map<std::tuple<int32_t, int32_t>, uint32_t> vertexIndices;
        std::vector<Shared::Vertex> vertices(unifiedVertexMap.size());
        uint32_t vertexIndex = 0;
        for (const auto &kv : unifiedVertexMap) {
            vertices[vertexIndex] = kv.second;
            vertexIndices[kv.first] = vertexIndex++;
        }
        unifiedVertexMap.clear();

        // Calculate triangle index buffer.
        std::vector<Shared::Triangle> triangles;
        for (int sIdx = 0; sIdx < shapes.size(); ++sIdx) {
            const tinyobj::shape_t &shape = shapes[sIdx];
            size_t idxOffset = 0;
            for (int fIdx = 0; fIdx < shape.mesh.num_face_vertices.size(); ++fIdx) {
                uint32_t numFaceVertices = shape.mesh.num_face_vertices[fIdx];
                if (numFaceVertices != 3) {
                    idxOffset += numFaceVertices;
                    continue;
                }

                tinyobj::index_t idx0 = shape.mesh.indices[idxOffset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[idxOffset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[idxOffset + 2];
                auto key0 = std::make_tuple(idx0.vertex_index, idx0.normal_index);
                auto key1 = std::make_tuple(idx1.vertex_index, idx1.normal_index);
                auto key2 = std::make_tuple(idx2.vertex_index, idx2.normal_index);

                triangles.push_back(Shared::Triangle{
                    vertexIndices.at(key0),
                    vertexIndices.at(key1),
                    vertexIndices.at(key2) });

                idxOffset += numFaceVertices;
            }
        }
        vertexIndices.clear();

        for (int tIdx = 0; tIdx < triangles.size(); ++tIdx) {
            const Shared::Triangle &tri = triangles[tIdx];
            Shared::Vertex &v0 = vertices[tri.index0];
            Shared::Vertex &v1 = vertices[tri.index1];
            Shared::Vertex &v2 = vertices[tri.index2];
            float3 gn = normalize(cross(v1.position - v0.position, v2.position - v0.position));
            v0.normal += gn;
            v1.normal += gn;
            v2.normal += gn;
        }
        for (int vIdx = 0; vIdx < vertices.size(); ++vIdx) {
            Shared::Vertex &v = vertices[vIdx];
            v.normal = normalize(v.normal);
        }

        vertexBufferBunny.initialize(cuContext, cudau::BufferType::Device, vertices);
        triangleBufferBunny.initialize(cuContext, cudau::BufferType::Device, triangles);

        geomInstBunny.setVertexBuffer(&vertexBufferBunny);
        geomInstBunny.setTriangleBuffer(&triangleBufferBunny);
        geomInstBunny.setNumMaterials(1, nullptr);
        geomInstBunny.setMaterial(0, 0, matBunny);
        geomInstBunny.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        geomInstBunny.setUserData(geomInstIndex);

        geomData[geomInstIndex].vertexBuffer = vertexBufferBunny.getDevicePointer();
        geomData[geomInstIndex].triangleBuffer = triangleBufferBunny.getDevicePointer();

        ++geomInstIndex;
    }

    geomDataBuffer.unmap();



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure gasRoom = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasRoomMem;
    cudau::Buffer gasRoomCompactedMem;
    gasRoom.setConfiguration(true, false, true, false);
    gasRoom.setNumMaterialSets(1);
    gasRoom.setNumRayTypes(0, Shared::NumRayTypes);
    gasRoom.addChild(geomInstRoom);
    gasRoom.prepareForBuild(&asMemReqs);
    gasRoomMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure gasAreaLight = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasAreaLightMem;
    cudau::Buffer gasAreaLightCompactedMem;
    gasAreaLight.setConfiguration(true, false, true, false);
    gasAreaLight.setNumMaterialSets(1);
    gasAreaLight.setNumRayTypes(0, Shared::NumRayTypes);
    gasAreaLight.addChild(geomInstAreaLight);
    gasAreaLight.prepareForBuild(&asMemReqs);
    gasAreaLightMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure gasBunny = scene.createGeometryAccelerationStructure();
    cudau::Buffer gasBunnyMem;
    cudau::Buffer gasBunnyCompactedMem;
    gasBunny.setConfiguration(true, false, true, false);
    gasBunny.setNumMaterialSets(1);
    gasBunny.setNumRayTypes(0, Shared::NumRayTypes);
    gasBunny.addChild(geomInstBunny);
    gasBunny.prepareForBuild(&asMemReqs);
    gasBunnyMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    // JP: Geometry Acceleration Structureをビルドする。
    // EN: Build geometry acceleration structures.
    asBuildScratchMem.initialize(cuContext, cudau::BufferType::Device, maxSizeOfScratchBuffer, 1);
    gasRoom.rebuild(cuStream, gasRoomMem, asBuildScratchMem);
    gasAreaLight.rebuild(cuStream, gasAreaLightMem, asBuildScratchMem);
    gasBunny.rebuild(cuStream, gasBunnyMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    // EN: Perform compaction for static meshes.
    size_t gasBoxCompactedSize;
    gasRoom.prepareForCompact(&gasBoxCompactedSize);
    gasRoomCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasBoxCompactedSize, 1);
    size_t gasAreaLightCompactedSize;
    gasAreaLight.prepareForCompact(&gasAreaLightCompactedSize);
    gasAreaLightCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasAreaLightCompactedSize, 1);
    size_t gasBunnyCompactedSize;
    gasBunny.prepareForCompact(&gasBunnyCompactedSize);
    gasBunnyCompactedMem.initialize(cuContext, cudau::BufferType::Device, gasBunnyCompactedSize, 1);

    gasRoom.compact(cuStream, gasRoomCompactedMem);
    gasRoom.removeUncompacted();
    gasAreaLight.compact(cuStream, gasAreaLightCompactedMem);
    gasAreaLight.removeUncompacted();
    gasBunny.compact(cuStream, gasBunnyCompactedMem);
    gasBunny.removeUncompacted();



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance instRoom = scene.createInstance();
    instRoom.setChild(gasRoom);

    float instAreaLightTr[] = {
        1, 0, 0, 0,
        0, 1, 0, 0.9f,
        0, 0, 1, 0
    };
    optixu::Instance instAreaLight = scene.createInstance();
    instAreaLight.setChild(gasAreaLight);
    instAreaLight.setTransform(instAreaLightTr);

    std::vector<optixu::Instance> instsBunny;
    const float GoldenRatio = (1 + std::sqrt(5.0f)) / 2;
    const float GoldenAngle = 2 * M_PI / (GoldenRatio * GoldenRatio);
    constexpr uint32_t NumBunnies = 100;
    for (int i = 0; i < NumBunnies; ++i) {
        float t = static_cast<float>(i) / (NumBunnies - 1);
        float r = 0.9f * std::pow(t, 0.5f);
        float x = r * std::cos(GoldenAngle * i);
        float z = r * std::sin(GoldenAngle * i);

        float tt = std::pow(t, 0.25f);
        float scale = (1 - tt) * 0.003f + tt * 0.0006f;
        float instBunnyTr[] = {
            scale, 0, 0, x,
            0, scale, 0, -1 + (1 - tt),
            0, 0, scale, z
        };
        optixu::Instance instBunny = scene.createInstance();
        instBunny.setChild(gasBunny);
        instBunny.setTransform(instBunnyTr);
        instsBunny.push_back(instBunny);
    }



    // JP: IAS作成時には各インスタンスのTraversable HandleとShader Binding Table中のオフセットが
    //     確定している必要がある。
    // EN: Traversable handle and offset in the shader binding table must be fixed for each instance
    //     when creating an IAS.
    cudau::Buffer shaderBindingTable;
    size_t sbtSize;
    scene.generateShaderBindingTableLayout(&sbtSize);
    shaderBindingTable.initialize(cuContext, cudau::BufferType::Device, sbtSize, 1);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    uint32_t numInstances;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(true, false, false);
    ias.addChild(instRoom);
    ias.addChild(instAreaLight);
    for (int i = 0; i < instsBunny.size(); ++i)
        ias.addChild(instsBunny[i]);
    ias.prepareForBuild(&asMemReqs, &numInstances);
    iasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    instanceBuffer.initialize(cuContext, cudau::BufferType::Device, numInstances);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    if (maxSizeOfScratchBuffer > asBuildScratchMem.sizeInBytes())
        asBuildScratchMem.resize(maxSizeOfScratchBuffer, 1);

    OptixTraversableHandle travHandle = ias.rebuild(cuStream, instanceBuffer, iasMem, asBuildScratchMem);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    // END: Setup a scene.
    // ----------------------------------------------------------------



    constexpr uint32_t renderTargetSizeX = 1024;
    constexpr uint32_t renderTargetSizeY = 1024;
    cudau::Array colorAccumBuffer;
    cudau::Array albedoAccumBuffer;
    cudau::Array normalAccumBuffer;
    colorAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                  cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                  renderTargetSizeX, renderTargetSizeY, 1);
    albedoAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    normalAccumBuffer.initialize2D(cuContext, cudau::ArrayElementType::Float32, 4,
                                   cudau::ArraySurface::Enable, cudau::ArrayTextureGather::Disable,
                                   renderTargetSizeX, renderTargetSizeY, 1);
    cudau::TypedBuffer<float4> linearColorBuffer;
    cudau::TypedBuffer<float4> linearAlbedoBuffer;
    cudau::TypedBuffer<float4> linearNormalBuffer;
    cudau::TypedBuffer<float4> linearOutputBuffer;
    linearColorBuffer.initialize(cuContext, cudau::BufferType::Device,
                                 renderTargetSizeX * renderTargetSizeY);
    linearAlbedoBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);
    linearNormalBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);
    linearOutputBuffer.initialize(cuContext, cudau::BufferType::Device,
                                  renderTargetSizeX * renderTargetSizeY);

    optixu::HostBlockBuffer2D<Shared::PCG32RNG, 1> rngBuffer;
    rngBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);
    {
        std::mt19937_64 rng(591842031321323413);

        rngBuffer.map();
        for (int y = 0; y < renderTargetSizeY; ++y)
            for (int x = 0; x < renderTargetSizeX; ++x)
                rngBuffer(x, y).setState(rng());
        rngBuffer.unmap();
    };



    // JP: デノイザーのセットアップ。
    // EN: Setup a denoiser.
    constexpr bool useTiledDenoising = false; // Change this to true to use tiled denoising.
    constexpr uint32_t tileWidth = useTiledDenoising ? 32 : 0;
    constexpr uint32_t tileHeight = useTiledDenoising ? 32 : 0;
    optixu::Denoiser denoiser = optixContext.createDenoiser(OPTIX_DENOISER_INPUT_RGB_ALBEDO_NORMAL);
    denoiser.setModel(OPTIX_DENOISER_MODEL_KIND_HDR, nullptr, 0);
    size_t stateSize;
    size_t scratchSize;
    size_t scratchSizeForComputeIntensity;
    uint32_t numTasks;
    denoiser.prepare(renderTargetSizeX, renderTargetSizeY, tileWidth, tileHeight,
                     &stateSize, &scratchSize, &scratchSizeForComputeIntensity,
                     &numTasks);;
    hpprintf("Denoiser State Buffer: %llu bytes\n", stateSize);
    hpprintf("Denoiser Scratch Buffer: %llu bytes\n", scratchSize);
    hpprintf("Compute Intensity Scratch Buffer: %llu bytes\n", scratchSizeForComputeIntensity);
    cudau::Buffer denoiserStateBuffer;
    cudau::Buffer denoiserScratchBuffer;
    denoiserStateBuffer.initialize(cuContext, cudau::BufferType::Device, stateSize, 1);
    denoiserScratchBuffer.initialize(cuContext, cudau::BufferType::Device,
                                     std::max(scratchSize, scratchSizeForComputeIntensity), 1);

    std::vector<optixu::DenoisingTask> denoisingTasks(numTasks);
    denoiser.getTasks(denoisingTasks.data());

    denoiser.setLayers(&linearColorBuffer, &linearAlbedoBuffer, &linearNormalBuffer, &linearOutputBuffer,
                       OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4, OPTIX_PIXEL_FORMAT_FLOAT4);
    denoiser.setupState(cuStream, denoiserStateBuffer, denoiserScratchBuffer);

    // JP: デノイザーは入出力にリニアなバッファーを必要とするため結果をコピーする必要がある。
    // EN: Denoiser requires linear buffers as input/output, so we need to copy the results.
    CUmodule moduleCopyBuffers;
    CUDADRV_CHECK(cuModuleLoad(&moduleCopyBuffers, (getExecutableDirectory() / "denoiser/ptxes/copy_buffers.ptx").string().c_str()));
    cudau::Kernel kernelCopyBuffers(moduleCopyBuffers, "copyBuffers", cudau::dim3(8, 8), 0);

    CUdeviceptr hdrIntensity;
    CUDADRV_CHECK(cuMemAlloc(&hdrIntensity, sizeof(float)));



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.materialData = materialDataBuffer.getDevicePointer();
    plp.geomInstData = geomDataBuffer.getDevicePointer();
    plp.imageSize = int2(renderTargetSizeX, renderTargetSizeY);
    plp.rngBuffer = rngBuffer.getBlockBuffer2D();
    plp.colorAccumBuffer = colorAccumBuffer.getSurfaceObject(0);
    plp.albedoAccumBuffer = albedoAccumBuffer.getSurfaceObject(0);
    plp.normalAccumBuffer = normalAccumBuffer.getSurfaceObject(0);
    plp.camera.fovY = 50 * M_PI / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 0, 3.5);
    plp.camera.orientation = rotateY3x3(M_PI);
    plp.lightGeomInstIndex = lightGeomInstIndex;

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(&shaderBindingTable);

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    cudau::Timer timerRender;
    cudau::Timer timerDenoise;
    timerRender.initialize(cuContext);
    timerDenoise.initialize(cuContext);
    
    // JP: レンダリング
    // EN: Render
    constexpr uint32_t numSamples = 8;
    timerRender.start(cuStream);
    for (int frameIndex = 0; frameIndex < numSamples; ++frameIndex) {
        plp.numAccumFrames = frameIndex;
        CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
        pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    }

    // JP: 結果をリニアバッファーにコピーする。(法線の正規化も行う。)
    // EN: Copy the results to the linear buffers (and normalize normals).
    cudau::dim3 dimCopyBuffers = kernelCopyBuffers.calcGridDim(renderTargetSizeX, renderTargetSizeY);
    kernelCopyBuffers(cuStream, dimCopyBuffers,
                      colorAccumBuffer.getSurfaceObject(0),
                      albedoAccumBuffer.getSurfaceObject(0),
                      normalAccumBuffer.getSurfaceObject(0),
                      linearColorBuffer.getDevicePointer(),
                      linearAlbedoBuffer.getDevicePointer(),
                      linearNormalBuffer.getDevicePointer(),
                      uint2(renderTargetSizeX, renderTargetSizeY));
    timerRender.stop(cuStream);

    // JP: パストレーシング結果のデノイズ。
    //     毎フレーム呼ぶ必要があるのはcomputeIntensity()とinvoke()。
    //     computeIntensity()は自作することもできる。
    //     サイズが足りていればcomputeIntensity()のスクラッチバッファーとしてデノイザーのものが再利用できる。
    // EN: Denoise the path tracing the result.
    //     computeIntensity() and invoke() should be calld every frame.
    //     You can also create a custom computeIntensity().
    //     Reusing the scratch buffer for denoising for computeIntensity() is possible if its size is enough.
    timerDenoise.start(cuStream);
    denoiser.computeIntensity(cuStream, denoiserScratchBuffer, hdrIntensity);
    for (int i = 0; i < denoisingTasks.size(); ++i)
        denoiser.invoke(cuStream, false, hdrIntensity, 0.0f, denoisingTasks[i]);
    timerDenoise.stop(cuStream);

    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    hpprintf("Render %u [spp]: %.3f[ms]\n", numSamples, timerRender.report());
    hpprintf("Denoise: %.3f[ms]\n", timerDenoise.report());

    timerDenoise.finalize();
    timerRender.finalize();



    // JP: 結果とデノイズ用付随バッファーの画像出力。
    // EN: Output the result and buffers associated to the denoiser as images.
    auto colorPixels = colorAccumBuffer.map<float4>();
    auto albedoPixels = albedoAccumBuffer.map<float4>();
    auto normalPixels = normalAccumBuffer.map<float4>();
    auto outputPixels = linearOutputBuffer.map();
    std::vector<uint32_t> colorImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> albedoImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> normalImageData(renderTargetSizeX * renderTargetSizeY);
    std::vector<uint32_t> outputImageData(renderTargetSizeX * renderTargetSizeY);
    for (int y = 0; y < renderTargetSizeY; ++y) {
        for (int x = 0; x < renderTargetSizeX; ++x) {
            uint32_t linearIndex = renderTargetSizeX * y + x;

            float4 color = colorPixels[linearIndex];
            color.x = sRGB_gamma_s(1 - std::exp(-color.x));
            color.y = sRGB_gamma_s(1 - std::exp(-color.y));
            color.z = sRGB_gamma_s(1 - std::exp(-color.z));
            uint32_t &dstColor = colorImageData[linearIndex];
            dstColor = (std::min<uint32_t>(255, 255 * color.x) << 0) |
                       (std::min<uint32_t>(255, 255 * color.y) << 8) |
                       (std::min<uint32_t>(255, 255 * color.z) << 16) |
                       (std::min<uint32_t>(255, 255 * color.w) << 24);

            float4 albedo = albedoPixels[linearIndex];
            uint32_t &dstAlbedo = albedoImageData[linearIndex];
            dstAlbedo = (std::min<uint32_t>(255, 255 * albedo.x) << 0) |
                        (std::min<uint32_t>(255, 255 * albedo.y) << 8) |
                        (std::min<uint32_t>(255, 255 * albedo.z) << 16) |
                        (std::min<uint32_t>(255, 255 * albedo.w) << 24);

            float4 normal = normalPixels[linearIndex];
            uint32_t &dstNormal = normalImageData[linearIndex];
            dstNormal = (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.x)) << 0) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.y)) << 8) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.z)) << 16) |
                        (std::min<uint32_t>(255, 255 * (0.5f + 0.5f * normal.w)) << 24);

            float4 output = outputPixels[linearIndex];
            output.x = sRGB_gamma_s(1 - std::exp(-output.x));
            output.y = sRGB_gamma_s(1 - std::exp(-output.y));
            output.z = sRGB_gamma_s(1 - std::exp(-output.z));
            uint32_t &dstOutput = outputImageData[linearIndex];
            dstOutput = (std::min<uint32_t>(255, 255 * output.x) << 0) |
                        (std::min<uint32_t>(255, 255 * output.y) << 8) |
                        (std::min<uint32_t>(255, 255 * output.z) << 16) |
                        (std::min<uint32_t>(255, 255 * output.w) << 24);
        }
    }
    linearOutputBuffer.unmap();
    normalAccumBuffer.unmap();
    albedoAccumBuffer.unmap();
    colorAccumBuffer.unmap();

    stbi_write_bmp("color.bmp", renderTargetSizeX, renderTargetSizeY, 4, colorImageData.data());
    stbi_write_bmp("albedo.bmp", renderTargetSizeX, renderTargetSizeY, 4, albedoImageData.data());
    stbi_write_bmp("normal.bmp", renderTargetSizeX, renderTargetSizeY, 4, normalImageData.data());
    stbi_write_bmp("color_denoised.bmp", renderTargetSizeX, renderTargetSizeY, 4, outputImageData.data());



    CUDADRV_CHECK(cuMemFree(plpOnDevice));


    
    CUDADRV_CHECK(cuMemFree(hdrIntensity));

    CUDADRV_CHECK(cuModuleUnload(moduleCopyBuffers));
    
    denoiserScratchBuffer.finalize();
    denoiserStateBuffer.finalize();
    
    denoiser.destroy();
    
    rngBuffer.finalize();

    linearOutputBuffer.finalize();
    linearNormalBuffer.finalize();
    linearAlbedoBuffer.finalize();
    linearColorBuffer.finalize();

    normalAccumBuffer.finalize();
    albedoAccumBuffer.finalize();
    colorAccumBuffer.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    shaderBindingTable.finalize();

    for (int i = instsBunny.size() - 1; i >= 0; --i)
        instsBunny[i].destroy();
    instAreaLight.destroy();
    instRoom.destroy();

    gasBunnyCompactedMem.finalize();
    gasAreaLightCompactedMem.finalize();
    gasRoomCompactedMem.finalize();
    gasBunnyMem.finalize();
    gasBunny.destroy();
    gasAreaLightMem.finalize();
    gasAreaLight.destroy();
    gasRoomMem.finalize();
    gasRoom.destroy();

    triangleBufferBunny.finalize();
    vertexBufferBunny.finalize();
    geomInstBunny.destroy();
    
    triangleBufferAreaLight.finalize();
    vertexBufferAreaLight.finalize();
    geomInstAreaLight.destroy();

    matIndexBufferRoom.finalize();
    triangleBufferRoom.finalize();
    vertexBufferRoom.finalize();
    geomInstRoom.destroy();

    geomDataBuffer.finalize();

    scene.destroy();

    matBunny.destroy();
    matAreaLight.destroy();
    CUDADRV_CHECK(cuTexObjectDestroy(matFloorData.texture));
    arrayFloor.finalize();
    matFloor.destroy();
    matRightWall.destroy();
    matLeftWall.destroy();
    CUDADRV_CHECK(cuTexObjectDestroy(matFarSideWallData.texture));
    arrayFarSideWall.finalize();
    matFarSideWall.destroy();
    matCeiling.destroy();

    materialDataBuffer.finalize();

    visibilityHitProgramGroup.destroy();
    shadingHitProgramGroup.destroy();

    emptyMissProgram.destroy();
    missProgram.destroy();
    pathTracingRayGenProgram.destroy();

    moduleOptiX.destroy();

    pipeline.destroy();

    optixContext.destroy();

    CUDADRV_CHECK(cuStreamDestroy(cuStream));
    CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}

int32_t main(int32_t argc, const char* argv[]) {
    try {
        mainFunc(argc, argv);
    }
    catch (const std::exception &ex) {
        hpprintf("Error: %s\n", ex.what());
    }

    return 0;
}
