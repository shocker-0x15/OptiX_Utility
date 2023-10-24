/*

JP: このサンプルはカーブを扱う方法を示します。
    カーブとレイの交叉判定はOptiXによって内部的に扱われます。
    OptiXがサポートするカーブは以下のとおりです。
    - リニア
    - 2次Bスプライン
    - 3次Bスプライン
    - Catmull-Romスプライン
    - 3次ベジェ曲線
    - リボン(2次Bスプライン)
    カーブは頂点バッファーと頂点ごとの幅を表すバッファー、そしてインデックスバッファーから構成されます。
    リボンは加えて頂点ごとの法線ベクトルを表すバッファーを指定することができます。

EN: This sample shows how to handle curves.
    Intersection test between a ray and a curve is handled internally by OptiX.
    The curve supported by OptiX are the following:
    - Linear
    - Quadratic B-Spline
    - Cubic B-Spline
    - Catmull-Rom Spline
    - Cubic Bezier Curves
    - Ribbon (Quadratic B-Spline)
    Curves consist of a vertex buffer and a buffer for the width at each vertex and an index buffer.
    Ribbon can takes an additional buffer for the normal at each vertex.

*/

#include "curve_primitive_shared.h"

template <OptixPrimitiveType curveType>
static void generateCurves(
    std::vector<Shared::CurveVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float baseWidth);

static void generateBezierCurves(
    std::vector<Shared::CurveVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float baseWidth);

static void generateRibbons(
    std::vector<Shared::RibbonVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float width);

int32_t main(int32_t argc, const char* argv[]) try {
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
        std::max(optixu::calcSumDwords<float2>(),
                 optixu::calcSumDwords<float>()),
        "plp", sizeof(Shared::PipelineLaunchParameters),
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING,
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
        OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_FLAT_QUADRATIC_BSPLINE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CATMULLROM |
        OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BEZIER);

    const std::vector<char> optixIr =
        readBinaryFile(getExecutableDirectory() / "curve_primitive/ptxes/optix_kernels.optixir");
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

    /*
    JP: 各種カーブ用のヒットグループを作成する。
        各種カーブには三角形と同様、ビルトインのIntersection Programが使われるのでユーザーが指定する必要はない。
        カーブを含むことになるASと同じビルド設定を予め指定しておく必要がある。
    EN: Create a hit group for each of curve types.
        Each curve type uses a built-in intersection program similar to triangle,
        so the user doesn't need to specify it.
        The same build configuration as an AS having the curve is required.
    */
    constexpr OptixCurveEndcapFlags curveEndcap = OPTIX_CURVE_ENDCAP_ON;
    constexpr optixu::ASTradeoff curveASTradeOff = optixu::ASTradeoff::PreferFastTrace;
    constexpr optixu::AllowUpdate curveASUpdatable = optixu::AllowUpdate::No;
    constexpr optixu::AllowCompaction curveASCompactable = optixu::AllowCompaction::Yes;
    constexpr auto useEmbeddedVertexData = optixu::AllowRandomVertexAccess(Shared::useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForLinearCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR, OPTIX_CURVE_ENDCAP_DEFAULT,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForQuadraticCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForCubicCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForCatmullRomCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForCubicBezierCurves = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);
    optixu::HitProgramGroup hitProgramGroupForQuadraticRibbons = pipeline.createHitProgramGroupForCurveIS(
        OPTIX_PRIMITIVE_TYPE_FLAT_QUADRATIC_BSPLINE, curveEndcap,
        moduleOptiX, RT_CH_NAME_STR("closesthit"),
        emptyModule, nullptr,
        curveASTradeOff, curveASUpdatable, curveASCompactable, useEmbeddedVertexData);

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

    optixu::Material matForTriangles = optixContext.createMaterial();
    matForTriangles.setHitGroup(Shared::RayType_Primary, hitProgramGroupForTriangles);
    optixu::Material matForLinearCurves = optixContext.createMaterial();
    matForLinearCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForLinearCurves);
    optixu::Material matForQuadraticCurves = optixContext.createMaterial();
    matForQuadraticCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForQuadraticCurves);
    optixu::Material matForCubicCurves = optixContext.createMaterial();
    matForCubicCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCubicCurves);
    optixu::Material matForCatmullRomCurves = optixContext.createMaterial();
    matForCatmullRomCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCatmullRomCurves);
    optixu::Material matForCubicBezierCurves = optixContext.createMaterial();
    matForCubicBezierCurves.setHitGroup(Shared::RayType_Primary, hitProgramGroupForCubicBezierCurves);
    optixu::Material matForQuadraticRibbons = optixContext.createMaterial();
    matForQuadraticRibbons.setHitGroup(Shared::RayType_Primary, hitProgramGroupForQuadraticRibbons);

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
            { make_float3(-1.5f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(0, 0) },
            { make_float3(-1.5f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(0, 5) },
            { make_float3(1.5f, 0.0f, 1.0f), make_float3(0, 1, 0), make_float2(5, 5) },
            { make_float3(1.5f, 0.0f, -1.0f), make_float3(0, 1, 0), make_float2(5, 0) },
        };

        Shared::Triangle triangles[] = {
            { 0, 1, 2 }, { 0, 2, 3 },
        };

        floorVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices, lengthof(vertices));
        floorTriangleBuffer.initialize(cuContext, cudau::BufferType::Device, triangles, lengthof(triangles));

        Shared::GeometryData geomData = {};
        geomData.vertexBuffer = floorVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.triangleBuffer = floorTriangleBuffer.getROBuffer<enableBufferOobCheck>();

        floorGeomInst.setVertexBuffer(floorVertexBuffer);
        floorGeomInst.setTriangleBuffer(floorTriangleBuffer);
        floorGeomInst.setNumMaterials(1, optixu::BufferView());
        floorGeomInst.setMaterial(0, 0, matForTriangles);
        floorGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        floorGeomInst.setUserData(geomData);
    }

    uint32_t numX = 5;
    uint32_t numZ = 15;
    float baseWidth = 0.03f;

    // JP: カーブ用GeometryInstanceは生成時に指定する必要がある。
    // EN: GeometryInstance for curves requires to be specified at the creation.

    // Linear Segments
    optixu::GeometryInstance linearCurveGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::LinearSegments);
    cudau::TypedBuffer<Shared::CurveVertex> linearCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> linearCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves<OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR>(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            baseWidth);

        linearCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        linearCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = linearCurveVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = linearCurveSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

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
    optixu::GeometryInstance quadraticCurveGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::QuadraticBSplines);
    cudau::TypedBuffer<Shared::CurveVertex> quadraticCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> quadraticCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves<OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE>(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            baseWidth);

        quadraticCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        quadraticCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = quadraticCurveVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = quadraticCurveSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

        quadraticCurveGeomInst.setVertexBuffer(optixu::BufferView(
            quadraticCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            quadraticCurveVertexBuffer.numElements(), quadraticCurveVertexBuffer.stride()));
        quadraticCurveGeomInst.setWidthBuffer(optixu::BufferView(
            quadraticCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            quadraticCurveVertexBuffer.numElements(), quadraticCurveVertexBuffer.stride()));
        quadraticCurveGeomInst.setSegmentIndexBuffer(quadraticCurveSegmentIndexBuffer);
        quadraticCurveGeomInst.setCurveEndcapFlags(curveEndcap);
        quadraticCurveGeomInst.setMaterial(0, 0, matForQuadraticCurves);
        quadraticCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        quadraticCurveGeomInst.setUserData(geomData);
    }

    // Cubic B-Splines
    optixu::GeometryInstance cubicCurveGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::CubicBSplines);
    cudau::TypedBuffer<Shared::CurveVertex> cubicCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> cubicCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves<OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE>(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            baseWidth);

        cubicCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        cubicCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = cubicCurveVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = cubicCurveSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

        cubicCurveGeomInst.setVertexBuffer(optixu::BufferView(
            cubicCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            cubicCurveVertexBuffer.numElements(), cubicCurveVertexBuffer.stride()));
        cubicCurveGeomInst.setWidthBuffer(optixu::BufferView(
            cubicCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            cubicCurveVertexBuffer.numElements(), cubicCurveVertexBuffer.stride()));
        cubicCurveGeomInst.setSegmentIndexBuffer(cubicCurveSegmentIndexBuffer);
        cubicCurveGeomInst.setCurveEndcapFlags(curveEndcap);
        cubicCurveGeomInst.setMaterial(0, 0, matForCubicCurves);
        cubicCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        cubicCurveGeomInst.setUserData(geomData);
    }

    // Catmull-Rom Splines
    optixu::GeometryInstance catmullRomCurveGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::CatmullRomSplines);
    cudau::TypedBuffer<Shared::CurveVertex> catmullRomCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> catmullRomCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateCurves<OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM>(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            baseWidth);

        catmullRomCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        catmullRomCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = catmullRomCurveVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = catmullRomCurveSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

        catmullRomCurveGeomInst.setVertexBuffer(optixu::BufferView(
            catmullRomCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            catmullRomCurveVertexBuffer.numElements(), catmullRomCurveVertexBuffer.stride()));
        catmullRomCurveGeomInst.setWidthBuffer(optixu::BufferView(
            catmullRomCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            catmullRomCurveVertexBuffer.numElements(), catmullRomCurveVertexBuffer.stride()));
        catmullRomCurveGeomInst.setSegmentIndexBuffer(catmullRomCurveSegmentIndexBuffer);
        catmullRomCurveGeomInst.setCurveEndcapFlags(curveEndcap);
        catmullRomCurveGeomInst.setMaterial(0, 0, matForCatmullRomCurves);
        catmullRomCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        catmullRomCurveGeomInst.setUserData(geomData);
    }

    // Cubic Bezier Curves
    optixu::GeometryInstance cubicBezierCurveGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::CubicBezier);
    cudau::TypedBuffer<Shared::CurveVertex> cubicBezierCurveVertexBuffer;
    cudau::TypedBuffer<uint32_t> cubicBezierCurveSegmentIndexBuffer;
    {
        std::vector<Shared::CurveVertex> vertices;
        std::vector<uint32_t> indices;
        generateBezierCurves(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            baseWidth);

        cubicBezierCurveVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        cubicBezierCurveSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.curveVertexBuffer = cubicBezierCurveVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = cubicBezierCurveSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

        cubicBezierCurveGeomInst.setVertexBuffer(optixu::BufferView(
            cubicBezierCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, position),
            cubicBezierCurveVertexBuffer.numElements(), cubicBezierCurveVertexBuffer.stride()));
        cubicBezierCurveGeomInst.setWidthBuffer(optixu::BufferView(
            cubicBezierCurveVertexBuffer.getCUdeviceptr() + offsetof(Shared::CurveVertex, width),
            cubicBezierCurveVertexBuffer.numElements(), cubicBezierCurveVertexBuffer.stride()));
        cubicBezierCurveGeomInst.setSegmentIndexBuffer(cubicBezierCurveSegmentIndexBuffer);
        cubicBezierCurveGeomInst.setCurveEndcapFlags(curveEndcap);
        cubicBezierCurveGeomInst.setMaterial(0, 0, matForCubicBezierCurves);
        cubicBezierCurveGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        cubicBezierCurveGeomInst.setUserData(geomData);
    }

    // Flat Quadratic B-Splines
    optixu::GeometryInstance quadraticRibbonGeomInst =
        scene.createGeometryInstance(optixu::GeometryType::FlatQuadraticBSplines);
    cudau::TypedBuffer<Shared::RibbonVertex> quadraticRibbonVertexBuffer;
    cudau::TypedBuffer<uint32_t> quadraticRibbonSegmentIndexBuffer;
    {
        std::vector<Shared::RibbonVertex> vertices;
        std::vector<uint32_t> indices;
        generateRibbons(
            &vertices, &indices,
            -1.0f / 4.0f + 0.05f, 1.0f / 4.0f - 0.05f, numX,
            -1.0f + 0.05f, 1.0f - 0.05f, numZ,
            0.6f * baseWidth);

        quadraticRibbonVertexBuffer.initialize(cuContext, cudau::BufferType::Device, vertices);
        quadraticRibbonSegmentIndexBuffer.initialize(cuContext, cudau::BufferType::Device, indices);

        Shared::GeometryData geomData = {};
        geomData.ribbonVertexBuffer = quadraticRibbonVertexBuffer.getROBuffer<enableBufferOobCheck>();
        geomData.segmentIndexBuffer = quadraticRibbonSegmentIndexBuffer.getROBuffer<enableBufferOobCheck>();

        quadraticRibbonGeomInst.setVertexBuffer(optixu::BufferView(
            quadraticRibbonVertexBuffer.getCUdeviceptr() + offsetof(Shared::RibbonVertex, position),
            quadraticRibbonVertexBuffer.numElements(), quadraticRibbonVertexBuffer.stride()));
        // JP: リボンは法線を設定できる。
        // EN: Ribbon can set normals.
        quadraticRibbonGeomInst.setNormalBuffer(optixu::BufferView(
            quadraticRibbonVertexBuffer.getCUdeviceptr() + offsetof(Shared::RibbonVertex, normal),
            quadraticRibbonVertexBuffer.numElements(), quadraticRibbonVertexBuffer.stride()));
        quadraticRibbonGeomInst.setWidthBuffer(optixu::BufferView(
            quadraticRibbonVertexBuffer.getCUdeviceptr() + offsetof(Shared::RibbonVertex, width),
            quadraticRibbonVertexBuffer.numElements(), quadraticRibbonVertexBuffer.stride()));
        quadraticRibbonGeomInst.setSegmentIndexBuffer(quadraticRibbonSegmentIndexBuffer);
        quadraticRibbonGeomInst.setCurveEndcapFlags(curveEndcap);
        quadraticRibbonGeomInst.setMaterial(0, 0, matForQuadraticRibbons);
        quadraticRibbonGeomInst.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
        quadraticRibbonGeomInst.setUserData(geomData);
    }



    size_t maxSizeOfScratchBuffer = 0;
    OptixAccelBufferSizes asMemReqs;

    cudau::Buffer asBuildScratchMem;

    // JP: Geometry Acceleration Structureを生成する。
    // EN: Create geometry acceleration structures.
    optixu::GeometryAccelerationStructure floorGas = scene.createGeometryAccelerationStructure();
    cudau::Buffer floorGasMem;
    floorGas.setConfiguration(
        optixu::ASTradeoff::PreferFastTrace,
        optixu::AllowUpdate::No,
        optixu::AllowCompaction::Yes);
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

    optixu::GeometryAccelerationStructure linearCurvesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::LinearSegments);
    cudau::Buffer linearCurvesGasMem;
    linearCurvesGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    linearCurvesGas.setNumMaterialSets(1);
    linearCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    linearCurvesGas.addChild(linearCurveGeomInst);
    linearCurvesGas.prepareForBuild(&asMemReqs);
    linearCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure quadraticCurvesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::QuadraticBSplines);
    cudau::Buffer quadraticCurvesGasMem;
    quadraticCurvesGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    quadraticCurvesGas.setNumMaterialSets(1);
    quadraticCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    quadraticCurvesGas.addChild(quadraticCurveGeomInst);
    quadraticCurvesGas.prepareForBuild(&asMemReqs);
    quadraticCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure cubicCurvesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::CubicBSplines);
    cudau::Buffer cubicCurvesGasMem;
    cubicCurvesGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    cubicCurvesGas.setNumMaterialSets(1);
    cubicCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    cubicCurvesGas.addChild(cubicCurveGeomInst);
    cubicCurvesGas.prepareForBuild(&asMemReqs);
    cubicCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure catmullRomCurvesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::CatmullRomSplines);
    cudau::Buffer catmullRomCurvesGasMem;
    catmullRomCurvesGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    catmullRomCurvesGas.setNumMaterialSets(1);
    catmullRomCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    catmullRomCurvesGas.addChild(catmullRomCurveGeomInst);
    catmullRomCurvesGas.prepareForBuild(&asMemReqs);
    catmullRomCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure cubicBezierCurvesGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::CubicBezier);
    cudau::Buffer cubicBezierCurvesGasMem;
    cubicBezierCurvesGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    cubicBezierCurvesGas.setNumMaterialSets(1);
    cubicBezierCurvesGas.setNumRayTypes(0, Shared::NumRayTypes);
    cubicBezierCurvesGas.addChild(cubicBezierCurveGeomInst);
    cubicBezierCurvesGas.prepareForBuild(&asMemReqs);
    cubicBezierCurvesGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);

    optixu::GeometryAccelerationStructure quadraticRibbonsGas =
        scene.createGeometryAccelerationStructure(optixu::GeometryType::FlatQuadraticBSplines);
    cudau::Buffer quadraticRibbonsGasMem;
    quadraticRibbonsGas.setConfiguration(
        curveASTradeOff, curveASUpdatable, curveASCompactable,
        useEmbeddedVertexData);
    quadraticRibbonsGas.setNumMaterialSets(1);
    quadraticRibbonsGas.setNumRayTypes(0, Shared::NumRayTypes);
    quadraticRibbonsGas.addChild(quadraticRibbonGeomInst);
    quadraticRibbonsGas.prepareForBuild(&asMemReqs);
    quadraticRibbonsGasMem.initialize(cuContext, cudau::BufferType::Device, asMemReqs.outputSizeInBytes, 1);
    maxSizeOfScratchBuffer = std::max(maxSizeOfScratchBuffer, asMemReqs.tempSizeInBytes);



    // JP: GASを元にインスタンスを作成する。
    // EN: Create instances based on GASs.
    optixu::Instance floorInst = scene.createInstance();
    floorInst.setChild(floorGas);

    float linearCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, -1.25f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance linearCurvesInst = scene.createInstance();
    linearCurvesInst.setChild(linearCurvesGas);
    linearCurvesInst.setTransform(linearCurvesInstXfm);

    float quadraticCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, -0.75f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance quadraticCurvesInst = scene.createInstance();
    quadraticCurvesInst.setChild(quadraticCurvesGas);
    quadraticCurvesInst.setTransform(quadraticCurvesInstXfm);

    float cubicCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, -0.25f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance cubicCurvesInst = scene.createInstance();
    cubicCurvesInst.setChild(cubicCurvesGas);
    cubicCurvesInst.setTransform(cubicCurvesInstXfm);

    float catmullRomCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, 0.25f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance catmullRomCurvesInst = scene.createInstance();
    catmullRomCurvesInst.setChild(catmullRomCurvesGas);
    catmullRomCurvesInst.setTransform(catmullRomCurvesInstXfm);

    float cubicBezierCurvesInstXfm[] = {
        1.0f, 0.0f, 0.0f, 0.75f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance cubicBezierCurvesInst = scene.createInstance();
    cubicBezierCurvesInst.setChild(cubicBezierCurvesGas);
    cubicBezierCurvesInst.setTransform(cubicBezierCurvesInstXfm);

    float quadraticRibbonsInstXfm[] = {
        1.0f, 0.0f, 0.0f, 1.25f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    optixu::Instance quadraticRibbonsInst = scene.createInstance();
    quadraticRibbonsInst.setChild(quadraticRibbonsGas);
    quadraticRibbonsInst.setTransform(quadraticRibbonsInstXfm);



    // JP: Instance Acceleration Structureを生成する。
    // EN: Create an instance acceleration structure.
    optixu::InstanceAccelerationStructure ias = scene.createInstanceAccelerationStructure();
    cudau::Buffer iasMem;
    cudau::TypedBuffer<OptixInstance> instanceBuffer;
    ias.setConfiguration(optixu::ASTradeoff::PreferFastTrace);
    ias.addChild(floorInst);
    ias.addChild(linearCurvesInst);
    ias.addChild(quadraticCurvesInst);
    ias.addChild(cubicCurvesInst);
    ias.addChild(catmullRomCurvesInst);
    ias.addChild(cubicBezierCurvesInst);
    ias.addChild(quadraticRibbonsInst);
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
    catmullRomCurvesGas.rebuild(cuStream, catmullRomCurvesGasMem, asBuildScratchMem);
    cubicBezierCurvesGas.rebuild(cuStream, cubicBezierCurvesGasMem, asBuildScratchMem);
    quadraticRibbonsGas.rebuild(cuStream, quadraticRibbonsGasMem, asBuildScratchMem);

    // JP: 静的なメッシュはコンパクションもしておく。
    //     複数のメッシュのASをひとつのバッファーに詰めて記録する。
    // EN: Perform compaction for static meshes.
    //     Record ASs of multiple meshes into single buffer back to back.
    struct CompactedASInfo {
        optixu::GeometryAccelerationStructure gas;
        cudau::Buffer* mem;
        size_t offset;
        size_t size;
    };
    CompactedASInfo gasList[] = {
        { floorGas, &floorGasMem, 0, 0 },
        { linearCurvesGas, &linearCurvesGasMem, 0, 0 },
        { quadraticCurvesGas, &quadraticCurvesGasMem, 0, 0 },
        { cubicCurvesGas, &cubicCurvesGasMem, 0, 0 },
        { catmullRomCurvesGas, &catmullRomCurvesGasMem, 0, 0 },
        { cubicBezierCurvesGas, &cubicBezierCurvesGasMem, 0, 0 },
        { quadraticRibbonsGas, &quadraticRibbonsGasMem, 0, 0 },
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
        info.gas.compact(
            cuStream,
            optixu::BufferView(compactedASMem.getCUdeviceptr() + info.offset, info.size, 1));
    }
    // JP: removeUncompacted()はcompact()がデバイス上で完了するまでホスト側で待つので呼び出しを分けたほうが良い。
    // EN: removeUncompacted() waits on host-side until the compact() completes on the device,
    //     so separating calls is recommended.
    for (int i = 0; i < lengthof(gasList); ++i) {
        gasList[i].gas.removeUncompacted();
        gasList[i].mem->finalize();
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
    optixu::HostBlockBuffer2D<float4, 1> accumBuffer;
    accumBuffer.initialize(cuContext, cudau::BufferType::Device, renderTargetSizeX, renderTargetSizeY);



    Shared::PipelineLaunchParameters plp;
    plp.travHandle = travHandle;
    plp.imageSize.x = renderTargetSizeX;
    plp.imageSize.y = renderTargetSizeY;
    plp.resultBuffer = accumBuffer.getBlockBuffer2D();
    plp.camera.fovY = 50 * pi_v<float> / 180;
    plp.camera.aspect = static_cast<float>(renderTargetSizeX) / renderTargetSizeY;
    plp.camera.position = make_float3(0, 1.3f, 2.6f);
    plp.camera.orientation = rotateX3x3(-pi_v<float> / 7) * rotateY3x3(pi_v<float>);
    //plp.camera.position = make_float3(0, 0.01f, 2.5f);
    //plp.camera.orientation = rotateY3x3(pi_v<float>);

    pipeline.setScene(scene);
    pipeline.setHitGroupShaderBindingTable(hitGroupSBT, hitGroupSBT.getMappedPointer());

    CUdeviceptr plpOnDevice;
    CUDADRV_CHECK(cuMemAlloc(&plpOnDevice, sizeof(plp)));



    CUDADRV_CHECK(cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream));
    pipeline.launch(cuStream, plpOnDevice, renderTargetSizeX, renderTargetSizeY, 1);
    CUDADRV_CHECK(cuStreamSynchronize(cuStream));

    saveImage("output.png", accumBuffer, false, false);



    CUDADRV_CHECK(cuMemFree(plpOnDevice));



    accumBuffer.finalize();



    hitGroupSBT.finalize();

    compactedASMem.finalize();

    asBuildScratchMem.finalize();

    instanceBuffer.finalize();
    iasMem.finalize();
    ias.destroy();

    quadraticRibbonsInst.destroy();
    cubicBezierCurvesInst.destroy();
    catmullRomCurvesInst.destroy();
    cubicCurvesInst.destroy();
    quadraticCurvesInst.destroy();
    linearCurvesInst.destroy();
    floorInst.destroy();

    quadraticRibbonsGasMem.finalize();
    quadraticRibbonsGas.destroy();
    cubicBezierCurvesGasMem.finalize();
    cubicBezierCurvesGas.destroy();
    catmullRomCurvesGasMem.finalize();
    catmullRomCurvesGas.destroy();
    cubicCurvesGasMem.finalize();
    cubicCurvesGas.destroy();
    quadraticCurvesGasMem.finalize();
    quadraticCurvesGas.destroy();
    linearCurvesGasMem.finalize();
    linearCurvesGas.destroy();
    floorGasMem.finalize();
    floorGas.destroy();

    quadraticRibbonSegmentIndexBuffer.finalize();
    quadraticRibbonVertexBuffer.finalize();
    quadraticRibbonGeomInst.destroy();

    cubicBezierCurveSegmentIndexBuffer.finalize();
    cubicBezierCurveVertexBuffer.finalize();
    cubicBezierCurveGeomInst.destroy();

    catmullRomCurveSegmentIndexBuffer.finalize();
    catmullRomCurveVertexBuffer.finalize();
    catmullRomCurveGeomInst.destroy();

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

    matForQuadraticRibbons.destroy();
    matForCubicBezierCurves.destroy();
    matForCatmullRomCurves.destroy();
    matForCubicCurves.destroy();
    matForQuadraticCurves.destroy();
    matForLinearCurves.destroy();
    matForTriangles.destroy();



    shaderBindingTable.finalize();

    hitProgramGroupForQuadraticRibbons.destroy();
    hitProgramGroupForCubicBezierCurves.destroy();
    hitProgramGroupForCatmullRomCurves.destroy();
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
    //CUDADRV_CHECK(cuCtxDestroy(cuContext));

    return 0;
}
catch (const std::exception &ex) {
    hpprintf("Error: %s\n", ex.what());
    return -1;
}



template <OptixPrimitiveType curveType>
static void generateCurves(
    std::vector<Shared::CurveVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float baseWidth) {
    std::mt19937 rng(390318410);
    std::uniform_int_distribution<uint32_t> uSeg(3, 5);
    std::uniform_real_distribution<float> u01;

    uint32_t curveDegree;
    if (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR)
        curveDegree = 1;
    else if (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE)
        curveDegree = 2;
    else if (curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE ||
             curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CATMULLROM ||
             curveType == OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BEZIER)
        curveDegree = 3;
    else
        Assert_ShouldNotBeCalled();

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
                float3 pos = float3(x + 0.6f * deltaX * (u01(rng) - 0.5f),
                                    0.1f * (s + 1),
                                    z + 0.6f * deltaZ * (u01(rng) - 0.5f));
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
}

static void generateBezierCurves(
    std::vector<Shared::CurveVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float baseWidth) {
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


            // Base
            {
                float3 pos = float3(x, 0.0f, z);
                float width = baseWidth;
                vertices->push_back(Shared::CurveVertex{ pos, width });
            }
            float3 lastHandleDir = normalize(float3(u01(rng) - 0.5f, 1.0f, u01(rng) - 0.5f));

            for (int s = 0; s < numSegments; ++s) {
                float3 nextPos(x + 0.6f * deltaX * (u01(rng) - 0.5f),
                               0.1f * (s + 1),
                               z + 0.6f * deltaZ * (u01(rng) - 0.5f));
                float3 lastPos = vertices->back().position;
                float lastWidth = vertices->back().width;
                float p = (float)(s + 1) / numSegments;
                float width = baseWidth * (1 - p);
                float3 firstHandlePos = lastPos + 0.03f * lastHandleDir;
                vertices->push_back(Shared::CurveVertex{ firstHandlePos, lastWidth * 2.0f / 3 + width * 1.0f / 3 });
                lastHandleDir = normalize(float3(u01(rng) - 0.5f, 1.0f, u01(rng) - 0.5f));
                float3 secondHandlePos = nextPos - 0.03f * lastHandleDir;
                vertices->push_back(Shared::CurveVertex{ secondHandlePos, lastWidth * 1.0f / 3 + width * 2.0f / 3 });
                vertices->push_back(Shared::CurveVertex{ nextPos, width });
            }

            for (int s = 0; s < vertices->size() - indexStart - 1; s += 3)
                indices->push_back(indexStart + s);
        }
    }
}

void generateRibbons(
    std::vector<Shared::RibbonVertex>* vertices,
    std::vector<uint32_t>* indices,
    float xStart, float xEnd, uint32_t numX,
    float zStart, float zEnd, uint32_t numZ,
    float width) {
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

            float curNormalAngle = 2 * M_PI * u01(rng);

            // Beginning phantom points
            {
                float3 pos = float3(0.0f, 0.0f, 0.0f);
                float3 normal = float3(
                    std::cos(curNormalAngle), 0.0f, std::sin(curNormalAngle));
                vertices->push_back(Shared::RibbonVertex{ pos, normal, width });
            }

            // Base
            {
                float3 pos = float3(x, 0.0f, z);
                float3 normal = float3(
                    std::cos(curNormalAngle), 0.0f, std::sin(curNormalAngle));
                vertices->push_back(Shared::RibbonVertex{ pos, normal, width });
            }
            for (int s = 0; s < numSegments; ++s) {
                float p = (float)(s + 1) / numSegments;
                float3 pos = float3(x + 0.6f * deltaX * (u01(rng) - 0.5f),
                                    0.1f * (s + 1),
                                    z + 0.6f * deltaZ * (u01(rng) - 0.5f));
                curNormalAngle += M_PI / 3 * (u01(rng) - 0.5f);
                float3 normal = float3(
                    std::cos(curNormalAngle), 0.0f, std::sin(curNormalAngle));
                vertices->push_back(Shared::RibbonVertex{ pos, normal, width });
            }

            // Ending phantom points
            {
                float3 pm1 = (*vertices)[vertices->size() - 1].position;
                float3 pm2 = (*vertices)[vertices->size() - 2].position;
                float3 d = pm1 - pm2;
                float3 pos = pm1 + d;
                float3 normal = float3(
                    std::cos(curNormalAngle), 0.0f, std::sin(curNormalAngle));
                vertices->push_back(Shared::RibbonVertex{ pos, normal, width });
            }

            // Modify the beginning phantom points
            {
                float3 p1 = (*vertices)[indexStart + 1].position;
                float3 p2 = (*vertices)[indexStart + 2].position;
                float3 d = p1 - p2;
                (*vertices)[indexStart].position = p1 + d;
            }

            // Fix normal vector to be parpendicular to the curve.
            // TODO: still seems not smooth looking.
            for (int s = 0; s <= numSegments; ++s) {
                float3 p0 = (*vertices)[indexStart + s].position;
                float3 p1 = (*vertices)[indexStart + s + 1].position;
                float3 p2 = (*vertices)[indexStart + s + 2].position;
                float3 dir = normalize((p0 - 2 * p1 + p2) * 0.5f + (-p0 + p1));
                float3 &n = (*vertices)[indexStart + s + 1].normal;
                n = normalize(n - dot(dir, n) * dir);
            }
            (*vertices)[indexStart].normal = (*vertices)[indexStart + 1].normal;
            (*vertices)[indexStart + numSegments + 2].normal = (*vertices)[indexStart + 1].normal;

            for (int s = 0; s < vertices->size() - indexStart - 3; ++s)
                indices->push_back(indexStart + s);
        }
    }
}
