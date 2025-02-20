# OptiX Utility

![example](example.png)

[OptiX](https://developer.nvidia.com/optix)はOptiX 7以降[Direct X Raytracing (DXR)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)にそっくりなローレベル指向なAPIになりました。<!--
-->細かいところに制御が効く一方で、何をするにも煩雑なセットアップコードを書く必要が出てきました。<!--
-->このOptiX Utilityは細かい制御性はできる限り保持したまま定形処理になりがちな部分を隠蔽したクラス・関数を提供することを目的としています。

[OptiX](https://developer.nvidia.com/optix) has transformed into a low-level-oriented API similar to [Direct X Raytracing (DXR)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html) since OptiX 7.
While it offers fine-grained control, it also requires users to write cumbersome setup code for almost any operation.
The purpose of this OptiX Utility is to provide classes and functions that encapsulate boilerplate code, while preserving as much fine control as possible.

## 組込方法 / Integration
OptiX Utilityを使うプログラムに以下を追加します。\
Add the followings to your program which uses OptiX Utility:

- optix_util.h
- optix_util_private.h
- optix_util.cpp

オプションとして、プログラムがCUDAのメモリ確保などを実装していない場合は cuda_util.h, cuda_util.cpp と optixu_on_cudau.h も追加します。CUDA UtilityにOpenGL連携機能が必要ない場合はコンパイルオプションとして`CUDA_UTIL_DONT_USE_GL_INTEROP`を定義してください。\
Optionally add cuda_util.h, cuda_util.cpp and optixu_on_cudau.h as well if the program doesn't have functionalities like memory allocation for CUDA. Define `CUDA_UTIL_DONT_USE_GL_INTEROP` as a compile option when you don't need OpenGL interoperability in CUDA Utility.

## 機能 / Features
Currently based on OptiX 9.0.0

[&raquo;] は対応する機能を使うサンプルコードへのリンク(抜粋)です。\
[&raquo;] links to a sample code (excerpt) which uses a corresponding feature.

- Traversable types
  - Single GAS [[&raquo;]](samples/single_gas/)
  - Single-level instancing [[&raquo;]](samples/single_level_instancing/)
  - Multi-level instancing  [[&raquo;]](samples/multi_level_instancing/)
- Primitive types
  - Triangles [[&raquo;]](samples/single_gas/)
  - Curves [[&raquo;]](samples/curve_primitive/)
    - 1st order: Round linear segments
    - 2nd order: Round quadratic B-splines, Ribbons (Flat quadratic B-splines)
    - 3rd order: Round cubic B-splines, Round Catmull-Rom splines, Round cubic B&#233;zier curves
    - Each curve type (except for linear and ribbon) has Rocaps (Roving Capsules) variants
  - Spheres [[&raquo;]](samples/sphere_primitive/)
  - User-defined custom primitives [[&raquo;]](samples/custom_primitive/)
- Opacity micro-map (OMM) [[&raquo;]](samples/opacity_micro_map/)
- Motion blur types
  - Instance motion blur [[&raquo;]](samples/multi_level_instancing/)
  - Deformation blur [[&raquo;]](samples/deformation_blur/)
- Acceleration structure management
  - Full build [[&raquo;]](samples/single_gas/) [[&raquo;]](samples/single_level_instancing/)
  - Fast update [[&raquo;]](samples/as_update/)
  - Compaction [[&raquo;]](samples/single_gas/) [[&raquo;]](samples/single_level_instancing/)
- Shader binding table management
  - Automatic build
  - Memory management is still under user control
- Geometry instancing with different material sets [[&raquo;]](samples/material_sets/)
- Callable programs [[&raquo;]](samples/callable_program/)
- OptiX-IR support for better debugging\
  \* but fow now (OptiX 8.1.0 / CUDA 12.6.2 and the 566.03 diver), OptiX-IR itself causes some weird behavior, so using traditional ptx input is recommended until we get the update...
- OptiX AI denoiser (AOV Output Not Tested) [[&raquo;]](samples/denoiser/) [[&raquo;]](samples/temporal_denoiser/)
  - HDR
  - HDR + Upscaling 2x
  - HDR Temporal
  - HDR Temporal + Upscaling 2x
- Automatic payload/attribute packing/unpacking in kernel code
  - supports hit objects as well for shader execution reordering (SER)
- Payload usage annotation to reduce register consumption in complex pipelines [[&raquo;]](samples/payload_annotation/)

### TODO
- Clusters API
- Support new HitObject functionalities
- Support SBT offset summation accross all instances in the traversal graph
- Test NVRTC compilation.
- Test flow vector trustworthiness guiding
- Test AOV denoisers
- Test Linux environment
- Parallel module compilation
- AS relocation
- OMM relocation
- Multi-GPU

## 構成要素 / Components
- **optix_util.h, optix_util_private.h, optix_util.cpp**\
  OptiXのオブジェクトをホスト側で管理するためのAPIと、デバイス側の関数ラッパーを提供しています。\
  This provides API to manage OptiX objects on host-side and device-side function wrappers.
- **cuda_util.h, cuda_util.cpp**\
  このCUDAユーティリティはCUDAのbufferやarrayの生成、そしてカーネルの実行のためのクラス・関数を提供しています。\
  現在のOptiXはCUDAに基づいたAPIになっているため、ユーザーはOptiXのコードと併せて頻繁に純粋なCUDAのコードも扱う必要があります。\
  これにはOptiX関連のコードは含まれず、OptiX Utilityとも直接関係しません。\
  This CUDA Utility provides classes and functions for CUDA buffer, array creation, and kernel execution.
  OptiX is now CUDA-centric API, so the user often needs to manage pure CUDA code along with OptiX code.\
  This doesn't contain any OptiX-related code and is not directly related to the OptiX Utility.
- **optixu_on_cudau.h**\
  OptiX UtilityをCUDA Utilityと組み合わせて使うための関数といくつかの補助クラスを定義した取るに足らないファイルです。\
  This trivial file defines a function to use OptiX Utility combined with the CUDA Utility and defines several auxiliary classes.
- **samples**\
  OptiX Utilityの基本的な使い方を網羅した複数のサンプルがあります。\
  Multiple samples cover basic usage of the OptiX Utility.

## コード例 / Code example
### ホスト側 / Host-side
OptiX UtilityはシェーダーバインディングテーブルのセットアップといったOptiXカーネルを実行するまでに必要な面倒な手続きを可能な限り隠蔽します。

OptiX utility hides troublesome procedures like setting up shader binding table required to execute OptiX kernels as much as possible.
```cpp
// Create an OptiX context from a CUDA context (Driver API).
optixu::Context optixContext = optixu::Context::create(cuContext);

// Create a pipeline and associated programs (groups) then link the pipeline.
optixu::Pipeline pipeline = optixContext.createPipeline();
pipeline.setPipelineOptions(
    optixu::calcSumDwords<PayloadSignature>(),
    optixu::calcSumDwords<AttributeSignature>(),
    "plp", sizeof(PipelineLaunchParameters),
    OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY,
    OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH,
    OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE);
optixu::Module mainModule =
    pipeline.createModuleFromOptixIR(
        optixIr, OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT,
        OPTIX_COMPILE_OPTIMIZATION_DEFAULT, OPTIX_COMPILE_DEBUG_LEVEL_NONE);
optixu::Program rayGenProgram = pipeline.createRayGenProgram(module, RT_RG_NAME_STR("pathtracing"));
// ...
optixu::HitProgramGroup searchRayHitProgramGroup =
    pipeline.createHitProgramGroupForTriangleIS(
        mainModule, RT_CH_NAME_STR("shading"), emptyModule, nullptr);
optixu::HitProgramGroup visibilityRayHitProgramGroup =
    pipeline.createHitProgramGroupForTriangleIS(
        emptyModule, nullptr, mainModule, RT_AH_NAME_STR("visibility"));
// ...
pipeline.link(2);

// Allocate a shader binding table (scene independent part).
cudau::Buffer sbt;
size_t sbtSize;
scene.generateShaderBindingTableLayout(&sbtSize);
//...
pipeline.setShaderBindingTable(sbt, sbt.getMappedPointer());

// Create materials.
optixu::Material defaultMat = optixContext.createMaterial();
defaultMat.setHitGroup(RayType::Search, searchRayHitProgramGroup);
defaultMat.setHitGroup(RayType::Visibility, visibilityRayHitProgramGroup);
// ...
defaultMat.setUserData(...);

// Create a scene.
optixu::Scene scene = optixContext.createScene();

// Create geometry instances (triangles or curves or user-defined custom primitives).
optixu::GeometryInstance geomInst0 = scene.createGeometryInstance();
cudau::TypedBuffer<Vertex> vertexBuffer;
cudau::TypedBuffer<Triangle> triangleBuffer;
// ...
geomInst0.setVertexBuffer(vertexBuffer);
geomInst0.setTriangleBuffer(triangleBuffer);
geomInst0.setUserData(...);
geomInst0.setNumMaterials(1, BufferView());
geomInst0.setGeometryFlags(0, OPTIX_GEOMETRY_FLAG_NONE);
geomInst0.setMaterial(0, 0, defaultMat);

OptixAccelBufferSizes asMemReqs;

// Create geometry acceleration structures.
optixu::GeometryAccelerationStructure gas0 = scene.createGeometryAccelerationStructure();
gas0.setConfiguration(
    optixu::ASTradeoff::PreferFastTrace,
    optixu::AllowUpdate::Yes, optixu::AllowCompaction::Yes); // Builder preference.
gas0.addChild(geomInst0);
gas0.addChild(geomInst1);
gas0.addChild(...);
gas0.setUserData(...);
optixu::GeometryAccelerationStructure gas1 = scene.createGeometryAccelerationStructure();
// ...
cudau::Buffer gas0Mem;
gas0.prepareForBuild(&asMemReqs);

// Create instances.
optixu::Instance inst0 = scene.createInstance();
inst0.setChild(gas0);
inst0.setTransform(...);
optixu::Instance inst1 = scene.createInstance();
// ...

// Create instance acceleration structures.
optixu::InstanceAccelerationStructure ias0 = scene.createInstanceAccelerationStructure();
ias0.setConfiguration(
    optixu::ASTradeoff::PreferFastBuild,
    optixu::AllowUpdate::Yes, optixu::AllowCompaction::Yes); // Builder preference.
ias0.addChild(inst0);
ias0.addChild(inst1);
ias0.addChild(...);
optixu::InstanceAccelerationStructure ias1 = scene.createInstanceAccelerationStructure();
// ...
cudau::TypedBuffer<OptixInstance> instBuffer;
cudau::Buffer ias0Mem;
ias0.prepareForBuild(&asMemReqs);

// Build acceleration structures.
cudau::Buffer asBuildScratchMem;
// ...
OptixTraversableHandle gas0Handle = gas0.rebuild(cuStream, gas0Mem, asBuildScratchMem);
// ...
OptixTraversableHandle ias0Handle = ias0.rebuild(cuStream, instBuffer, ias0Mem, asBuildScratchMem);

// Allocate a shader binding table (scene dependent part).
cudau::Buffer hitGroupSbt;
size_t hitGroupSbtSize;
scene.generateShaderBindingTableLayout(&hitGroupSbtSize);
// ...

// Associate the pipeline and the scene/shader binding table.
pipeline.setScene(scene);
pipeline.setHitGroupShaderBindingTable(hitGroupSbt, hitGroupSbt.getMappedPointer());

// Setup pipeline launch parameters and allocate memory for it on the device.
PipelineLaunchParameter plp;
// ...
CUdeviceptr plpOnDevice;
cuMemAlloc(&plpOnDevice, sizeof(plp));

// Launch the pipeline!
cuMemcpyHtoDAsync(plpOnDevice, &plp, sizeof(plp), cuStream);
pipeline.launch(cuStream, plpOnDevice, width, height, 1);
//...
```

### デバイス側 / Device-side
OptiX Utilityはペイロードのパッキングを簡単にしたりカーネル間通信における型の不一致を回避するため、デバイス側の組み込み関数のラッパーを提供しています。

OptiX utility provides template wrappers for device-side built-in functions to ease packing of payloads and to avoid type inconsistency for inter-kernel communications.
```cpp
// Define payload signatures.
using SearchRayPayloadSignature = optixu::PayloadSignature<PCG32RNG, ExtraPayload*>;
using VisibilityRayPayloadSignature = optixu::PayloadSignature<float>;
// ...
CUDA_DEVICE_KERNEL void RT_RG_NAME(pathtracing)() {
    // ...
    ExtraPayload* exPayloadPtr = &exPayload;
    while (true) {
        // ...
        SearchRayPayloadSignature::trace(
            traversable, origin, direction,
            0.0f, FLT_MAX, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Search, NumRayTypes, RayType_Search,
            rng, exPayloadPtr);
        // ...
    }
    // ...
}
// ...
CUDA_DEVICE_KERNEL void RT_CH_NAME(shading)() {
    auto sbtr = reinterpret_cast<HitGroupSBTRecordData*>(optixGetSbtDataPointer());
    // ...
    PCG32RNG rng;
    ExtraPayload* exPayloadPtr;
    SearchRayPayloadSignature::get(&rng, &exPayloadPtr);
    // ...
    {
        // ...
        float visibility = 1.0f;
        VisibilityRayPayloadSignature::trace(
            traversable, p, shadowRayDir, 0.0f, dist * 0.999f, 0.0f, 0xFF, OPTIX_RAY_FLAG_NONE,
            RayType_Visibility, NumRayTypes, RayType_Visibility,
            visibility);
        // ...
    }
    // ...
    SearchRayPayloadSignature::set(&rng, nullptr);
}
// ...
CUDA_DEVICE_KERNEL void RT_AH_NAME(visibility)() {
    float visibility = 0.0f;
    VisibilityRayPayloadSignature::set(&visibility);

    optixTerminateRay();
}
// ...
```

## 動作環境 / Confirmed Environment
現状以下の環境で動作を確認しています。\
I've confirmed that the programs run correctly in the following environment.

* Windows 11 (24H2) & Visual Studio Community 2022 (17.13.1)
* Ryzen 9 7950X, 64GB, RTX 4080 16GB
* NVIDIA Driver 572.47

動作させるにあたっては以下のライブラリが必要です。\
It requires the following libraries.

* [CUDA](https://developer.nvidia.com/cuda-downloads) 12.8 (probably works with lower CUDA versions)\
  Note that CUDA (<= 12.5.0) has compilation issues for C++20 with Visual Studio 2022 17.10.\
  Use CUDA 12.5 Update 1 or newer for C++20.
* [OptiX](https://developer.nvidia.com/designworks/optix/download) 9.0.0 (requires Turing or later generation NVIDIA GPU)

## ライセンス / License
Released under the Apache License, Version 2.0 (See [LICENSE.md](LICENSE.md))

----
2025 [@Shocker_0x15](https://twitter.com/Shocker_0x15), [@bsky.rayspace.xyz](https://bsky.app/profile/bsky.rayspace.xyz)
