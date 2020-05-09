# OptiX Utility

[OptiX](https://developer.nvidia.com/optix)はOptiX 7以降[Direct X Raytracing (DXR)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html)にそっくりなローレベル指向なAPIになりました。<!--
-->細かいところに制御が効く一方で、何をするにも煩雑なセットアップコードを書く必要が出てきました。<!--
-->このOptiX Utilityは細かい制御性はできる限り保持したまま定形処理になりがちな部分を隠蔽したクラス・関数を提供することを目的としています。

[OptiX](https://developer.nvidia.com/optix) changes its form since OptiX 7 into low-level oriented API like [Direct X Raytracing (DXR)](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html).
It provides fine-level controllability but requires the user to write troublesome setup code to do anything.
The purpose of this OptiX Utility is to provide classes and functions which encapsulates parts which tend to be boilerplate code while keeping fine controllability.

- cuda_util.h, cuda_util.cpp \
  このCUDAユーティリティはCUDAのbufferやarray、そしてカーネルの実行のためのクラス・関数を提供しています。\
  現在のOptiXはCUDAに基づいたAPIになっているため、ユーザーはOptiXのコードと併せて頻繁に純粋なCUDAのコードも扱う必要があります。\
  これにはOptiX関連のコードは含まれません。\
  This CUDA Utility provides classes and functions for CUDA buffer, CUDA array creation and CUDA kernel execution.
  OptiX is now CUDA-centric API, so the user often needs to manage pure CUDA code along with OptiX code.\
  This doesn't contain any OptiX-related code.
- optix_util.h, optix_util_private.h, optix_util.cpp\
  optix_util.hはOptiXのオブジェクトをホスト側で管理するためのAPIを公開し、デバイス側の関数ラッパーも提供しています。\
  これは上記CUDAユーティリティに依存しています。\
  optix_util.h exposes API to manage OptiX objects on host-side and provides device-side function wrappers as well.\
  This depends on the CUDA Utility above.

----
2020 [@Shocker_0x15](https://twitter.com/Shocker_0x15)
