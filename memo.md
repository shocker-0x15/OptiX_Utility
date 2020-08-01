# optix_util方針

- GAS/IAS/GeomInstとPipelineを切り離す。
  異なるPipeline間でGAS/IASを再利用できるようにする。
  - GASのBuildInputのnumSbtRecords, sbtIndexOffsetBuffer, flagsが同じになる必要があるため
    メッシュ中のマテリアルの「区別」とジオメトリフラグはジオメトリに属するものとする。
    マテリアルそのものはジオメトリとは切り離される。
- 面倒なところ、特にShader Binding Tableは隠蔽したい。
- ~~SBTの部分的なアップデートを可能にする。~~\
  これはちょっと綺麗に実装する方法が思いつかなかった。一旦シンプルに作る。
- ~~本来ある自由度は基本的には維持する。\
  例: SBT中のレコードはレイタイプ(= optixTrace時のオフセット)ごとに異なる値を持てる。~~\
  レイタイプ全体で同じでもあまり困らなさそうなので一旦シンプルなものを作る。



## クラス

### Context
- OptiXのコンテキストを持つ。
- マテリアルを作成する。
- シーンを作成する。
- パイプラインを作成する。

### Material
- OptiXのマテリアルを持つ。
- OptiX 6と同様にレイタイプごとにHitGroupを登録する。
- ~~OptiX 6とは異なり、あるマテリアル全体で共通のSBTデータを持つのではなく、レイタイプごとに持つ。
  HitGroupと併せて登録する。HitGroupはパイプラインに属しているため、SBTデータはPipelineごとに区別して保持する。
  つまりパイプラインとレイタイプをキーとしてSBTレコードを保持する。~~\
  一旦シンプルにマテリアルごとに単一のデータを持たせることにする。
- ~~SBTレコードのデータはレコード中に直接持たせるのではなく、Contextごとのマテリアル用バッファー内に持ち、
  SBTレコード中はインデックスのみを持つようにする。内容によらずSBTレコードのマテリアルに対応する箇所は固定サイズになる。
  また、データを更新してもSBT自体の更新が必要無い。~~\
  SBTレコードにはuint32_tの値(典型的には何らかのバッファーのインデックス)をユーザーが指定して持たせるだけにする。
  マテリアルに関連したデータはユーザーの世界で管理する。

### Scene
- SBTのHitGroupレコードに関連する情報を保持する。
- GeometryInstanceを作成する。
- GASとIASを作成する。

### GeometryInstance
- GASのBuildInputと一対一対応。
- マテリアル数(==ジオメトリフラグ数)を設定する。
  マテリアルの区別はパイプラインとは切り離された概念。
  - 各マテリアルごとにジオメトリのフラグを設定する。
- 固有のSBTデータを持つことができる。持つデータはパイプライン間で同じとする(パイプラインごとに異なるデータとすることも容易か)。
- ~~マテリアルと同様に、SBTデータは直接持つのではなく、Sceneごとのジオメトリインスタンス用バッファー内に持ち、
  SBTレコード中はインデックスのみを持つようにする。内容によらずSBTレコードのマテリアルに対応する箇所は固定サイズになる。
  また、データを更新してもSBT自体の更新が必要無い。~~\
  SBTレコードにはuint32_tの値(典型的には何らかのバッファーのインデックス)をユーザーが指定して持たせるだけにする。
  ジオメトリに関連したデータはユーザーの世界で管理する。
- インスタンシング時にマテリアルを切り替えられるように、マテリアルセットの概念を取り入れる。
  実際のMaterialはマテリアルのインデックスだけでなく、マテリアルセットのインデックスを併せて登録する。

### GeometryAccelerationStructure (GAS)
- OptiXTraversableHandleを持つ。
- 子として登録した複数のGeometryInstanceからAcceleration Structureのビルドを行う。
  コンパクション・アップデートも行う。
- インスタンシング時にマテリアルを切り替えられるように、マテリアルセットの概念を取り入れる。
  マテリアルセット数を設定するAPIを持つ。
- マテリアルセットごとにレイタイプ数を設定する。
- 固有のSBTデータを持つことができる。持つデータはパイプライン間で同じとする(パイプラインごとに異なるデータとすることも容易か)。

### InstanceAccelerationStructure (IAS)
- OptiXTraversableHandleを持つ。
- 子として登録した複数のインスタンスからAcceleration Structureのビルドを行う。
  コンパクション・アップデートも行う。

### Pipeline
- aaa



## Shader Binding Table

### HitGroupテーブルのレイアウト
GASとMaterialSetの組み合わせに対応するデータのまとまりが並ぶ。以下は一例。
```
| GAS 0 - MatSet 0 | GAS 0 - MatSet 1 | GAS 1 - MatSet 0 | GAS 2 - MatSet 0|
```
GASとMaterialSetの組み合わせごとに以下の例のようなデータのまとまりを持つ。
```
| GAS-MatSet                                                |
| Input 0           | Input 1                     | Input 2 |
| Mat 0-0 | Mat 0-1 | Mat 1-0 | Mat 1-1 | Mat 1-2 | Mat 2-0 |
 <--- #SBTRecords -> <------------- #SBTRecords -> <------->
```
さらにそれぞれのMaterialごとに複数個のレコードが並ぶ。この個数は上記のGAS-MatSetごとに同一の値。
この個数は典型的にはレイタイプ数になる。現状のMaterialの実装ではレイタイプごとにデータを区別しないため、各レコードの違いはヘッダー部分(HitGroupの違い)のみ。
```
| Mat       |
| 0 | 1 | 2 |
```
トレース命令には次のようなSBTレイアウトに関連するパラメターがある。
```
optixTrace(...,
           sbt-offset-from-trace-call,
           sbt-stride-from-trace-call,
           ...);

sbt-index = sbt-instance-offset + 
            (sbt-GAS-index * sbt-stride-from-trace-call) + 
            sbt-offset-from-trace-call
```
- sbt-instance-offsetはインスタンス(GAS-MatSetの組み合わせに対応)のオフセットを指定する。
- sbt-GAS-indexはOptiXによって内部的に自動計算される。上記レイアウト内のMat列におけるインデックスに対応する(例:Mat 1-0が2、Mat 2-0が5)。
- 典型的にはsbt-stride-from-trace-callはレイタイプ数、sbt-offset-from-trace-callがレイタイプ(に対応するインデックス)を表す。
- レイアウトはIASには依存しない。逆にIASはレイアウトに依存する。\
  関連するGASごとのSBTレイアウトに関する情報が確定した時点でレイアウトを計算できる。\
  ```
  [GASのBuildInput確定] <--- [HitGroup SBT Layout] <-+- [IASビルド]
                                                     |
                             [GASビルド] <-----------+
  ```
  - SBTに関してはできれば隠したままにしたい。
  - ユーザーがASに関して一切何もせずパイプラインのローンチ => パイプラインにセットされているシーンに関してSBTレイアウトの計算、SBTのセットアップ、GASビルド、IASビルド、ローンチ。\
  この場合ユーザーがTraversableHandleを得る機会が無いので、Util規定のハンドルを収めたバッファーをユーザーに必ずセットしてもらい、ユーザーにはバッファーインデックスをAS作成時に渡しておき、カーネルではバッファーとインデックスを組み合わせてアクセスしてもらう、というルールを定めなければならない。ASのダブルバッファリングなどをどう扱うかも考えるのが面倒。
  - TraversableHandleはユーザー管理の世界。\
  ダブルバッファリングの扱いなどもユーザー責任。基本的にユーザーは明示的にGASビルドとIASビルドを行う。IASビルドを行った時点で、Hit Group SBT Layout計算を行う。
- MaterialSetの概念はインスタンシングに使用するが、レイタイプ数をGASのMaterialSetごとに変えられるため、同じシーン中で異なるレイのセットを対象とするトレースを複数個使い分けられる。



## 各Dirty化条件
Dirty化とは:
ASに関してはリビルド(!=アップデート)しないといけない状態。
HitGroup SBT Layoutに関してはレイアウトが現在のGASと一貫性がなくなっている状態。
HitGroup SBTに関しては各レコードのヘッダーやデータ部分が古くなっている状態。

GASのDirty化条件:
1. GASのビルド設定の更新 (Auto)
1. GASに対するGeomInstの追加・削除 (Auto)
1. GASに所属するGeomInstのジオメトリ・ジオメトリフラグ・マテリアル数の更新

HitGroup SBT LayoutのDirty化条件:
1. Sceneに所属するGASのDirty化 (Auto)
1. Sceneに所属するGASのNumMaterialSets変更 (Auto)
1. Sceneに所属するGASのNumRayTypes変更 (Auto)

IASのDirty化条件:
1. HitGroup SBT LayoutがDirty化 (Auto)
1. IASに対するInstanceの追加・削除 (Auto)
1. IASのビルド設定の更新 (Auto)
1. IASに所属するInstanceのデータの更新・Instanceが持つGASのTraversableHandleの更新

HitGroup SBTのDirty化条件:
1. HitGroup SBT LayoutのDirty化
1. GeomInstのデータ更新\
   これはレコード内に直接データを記録しないようにしたことで不要になる。
   ただしレコードへのインデックスを更新する場合はSBTを直接書き換える必要があるためDirty化する。
1. GeomInstに所属するMaterialのデータ更新\
   これはレコード内に直接データを記録しないようにしたことで不要になる。
   ただしレコードへのインデックスを更新する場合はSBTを直接書き換える必要があるためDirty化する。
1. MaterialのHitGroupのアップデート
1. GASのデータ更新

(Auto)は該当の条件によるDirty化が自動で行われることを示す。
例: GASに対するGeomInstの追加・削除 (Auto)
    この操作を行ったときにGASが自動でdirty状態になることを示す。

あるべき姿
- パイプラインのローンチ時にSBTを正しい状態へと更新する。
  あるジオメトリのSBTレコードが更新されたとき、あるマテリアルのSBTレコードが更新されたとき
  SBT中の対応する箇所だけ更新する。

気づき:
- ユーザーが書くOptiXカーネル内でTraversableHandleを直接扱う現状の実装では、
  パイプラインローンチ前にGAS, SBT Layout, IASのセットアップをユーザー側で終えておく必要がある。
  (パイプラインのローンチパラメターにTraversableHandleの値を含める必要があるため)
  逆に言えばUtil側ではSBTLayoutの生成まではローンチ時には終わっているとみなすことができる。

- Privからしかアクセスできない場合と1行で処理が終わるほど簡単・明快な場合はutil_private.hに書く。
- module, program group, pipelineの設定変更時のリコンパイル
  program groupのpipeline optionsを記憶しておいてlink時に照合？
- pipelineのsceneセット解除
- materialのhitgroup変更時のSBT無効化
  materialごとに関与しているsceneとそのカウントを覚えておいて変更時にSBT無効化通知？

CPUでジオメトリのデータ更新をしたい場合:
- GeometryInstanceのVertexBufferなどは2ついる。
  1. CPUで更新したデータを含んだVertex/TriangleBufferをGeometryInstanceにセット。
  1. GASのprepareForBuildを呼ぶ。
  1. 新しいバッファーを用意してGASのrebuild/updateを呼ぶ。\
     ユーザーが持っている古いバッファーのメモリとハンドルは何も影響を受けない。
  1. 