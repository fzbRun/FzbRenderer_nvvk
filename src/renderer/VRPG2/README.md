# VRPG2

自行实现的 VRPG（path guiding）渲染器框架：单套光栅体素化 → 几何聚类 → 光照注入 → 光照聚类 →
RIS 候选 → 引导采样 + BSDF 的 MIS。所有模块共用一个 `VRPG2Renderer`，设置与缓冲通过
`VRPG2Renderer::setting` / `VRPG2Renderer::source` 共享。

> 状态：**框架已完成、可编译**。`PathGuiding/shaders/pathGuiding.slang` 在引擎内
> `compileSlangShader` 时会导致进程退出（离线 `slangc` 能编译通过），这是当前唯一已知阻塞点，
> 见文末「已知问题」。

## 模块与文件

```
src/renderer/VRPG2/
  VRPG2.h / VRPG2.cpp              渲染器本体：帧编排、UI、模块生命周期
  VRPG2Shaderio.h                  共享 IO：MAX_OCTREE_LAYER_VRPG2、OctreeLayerNodeCount_VRPG2、
                                   OctreeNodeInfo_VRPG2、GlobalInfo_VRPG2、ClusterMaxCount_VRPG
  GeometryInjection/               几何注入（32³ VGB，单套光栅化）
    GeometryInjection.h/.cpp
    GeometryInjectionShaderio.h    GeometryInjectionPushConstant_VRPG、VoxelData_VRPG
    shaders/init.slang             computeMain_clearVGB
    shaders/injection.slang        vertexMain / geometryMain / fragmentMain（写 VGB）
  GeometryClustering/              几何聚类（Octree + SVO + nearby）
    GeometryClustering.h/.cpp
    GeometryClusteringShaderio.h   ClusterData_Octree_VRPG / ClusterInfo_Octree_VRPG / SVO / nearby
    shaders/init.slang             computeMain_init
    shaders/clustering.slang       GetHasDataClusterBlockInfo / DispatchClustering / Clustering1 / Clustering2
    shaders/createSVO.slang        createSVO1..4
    shaders/getNearbyCluster.slang dispatchGetNearbyCluster / getNearbyClusters1..2
  LightInjection/                  光照注入（把 radiance 写进几何 VGB 的 irradiance 字段）
    LightInjection.h/.cpp
    LightInjectionShaderio.h       LightInjectPushConstant_VRPG
    shaders/injection.slang        getHasGeometryVoxels / dispatchLightInjection / LightInject
  LightClustering/                 光照聚类（光树叶层 = 几何体素层，向上聚合）
    LightClustering.h/.cpp
    LightClusteringShaderio.h      LightNodeData_Octree_VRPG（radiance/halfAngle/meanNormal/aabb）
    shaders/init.slang             computeMain_init（叶层）
    shaders/clustering.slang       GetHasDataLightNodeBlockInfo / DispatchClustering / Clustering1 / Clustering2
  RIS/                             候选采样（两种策略，编译期宏切换）
    RIS.h/.cpp
    RISShaderio.h                  RISPushConstant_VRPG2、RISBindingPoints_VRPG2、候选结构、模式宏
    shaders/getCandidateNodes.slang  模式 1: initCandidatePool / expandCandidatePool / emitClusterPairs
                                     模式 0: lightCut
    shaders/getResampleWeights.slang getCandidateLeafNodes / clusterPairHitTest /
                                     getCandidateNodeWeight / createCandidatePrefixWSum
  PathGuiding/                     导引查询 + 路径追踪 + MIS
    PathGuiding.h/.cpp
    PathGuidingShaderio.h          PathGuidingPushConstant_VRPG2、PathGuidingBindingPoints_VRPG2
    shaders/guiding.slang          G 树 label 下降、前缀二分、球面矩形采样、getPgPdf2
    shaders/pathGuiding.slang      computeMain_VRPG（spp 循环 + NEE + BSDF/guide MIS）
```

## 数据流

```
scene ──几何注入──► 32³ VGB（6 个法线分区，VoxelData_VRPG: aabbI / sumNormal_G / irradiance）
                        │
                        ├──几何聚类──► ClusterData_Octree[layer]   (meanNormal, aabb, fillRate, indivisible)
                        │              ClusterInfo_Octree[layer]   (label_indivisible)
                        │              ClusterInfo_SVO            (每个 indivisible 簇的 layer/node)
                        │              nearbyClusterInfo
                        │
                        ├──光照注入──► 同一个 VGB 的 irradiance 字段（每体素 radiance 累加）
                        │
                        └──光照聚类──► LightNodeData_Octree[layer]（叶层 = 几何体素层，向上聚合）
                                          │
                                    RIS ──► clusterPairInfo / candidateSubTreeInfo / candidateLeafNodeData
                                          / candidateSampleData / candidateSamplePrefixWSum
                                          │
                              导引路径追踪 ──► gBuffers(eImgRendered) ──► tonemap ──► viewport
```

### 帧顺序（`VRPG2Renderer::render`）

```
updateDataPerFrame                     // 各模块 push constant 与 change flag
if geometryChange   : 几何注入 + barrier
if clusteringChange : 几何聚类 + barrier
if lightChange      : 光照注入 + 光照聚类 + RIS + barrier
                       （RIS 的候选表只依赖光树，所以跟光树一起刷新）
pathGuiding->render  + barrier         // 读候选表 + G 树 label + 光树
tonemapper
```

注意：几何聚类必须在光照注入之前（光树叶层直接复用几何体素层），而 RIS 必须整段连续
dispatch（它用 `globalInfoBuffer.cmd` 做乒乓，中间不能插入别的 dispatch）。

## 缓冲归属（`VRPG2Renderer::source`）

| buffer | 结构体 | 谁写 | 谁读 |
|---|---|---|---|
| `VGBs[6]` | `VoxelData_VRPG` | 几何注入（aabb/normal）、光照注入（irradiance） | 几何聚类、光照聚类 |
| `globalInfoBuffer` | `GlobalInfo_VRPG2`（cmd + ClusterCount + layerInfos_G） | 几何聚类 | 几何聚类 / RIS / 导引 |
| `clusterDataBuffer_Octree[]` | `ClusterData_Octree_VRPG` | 几何聚类 | RIS / 导引 |
| `clusterInfoBuffer_Octree[]` | `ClusterInfo_Octree_VRPG` | 几何聚类 | RIS / 导引 |
| `clusterInfoBuffer_SVO` | `OctreeNodeInfo_VRPG2` | 几何聚类（createSVO） | RIS |
| `nearbyClusterInfoBuffer` | `NearbyClusterInfo_VRPG` | 几何聚类 | （jitter 关闭时不用） |
| `lightNodeDataBuffer_Octree[]` | `LightNodeData_Octree_VRPG` | 光照聚类 | RIS / 导引 |
| `clusterPairInfoBuffer` | `ClusterPairInfo_VRPG2` | RIS | RIS |
| `clusterPairHitInfoBuffer` | `ClusterPairHitInfo_VRPG2` | RIS | RIS |
| `candidatePoolInfoBuffer` | `CandidatePoolInfo_VRPG2` | RIS | RIS / 导引 |
| `candidatePoolNodeInfoBuffer` | `CandidatePoolNodeInfo_VRPG2` | RIS（模式 1） | RIS（模式 1） |
| `candidateSubTreeInfoBuffer` | `CandidateSubTreeInfo_VRPG2` | RIS | 导引 |
| `candidateLeafNodeDataBuffer` | `CandidateLeafNodeData_VRPG2` | RIS | RIS / 导引 |
| `candidateSampleDataBuffer` | `CandidateSampleData_VRPG2`（w, p_hat） | RIS | 导引 |
| `candidateSamplePrefixWSumBuffer` | `float` | RIS | 导引（二分选候选） |

布局约定：候选按 `label * M + i` 存放（label 为几何簇的全局 label），权重按
`(k * ClusterMaxCount + label) * M + i` 存放，`k` 是 `OUTGOING_COUNT_VRPG2 = 32` 个方向分箱。
不做时空复用，因此没有 reservoir / history 缓冲。

## RIS 两种模式

`RIS/RISShaderio.h`：

```cpp
#ifndef VRPG_Candidate_Samples_SampleMode
#define VRPG_Candidate_Samples_SampleMode 1   // 改这里；0 = greedy relative error cut
#endif
```

| 宏 | M | 入口 |
|---|---|---|
| `1` 默认 | 32 | `initCandidatePool` → `expandCandidatePool` × 6 → `emitClusterPairs` |
| `0` | 64 | `initCandidatePool` → `lightCut` |

- 模式 1：每个几何簇维护一个 ≤M 的 pool（光树节点），每轮把 8 个子节点按上界
  `B = I·cos_G·cos_L/d²` 展开；装不下时把 `B` 最小的项回退成 `noExpand`（父节点成为 cut 叶）。
- 模式 0：反复展开"误差最大"的节点，直到 `maxError <= LIGHTCUT_RELATIVE_ERROR_VRPG2·ΣI`。
- 光树**不需要 label**：cut 只用 `LightNodeData_Octree` 的 radiance / meanNormal / aabb，
  到 `lightOctreeMaxLayer` 或预算耗尽即停。label 只存在于几何树（供导引定位命中点所属簇）。

## 已知问题（当前唯一阻塞点）

**现象**：Debug 运行时所有模块 shader 依次编译成功，最后在
`PathGuiding/shaders/pathGuiding.slang` 的编译过程中进程退出（无 validation 报错、无异常输出）。
同一文件用 `slangc -target spirv -entry computeMain_VRPG -stage compute` 离线编译成功。

排查建议（按顺序）：

1. 把 `computeMain_VRPG` 临时替换成"只写 `outImage` 常数"的最小版本，确认引擎路径可用；
2. 逐步加回 `#include "guiding.slang"`、NEE、guide 分支、BSDF 分支，定位触发点；
3. 对比旧 `VRPG/shaders/VRPG.slang` 的编译参数与 include 结构（它是能编译的）；
4. 若与单文件体积有关，把 `guiding.slang` 拆分为独立函数文件（`guidingFunctions.slang`）。

## 尚未实现 / 后续

- `GeometryClustering/shaders/debug.slang`、`LightClustering/shaders/debug.slang`、
  `LightInjection/shaders/debug.slang` 目前是空文件（聚类/注入的调试图未实现）；
  `GeometryClustering::postProcess` 因此是空实现。
- 邻域抖动（nearby jitter）与 RIS 时空复用按要求未实现（导引端已删除对应的绑定与分支）。
- `LightClustering` 的 cut 叶判定当前是"叶子层即终点"，如果以后需要更激进的剪枝，
  可在 `LightNodeData_Octree_VRPG` 里补 `indivisible`/误差字段。
