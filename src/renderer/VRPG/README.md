# VRPG renderer

This directory is the VRPG port described in the CVM paper. It is intentionally
self-contained: it does not modify or include the `FzbPathGuidingRenderer`
translation units. The implementation was bootstrapped from the existing
six-normal-path-guiding infrastructure, then the public shaderio/class names
were renamed with the `_VRPG` suffix so both renderers can live in the same
binary.

## Pipeline mapping

| Paper stage | Implementation |
| --- | --- |
| Geometry injection (AABB, mean normal, fill) | `RasterVoxelization_VRPG` |
| Six normal-partitioned geometry/light grids | `Octree_VRPG`, `Clustering.slang` |
| Light injection / power accumulation | `LightInject_VRPG` |
| Light-tree cut, candidate subtrees | `Octree/shaders/GetClusterWeight.slang` |
| Guide table `(geometric cluster C, direction d_k) -> M candidates` | `candidateSampleDataBuffer`, `candidateSubTreeInfoBuffer`, `candidateLeafNodeDataBuffer` |
| RIS weights / final density | `GetClusterWeight.slang` + `shaders/VRPG.slang` |
| BSDF/guide MIS | `shaders/VRPG.slang` (`computeMain_VRPG`) |

## Defaults

- `Candidate_Samples_Count` = 128  (paper `M`)
- `OUTGOING_COUNT_VRPG` = 32       (paper `K`, Fibonacci directions)
- `HITTEST_COUNT_VRPG` = 8         (paper `S`)
- `IndivisibleNodeCount_G_VRPG` = 2500
- Geometry VGB: `voxelCount=32`, raster resolution `1024x1024`
- Light VGB: `voxelCount=8`, raster resolution `256x256`
- `maxDepth` and `spp` come from `rendererInfo.xml`

Only the geometry VGB is rasterized from the scene. The light VGB is rebuilt
every frame by a compute pass that clusters geometry voxels into light voxels
(`computeMain_clusterGeometryVGB`), then light injection runs on those voxels.
Per-frame light injection uses the dynamic sample count instead of the expensive
static-only count.

The geometry and light grids are separate `RasterVoxelization_VRPG` instances.
Geometry injection feeds the G-tree; light injection feeds a separate light VGB.
The E-tree is now built directly from that light VGB at
`lightOctreeMaxLayer` (`LightTree.slang`): leaves are light voxels and the upper
layers are bottom-up aggregates. G and E therefore have independent resolutions
and independent node counts.

Candidate cuts and leaf selection are generated per geometric cluster. The
cluster-pair, candidate-subtree and candidate-leaf buffers use `G*M`; only the
per-outgoing weights/prefix sums use `K*G*M`. Material data is not stored in the
geometry VGB or G-tree.

## Runtime controls

`VRPGSettings` exposes:

- **Spatiotemporal Reuse**: builds one reservoir per `(outgoing, G)`, merges
  the same outgoing reservoir from nearby G clusters, then merges the previous
  frame's reservoir for that `(outgoing, G)`.
- **Temporal Blend**: weight multiplier for the previous-frame reservoir.
- **Jitter Sampling**: toggles the nearby-cluster dithering at runtime
  (`octree->ditherSample`).
- **Jitter Radius**: scales the cluster-relative nearby jitter radius.
- **Rebuild Per Frame**: runs raster voxelization / light injection every
  frame (otherwise static scenes rebuild on frame 0 only).
- Max Frames / bounces / spp as before.

`M`, `K`, and `S` are compile-time capacities in this first port because they
size the candidate/prefix buffers and shader arrays. Changing them requires a
rebuild. Spatiotemporal reuse is also guarded by `SpatialReuse_VRPG`; it is
currently left off because the existing spatial-reuse entry point uses a
workgroup size of `K * M`, which exceeds the Vulkan per-group limit at the
paper defaults. The panel reports this state explicitly.

## Registration

- `src/renderer/Renderer.cpp`: `"VRPG"` -> `VRPGRenderer`
- `rendererInfo/rendererInfo.xml`: `<renderer type="VRPG">`
- CMake picks up this directory through its existing `src/*.*` glob.
