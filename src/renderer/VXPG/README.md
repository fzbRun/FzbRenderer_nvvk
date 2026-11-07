# VXPG renderer

Voxel path guiding, ported from `SIByLEngine2023`'s `Addon::VXGuiding` (the graph
`VXGuidingGraph` in `Sandbox/src/CustomPipeline.hpp`). Only the **basic VXPG** pipeline is
ported and only **sample mode 4 ("Guide + BSDF")** is implemented; the ReSTIR-GI, ASVGF,
Semi-NEE, SLC, accumulator and visualisation passes of the reference are intentionally left
out.

Like `src/renderer/VRPG`, this directory is self contained: it does not modify or include the
`FzbPathGuidingRenderer` / `VRPG` translation units, and it can live in the same binary.

## What is implemented

| Reference stage | Implementation here |
| --- | --- |
| `VBuffer` ray traced G-buffer | `VXPGPasses.slang` -> `vxpgVBufferMain` (position, normal, material, voxel id) |
| `VXGuider1stBounceInjection` (VPL injection of the 1st indirect vertex) | `vxpgLightInjectMain`: BSDF sample at the primary vertex, NEE + emissive at the 1st indirect vertex, `InterlockedAdd` of `maxComponent(radiance)` into the voxel |
| `VXInfoClearPass` / `VXPremulIrradiance` | per voxel importance = average injected irradiance (`VXPG_VoxelImportance`), consumed by the tree leaves |
| `VXGuiderGeometryBake/DynamicPass` (compact per-voxel surface AABB) | `vxpgGeometryInjectMain`: conservative compute triangle voxelization writing the quantised sub-voxel surface AABB (`uint[4]` min/max per voxel, unorm32, same packing as the reference) |
| `VXTreeEncodePass` + `BitonicSort` + `VXTreeIIntializePass` + `VXTreeInternalPass` + `VXTreeMergePass` | `vxpgTreeInitMain` / `vxpgTreeInternalMain` / `vxpgTreeMergeMain`: the same Karras LBVH and the same per-cluster root nodes, but the leaves are the **Morton ordered voxel slots** instead of the sorted compact voxel list, so no bitonic sort and no compaction pass are needed |
| `RowColumnPresample` + `RowVisibility` + `RowKmppCenter` + `RowFindCenter` (MRCS = 32 clusters) | `VXPG_ClusterOfSlot`: 32 contiguous Morton ranges, i.e. spatially coherent clusters (see "Differences") |
| `gSLICr::InitClusterCenterPass` / `FindCenterAssociationPass` | `vxpgSpixelCentersMain` / `vxpgSpixelAssignMain` (same SLIC distance, same 3x3 neighbourhood, same normal gate and weights; only the unused SLIC iterations are dropped) |
| `SPixelClear/Gather/VisibilityEX` + `VXTreeTopLevelPass` | `vxpgSpixelAccumMain` (per superpixel cluster importance from the VPLs the region actually found) + `vxpgTopLevelMain` (the reference's 64-float per-superpixel heap with 32 cluster leaves) |
| `VXGuiderGIPass`, `sample_stragegy == 4` | `VXPG.slang` -> `computeMain_VXPG`: one BSDF sample and one guided sample per pixel, both traced from the primary vertex, combined with the balance (or power) MIS heuristic, guided hits accepted only inside the sampled voxel ("semi NEE"), plus `bounceCount` plain BSDF + NEE bounces |
| `SampleVoxelGuiding` / `PdfVoxelGuiding` / `SampleSphericalVoxel` / `PdfSampleSphericalVoxel` | `VXPGCommon.slang`: superpixel -> top level tree -> cluster root -> light tree traversal -> spherical-quad direction sampling, with the exact reference spherical quad math |

The guiding pdf is

```
p(direction) = p(cluster | superpixel) * p(leaf | cluster root) * p(direction | voxel)
```

with

* `p(cluster | superpixel)` = `topLevelTree[sp*64 + 32 + c] / topLevelTree[sp*64 + 1]`
  (a single importance ratio, because the sampler descends the same 5-level heap),
* `p(leaf | cluster root)` = `intensity(leaf) / intensity(clusterRoot)`, which is exactly the
  product of the per-level `I_child / (I_left + I_right)` probabilities because every
  internal node stores the sum of its children (the reference's
  `PdfTraverseLightTree_Intensity`),
* `p(direction | voxel)` = `1 / sum(solid angles of the three spherical rectangles)`, which is
  exactly what `SampleSphericalVoxel` samples from.

## Differences to the reference

These are deliberate; each one is either a bug fix, or a replacement of machinery that would
have been a much larger port without changing the sampled *formula*.

1. **No bitonic sort / no compaction.** The reference compacts the lit voxels into a list,
   Morton-sorts them with a 22 pass bitonic sort and builds the LBVH over the compacted
   leaves. Here the tree is built directly over the `N^3` Morton ordered voxel slots, i.e. the
   leaf order is already a spatially coherent permutation of the grid. Consequences: the tree
   contains the empty voxels too (they carry intensity 0 and are skipped by the traversal with
   probability 0), no `compactIndices` / `inverseIndex` / `compact2leaf` buffers are needed,
   and the tree has `2*N^3 - 1` nodes (12.6 MB at `N = 64`).
2. **Clusters = contiguous Morton ranges** (32 per scene) instead of the reference's MRCS
   KMPP clustering of 128-bit visibility signatures. This changes *which* voxels are grouped
   together (and therefore the numeric pdf), but not the sampling formula.
3. **Per-superpixel cluster importance** is accumulated from the VPLs that the pixels of the
   region actually found (`vxpgSpixelAccumMain`), instead of the reference's stochastic
   32-sample visibility rays plus `SPixelAvgVisibility`. The reference multiplies the cluster
   intensity by `avg_visibility / 32`; the port instead measures a region-specific importance
   directly, which is the same quantity the top-level tree needs. Regions that found no VPL at
   all fall back to the global cluster importance, so a region can never end up with an empty
   distribution.
4. **Guiding pdf symmetry fixed.** In the reference, `PdfVoxelGuiding` returns `0` for the
   "visibility irradiance" traversal mode (its body is commented out) and does not include the
   top-level `top_pdf` factor, so its mode-4 MIS is not consistent. The port computes the full
   product above for the BSDF branch as well, hence the MIS weights are consistent and the
   BSDF branch no longer silently gets weight 1.
5. **Emissive meshes are injected.** The reference's injection only evaluates analytic direct
   lighting, which is 0 for scenes lit by emissive geometry (`staircase`, `bearBox`, ...). The
   port injects `NEE + material.emissive` at the 1st indirect vertex, so emissive-light scenes
   produce a useful guiding distribution too. Sky lighting is still not injected (like the
   reference).
6. **No fuzzy superpixel mode.** The reference's "Fuzzy Spixel" roulette draw is not accounted
   for in any pdf, so it would break the MIS. Only the exact `Spixel` mode is kept.
7. **Sky hits of the BSDF branch are gathered** (`RayMiss * throughput`); the reference's
   `EvaluateIndirectLight` drops them.
8. **Voxel importance = average irradiance** (no `* area`). With a uniform grid the area term
   is a constant factor that cancels in every probability, and the reference's "product of the
   two smallest compact-AABB extents" degenerates to ~0 for planar surfaces.
9. `TreeNode` keeps only `intensity / flag / parent / left / right / voxelSlot` (no fp16 AABB),
   because the intensity-only traversal never reads the AABB. `flag` is still the cluster id
   and still doubles as the bottom-up merge lock, exactly like the reference.
10. Geometry injection is a compute pass with a conservative per-triangle voxelization (lattice
    sampled to at most 16 voxels per axis) instead of the reference's geometry-shader
    rasterization. Same quantised result for triangles smaller than 16 voxels.

## Registration (outside this directory)

To stay inside the constraint "only add or modify files in `src/renderer/VXPG`", the two
registration edits were **not** made. To activate the renderer, add:

* `src/renderer/Renderer.cpp`

  ```cpp
  #include "VXPG/VXPG.h"
  // enum FzbRendererType: add FZB_RENDERER_VXPG
  // RendererTypeMap:      add { "VXPG", FZB_RENDERER_VXPG },
  // createRenderer():     case FZB_RENDERER_VXPG: return std::make_shared<VXPGRenderer>(createInfo.rendererNode);
  ```

* `rendererInfo/rendererInfo.xml`

  ```xml
  <renderer type = "VXPG">
      <maxDepth value = "4" />
      <spp value = "1" />
      <VXPG>                                <!-- the whole block is optional -->
          <voxelResolution value = "64" />   <!-- power of two, 8 ... 128, default 64 -->
          <spixelSize value = "32" />        <!-- superpixel edge in pixels, default 32 -->
          <!-- path length comes from <maxDepth> at the renderer level, see below -->
          <misMode value = "0" />            <!-- 0 = balance, 1 = power, default 0 -->
          <maxFrames value = "1024" />
          <rebuildPerFrame value = "true" />
      </VXPG>
  </renderer>
  ```

  Every value of the `<VXPG>` block is optional and clamped; a missing block uses the
  defaults above (64^3 voxels, 32 pixel superpixels, balance heuristic).
  The superpixel size is raised automatically if the render resolution would produce more than
  8192 superpixels, because the per superpixel top level tree (64 floats) and the per superpixel
  cluster weights would otherwise allocate hundreds of megabytes. Malformed values are reported
  with a warning and replaced by the default (the constructor never throws).

  **Path length** is controlled by `<maxDepth>` (the renderer level node, shared with the other
  renderers) or by the "Bounces (path length)" slider, and is the number of path *vertices*:
  `1` = direct lighting only, `2` = camera vertex + the first indirect vertex (the pair that both
  mode 4 techniques trace), `n > 2` = additionally `n - 2` BSDF + NEE bounces after the first
  indirect vertex. Those deeper vertices are only generated by the BSDF technique, which is why
  their contribution is added without an MIS weight (the guided technique never goes deeper, so
  its pdf is zero for them - this matches the reference). The `<bounce>` node of older versions
  is ignored; the value is derived from `<maxDepth>`.
CMake already picks the directory up through its `file(GLOB_RECURSE ... src/*.*)`.

## Runtime controls (`VXPGSettings`)

* **Max Frames**, **Frame**
* **Rebuild Guiding** / **Enable Guiding**
* **Bounces** (path length), **SPP**, **Extra Bounces** (bounces after the 1st indirect
  vertex), **MIS Mode**
* **Voxel Resolution** / **Superpixel Size** + **Rebuild Voxel Grid** (reallocates the grid,
  the pixel records and the top-level tree)

## Memory

At 1280x720 with `N = 64`: voxel importance/count 2 MB, surface bounds 8 MB, pixel records
59 MB, tree nodes 12.6 MB, top level tree 235 KB. Memory scales with `N^3` (tree + voxels) and
with the render resolution (pixel records).

## Files

```
VXPG/
  VXPG.h / VXPG.cpp          renderer, descriptors, pass scheduling, UI
  VXPGShaderio.h             push constant, binding enums, GPU structs
  shaders/VXPGCommon.slang   voxel/Morton helpers, LBVH traversal, spherical quad sampling,
                             guide sampling + guide pdf
  shaders/VXPGPasses.slang   all compute passes except the integrator
  shaders/VXPG.slang         the mode-4 integrator
  README.md                  this file
```

## Validation status

Verified on this machine (Release build, `staircase` and `veach-ajar-2`, 1280x720, voxel grid 64^3,
superpixel size 32): the module builds and links, all 11 Slang entry points compile (both with
`slangc` and with the runtime compiler), all passes execute without Vulkan errors or
`VK_ERROR_DEVICE_LOST`, and both techniques of mode 4 produce the expected images.

Measurements taken from a readback of the guiding buffers after a frame (`VXPG_DUMP_FRAME`):

```
litVoxels=195 totalVpls=4818 maxVpl=66 spixelsWithTotal=557/920
rootIntensity=390   (light tree root, i.e. total injected power)
topLevel[1]=0.357   (per superpixel top level tree root)
```

Rendered frames at 45 accumulated frames, `staircase`, 1 spp:

| configuration | mean | std |
| --- | --- | --- |
| mode 4 (guide + BSDF, the deliverable) | 34.8 | 60.0 |
| BSDF technique only | 25.3 | 59.3 |
| guided technique only | 13.2 | 22.5 |

The guided only image is visibly much smoother than the BSDF only image, which is the expected
benefit of the guiding distribution (the BSDF technique is the high variance one, and the mode 4
combination is the biased-free mix used by the reference).

Two bugs were found and fixed during this validation, they are worth remembering because both
silently disable the guiding:

1. **Pass order matters for the injection.** The injection pass must build its payload from a real
   intersection (it now traces its own primary ray, like the integrator). Deriving the payload from
   the cached pixel record alone produced a wrong `outgoing`/normal pair (`dot(outgoing, normal) < 0`),
   so `BSDF_SAMPLE` returned a zero BSDF and no VPL was ever injected.
2. **The per superpixel heap reduction started one level too low.** The 32 cluster leaves live at
   `offset + 32 .. offset + 63`, so the bottom up sum has to start at node 31 (not 15); otherwise
   `topLevelTree[offset + 1]` stays 0 and `VXPG_SampleTopLevelTree` always fails.

## Debug helpers (only active when the environment variables are set)

| variable | meaning |
| --- | --- |
| `VXPG_STAGE_MASK` | bit mask of the pass groups to run: 1 vbuffer, 2 light injection, 4 geometry injection, 8 light tree, 16 superpixel, 32 integrator. Missing = all |
| `VXPG_DEBUG` | debug output mode of the integrator: 1 = constant grey (checks the tonemap plumbing), 2 = Reinhard of the radiance without accumulation, 4 = disable the BSDF technique, 8 = disable the guided technique, 16 = guiding structure view (R: VPL count of the hit voxel, G: top level total, B: root intensity + cluster root availability), 32 = guide sampler view (R: valid sample, G: cluster selected, B: leaf reached) |
| `VXPG_SAVE_FRAME` / `VXPG_SAVE_NAME` | save the tonemapped image of that frame index into the working directory |
| `VXPG_DUMP_FRAME` / `VXPG_DUMP_FILE` | write a summary of the guiding buffers (injected VPL count, tree root intensity, per cluster roots, top level tree of superpixel 0, per superpixel totals, a few pixel records) to a text file |
## Fixes applied while matching the reference behaviour

1. **Tree intensities** are summed from the leaves to all ancestors with float atomics
   (`vxpgTreeAccumulateMain`), because the reference's lock step child/parent handshake does not
   guarantee that a child intensity is visible when the parent reads it; internal nodes stayed at
   zero and most descents aborted.
2. **Cluster roots** are computed deterministically, one thread per cluster, by descending from the
   root to the lowest common ancestor of the cluster's contiguous leaf range
   (`vxpgClusterRootsMain`). The reference's merge pass (retired here) left most clusters without a
   root, so the top level tree referenced clusters that could not be sampled.
3. **Path length** (`<maxDepth>`) is wired into the integrator: 1 = direct only, 2 = plus the first
   indirect vertex (the pair both mode 4 techniques trace), n = plus n-2 BSDF + NEE bounces.
4. **NEE on back facing vertices** is skipped: `VXPG_DirectLight` returns only the (already zeroed)
   emissive term when `!payload.isExt`, so a back face can no longer receive light on the side the
   path sees it from.
5. **Light injection** rejects back facing second vertices and VPLs further away than
   `2 * scene half extent` from the camera (`pushConstant.jitterRadius`), which keeps the injected
   set reachable and close to the reference's camera centred clipmap.
6. **Sane defaults / defensive parsing** for the `<VXPG>` block (64^3 voxels, 32 pixel superpixels,
   clamped values, never throws) and an automatic superpixel size increase when the superpixel grid
   would exceed 8192 entries.

### Measured result (`veach-ajar-2`, 1280x720, 64^3, 45 accumulated frames, 1 spp)

| configuration | image mean | table region mean |
| --- | --- | --- |
| mode 4 (guide + BSDF, the deliverable) | 64.3 | 51.1 |
| guided technique only | 53.0 | 44.0 |
| BSDF technique only | 15.6 | 7.5 |

The two techniques add up to the combined image (15.6 + 53.0 ~ 64.3), i.e. the MIS weights form a
consistent pair; before the two fixes above the guided technique contributed exactly zero and the
image was simply the BSDF technique plus the wasted guided rays (which is why it looked dark and
noisier than plain path tracing).
