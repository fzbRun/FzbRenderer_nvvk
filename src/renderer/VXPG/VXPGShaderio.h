#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VXPG_SHADER_IO_H
#define FZBRENDERER_VXPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

//------------------------------------------------------------------------------------------
// VXPG (voxel path guiding) port of SIByLEngine2023 "VXGuiding".
//
// The port keeps the reference data flow:
//   light injection (VPLs of the 1st indirect vertex) -> per voxel irradiance
//   geometry injection                              -> per voxel sub-voxel surface AABB
//   compact voxels                                  -> compact list + inverse index
//   superpixel clustering (gSLICr style)            -> spixel index + 4 fuzzy neighbours
//   light tree over the compact voxels              -> per voxel power hierarchy
//   per superpixel cluster importance (top level)   -> region dependent distribution
//   guide sampling + BSDF sampling with MIS         -> "Guide + BSDF" (sample mode 4)
//
// Differences to the reference implementation are documented in README.md next to this
// file. In short: the LBVH is built directly on the Morton ordered dense voxel grid (no
// bitonic sort / no compaction pass) and the MRCS visibility-signature clustering is
// replaced by a per-superpixel accumulation of the injected VPL power.
//------------------------------------------------------------------------------------------

#define VXPG_CLUSTER_COUNT 32                 // clusters per superpixel (top level tree leaves)
#define VXPG_TOP_LEVEL_STRIDE 64              // floats per superpixel in the top level tree
#define VXPG_TOP_LEVEL_LEAF_OFFSET 32         // leaf i lives at [offset + 32 + i]
#define VXPG_MAX_VOXEL_RESOLUTION 128
#define VXPG_REP_SP_COUNT 128        // stratified jittered representative shading points (paper 5.2)
#define VXPG_SIGNATURE_WORDS 4       // 128 bit visibility signature
#define VXPG_KMPP_MAX_VOXELS 8192    // clustering works on the compact lit voxel list         // upper bound of the (cubic) voxel grid
#define VXPG_THREADGROUP_SIZE_X 16
#define VXPG_THREADGROUP_SIZE_Y 16
#define VXPG_CLEAR_THREADGROUP_SIZE 512
#define VXPG_TREE_THREADGROUP_SIZE 256
#define VXPG_IRRADIANCE_SCALE 100.0f          // same packing as the reference (uint(irr*100))

// Sample mode. Only mode 4 of the reference ("Guide + BSDF") is implemented.
enum class VXPG_SampleMode : int
{
	eGuidePlusBsdf = 4,
};

// Sample mode 4 uses a two technique MIS. 0 = balance heuristic, 1 = power heuristic.
enum class VXPG_MisMode : int
{
	eBalance = 0,
	ePower = 1,
};

// Superpixel visibility mode of the reference ("Spixel" / "Fuzzy Spixel").
enum class VXPG_VisibilityMode : int
{
	eSpixel = 0,
	eFuzzySpixel = 1,
};

// One record per pixel. Filled by the VBuffer pass, extended by the light injection pass.
struct VXPGPixelData
{
	float4 position;       // xyz: primary hit position, w: 1 if valid, 0 otherwise
	float4 normalMaterial; // xyz: shading normal,   w: asfloat(material index)
	float4 vplPos;         // xyz: injected VPL position (1st indirect vertex), w: asfloat(octa normal)
	uint4  info;           // x: superpixel id, y: primary voxel flat id, z: VPL voxel flat id, w: flags
	float4 texCoordsIsExt; // xy: texture coordinates, z: hasTexture w: 1 if ext, 0 otherwise
};

// Light tree node. Internal nodes are [0, numVoxels-1), leaves [numVoxels-1, 2*numVoxels-1).
// The reference stores the AABB in the node as well; only the traversal weight is used by the
// intensity-only traversal, so the port keeps the node minimal.
// SLIC cluster accumulators (kept apart from VXPGSpixelInfo, which holds the *means*: mixing the
// two in one buffer made every iteration divide the running sums again and collapsed the centres).
struct VXPGSpixelAccum
{
	float4 posSum;      // xyz sum of positions, w = sample count
	float4 normalSum;   // xyz sum of normals
	float2 xySum;       // sum of pixel coordinates
	uint2  pad;
};

struct VXPGTreeNode
{
	float intensity;   // leaf: importance of the voxel, internal: sum of the children
	uint  flag;        // leaf: cluster id (0xffffffff = mixed), used as merge lock in treeMerge
	uint  parentIdx;   // 0xffffffff for the root
	uint  leftIdx;     // 0xffffffff for leaves
	uint  rightIdx;    // 0xffffffff for leaves
	uint  voxelSlot;   // leaf: morgon ordered voxel slot, 0xffffffff for internal nodes
};

// Superpixel centre record (same role as gSLICr's "spixel_info").
struct VXPGSpixelInfo
{
	float4 pos;        // shading position of the centre
	float4 normal;     // shading normal of the centre
	float2 center;     // pixel coordinate of the centre
	int    noPixels;   // unused by the port (kept for layout parity)
	int    id;         // spixel flat id
};

// Compact per pixel record for the SLIC passes: they only need the shading position, the normal
// and the validity flag, so they read 32 bytes here instead of the 80 byte VXPGPixelData.
struct VXPGPixelCompact
{
	float4 posValid;   // xyz: shading position, w: 1 when the pixel has a valid primary hit
	float4 normal;     // xyz: shading normal (w unused)
};

struct VXPGGlobalInfo
{
	DispatchIndirectCommand cmd;     // 12 bytes: x/y/z group counts for vkCmdDispatchIndirect
	uint32_t               padding0; // explicit: puts globalMean at offset 16 in both layouts
	float3                 globalMean;   // xyz: sum of the lit voxel centres (divided by litCount)
	uint32_t               padding1;     // keeps the size a multiple of 16
};

// Push constants shared by every VXPG pass. Kept 16 byte friendly: a float3 is always
// followed by a float so that the C++ and the Slang (ScalarDataLayout) layout agree.
struct VXPGPushConstant
{
	int frameIndex;
	int maxFrameCount;
	int spp;
	int maxDepth;

	int voxelResolution;    // N, the cubic voxel grid resolution
	int spixelSize;         // S, superpixel edge length in pixels
	int spixelCount;        // ceil(W/S) * ceil(H/S)
	int clusterCount;       // VXPG_CLUSTER_COUNT

	int mode;               // VXPG_SampleMode (4 = guide + bsdf)
	int misMode;            // VXPG_MisMode
	int visibilityMode;     // VXPG_VisibilityMode
	int bounceCount;        // extra BSDF + NEE bounces after the 1st indirect vertex

	int instanceIndex;      // geometry injection: current instance
	int triangleCount;      // geometry injection: triangles of the current instance
	int maxTreeLevel;       // log2(N)
	int flags;              // bit0: guide sampling enabled, bit1: fuzzy spixel

	float fireflyClamp;      // clamp for the indirect contributions, 0 = disabled
	float time;
	int randomFrameIndex;
	float jitterRadius;      // guide sampling: shading point (SP) jitter, in voxel sizes
	float padding0;          // guide sampling: superpixel lookup jitter, in pixels

	float3 VGBStartPos;     // world space min corner of the voxel grid cube
	float  VGBExtent;       // half extent of the voxel grid cube

	float3 VGBVoxelSize;    // world space size of one voxel
	float  padding1;         // light injection: VPL position jitter, in voxel sizes

	SceneInfo* sceneInfoAddress;
	uint2 screenSize;
	uint2 threadGroupCount;
	uint2 dispatchOffset;
	uint2 padding2;

	float kmppWeightE;  // Eq. (4) w_E: weight of the |E_p - E_q| irradiance term of the KMPP distance
	float neeTMin;      // shadow ray TMin for the directional light NEE (1e-3 in the shared helper
	                    // leaks light along wall corners: an occluder closer than 1 mm / sin(angle)
	                    // is skipped, which produced the bright hairlines on the edges)
	float3 padding3;
};

// Static descriptor set bindings. 0 and 1 are reserved by pathTracingCommon.slang
// (textures[] and outImage), exactly like the other renderers of this project.
enum class VXPGStaticBindingPoints
{
	eVoxelIrradiance = 2,   // RWStructuredBuffer<uint>   N^3, packed sum (uint(irradiance*100))
	eVoxelVplCount,         // RWStructuredBuffer<uint>   N^3, number of injected VPLs
	eVoxelBoundMin,         // RWStructuredBuffer<uint4>  N^3, packed sub-voxel surface AABB min
	eVoxelBoundMax,         // RWStructuredBuffer<uint4>  N^3, packed sub-voxel surface AABB max
	ePixelData,             // RWStructuredBuffer<VXPGPixelData>  W*H
	eTreeNodes,             // RWStructuredBuffer<VXPGTreeNode>   2*N^3-1
	eClusterRoots,          // RWStructuredBuffer<int>            32
	eTopLevelTree,          // RWStructuredBuffer<float>          spixelCount*64
	eSpixelInfo,            // RWStructuredBuffer<VXPGSpixelInfo> spixelCount
	eSpixelClusterWeight,   // RWStructuredBuffer<int>            spixelCount*32 (fixed point 1/1000)
	eSpixelClusterTotal,    // RWStructuredBuffer<int>            spixelCount    (fixed point 1/1000)
	eLeafVoxel,             // RWStructuredBuffer<uint>           N^3, leaf slot -> flat voxel index
	eVoxelLeaf,             // RWStructuredBuffer<uint>           N^3, flat voxel index -> leaf slot
	eClusterRange,          // RWStructuredBuffer<uint2>          VXPG_CLUSTER_COUNT, leaf range per cluster
	eRepSP,                 // RWStructuredBuffer<float4>         128 stratified jittered representative SPs
	eVoxelSignature,        // RWStructuredBuffer<uint4>          N^3, 128 bit visibility signature
	eLitVoxelList,          // RWStructuredBuffer<uint>           KMPP_MAX_VOXELS, compact lit voxel list
	eLitVoxelCount,         // RWStructuredBuffer<uint>           1, number of entries in the list
	eClusterCentroid,       // RWStructuredBuffer<uint4>          VXPG_CLUSTER_COUNT, signature centroids
	eVoxelCluster,          // RWStructuredBuffer<uint>           N^3, cluster id per voxel
	eSpixelAccum,           // RWStructuredBuffer<VXPGSpixelAccum> spixelCount, SLIC cluster sums
	eGlobalInfo,
	eClusterCentroidE,   // RWStructuredBuffer<float>         VXPG_CLUSTER_COUNT, centroid irradiance
	ePixelSpixelId,      // RWStructuredBuffer<uint>          pixels, spixel id of each pixel
	ePixelCompact,       // RWStructuredBuffer<VXPGPixelCompact> pixels, SLIC input
};

// Dynamic descriptor set bindings.
enum class VXPGDynamicBindingPoints
{
	eTlas = 0,
};

NAMESPACE_SHADERIO_END()
#endif
