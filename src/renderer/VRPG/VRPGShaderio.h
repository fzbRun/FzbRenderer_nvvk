#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VRPG_SHADER_IO_H
#define FZBRENDERER_VRPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Candidate_Samples_Count 64

// Compile-time VRPG switches (change requires rebuilding the app).
#define VRPG_USE_WAVEFRONT 0      // 0 stable single dispatch, 1 experimental wavefront
#define VRPG_RANDOM_FRAME_SOURCE 1
#define VRPG_CAMERA_REPROJECTION 0 // 0 off, 1 reproject static camera reset
#define VRPG_PATH_SAMPLE_MODE 0   // 0 double MIS, 1 one-sample MIS, 2 guide only
#define VRPG_JITTER_SAMPLING 1
#define VRPG_NEARBY_NODE_COUNT 4
#define VRPG_SPATIAL_REUSE 0
#define VRPG_TEMPORAL_REUSE 0

//#define SpatialReuse_VRPG
#define Pairwise_MIS

#define OCTREE_E_CLUSTER


#define HITTEST_COUNT_PER_CHILDNODE_VRPG 4		//must <= 4

#define NEARBY_NODE_COUNT_VRPG VRPG_NEARBY_NODE_COUNT	// compile-time count
//#define COMMON_NEARBYNODE_JITTER

#define FZB_PATHGUIDING_THREADGROUP_SIZE_X 16
#define FZB_PATHGUIDING_THREADGROUP_SIZE_Y 16

struct VRPGPushConstant
{
	int maxFrameCount;
	int frameIndex = 0;
	int randomFrameIndex = 0;
	int staticScene = 0;
	float time;

	int maxDepth = 6;
	int spp = 1;

	int spatialRuse = 0;
	int ditherSample = 1;
	int nearbyNodeCount = NEARBY_NODE_COUNT_VRPG;
	int pathSampleMode = 2;
	float jitterRadiusScale = 1.5f;

	float3 VGBVoxelSize;
	int maxOctreeLayer;
	float4 VGBStartPos_Size;

	float4 lightVGBStartPos_Size;
	float4 lightVGBVoxelSize;
	int lightMaxOctreeLayer;

	float3x3 randomRotateMatrix;
	uint2 dispatchOffset;
	int padding1;

	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;

};

enum class StaticBindingPoints_VRPG
{
	eOctreeNodeInfo_G = 2,
	eOctreeNodeData_E,

	eCandidateSubTreeData,
	eCandidateLeafNodeData_VRPG,

	eCandidateSampleData,
	eCandidateSamplePrefixWSum,

	eGlobalInfo,

	eCandidateSampleData2,
	eCandidateSampleAABB,

	eOctreeNodeData_G,
	eNearbyNodeInfos,
	eReservoir,

#ifndef NDEBUG
	eDepthImage,
	ePGValueImage,
	ePGValue2Image,
	ePGVarianceImage,
#endif
};
enum class DynamicBindingPoints_VRPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};
struct GlobalInfo_VRPG {
	uint SVOMaxLayer_G;
	uint indivisibleNodeCount_G;
	uint totalNodeCount_E;
};

//------------------------------------------------------------------------------------------
// Experimental wavefront path tracer (shaders/VRPGWavefront.slang).
// This is a separate path from the stable single-dispatch VRPG shader; it owns a
// dedicated path-state buffer and does not change any existing binding values.
//------------------------------------------------------------------------------------------
enum class StaticBindingPoints_VRPGWavefront {
	ePathState = 32,
};

#define FZB_WAVEFRONT_THREADGROUP_SIZE_X 16
#define FZB_WAVEFRONT_THREADGROUP_SIZE_Y 16

// One entry per pixel. All members are 16-byte aligned so the Slang
// ScalarDataLayout representation matches this C++ struct byte for byte.
struct WavefrontPathState
{
	float4 accumulatedRadiance;   // xyz: radiance accumulated by the path so far
	float4 throughput;            // xyz: path throughput (bsdf * cos / pdf)
	float4 rayOrigin;             // xyz: origin of the current (already traced) ray
	float4 rayDirection;          // xyz: direction of the current (already traced) ray
	float4 hitPos;                // xyz: shading position of the pending hit
	float4 hitNormal;             // xyz: shading normal (normal map already applied)
	float4 outgoing;              // xyz: direction towards the previous vertex
	float4 uvFlags;               // xy: texture coordinates, z: uvValid, w: frontFacing
	uint4  info;                  // x: active, y: bounceIndex, z: materialIndex, w: randomSeed
};

#ifdef __cplusplus
static_assert(sizeof(WavefrontPathState) == 9 * 16,
	"WavefrontPathState must stay 16-byte aligned for the Slang ScalarDataLayout buffer");
#endif

NAMESPACE_SHADERIO_END()
#endif
