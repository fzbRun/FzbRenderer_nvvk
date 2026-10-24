#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VRPG_SHADER_IO_H
#define FZBRENDERER_VRPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Candidate_Samples_Count 128

//#define SpatialReuse_VRPG
#define Pairwise_MIS

#define OCTREE_E_CLUSTER


#define HITTEST_COUNT_PER_CHILDNODE_VRPG 4		//must <= 4

#define NEARBY_NODE_COUNT_VRPG 4	//dont't change!!!
//#define COMMON_NEARBYNODE_JITTER

#define FZB_PATHGUIDING_THREADGROUP_SIZE_X 16
#define FZB_PATHGUIDING_THREADGROUP_SIZE_Y 16

struct VRPGPushConstant
{
	int maxFrameCount;
	int frameIndex = 0;
	float time;

	int maxDepth = 6;
	int spp = 1;

	int spatialRuse = 0;
	int ditherSample = 1;
	int nearbyNodeCount = NEARBY_NODE_COUNT_VRPG;
	int padding;
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

NAMESPACE_SHADERIO_END()
#endif
