#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_FZB_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_FZB_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define RIS_Version
//#define StochasticLightcuts_RIS
#define Candidate_Samples_Count 32

#define OCTREE_E_CLUSTER

//#define GEOMETRY_CLUSTER_WITH_E

//#define ADAPTIVE_IMPORTANCE_SAMPLING	//Adaptive importance sampling
#define HITTEST_COUNT_PER_CHILDNODE_FZBPG 4		//must <= 4
#define ADAPTIVE_IMPORTANCE_SAMPLING_MAX_LAYER 3

//#define NEARBYNODE_JITTER_FZBPG
#define NEARBY_NODE_COUNT_FZBPG 4	//dont't change!!!
//#define COMMON_NEARBYNODE_JITTER

#define FZB_PATHGUIDING_THREADGROUP_SIZE_X 16
#define FZB_PATHGUIDING_THREADGROUP_SIZE_Y 16

struct FzbPathGuidingPushConstant
{
	float3x3 randomRotateMatrix;
	float3 VGBVoxelSize;
	int maxDepth = 6;
	int spp = 1;
	float time;
	int maxOctreeLayer;
	float4 VGBStartPos_Size;
	int maxFrameCount;
	int frameIndex = 0;
	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;

#ifndef StochasticLightcuts_RIS
	uint* indivisibleNodeInfoBufferAddress_E;
	float* weightSumBufferAddress;
#endif
};

#ifdef StochasticLightcuts_RIS
enum class StaticBindingPoints_FzbPG
{
	eOctreeNodeInfo_G = 2,
	eOctreeNodeData_E,
	eCandidateNodeData_E,
	eGlobalInfo,

#ifndef NDEBUG
	eDepthImage,
	ePGValueImage,
	ePGValue2Image,
	ePGVarianceImage,
#endif
};
#else
enum class StaticBindingPoints_FzbPG
{
	//eTextures = 0,
	//eOutImage = 1,
	eOctreeData_G = 2,
	eClusterLayerData_E,
	eOctreeNodePairWeight,
	eGlobalInfo,
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	eOctreeNodePairData,
#endif
#ifdef NEARBYNODE_JITTER_FZBPG
	eOctreeClusterData_G,
	eNearbyNodeInfos,
#endif
#ifndef NDEBUG
	eDepthImage,
	ePGValueImage,
	ePGValue2Image,
	ePGVarianceImage,
#endif
};
#endif
enum class DynamicBindingPoints_FzbPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};
struct GlobalInfo_FzbPG {
	uint SVOMaxLayer_G;
	uint indivisibleNodeCount_G;
	uint totalNodeCount_E;
};

NAMESPACE_SHADERIO_END()
#endif
