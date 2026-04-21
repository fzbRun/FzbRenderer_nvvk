#pragma once
/*
#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_FZB_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_FZB_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define MERGE_SIMILIAR_E

#define NEARBYNODE_JITTER
#define NEARBY_NODE_COUNT 4

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
};

enum class StaticBindingPoints_FzbPG
{
	//eTextures = 0,
	//eOutImage = 1,
	eOctreeArray_G = 2,
#ifdef NEARBYNODE_JITTER
	eOctreeNearbyNodeInfos,
#endif
	eNodeData_E,
	eGlobalInfo,
	eWeights,
#ifndef NDEBUG
	eDepthImage,
#endif
};
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
*/