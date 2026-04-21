#pragma once
/*
#include <common/Shader/shaderStructType.h>
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"

#ifndef FZBRENDERER_OCTREE_FZBPG_SHADERIO_H
#define FZBRENDERER_OCTREE_FZBPG_SHADERIO_H

#define MAX_OCTREE_LAYER 7
#define IndivisibleNodeCount_G 1024

NAMESPACE_SHADERIO_BEGIN()

struct OctreePushConstant_FzbPG {
	uint32_t octreeMaxLayer;
	uint32_t currentLayer;
	float voxelVolume;
	uint32_t octreeNodeTotalCount;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;
	uint32_t VGBVoxelTotalCount;
	float frameIndex;
	SceneInfo* sceneInfoAddress;
	float4 VGBStartPos_Size;
	float4 VGBVoxelSize;
#ifndef NDEBUG
	uint32_t showOctreeNodeTotalCount;
	int normalIndex;
#endif
};

enum class BindingPoints_Octree_FzbPG : uint32_t {
	eVGB = 0,
	eOctreeClusterData_G,
	eOctreeData_G,
	eOctreeClusterData_E,
	eClusterLayerData_E,
	eBlockInfos_G,
	eBlockInfos_E,
	eHasDataBlockIndices_G,
	eHasDataBlockIndices_E,
	eHasDataBlockCount,
	eGlobalInfo,
	eDivisibleNodeInfos_G,
	eThreadGroupInfos,
	eIndivisibleNodeInfos_G,
	eIndivisibleNodeInfos_E,
};
//------------------------------------------------------------------------------------------
#define OCTREE_CLUSTER_LAYER 2
#define OCTREE_NODECOUNT_E 440		//8 + 48 + 384
#define NODECOUNT_E 384
//static const uint OctreeLayerInfo_E[3] = { 8, 48, 384 };
//static const uint OctreeLayerStartIndex_E[3] = { 0, 8, 56 };

struct OctreeNodeClusterData_G {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint indivisible;
};
struct OctreeNodeData_G {
	uint32_t label_indivisible;
};
struct OctreeNearbyNodeInfo {
	int nearbyNodeInfos[NEARBY_NODE_COUNT];		//0-3 is layer, 4 - 31 is nodeIndex
};

struct OctreeNodeClusterData_E {
	float E;
	float4 meanNormal;
	AABB aabb;
};
struct OctreeNodeData_E {
	uint32_t label_indivisible;
};
struct OctreeNodePairData {
	float pdf;
	AABB aabb;		//node_E's visible aabb for a node_G
};
//------------------------------------------------------------------------------------------
struct HasDataOctreeBlockCount {
	uint32_t count_G;
	uint32_t count_E;
};
struct OctreeLayerInfo {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};
struct OctreeGlobalInfo {
	DispatchIndirectCommand cmd;
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	OctreeLayerInfo layerInfos_G[MAX_OCTREE_LAYER];
};
struct OctreeThreadGroupInfo {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
};
*/

NAMESPACE_SHADERIO_END()
#endif