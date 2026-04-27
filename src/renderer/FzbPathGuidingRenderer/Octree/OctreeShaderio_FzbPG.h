#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"

#ifndef FZBRENDERER_OCTREE_FZBPG_SHADERIO_H
#define FZBRENDERER_OCTREE_FZBPG_SHADERIO_H

#define MAX_OCTREE_LAYER_FZBPG 5
#define IndivisibleNodeCount_G_FZBPG 1024

NAMESPACE_SHADERIO_BEGIN()

struct OctreePushConstant_FzbPG {
	float3x3 randomRotateMatrix;
	uint32_t octreeMaxLayer;
	uint32_t currentLayer;
	float voxelVolume;
	uint32_t octreeNodeTotalCount;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;
	uint32_t VGBVoxelTotalCount;
	SceneInfo* sceneInfoAddress;
	float frameIndex;
	float4 VGBStartPos_Size;
	float4 VGBVoxelSize;
#ifndef NDEBUG
	int showOctreeNodeTotalCount;
	int normalIndex;
	int sampleNodeLabel_G = 1;
	int sampleNodeLabel_E = 1;
	int showVisibleAABB = 1;
#endif
};

enum class BindingPoints_Octree_FzbPG : uint32_t {
	eVGB = 2,
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
#ifdef USE_VISIBLE_AABB_FZBPG
	eOctreeNodePairVisibleData,
	eOctreeNodePairE,
#endif
	eOctreeNodePairWeight,
#ifdef NEARBYNODE_JITTER_FZBPG
	eNearbyNodeTempInfos,
	eNearbyNodeInfos,
#endif
};
//------------------------------------------------------------------------------------------
#define OCTREE_CLUSTER_LAYER_FZBPG 2		//don't change!!!!!
#define OCTREE_NODECOUNT_E_FZBPG 440		//8 + 48 + 384
#define CLUSTER_LAYER_NODECOUNT_E_FZBPG 384
static const uint OctreeLayerNodeCount_FzbPG[MAX_OCTREE_LAYER_FZBPG] = { 8, 48, 384, 3072, 24576 };
static const uint OctreeLayerStartIndex_FzbPG[3] = { 0, 8, 56 };

struct OctreeNodeClusterData_G_FzbPG {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint indivisible;
#ifdef GEOMETRY_CLUSTER_WITH_E
	float E;
#endif
};
/*
label_indivisible is made of two data
the first bite is indivisible, 0 mean divisible, 1 mean indivisible
the 1 - 31 bite is label, 0 mean no data, other mean the node's index of this layer indivisible or divisible node
*/

struct OctreeNodeData_G_FzbPG {
	uint32_t label_indivisible;
};

/*
Octree_E is OCTREE_CLUSTER_LAYER - octreeMaxLayer
every node is indivisible, has all childNode's aabb, E and normal
*/
struct OctreeNodeClusterData_E_FzbPG {
#ifdef USE_VISIBLE_AABB_FZBPG
#else
	float pdf;
#endif
	float E;
	float4 meanNormal;
	AABB aabb;
};
/*
Octree_E is only 0 - OCTREE_CLUSTER_LAYER
the OCTREE_CLUSTER_LAYER layer node must is indivisible, and 0-OCTREE_CLUSTER_LAYER - 1 layer node must is divisible
so we only record the OCTREE_CLUSTER_LAYER layer node's label
*/
struct OctreeNodeData_E_FzbPG {
#ifdef USE_VISIBLE_AABB_FZBPG
#else
	AABB aabb;
	float pdf;
#endif
	uint32_t label;
};

//------------------------------------------------------------------------------------------
struct HasDataOctreeBlockCount_FzbPG {
	uint32_t count_G;
	uint32_t count_E;
};
struct OctreeLayerInfo_FzbPG {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};
struct OctreeGlobalInfo_FzbPG {
	DispatchIndirectCommand cmd;
#ifdef NEARBYNODE_JITTER_FZBPG
	DispatchIndirectCommand cmd2;
#endif
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	OctreeLayerInfo_FzbPG layerInfos_G[MAX_OCTREE_LAYER_FZBPG];
};
struct OctreeThreadGroupInfo_FzbPG {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
};
//------------------------------------------------------------------------------------------
#define OUTGOING_COUNT_FZBPG 64
#define HITTEST_COUNT_FZBPG 16		//not bigger than 32 or smaller than 8

#define OUTGOING_TYPE_FZBPG 0
#if OUTGOING_TYPE_FZBPG == 0
#define getOutgoing_FzbPG fibSpherePoint
#define inverseOutgoing_FzbPG inverseSF
#else 
#define getOutgoing hammersleySpherePoint
#define inverseOutgoing inverseSH
#endif

#define VISIBLEAABB_CLUSTER_LAYER 4

/*
nodePair consisting of indivisibleNode_G and indivisibleNode_E
aabb is the indivisibleNode_E's visible area for a indivisibleNode_G
pdf is (the visible area's E) / (OCTREE_CLUSTER_LAYER node's E that merge all child node' E)
*/
struct OctreeNodePairVisibleData_FzbPG {
	float pdf;
	AABB aabb;		//node_E's visible aabb for a node_G
};
//------------------------------------------------------------------------------------------
struct IndivisibleNodeNearbyNodeTempInfo_FzbPG {
	uint nodeLabel;
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT_FZBPG];
	float nearbyNodeDistances[NEARBY_NODE_COUNT_FZBPG];
};

struct OctreeNearbyNodeInfo_FzbPG {
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT_FZBPG];		//0-3 is layer, 4 - 31 is nodeIndex
};

#define CREATEOCTREE_CS_THREADGROUP_SIZE 256

#define GETOCTREELABEL_CS_THREADGROUP_SIZE 1024
#define GETOCTREELABEL4_CS_THREADGROUP_SIZE 512

#define INITWEIGHT_CS_THREADGROUP_SIZE 256
#define HITTEST_CS_THREADGROUP_SIZE 512
#define VISIBLEAABB_CLUSTER_CS_THREADGROUP_SIZE 512
#define GETPROBABILITY_CS_THREADGROUP_SIZE 1024

#define GETNEARBYNODES_CS_THREADGROUP_SIZE 512
#define GETNEARBYNODES2_CS_THREADGROUP_SIZE 1024

NAMESPACE_SHADERIO_END()
#endif