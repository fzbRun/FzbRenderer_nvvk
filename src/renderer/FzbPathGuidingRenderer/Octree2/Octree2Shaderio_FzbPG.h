#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"

#ifndef FZBRENDERER_OCTREE_2_FZBPG_SHADERIO_H
#define FZBRENDERER_OCTREE_2_FZBPG_SHADERIO_H

#define MAX_OCTREE_LAYER_FZBPG 7
#define IndivisibleNodeCount_G_FZBPG 2000

NAMESPACE_SHADERIO_BEGIN()

struct OctreePushConstant_FzbPG {
	float frameIndex;

	uint32_t VGBVoxelTotalCount;
	float voxelVolume;
	float4 VGBStartPos_Size;
	float4 VGBVoxelSize;

	float3x3 randomRotateMatrix;

	uint32_t octreeMaxLayer;
	uint32_t octreeNodeTotalCount;

	uint32_t currentLayer;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;

	SceneInfo* sceneInfoAddress;

	float4x4 padding0;
	float4x4 padding1;
	float4 padding2;
};

struct OctreeDebugPushConstant_FzbPG {
	int frameIndex;

	int mode = 0;

	int normalIndex_G = 0;
	int curLayer_G;
	int curLayerNodeCount_G;

	int normalIndex_G_2 = 0;
	int curLayer_G_2;
	int curLayerNodeCount_G_2;

	int normalIndex_E = 0;
	int curLayer_E;
	int curLayerNodeCount_E;

	SceneInfo* sceneInfoAddress;

	float4x4 padding0;
	float4x4 padding1;
	float4x4 padding2;
	float2 padding5;
};

enum class BindingPoints_Octree_FzbPG : uint32_t {
	eVGB = 2,

	eOctreeNodeData_G,
	eOctreeNodeInfo_G,
	eOctreeNodeData_E,

	eBlockInfos_G,
	eBlockInfos_E,
	eHasDataBlockIndices_G,
	eHasDataBlockIndices_E,
	eHasDataBlockCount,

	eGlobalInfo,

	eDivisibleNodeInfos_G,
	eThreadGroupInfos,

	eIndivisibleNodeInfos_G,

	eClusterPairInfo,
	eClusterPairGlobalInfo,
	eCandidateNodeData_E,
};
//------------------------------------------------------------------------------------------
//don't change!!!!!
#define OCTREE_CLUSTER_LAYER_FZBPG 2
static const uint OctreeLayerNodeCount_FzbPG[MAX_OCTREE_LAYER_FZBPG] = { 6, 48, 384, 3072, 24576, 196608 };
static const uint OctreeLayerStartIndex_FzbPG[3] = { 0, 8, 56 };

struct OctreeNodeData_G_FzbPG {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint indivisible;
};

struct OctreeNodeInfo_G_FzbPG {
	uint32_t label_indivisible;
};

struct OctreeNodeData_E_FzbPG {
	float3 radiance;
	float halfAngle;
	float4 meanNormal;
	AABB aabb;
};

struct CandidateNodeData_E_FzbPG {
	AABB aabb;
	float w;
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
	uint indivisibleNodeCount_G;
	OctreeLayerInfo_FzbPG layerInfos_G[MAX_OCTREE_LAYER_FZBPG];
};

struct OctreeThreadGroupInfo_FzbPG {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
};
//------------------------------------------------------------------------------------------
#define OUTGOING_COUNT_FZBPG 64		//not smaller than 8
#define HITTEST_COUNT_FZBPG 8		//not bigger than 32 or smaller than 8

#define OUTGOING_TYPE_FZBPG 0
#if OUTGOING_TYPE_FZBPG == 0
#define getOutgoing_FzbPG fibSpherePoint
#define inverseOutgoing_FzbPG inverseSF
#else 
#define getOutgoing hammersleySpherePoint
#define inverseOutgoing inverseSH
#endif

struct ClusterPairInfo {
	int layerIndex_G;
	int nodeIndex_G;
	int indivisibleNodeLabel;
	int layerIndex_E;
	int nodeIndex_E;
	int candidateNodeIndex;
};
struct ClusterPairGlobalInfo {
	uint clusterPairCount;
};
//------------------------------------------------------------------------------------------

#define CREATEOCTREE_CS_THREADGROUP_SIZE 256

#define GETOCTREELABEL_CS_THREADGROUP_SIZE 1024
#define GETOCTREELABEL4_CS_THREADGROUP_SIZE 512

#define INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE 256
#define GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE 128
#define GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE OUTGOING_COUNT_FZBPG * HITTEST_COUNT_FZBPG

#define OCTREE_RIS_THREADGROUP_SIZE OUTGOING_COUNT_FZBPG

#define GETNEARBYNODES_CS_THREADGROUP_SIZE 512
#define GETNEARBYNODES2_CS_THREADGROUP_SIZE 1024

NAMESPACE_SHADERIO_END()
#endif