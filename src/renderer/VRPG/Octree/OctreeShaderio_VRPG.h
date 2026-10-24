#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/VRPG/VRPGShaderio.h"

#ifndef FZBRENDERER_OCTREE_2_VRPG_SHADERIO_H
#define FZBRENDERER_OCTREE_2_VRPG_SHADERIO_H

#define MAX_OCTREE_LAYER_VRPG 8
#define IndivisibleNodeCount_G_VRPG 4000

NAMESPACE_SHADERIO_BEGIN()

struct OctreePushConstant_VRPG {
	int frameIndex;

	int candidateSampleBufferIndex = 0;

	uint32_t VGBVoxelTotalCount;      // geometry VGB
	float voxelVolume;
	float4 VGBStartPos_Size;          // geometry VGB
	float4 VGBVoxelSize;              // geometry VGB

	float4 lightVGBStartPos_Size;     // light VGB, w = light voxel count
	float4 lightVGBVoxelSize;         // light VGB

	float3x3 randomRotateMatrix;

	uint32_t octreeMaxLayer;
	uint32_t lightOctreeMaxLayer;
	uint32_t octreeNodeTotalCount;

	uint32_t currentLayer;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;
	int temporalReuse;
	float temporalBlend;

	SceneInfo* sceneInfoAddress;

	float clusterNormalThreshold;
	float clusterFillRateThreshold;
	float clusterVolumeTolerance;
	int lightBackfaceCull;
	int stableCandidateCut;

	float4x4 padding1;
};

struct OctreeDebugPushConstant_VRPG {
	int frameIndex;

	int mode = 0;
	int maxLayer;

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

struct ClusterPairDebugGlobalInfo_VRPG {
	int clusterIndex_G;
};
struct ClusterPairDebugPushConstant_VRPG {
	int frameIndex;
	int mode = 0;
	int mode2 = 0;
	int spatialReuse = 0;
	
	uint32_t octreeMaxLayer;
	uint32_t VGBVoxelTotalCount;
	float4 VGBStartPos_Size;
	float4 VGBVoxelSize;
	
	uint2 pixelIndex;
	uint2 screenSize;

	int outgoingIndex = 0;
	float padding;
	
	SceneInfo* sceneInfoAddress;
	ClusterPairDebugGlobalInfo_VRPG* globalInfoAddress;

	float4x4 padding0;
	float4x4 padding1;
	float4 padding2;
	float4 padding3;
};

enum class BindingPoints_Octree_VRPG : uint32_t {
	eVGB = 2,
	eLightVGB,

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

	eAdjClusterInfo_G,

	eClusterPairInfo_VRPG,
	eClusterPairGlobalInfo_VRPG,
	eClusetPairHitTestInfo,

	eCandidateSubTreeInfo_VRPG,
	eCandidateLeafNodeData_VRPG,

	eCandidateSampleData,
	eCandidateSamplePrefixWSum,

	eCandidateSampleIndex,

	eCandidateSampleAABB,
	eCandidateSampleData2,

	eNearbyNodeTempInfos,
	eNearbyNodeInfos,
	eCandidateSampleDataHistory,
	eSpatialReservoir,
	eReservoirCurrent,
	eReservoirHistory,
};
//------------------------------------------------------------------------------------------
#define MAX_ADJACENT_CLUSTER_COUNT 4

struct AdjClusterTempInfo_VRPG {
	uint nodeLabel;
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT_VRPG];
	float nearbyNodeDistances[NEARBY_NODE_COUNT_VRPG];
};
//------------------------------------------------------------------------------------------
//don't change!!!!!
#define OCTREE_CLUSTER_LAYER_VRPG 2
static const uint OctreeLayerNodeCount_VRPG[MAX_OCTREE_LAYER_VRPG] = { 6, 48, 384, 3072, 24576, 196608, 1572864 };
static const uint OctreeLayerStartIndex_VRPG[3] = { 0, 8, 56 };

struct OctreeNodeData_G_VRPG {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint indivisible;
};

struct OctreeNodeInfo_G_VRPG {
	uint32_t label_indivisible;
};

struct OctreeNodeData_E_VRPG {
	float3 radiance;
	float halfAngle;
	float4 meanNormal;
	AABB aabb;
};

//------------------------------------------------------------------------------------------
struct CandidateSubTreeInfo_VRPG {
	float ESum;
	int subTreeNodeIndex_layer;
};
struct CandidateLeafNodeData_VRPG {
	AABB aabb;
	float3 radiance;
#ifdef SpatialReuse_VRPG
	float3 normal;
#endif
};
//------------------------------------------------------------------------------------------
struct CandidateSampleData_VRPG {
	float w;
	float p_hat;
};

// One reservoir per (geometric cluster, outgoing direction).
struct ReservoirSample_VRPG {
	AABB aabb;
	float weight;       // total reservoir weight W
	float p_hat;        // target density estimate of the selected sample
	int normalIndex;    // light-tree normal partition of the selected sample
	int padding;
};

//------------------------------------------------------------------------------------------
struct HasDataOctreeBlockCount_VRPG {
	uint32_t count_G;
	uint32_t count_E;
};

struct OctreeLayerInfo_VRPG {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};

struct OctreeGlobalInfo_VRPG {
	DispatchIndirectCommand cmd;
	uint indivisibleNodeCount_G;
	OctreeLayerInfo_VRPG layerInfos_G[MAX_OCTREE_LAYER_VRPG];
};

struct OctreeThreadGroupInfo_VRPG {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
};
//------------------------------------------------------------------------------------------
#define OUTGOING_COUNT_VRPG 32		//not smaller than 8
#define HITTEST_COUNT_VRPG 8		//not bigger than 32 or smaller than 8

#define OUTGOING_TYPE_VRPG 0
#if OUTGOING_TYPE_VRPG == 0
#define getOutgoing_VRPG fibSpherePoint
#define inverseOutgoing_VRPG inverseSF
#else 
#define getOutgoing hammersleySpherePoint
#define inverseOutgoing inverseSH
#endif

struct ClusterPairInfo_VRPG {
	int layerIndex_G;
	int nodeIndex_G;
	int indivisibleNodeLabel;
	int layerIndex_E;
	int nodeIndex_E;
	int candidateNodeIndex;

	int leafNodeIndex_E;
	float leafNodePdf;
};
struct ClusterPairGlobalInfo_VRPG {
	uint clusterPairCount;
};
struct ClusterPairHitInfo_VRPG {
	float3 pos_G;
	float isHit;
	float3 normal_G;
	int materialIndex;
	float3 sampleDir;
	float d;
	float cosine;
	float cosine_Light;
};
//------------------------------------------------------------------------------------------
struct IndivisibleNodeNearbyNodeTempInfo_VRPG {
	uint nodeLabel;
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT_VRPG];
	float nearbyNodeDistances[NEARBY_NODE_COUNT_VRPG];
};

struct OctreeNearbyNodeInfo_VRPG {
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT_VRPG];		//0-3 is layer, 4 - 31 is nodeIndex
};
//------------------------------------------------------------------------------------------
#define CREATEOCTREE_CS_THREADGROUP_SIZE 256

#define GETOCTREELABEL_CS_THREADGROUP_SIZE 1024
#define GETOCTREELABEL4_CS_THREADGROUP_SIZE 512

#define INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE 256
#define GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE 128
#define CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE 256
#define GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE OUTGOING_COUNT_VRPG * HITTEST_COUNT_VRPG

#define OCTREE_RIS_THREADGROUP_SIZE OUTGOING_COUNT_VRPG

#define GETNEARBYNODES_CS_THREADGROUP_SIZE 512
#define GETNEARBYNODES2_CS_THREADGROUP_SIZE 1024

NAMESPACE_SHADERIO_END()
#endif