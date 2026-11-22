#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"

#ifndef FZBRENDERER_VRPG2_RIS_SHADER_IO_H
#define FZBRENDERER_VRPG2_RIS_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define OUTGOING_COUNT_VRPG2 32		// number of outgoing direction bins per cluster
#define HITTEST_COUNT_VRPG2 8		// visibility samples per candidate
#define LIGHTCUT_RELATIVE_ERROR_VRPG2 0.01f
#define LIGHTCUT_MAX_ITERATION_VRPG2 20

#define RIS_INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE 256
#define RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE 128
#define RIS_GET_CANDIDATE_LEAF_NODES_CS_THREADGROUP_SIZE 256
#define RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE 256
#define RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE (OUTGOING_COUNT_VRPG2 * HITTEST_COUNT_VRPG2)
#define RIS_PREFIX_WSUM_CS_THREADGROUP_SIZE 32

#define OUTGOING_TYPE_VRPG2 0
#if OUTGOING_TYPE_VRPG2 == 0
#define getOutgoing_VRPG2 fibSpherePoint
#define inverseOutgoing_VRPG2 inverseSF
#endif

struct ClusterPairDebugGlobalInfo_VRPG2 {
	int clusterIndex_G;
	int candidateNodeCounts[MAX_OCTREE_LAYER_VRPG2];
	OctreeNodeInfo_VRPG2 itertionResult[MAX_OCTREE_LAYER_VRPG2][Candidate_Samples_Count];
};
//------------------------------------------------------------
struct RISPushConstant_VRPG2 {
	int frameIndex;
	int randomFrameIndex = 0;

	int candidateSampleBufferIndex = 0;
	int iterationIndex = 0;

	uint32_t lightOctreeMaxLayer;
	float3x3 randomRotateMatrix;

	SceneInfo* sceneInfoAddress;
	ClusterPairDebugGlobalInfo_VRPG2* globalInfoAddress;
};

enum class RISBindingPoints_VRPG2 : uint32_t {
	eVGB = 2,

	eClusterData_Octree,
	eClusterInfo_Octree,
	eLightNodeData_Octree,

	eClusterInfo_SVO,

	eGlobalInfo,
	eRISGlobalInfo,

	eClusterPairInfo,
	eClusterPairHitInfo,

	eCandidateSubTreeInfo,
	eCandidateLeafNodeData,

	eCandidateSampleData,
	eCandidateSamplePrefixWSum,

	eCandidatePoolInfo,		// candidate count per cluster (both modes)

#if VRPG_Candidate_Samples_SampleMode == 1
	eCandidatePoolNodeInfo,	// pool contents (mode 1 only)
#endif

	eNearbyClusterInfos,	// debug pass only (VRPG: BindingPoints_Octree_VRPG::eNearbyNodeInfos)
};

//------------------------------------------------------------
// one candidate (a node of the light tree cut) of a geometric cluster
struct ClusterPairInfo_VRPG2 {
	int layerIndex_G;
	int nodeIndex_G;
	int clusterIndex;
	int layerIndex_E;
	int nodeIndex_E;
	int candidateNodeIndex;

	int leafNodeIndex_E;
	float leafNodePdf;
};

struct RISGlobalInfo_VRPG2 {
	uint32_t clusterPairCount;
};

struct ClusterPairHitInfo_VRPG2 {
	float3 pos_G;
	float isHit;
	float3 normal_G;
	int materialIndex;
	float3 sampleDir;
	float d;
	float cosine;
	float cosine_Light;
};

//------------------------------------------------------------
// light leaf reached by descending from a cut node, plus the probability of that descent
struct CandidateSubTreeInfo_VRPG2 {
	float ESum;
	int subTreeNodeIndex_layer;
};

struct CandidateLeafNodeData_VRPG2 {
	AABB aabb;
	float3 radiance;
};

//------------------------------------------------------------
// per (outgoing bin, cluster, candidate)
struct CandidateSampleData_VRPG2 {
	float w;
	float p_hat;
};

//------------------------------------------------------------
// pool based light cut (mode 1)
struct CandidatePoolInfo_VRPG2 {
	int candidateNodeCount;
};

// Same layout as VRPG's CandidatePoolNodeInfo_VRPG: a candidate that fails the visibility test is
// only marked invalid and stays in the pool until the compaction pass drops it.
struct CandidatePoolNodeInfo_VRPG2 {
	int valid;
	int layer_E;		// -1 for the root
	int nodeIndex_E;
};

//------------------------------------------------------------
// greedy light cut (mode 0)
struct LightCutNode_VRPG2 {
	int layerIndex;
	int nodeIndex;
	float error;
	float I;
};

//------------------------------------------------------------
// Cluster pair debug pass, ported from VRPG (OctreeShaderio_VRPG.h /
// Octree_VRPG::clusterPairDebug). The global info holds the geometric cluster that was picked by
// the debug compute shader from the pixel under the cursor.

struct ClusterPairDebugPushConstant_VRPG2 {
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

	int iteration;
	int normalIndex;
	int padding;

	SceneInfo* sceneInfoAddress;
	ClusterPairDebugGlobalInfo_VRPG2* globalInfoAddress;
};

// Debug GBuffer of the cluster pair pass (VRPG: GBuffers_Octree_VRPG::eClusterPairResult).


NAMESPACE_SHADERIO_END()
#endif