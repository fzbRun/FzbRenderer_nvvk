#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/VRPG2/VRPG2Shaderio.h"

#ifndef FZBRENDERER_Geometry_Clustering_VRPG_SHADERIO_H
#define FZBRENDERER_Geometry_Clustering_VRPG_SHADERIO_H

NAMESPACE_SHADERIO_BEGIN()

struct GeometryClusteringPushConstant_VRPG {
	int frameIndex;
	int randomFrameIndex = 0;

	uint32_t voxelTotalCount;      // geometry VGB
	float voxelVolume;
	float4 VGBStartPos_Size;          // geometry VGB
	float4 voxelSize;              // geometry VGB

	uint32_t octreeMaxLayer;
	uint32_t octreeNodeTotalCount;

	uint32_t currentLayer;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;

	SceneInfo* sceneInfoAddress;

	float clusteringThreshold;

	float4x4 padding1;
	float4x4 padding2;
	float4 padding3;
	float4 padding4;
	float2 padding5;
};

struct GeometryClusteringDebugPushConstant_VRPG {
	int frameIndex;

	int mode = 0;
	int maxLayer;

	int normalIndex_G = 0;
	int curLayer_G;
	int curLayerNodeCount_G;

	int normalIndex_G_2 = 0;
	int curLayer_G_2;
	int curLayerNodeCount_G_2;

	SceneInfo* sceneInfoAddress;
};
//------------------------------------------------------------------------------------------
//------------------------------------------------------------------------------------------
// Debug GBuffer of the geometry clustering (VRPG: GBuffers_Octree_VRPG). Only the geometry
// octree and the SVO are drawn here; the light tree lives in the LightClustering module.
enum class GBuffers_GeometryClustering_VRPG : uint32_t {
	eOctree = 0,
	eSVO,
	eBufferCount,
};
//------------------------------------------------------------------------------------------
enum class GeometryClusteringBindingPoints_VRPG : uint32_t {
	eVGB = 0,
	eGlobalInfo,

	eClusteringGlobalInfo,

	eClusterData_Octree,
	eClusterInfo_Octree,

	eClusterBlockInfos,
	eHasDataClusterBlockIndices,

	eClusterInfo_SVO,

	eSVODivisibleNodeInfos,
	eSVOThreadGroupInfos,

	eNearbyClusterTempInfos,
	eNearbyClusterInfos,
};
//--------------------------------------Octree----------------------------------------------------
struct ClusterData_Octree_VRPG {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint indivisible;
};

struct ClusterInfo_Octree_VRPG {
	uint32_t label_indivisible;
};
//--------------------------------------Global----------------------------------------------------
struct ClusteringGlobalInfo_VRPG {
	uint hasDataClusterBlockCount;
	OctreeLayerInfo_VRPG2 layerInfos_G[MAX_OCTREE_LAYER_VRPG2];
};
//-------------------------------------CreateOctree-----------------------------------------------------
struct ClusterBlockInfo {
	int hasData;
};
struct ClusterBlockIndex {
	int blockIndex;
};
//-------------------------------------SVO-----------------------------------------------------
#define ClusterInfo_SVO_VRPG OctreeNodeInfo_VRPG2
//-------------------------------------CreateSVO-----------------------------------------------------
struct SVODivisibleNodeInfo {
	int nodeIndex;
};

struct SVOThreadGroupInfo_VRPG {
	uint threadGroupDivisibleNodeCount;
	uint threadGroupIndivisibleNodeCount;
};
//-------------------------------------NearbyCluster-----------------------------------------------------
struct NearbyClusterTempInfo_VRPG {
	OctreeNodeInfo_VRPG2 nearbyClusterInfos[VRPG_NEARBY_CLUSTER_COUNT];
	float nearbyClusterDistances[VRPG_NEARBY_CLUSTER_COUNT];
};
struct NearbyClusterInfo_VRPG {
	OctreeNodeInfo_VRPG2 nearbyClusterInfos[VRPG_NEARBY_CLUSTER_COUNT];
};
//------------------------------------------------------------------------------------------
#define Clustering_CS_THREADGROUP_SIZE 256
#define CreateSVO_CS_THREADGROUP_SIZE 1024
#define GetNearbyCluster_CS_THREADGROUP_SIZE 512
#define GetNearbyCluster2_CS_THREADGROUP_SIZE 1024

NAMESPACE_SHADERIO_END()
#endif
