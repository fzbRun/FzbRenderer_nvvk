#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"

#ifndef FZBRENDERER_Light_Clustering_VRPG_SHADERIO_H
#define FZBRENDERER_Light_Clustering_VRPG_SHADERIO_H

NAMESPACE_SHADERIO_BEGIN()

struct LightClusteringPushConstant_VRPG {
	int frameIndex;
	int randomFrameIndex = 0;

	uint32_t nodeTotalCount;		//per normal
	float4 nodeGridStartPos_Size;
	float4 nodeSize;

	uint32_t currentLayer;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;

	SceneInfo* sceneInfoAddress;
};
//------------------------------------------------------------------------------------------
enum class LightClusteringBindingPoints_VRPG : uint32_t {
	eGlobalInfo = 0,

	eClusteringGlobalInfo,

	eLightNodeData_Octree,

	eLightNodeBlockInfos,
	eHasDataLightNodeBlockIndices,
};
//------------------------------------------------------------------------------------------
struct LightNodeData_Octree_VRPG {
	float3 radiance;
	float halfAngle;
	float4 meanNormal;
	AABB aabb;
};
//------------------------------------------------------------------------------------------
struct LightClusteringGlobalInfo_VRPG {
	uint hasDataLightNodeBlockCount;
};
//------------------------------------------------------------------------------------------
#define LightNodeBlockInfo ClusterBlockInfo
#define LightNodeBlockIndex ClusterBlockIndex
//------------------------------------------------------------------------------------------
#define LightClustering_CS_THREADGROUP_SIZE 256

//------------------------------------------------------------------------------------------
// Debug GBuffer of the light clustering (VRPG: GBuffers_Octree_VRPG::eLightClusterResult).
enum class GBuffers_LightClustering_VRPG : uint32_t {
	eLightTree = 0,
	eBufferCount,
};
//------------------------------------------------------------------------------------------
// Push constants of the light tree debug pass (VRPG: OctreeDebugPushConstant_VRPG, mode 1).
struct LightClusteringDebugPushConstant_VRPG {
	int frameIndex = 0;
	int mode = 0;
	int maxLayer = 0;

	int normalIndex_E = 0;
	int curLayer_E = 0;
	int curLayerNodeCount_E = 0;

	SceneInfo* sceneInfoAddress;
};

NAMESPACE_SHADERIO_END()
#endif