#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

#ifndef FZBRENDERER_OCTREE_SVOPG_SHADERIO_H
#define FZBRENDERER_OCTREE_SVOPG_SHADERIO_H

#define MAX_OCTREE_LAYER 8
#define IndivisibleNodeCount_G 1024

NAMESPACE_SHADERIO_BEGIN()

enum class BindingPoints_Octree_SVOPG : uint32_t {
	eVGB = 0,

	#ifdef CLUSTER_WITH_MATERIAL
	eVGBMaterialInfos,
	#endif

	eOctreeArray_G,
	eOctreeArray_E,
	eNodeData_E,
	eBlockInfos_G,
	eBlockInfos_E,
	eHasDataBlockIndices_G,
	eHasDataBlockIndices_E,
	eHasDataBlockCount,
	eGlobalInfo,
#ifndef USE_SVO
	eDivisibleNodeInfos_G,
	eThreadGroupInfos,

	eIndivisibleNodeInfos_G,
	eIndivisibleNodeInfos_E,
#endif
};

struct OctreePushConstant_SVOPG {
	uint32_t octreeMaxLayer;
	uint32_t currentLayer;
	float voxelVolume;
	uint32_t octreeNodeTotalCount;
	uint32_t currentLayerBlockCount;
	uint32_t currentLayerNodeCount;
	uint32_t VGBVoxelTotalCount;
	SceneInfo* sceneInfoAddress;
	float4 VGBStartPos_Size;
	float4 VGBVoxelSize;
#ifndef NDEBUG
	float frameIndex;
	uint32_t showOctreeNodeTotalCount;
	int normalIndex;
#endif
};

#define OCTREE_CLUSTER_LAYER 2
#define OCTREE_NODECOUNT_E 440		//8 + 48 + 384
#define NODECOUNT_E 384
static const uint OctreeLayerInfo_E[3] = { 8, 48, 384 };
static const uint OctreeLayerStartIndex_E[3] = { 0, 8, 56 };
struct OctreeNodeData_G {
	float4 meanNormal;
	AABB aabb;
	float fillRate;
	uint32_t indivisible;
	uint32_t label;

	#ifdef CLUSTER_WITH_MATERIAL
	uint materialCountSum;
	uint materialCounts[MAX_MATERIAL_COUNT];
	#endif

	#ifndef USE_SVO
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT];
	#endif
};
struct OctreeNodeData_E {
	float E;
	float pdf;
	float4 meanNormal;
	AABB aabb;
	uint32_t indivisible;
};

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
#ifndef USE_SVO
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	OctreeLayerInfo layerInfos_G[MAX_OCTREE_LAYER];
#endif
};

struct OctreeThreadGroupInfo {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
};

#define INITOCTREE_CS_THREADGROUP_SIZE 1024
#define CREATEOCTREE_CS_THREADGROUP_SIZE 256

#define GETOCTREELABEL_CS_THREADGROUP_SIZE 1024
#define GETINDIVISIBLENODEINFOS_CS_THREADGROUP_SIZE 512

NAMESPACE_SHADERIO_END()

#endif