#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_OCTREE_SVOPG_SHADERIO_H
#define FZBRENDERER_OCTREE_SVOPG_SHADERIO_H

#define MAX_OCTREE_LAYER 8

NAMESPACE_SHADERIO_BEGIN()

enum class BindingPoints_Octree_SVOPG : uint32_t {
	eVGB = 0,
	eVGBMaterialInfos,
	eOctreeArray_G,
	eOctreeArray_E,
	eNodeData_E,
	eBlockInfos_G,
	eBlockInfos_E,
	eHasDataBlockIndices_G,
	eHasDataBlockIndices_E,
	eHasDataBlockCount,
	eGlobalInfo,
	eWireframeMap,
	eBaseMap,
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
	uint materialCountSum;
	uint materialCounts[MAX_MATERIAL_COUNT];
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
struct OctreeGlobalInfo {
	DispatchIndirectCommand cmd;
};

#define INITOCTREE_CS_THREADGROUP_SIZE 1024
#define CREATEOCTREE_CS_THREADGROUP_SIZE 256

NAMESPACE_SHADERIO_END()

#endif