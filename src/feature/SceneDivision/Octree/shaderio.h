#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_OCTREE_SHADERIO_H
#define FZBRENDERER_OCTREE_SHADERIO_H

#define MAX_OCTREE_DEPTH 8

NAMESPACE_SHADERIO_BEGIN()
enum BindingPoints_Octree {
	eVGB_Octree = 0,
	eOctreeArray_G_Octree,
	eOctreeArray_E_Octree,
	eWireframeMap_Octree,
	eBaseMap_Octree,
};

struct OctreePushConstant {
	float2 entropyThreshold;
	uint32_t maxDepth;
	uint32_t currentDepth;
	float voxelVolume;
	float irradianceRelRatioThreshold;
	uint32_t OctreeNodeTotalCount;
	uint32_t currentLayerNodeCount;
	SceneInfo* sceneInfoAddress;
#ifndef NDEBUG
	float4 VGBStartPos_Size;
	float frameIndex;
	uint32_t clusteringLevel;
	uint32_t showOctreeNodeTotalCount;
	int normalIndex;
#endif
};

#define MAX_MATERIAL_TYPE_COUNT 5
struct OctreeNodeData_G {
	float4 meanNormal;
	AABB aabb;
	uint32_t indivisible;
	uint32_t label;
	uint32_t materialIndex;
};
struct OctreeNodeData_E {
	float3 irradiance;
	float notIgnoreRatio;
	float4 meanNormal;
	AABB aabb;
	uint32_t indivisible;
	uint32_t label;
};

NAMESPACE_SHADERIO_END()

#endif