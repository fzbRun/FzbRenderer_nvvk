#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_OCTREE_SHADERIO_H
#define FZBRENDERER_OCTREE_SHADERIO_H

NAMESPACE_SHADERIO_BEGIN()
enum BindingPoints_Octree {
	eVGB_Octree = 0,
	eOctreeArray_G_Octree,
	eOctreeArray_E_Octree,
	eWireframeMap_Octree,
	eBaseMap_Octree,
};

struct OctreePushConstant {
	uint32_t maxDepth;
	uint32_t currentDepth;
	float2 entropyThreshold;
	float irradianceRelRatioThreshold;
	SceneInfo* sceneInfoAddress;
#ifndef NDEBUG
	float frameIndex;
	uint32_t clusteringLevel;
	uint32_t showOctreeNodeTotalCount;
	float4 VGBStartPos_Size;
#endif
};

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