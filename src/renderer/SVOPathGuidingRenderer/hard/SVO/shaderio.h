#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/SVOPathGuidingRenderer/hard/Octree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

#ifndef FZBRENDERER_SVO_SVOPG_SHADER_IO_H
#define FZBRENDERER_SVO_SVOPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SVOSize_G 2000
#if SVOSize_G > OCTREE_NODECOUNT_E
#define SVOSize SVOSize_G
#else
#define SVOSize OCTREE_NODECOUNT_E
#endif
#define MAX_SVO_LAYER MAX_OCTREE_LAYER + 1

enum class BindingPoints_SVOPG {
	eOctreeArray_G_SVOPG = 0,
	eSVO_G_SVOPG,
	eSVOGlobalInfo_SVOPG,
	eSVODivisibleNodeIndices_G_SVOPG,
	eSVOThreadGroupInfos_SVOPG
};

typedef OctreeNodeData_E SVONodeData_E;

#define NEARBY_NODE_COUNT 4	//must 2 exponent
struct SVONodeData_G {
	float4 meanNormal;
	AABB aabb;
	uint32_t indivisible;
	uint32_t label;

	#ifdef CLUSTER_WITH_MATERIAL
	uint32_t materialIndex;
	#endif
	int nearbyNodeIndices[NEARBY_NODE_COUNT];
};

struct SVOPushConstant_SVOPG {
	uint32_t svoMaxLayer;		//Octree max layerIndex, 32x32x32 is 5
	uint32_t currentLayer;
#ifndef NDEBUG
	float frameIndex;
	SceneInfo* sceneInfoAddress;
#endif
};

struct SVOLayerInfo {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};

struct SVOThreadGroupInfo {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
	uint threadGroupDivisibleNodeCount_E;
	uint threadGroupIndivisibleNodeCount_E;
};

struct SVOGlobalInfo_SVOPG {
	DispatchIndirectCommand cmd;	//CS Dispatch size
#ifndef NDEBUG
	DrawIndexedIndirectCommand drawCmd;
#endif
	uint32_t totalNodeCount_G;
	SVOLayerInfo layerInfos_G[MAX_SVO_LAYER];
};

#define INITSVO_CS_THREADGROUP_SIZE 512
#define CREATESVO_CS_THREADGROUP_SIZE 512

NAMESPACE_SHADERIO_END()
#endif