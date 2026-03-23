#pragma once

#include <common/Shader/shaderStructType.h>
#include "../Octree/shaderio.h"

#ifndef FZBRENDERER_SVO_SHADERIO_H
#define FZBRENDERER_SVO_SHADERIO_H

#define THREADGROUP_SIZE 512u
#define THREADGROUP_SIZE2 256u
#define WARP_SIZE 32u
#define SVONodeCount_E_Layer1 8
#define SVONodeCount_E_Layer2 64
#define SVONodeCount_E_Layer3 512
#define SVONodeCount_E_FirstLayer3 SVONodeCount_E_Layer1 + SVONodeCount_E_Layer2 + SVONodeCount_E_Layer3
#define SVONodeCount_G_Layer1 8
#define SVONodeCount_G_Layer2 64
#define SVONodeCount_G_Layer3 512
#define SVONodeCount_G_FirstLayer3 SVONodeCount_G_Layer1 + SVONodeCount_G_Layer2 + SVONodeCount_G_Layer3

NAMESPACE_SHADERIO_BEGIN()

enum BindingPoints_SVO {
	eOctreeArray_G_SVO = 0,
	eOctreeArray_E_SVO,
	eSVOArray_G_SVO,
	eSVOArray_E_SVO,
	eSVOLayerInfos_G_SVO,
	eSVOLayerInfos_E_SVO,
	eSVOGlobalInfo_SVO,
	eSVODivisibleNodeIndices_G_SVO,
	eSVODivisibleNodeIndices_E_SVO,
	eSVOThreadGroupInfos_SVO
};

struct SVOPushConstant {
	uint32_t maxDepth;;
	uint32_t currentDepth;
	uint32_t sizes_G[MAX_OCTREE_DEPTH];
	uint32_t sizes_E[MAX_OCTREE_DEPTH];
#ifndef NDEBUG
	float frameIndex;
	SceneInfo* sceneInfoAddress;
#endif
};

typedef OctreeNodeData_G SVONodeData_G;
typedef OctreeNodeData_E SVONodeData_E;

struct SVOGloablInfo_SVO {
	DispatchIndirectCommand cmd;	//CS Dispatch size
#ifndef NDEBUG
	DrawIndexedIndirectCommand drawCmd;
	uint32_t totalNodeCount_G;
	uint32_t totalNodeCount_E;
#endif
};

struct SVOThreadGroupInfo {
	uint threadGroupDivisibleNodeCount_G;
	uint threadGroupIndivisibleNodeCount_G;
	uint threadGroupDivisibleNodeCount_E;
	uint threadGroupIndivisibleNodeCount_E;
};

struct SVOLayerInfo {
	uint32_t divisibleNodeCount;
	uint32_t indivisibleNodeCount;
};

NAMESPACE_SHADERIO_END()

#endif