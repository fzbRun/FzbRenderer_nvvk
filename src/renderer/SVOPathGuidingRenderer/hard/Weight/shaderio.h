#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/SVOPathGuidingRenderer/hard/Octree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/SVO/shaderio.h"

#ifndef FZBRENDERER_SVOWeight_SHADER_IO_H
#define FZBRENDERER_SVOWeight_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SVOIndivisibleNodeCount_G 1600
#define SVOIndivisibleNodeCount_E OCTREE_NODECOUNT_E

#define HITTEST_COUNT 8
#define OUTGOING_COUNT 64
struct SVOWeightPushConstant {
	uint32_t frameIndex;
	uint32_t currentLayer_E;
	SceneInfo* sceneInfoAddress;
#ifndef NDEBUG
	float3 samplePos;
	float3 outgoing;
	uint instanceIndex;
#endif
};

enum class StaticBindingPoints_SVOWeight {
	eSVO_G = 2,
	eNodeData_E,
	eSVOGlobalInfo,
	eGlobalInfo,
	eSVO_IndivisibleNodeInfos_G,
	eSVO_IndivisibleNodeInfos_E,
	eSVOWeights,
};

struct SVOWeightGlobalInfo {
	DispatchIndirectCommand cmd;
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	uint SVOMaxLayer_G;
#ifndef NDEBUG
	AABB sampelNodeAABB;
	uint sampelNodeLabel;
#endif
	SVOLayerInfo layerInfos_G[MAX_SVO_LAYER];
};
struct SVOIndivisibleNodeInfo {
	uint32_t layerIndex;
	uint32_t nodeIndex;
};

#define GETINDIVISIBLENODEINFO_CS_THREADGROUP_SIZE 512
#define INITWEIGHT_CS_THREADGROUP_SIZE 512
#define GETWEIGHT_CS_THREADGROUP_SIZE 512
#define GETPROBABILITY_CS_THREADGROUP_SIZE 512

NAMESPACE_SHADERIO_END()
#endif