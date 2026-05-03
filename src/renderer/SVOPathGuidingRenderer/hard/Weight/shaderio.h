#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/Octree/shaderio.h"
#ifdef USE_SVO
#include "renderer/SVOPathGuidingRenderer/hard/SVO/shaderio.h"
#endif

#ifndef FZBRENDERER_SVOWeight_SHADER_IO_H
#define FZBRENDERER_SVOWeight_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define HITTEST_COUNT 16
#define OUTGOING_COUNT 64

#define OUTGOING_TYPE 0
#if OUTGOING_TYPE == 0
#define getOutgoing fibSpherePoint
#define inverseOutgoing inverseSF
#else 
#define getOutgoing hammersleySpherePoint
#define inverseOutgoing inverseSH
#endif

struct SVOWeightPushConstant {
	float3x3 randomRotateMatrix;
	uint32_t frameIndex;
	uint32_t currentLayer_E;
	SceneInfo* sceneInfoAddress;
#ifndef NDEBUG
	float3 samplePos;
	float3 outgoing;
	uint instanceIndex;
	int sampleNodeLabel;
#endif
};

enum class StaticBindingPoints_SVOWeight {
#ifdef USE_SVO
	eSVO_G = 2,
#else
	eOctreeArray_G = 2,
#endif
	eNodeData_E,
	eTreeGlobalInfo,
	eGlobalInfo,
	eIndivisibleNodeInfos_G,
	eIndivisibleNodeInfos_E,
	eSVOWeights,
	eIndivisibleNodeNearbyNodeTempInfos,
#ifndef USE_SVO
	eOctreeNearbyNodeInofs,
#endif
};

struct SVOWeightGlobalInfo {
	DispatchIndirectCommand cmd;
	DispatchIndirectCommand cmd2;
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
#ifndef NDEBUG
	AABB sampelNodeAABB;
	uint sampelNodeLabel;
#endif

#ifdef USE_SVO
	uint SVOMaxLayer_G;
	SVOLayerInfo layerInfos_G[MAX_SVO_LAYER];
#else
	OctreeLayerInfo layerInfos_G[MAX_OCTREE_LAYER];
#endif
};
struct SVOIndivisibleNodeInfo {
	uint32_t layerIndex;
	uint32_t nodeIndex;
};

struct IndivisibleNodeNearbyNodeTempInfo {
#ifdef USE_SVO
	int nearbyNodeIndices[NEARBY_NODE_COUNT];
#else
	uint nodeLabel;
	int2 nearbyNodeInfos[NEARBY_NODE_COUNT];
#endif
	float nearbyNodeDistances[NEARBY_NODE_COUNT];
};

#define INITWEIGHT_CS_THREADGROUP_SIZE 512
#define GETWEIGHT_CS_THREADGROUP_SIZE 512
#define GETPROBABILITY_CS_THREADGROUP_SIZE 512
#define GETNEARBYNODES_CS_THREADGROUP_SIZE 256
#define GETNEARBYNODES2_CS_THREADGROUP_SIZE 1024

NAMESPACE_SHADERIO_END()
#endif