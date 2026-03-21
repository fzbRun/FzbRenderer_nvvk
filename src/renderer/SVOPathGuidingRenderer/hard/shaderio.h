#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/SceneDivision/Octree/shaderio.h"

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SVO_PATHGUIDING_THREADGROUP_SIZE_X 16
#define SVO_PATHGUIDING_THREADGROUP_SIZE_Y 16
struct SVOPathGuidingPushConstant
{
	float4 VGBStartPos_Size;
	int frameIndex = 0;
	int maxDepth = 3;
	float time;
	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;
};

enum StaticBindingPoints_SVOPG
{
	//eTextures = 0,
	//eOutImage = 1,	//First save the result of lightInject, and finally MIS
	eVGB_SVOPG = 2,
	eSVO_G_SVOPG,
	eSVO_E_SVOPG,
	eSVOLayerInfos_G_SVOPG,
	eSVOLayerInfos_E_SVOPG,
	eGlobalInfo_SVOPG,
	eWeights_SVOPG,
#ifndef NDEBUG
	eDepthImage_SVOPG,
#endif
};
enum DynamicBindingPoints_SVOPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};

struct GlobalInfo_SOVPG {
	uint SVOMaxLayer_G;
	uint SVOMaxLayer_E;
	uint indivisibleNodeCount_G;
	uint totalNodeCount_E;
};
struct SVONodeInfo_G_SVOPG {
	uint label;
	uint indivisible;
	AABB aabb;
};
struct SVONodeInfo_E_SVOPG {
	uint label;
	uint indivisible;
	AABB aabb;
};

//-------------------------------------------SVOWeight----------------------------------------
#define WEIGHT_HITTEST_COUNT 8
#define HITTEST_COUNT 8
#define OUTGOING_COUNT 64
struct SVOWeightPushConstant {
	uint32_t frameIndex;
	uint32_t layerNodeMaxCount_E;
	SceneInfo* sceneInfoAddress;
	uint32_t countdown;
#ifndef NDEBUG
	float3 samplePos;
	float3 outgoing;
	uint instanceIndex;
#endif
};

enum StaticBindingPoints_SVOWeight {
	eSVO_G_SVOWeight = 2,
	eSVO_E_SVOWeight,
	eSVOLayerInfos_G_SVOWeight,
	eSVOLayerInfos_E_SVOWeight,
	eGlobalInfo_SVOWeight,
	eSVO_IndivisibleNodeInfos_G_SVOWeight,
	eSVO_IndivisibleNodeInfos_E_SVOWeight,
	eSVOWeights_SVOWeight,
	eSVOWeightSums_SVOWeight,
};

struct SVOWeightGlobalInfo {
	DispatchIndirectCommand cmd;
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	uint SVOMaxLayer_G;
	uint SVOMaxLayer_E;
	uint totalNodeCount_E;
#ifndef NDEBUG
	AABB sampelNodeAABB;
	uint sampelNodeLabel;
#endif
};
struct SVOIndivisibleNodeInfo {
	uint32_t layerIndex;
	uint32_t nodeIndex;
};
struct IndivisibleNodeData_E {
	float3 normal;
	uint nodeIndex;
	float irradiance;
	AABB aabb;
};
struct IndivisibleNodeData_G {
	float3 normal;
	uint nodeLabel;
	AABB aabb;
	uint materialType;
};

NAMESPACE_SHADERIO_END()
#endif