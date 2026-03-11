#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct SVOPathGuidingPushConstant
{
	float4 VGBStartPos_Size;
	int frameIndex = 0;
	int maxDepth = 3;
	float time;
	float3x3 normalMatrix;
	SceneInfo* sceneInfoAddress;
};

enum StaticBindingPoints_SVOPG
{
	//eTextures = 0,
	//eOutImage = 1,	//First save the result of lightInject, and finally MIS
	eVGB_SVOPG = 2,
	eSVO_G_SVOPG,
	eSVO_E_SVOPG,
	eWeights_SVOPG,
	eOutImage_MIS_SVOPG,
	eDepthImage_SVOPG,
};
enum DynamicBindingPoints_SVOPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};

//-------------------------------------------SVOWeight----------------------------------------
#define WEIGHT_HITTEST_COUNT 8
#define HITTEST_COUNT 32
#define OUTGOING_COUNT 32
struct SVOWeightPushConstant {
	uint32_t frameIndex;
	uint32_t sizes[8];
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
	eSVOWeightSampleCounts_SVOWeight,
	eSVOWeights_SVOWeight,
	eSVOWeightSums_SVOWeight,
};
struct DispatchIndirectCommand {
	uint32_t    x;
	uint32_t    y;
	uint32_t    z;
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
struct HitTestResult {
	float V_Cosine;
	uint32_t layerIndex;
	uint32_t nodeIndex;
};

NAMESPACE_SHADERIO_END()
#endif