#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SVO_PATHGUIDING_THREADGROUP_SIZE_X 16
#define SVO_PATHGUIDING_THREADGROUP_SIZE_Y 16
struct SVOPathGuidingPushConstant
{
	float3x3 randomRotateMatrix;
	float4 VGBStartPos_Size;
	int frameIndex = 0;
	int maxDepth = 6;
	float time;
	float voxelLength;
	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;
};

enum class StaticBindingPoints_SVOPG
{
	//eTextures = 0,
	//eOutImage = 1,	//First save the result of lightInject, and finally MIS
	eSVO_G = 2,
	eNodeData_E,
	eGlobalInfo,
	eWeights,
#ifndef NDEBUG
	eDepthImage,
#endif
};
enum DynamicBindingPoints_SVOPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};

struct GlobalInfo_SOVPG {
	uint SVOMaxLayer_G;
	uint indivisibleNodeCount_G;
	uint totalNodeCount_E;
};

NAMESPACE_SHADERIO_END()
#endif
