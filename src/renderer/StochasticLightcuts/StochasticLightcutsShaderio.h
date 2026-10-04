#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_STOCHASTIC_LIGHTCUTS_SHADER_IO_H
#define FZBRENDERER_STOCHASTIC_LIGHTCUTS_SHADER_IO_H

#define Scene_Resolution 64

NAMESPACE_SHADERIO_BEGIN()

struct AreaLight_StochasticLightcuts {
	float3 startPos;
	float3 edge1;
	float3 edge2;
	float3 normal;
	float3 color;
};

struct LightNode {
	float3 normal;
	float halfAngle;
	float3 minimum;
	float3 maximum;
	float3 color;
};

struct StochasticLightcutsPushConstant {
	int mode = 0;

	int frameIndex = 0;
	int maxFrameCount;
	int areaLightCount;

	float3 sceneSize;
	float3 sceneStartPos;
	int lightNodeTotalCount;
	int lightTreeDepth;
	int curLayer;
	int curLayerBlockCount;
	int curLayerNodeStartIndex;
	int fatherLayerNodeStartIndex;

	SceneInfo* sceneInfoAddress;
	AreaLight_StochasticLightcuts* areaLightsAddress;
	LightNode* lightTreeAddress;

	uint* hasLightBlockCountAddress;
	uint* hasLightBlockArrayAddress;
	uint* hasLightBlockIndexAddress;

	DispatchIndirectCommand* cmd;

	uint2 screenSize;
};

struct CreateGBufferPushConstant_StochasticLightcuts {
	int instanceIndex;
	float4x4 normalMatrix;
	SceneInfo* sceneInfoAddress;

	float4x4 padding1;
	float4 padding2;
	float4 padding3;
	float4 padding4;
};

struct LightTreeDebugPushConstant_StochasticLightcuts {
	int frameIndex = 0;
	int lightTreeDepth;

	int curLayer = 0;
	int curLayerNodeCount;
	int curLayerNodeStartIndex;

	SceneInfo* sceneInfoAddress;
	LightNode* lightTreeAddress;
};

enum class StaticBindingPoints_StochasticLightcuts {
	eNormal_MaterialIndexImage = 2,
	eDepthImage,
};

NAMESPACE_SHADERIO_END()

#endif