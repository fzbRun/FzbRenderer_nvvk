#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_SHADOWMAP_SHADERIO_H
#define FZBRENDERER_SHADOWMAP_SHADERIO_H

NAMESPACE_SHADERIO_BEGIN()

//struct ShadowMapPushConstant {
//	float3x3 lightViewMatrix[6];		//directionLight use element 0
//	float3 displacementVector;		//the 4th column of viewMatrix
//	float4 lightProjectMatrixParameter;	//must ensure the aspect = 1
//	int instanceIndex;
//	SceneInfo* sceneInfoAddress;
//};
struct ShadowMapPushConstant {
	float4x4 lightVP;
	int lightIndex;
	int instanceIndex;
	SceneInfo* sceneInfoAddress;
#ifndef NDEBUG
	float4x4 inverse_lightVP;
	int frameIndex;
#endif
};

enum class BindingPoints_ShadowMap {
	eShadowMaps = 0,
	eDepthRestructResultMaps,
};

NAMESPACE_SHADERIO_END()

#endif