#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_TAA_SHADERIO_H
#define FZBRENDERER_TAA_SHADERIO_H

NAMESPACE_SHADERIO_BEGIN()

struct TAAPushConstant {
	float4x4 viewMatrix_lastFrame;
	uint2 screenSize;
	int frameIndex;
	float mergeRatio;
	float2 jitter;
	SceneInfo* sceneInfoAddress;
};

enum class BindingPoints_TAA {
	eRendereTarget_Current = 0,
	eRenderTarget_Last,
	eRenderTarget_Final,
	eVelocityImage,
	eDepthImage,
};

NAMESPACE_SHADERIO_END()

#endif
