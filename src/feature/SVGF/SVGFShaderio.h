#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_SVGF_SHADERIO_H
#define FZBRENDERER_SVGF_SHADERIO_H

#define SVGF_FILTER_COUNT 5

NAMESPACE_SHADERIO_BEGIN()

struct SVGFPushConstant {
	int frameIndex;
	int filterIndex;
	uint2 screenSize;
	float nearPlane;
	float farPlane;
	SceneInfo* sceneInfoAddress;

	float4x4 projMatrix;
};

enum class BindingPoints_SVGF {
	eAlbedoImage,
	eDepthImage,
	eNormalImage,
	eVelocityImage,
	eVertexInfoImage,
	eResultImage,

	eDepthGradientImage,
	eIrradianceImage,

	eMoment1Image,
	eMoment2Image,
	eVarianceImage,

	eFilterImages,

	eHistoryDepthImage,
	eHistoryNormalImage,
	eHistoryVertexInfoImage,
};

NAMESPACE_SHADERIO_END()

#endif