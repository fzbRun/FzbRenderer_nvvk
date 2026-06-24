#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3 fogStartPos = { 0.0f, 0.0f, 0.0f };
	int frameIndex;
	float3x3 normalMatrix;
	uint3 fogVoxelGridSize = { 4, 4, 4 };
	float3 fogVoxelSize = { 1.0f, 1.0f, 1.0f };
	int instanceIndex;
	SceneInfo* sceneInfoAddress;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eVolumetricFogImage,
	eVolumetricFogImage_sampler,
	eRenderedImage,
};

NAMESPACE_SHADERIO_END()
#endif