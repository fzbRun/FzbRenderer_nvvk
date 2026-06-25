#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3 fogStartPos = { -0.5f, 0.2f, -0.2f };
	int frameIndex;
	float3x3 normalMatrix;
	uint3 fogVoxelGridSize = { 8, 8, 8 };
	float3 fogVoxelSize = { 0.1f, 0.15f, 0.1f };
	int instanceIndex;
	float2 extinctionCoefficient = { 4.0, 10.0 };
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eVolumetricFogImage,
	eVolumetricFogImage_sampler,
	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG
	eVoxelGridImage,
#endif
};

NAMESPACE_SHADERIO_END()
#endif