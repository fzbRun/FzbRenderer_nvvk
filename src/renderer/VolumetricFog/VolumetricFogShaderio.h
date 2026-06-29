#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3 fogStartPos = { -1.5f, -0.75f, -1.75f };
	int frameIndex;
	float3x3 normalMatrix;
	uint3 fogVoxelGridSize = { 16, 16, 16 };
	float3 fogVoxelSize = { 0.2f, 0.2f, 0.2f };
	int instanceIndex;
	float2 absorption = { 0.0, 0.3 };
	float scattering = 0.3f;
	float phase = -0.5;
	float lightAttenuationStrength = 0.2f;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,
	eVolumetricFogImage,
	eLightAttenuationEstimatorBuffer,
	eVolumetricFogImage_sampler,
	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG
	eVoxelGridImage,
#endif
};

NAMESPACE_SHADERIO_END()
#endif