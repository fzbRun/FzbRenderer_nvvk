#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3 fogStartPos = { -1.5f, 0.0f, -1.5f };
	int frameIndex;
	float3x3 normalMatrix;
	uint3 fogVoxelGridSize = { 16, 16, 16 };
	float3 fogVoxelSize = { 0.2f, 0.2f, 0.2f };
	int instanceIndex;
	float2 extinctionCoefficient = { 0.0, 1.0 };
	float scatterCoefficient = 0.7f;
	float asymmetricParameters = 0.2;
	float lightAttenuationStrength = 1.0f;
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