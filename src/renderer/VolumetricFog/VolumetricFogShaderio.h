#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3 fogStartPos = { 5084.0f, -77.0f, -4486.0f };
	int frameIndex;
	float3x3 normalMatrix;
	uint3 fogVoxelGridSize = { 16, 16, 16 };
	float3 fogVoxelSize = { 7.0f, 2.0f, 5.0f };
	float2 absorption = { 0.0, 0.01 };
	float scattering = 0.03f;
	float phase = -0.7;
	float lightAttenuationStrength = 1.0f;
	uint localVolumetricFogCount;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;
	int instanceIndex;
#ifndef NDEBUG
	int showVoxelGridIndex = -1;
#endif
};

struct VolumetricFogInfo {
	float3 fogStartPos;
	uint3 fogVoxelGridSize;
	float3 fogVoxelSize;
	float2 absorption;
	float scattering;
	float phase;
	float lightAttenuationEstimator;
};

struct GlobalInfo_VolumetricFog {
	uint adjacentLocalVolumetricFogCount;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,

	eGlobalInfoBuffer,

	eVolumetricFogImage,
	eLightAttenuationEstimatorBuffer,

	eLocalVolumetricFogInfosBuffer,
	eLocalVolumetricFogImage,

	eVolumetricFogImage_sampler,
	eAdjacentLocalVolumetricFogIndexBuffer,
	eLocalVolumetricFogImage_sampler,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif