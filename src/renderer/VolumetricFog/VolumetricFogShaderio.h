#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct VolumetricFogPushConstant
{
	float3x3 normalMatrix;
	int frameIndex;

	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;
	int instanceIndex;

	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;
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
	uint visibleVolumetricFogCount;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,

	eGlobalInfoBuffer,

	eVolumetricFogInfosBuffer,
	eVolumetricFogImages,

	eVisibleVolumetricFogIndexBuffer,
	eVolumetricFogImages_sampler,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif