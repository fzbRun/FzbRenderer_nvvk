#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Jacobi_Iteration_Count 60u

struct VolumetricFogPushConstant
{
	float3x3 normalMatrix;
	float3 instanceVelocity;

	int instanceIndex;
	int volumetricFogFluidIndex;

	float dt;
	uint iteration = 0;
	
	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;

	float ambientFogDensity = 0.0f;

	int frameIndex;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;
};

enum class VolumetricFogType {
	Height,
	Fluid,
};
struct VolumetricFogInfo {
	float3 fogStartPos;
	uint3 fogVoxelGridSize;		//不要设置为1x1x1，否则有bug
	float3 fogVoxelSize;
	float2 absorption;
	float scattering;
	float phase;
	float lightAttenuationEstimator;

	VolumetricFogType type;
	float viscosity;
	uint fluidIndex;
	float FIntensity;
};
struct VolumetricFogVoxelInfo {
	float4 dirtyVelocity_pressure[2];
	bool isBoundary;
	float3 density;
	//float3 temperature;
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
	eVolumetricFogVoxelInfoBuffer,
	eVolumetricFogVoxelVelocityImage,
	eVolumetricFogVoxelVelocityImage_sample,

	eVolumetricFogExtinctionImages,
	eVolumetricFogExtinctionImages_sampler,

	eVisibleVolumetricFogIndexBuffer,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif