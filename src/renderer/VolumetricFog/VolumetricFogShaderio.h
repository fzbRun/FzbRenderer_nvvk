#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Jacobi_Iteration_Count 60u

struct VolumetricFogPushConstant{
	float3x3 normalMatrix;
	float3 instanceVelocity;

	int instanceIndex;

	float dt;
	float time;
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
	Noise,
};
struct VolumetricFogInfo {
	float3 fogStartPos;
	uint3 fogVoxelGridSize;		//不要设置为1x1x1，否则有bug
	float3 fogVoxelSize;
	float3 color;
	float ambientIntensity;
	float2 absorption;
	float scattering;
	float phase;

	VolumetricFogType type;
	int volumetricFogTypeIndex;
};
struct HeightFogInfo {
	float heightScale;
};
struct FluidFogInfo {
	int startUp;
	float viscosity;
	float FIntensity;
	float lightAttenuationEstimator;
};
struct NoiseFogInfo {
	float3 cloudScale;
	float cloudFlowSpeed;
	float2 cloudCoverage;
	float2 cloudTypePreference;
	float weatherScale;
};

struct VolumetricFogFluidVoxelInfo {
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
	eVisibleVolumetricFogIndexBuffer,
	eVolumetricFogInfosBuffer,

	//高度
	eVolumetricFogHeightInfoBuffer,

	//流体
	eVolumetricFogFluidInfoBuffer,
	eVolumetricFogFluidVoxelInfoBuffer,
	eVolumetricFogFluidVoxelVelocityImage,
	eVolumetricFogFluidVoxelVelocityImage_sample,
	eVolumetricFogFluidExtinctionImages,
	eVolumetricFogFluidExtinctionImages_sampler,

	//噪声
	eVolumetricFogNoiseInfoBuffer,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif