#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define Jacobi_Iteration_Count 40u

#define MAX_VOLUMETRIC_FOG_COUNT 10
#define MAX_HEIGHT_FOG_COUNT 3
#define MAX_FLUID_FOG_COUNT 3
#define MAX_NOISE_FOG_COUNT 3

struct VolumetricFogPushConstant{
	float3x3 normalMatrix;
	float3 instanceVelocity;

	int instanceIndex;
	int fluidFogIndex;

	float dt;
	float time;
	uint iteration = 0;
	
	int randomStepping;
	int sampleCount;
	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;
	uint heightFogCount;
	uint fluidFogCount;
	uint noiseFogCount;

	int frameIndex;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;

	int useEnvAccFog;
	int compressionPrecision;
	int forwardSampleCount;
	float cameraNearPlane;
	uint3 frustumGridSize;
	float cameraFarPlane;
	float tanCameraFov_2;	//fov / 2
	float aspectRatio;

#ifndef NDEBUG
	SceneInfo* showCameraInfoAddress;
#endif
};

enum class VolumetricFogType {
	Height,
	Fluid,
	Noise,
};
struct VolumetricFogInfo {
	float3 fogStartPos;
	uint3 fogVoxelGridSize;
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
	float restoreSpeed;
	float3 fogStartPos_lastTime;
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
	float4 voxelFogInfo;
};

struct GlobalInfo_VolumetricFog {
	AABB fluidAABB;
	int fluidStartUp;
	uint visibleVolumetricFogCount;
	uint visibleVolumetricFogIndices[MAX_VOLUMETRIC_FOG_COUNT];
	float envFogLightAttenuationEstimator;
	float envFogLightAttenuationLength;
};


enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,

	eGlobalInfoBuffer,
	eVolumetricFogInfosBuffer,

	eVolumetricFogHeightInfoBuffer,

	eVolumetricFogFluidInfoBuffer,
	eVolumetricFogFluidVoxelInfoBuffer,
	eVolumetricFogFluidVoxelVelocityImage,
	eVolumetricFogFluidVoxelVelocityImage_sample,
	eVolumetricFogFluidVoxelInfoImages,
	eVolumetricFogFluidVoxelInfoImages_sampler,

	eVolumetricFogNoiseInfoBuffer,

	eVolumetricFogAttenuationImage,
	eVolumetricFogAttenuation2Image,
	eVolumetricFogLImage,

	eEnvVolumetricFogInfoImage,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif