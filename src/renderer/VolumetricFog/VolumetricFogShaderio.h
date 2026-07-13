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
	int fluidFogIndex;

	float dt;
	float time;
	uint iteration = 0;
	
	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;

	//float4x4 instanceTransformMatrix_lastTime;

	int frameIndex;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;

	float3 mouseForcePosition;
	float mouseForceStrength;
	float mouseForceRadius;
};

enum class VolumetricFogType {
	Height,
	Fluid,
	Noise,
};
struct VolumetricFogInfo {
	float3 fogStartPos;
	uint3 fogVoxelGridSize;		//��Ҫ����Ϊ1x1x1��������bug
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
	float4 voxelFogInfo;
	//float3 temperature;
};

struct GlobalInfo_VolumetricFog {
	AABB fluidAABB;
	int fluidStartUp;
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

	//�߶�
	eVolumetricFogHeightInfoBuffer,

	//����
	eVolumetricFogFluidInfoBuffer,
	eVolumetricFogFluidVoxelInfoBuffer,
	eVolumetricFogFluidVoxelVelocityImage,
	eVolumetricFogFluidVoxelVelocityImage_sample,
	eVolumetricFogFluidVoxelInfoImages,
	eVolumetricFogFluidVoxelInfoImages_sampler,

	//����
	eVolumetricFogNoiseInfoBuffer,

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif