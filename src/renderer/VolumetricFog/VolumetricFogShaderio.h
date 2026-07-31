#pragma once

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>
#include <renderer/VolumetricFog/HeightFog/HeightFogShaderio.h>
#include <renderer/VolumetricFog/FluidFog/FluidFogShaderio.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define USE_TAA
#define USE_SVGF

#define Jacobi_Iteration_Count 40u

#define MAX_VOLUMETRIC_FOG_COUNT 10
#define MAX_NOISE_FOG_COUNT 3

//#define USE_ENVFOG
//#define Uniform_EnvFog_Grid
#define Fog_Acc_Stepping
#define BLUR_FOG

#define Interpolation_Manual

struct VolumetricFogPushConstant{
	float3x3 normalMatrix;
	float3 instanceVelocity;

	uint2 screenSize;

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

	int useAccFog;
	float compressionParams;
	//int forwardSampleCount;
	float cameraNearPlane;
	uint3 frustumGridSize;
	float cameraFarPlane;
	float tanCameraFov_2;	//fov / 2
	float aspectRatio;

	//float3 cameraMoveDir;
	float jitterStrength0;
	float jitterStrength1;

	int useFogBlur = true;

#ifndef NDEBUG
	SceneInfo* showCameraInfoAddress;
#endif
};

struct GlobalInfo_VolumetricFog {
	float4x4 VPMatrix_lastFrame;

	AABB fluidAABB;
	int fluidStartUp;
	uint visibleVolumetricFogCount;
	uint visibleVolumetricFogIndices[MAX_VOLUMETRIC_FOG_COUNT];
	float envFogLightAttenuationEstimator;
	float envFogLightAttenuationLength;

	float3 envStartPos;
	uint3 envGridSize;
	float3 envVoxelSize;
};

struct NoiseFogInfo {
	float3 cloudScale;
	float cloudFlowSpeed;
	float2 cloudCoverage;
	float2 cloudTypePreference;
	float weatherScale;
};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,

	eGlobalInfoBuffer,
	eFogInfosBuffer,

	eHeightFogInfoBuffer,

	eFluidFogInfoBuffer,
	eFluidFogVoxelInfoBuffer,
	eFluidFogVoxelVelocityImage,
	eFluidFogVoxelVelocityImage_sample,
	eFluidFogVoxelInfoImages,
	eFluidFogVoxelInfoImages_sampler,

	eNoiseFogInfoBuffer,

	eFogAttenuationImage,
	eFogAttenuationImage_sample,
	eFogLImage,
	eFogLImage_sample,

	eEnvFogInfoImage,
	eEnvFogInfoImage_sample,

#ifdef BLUR_FOG
	eRenderedFogImage,
	eDepthGradientImage,
	eFogVarianceImages,
	eFilterImages,
#endif

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif