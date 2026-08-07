#pragma once

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>
#include <renderer/VolumetricFog/HeightFog/HeightFogShaderio.h>
#include <renderer/VolumetricFog/FluidFog/FluidFogShaderio.h>
#include <renderer/VolumetricFog/GridFog/GridFogShaderio.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define USE_TAA
#define USE_SVGF

#define Jacobi_Iteration_Count 20u

#define MAX_VOLUMETRIC_FOG_COUNT 10

#define FOG_ATTENUATION_ESTIMATION

#define FLUID_SIMPLIFY_VISCOSITY
//#define FLUID_SIMPLIFY_PRESSURE

#define FLUID_SIMULATION_OBJECT_SAVE_FOG

#define USE_ENVFOG
//#define Uniform_EnvFog_Grid

#define Fog_Acc_Stepping

#define FOG_ACC_DELETE_NOFOGVOXEL

//#define FOG_ACC_DIVIDE_PART
//#define FOG_ACC_ONE_DISPATCH
//#define FOG_ACC_SERIAL
#define FOG_ACC_TWO_PASS

#ifdef FOG_ACC_DIVIDE_PART
#define FOG_ACC_THREADGROUP_SIZE 32
#elif defined(FOG_ACC_ONE_DISPATCH)
#define FOG_ACC_THREADGROUP_SIZE 8
#elif defined(FOG_ACC_SERIAL)
#define FOG_ACC_THREADGROUP_SIZE 32
#elif defined(FOG_ACC_TWO_PASS)
#define FOG_ACC_THREADGROUP_SIZE 32
#endif

#define BLUR_FOG
#define BLUR_FOG_THREADGROUP_SIZE 256
#define LINEAR_DEPTH

#define BLUR_FOG_VOXEL
//#define BLUR_FOG_VOXEL_PASS2

//#define Interpolation_Manual

struct VolumetricFogPushConstant {
	float3x3 normalMatrix;
	float3 instanceVelocity;

	uint2 screenSize;

	int instanceIndex;

	float dt;
	float time;
	uint iteration = 0;

	int randomStepping;
	int sampleCount;
	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;
	uint heightFogCount;
	uint fluidFogCount;
	uint gridFogCount;

	int frameIndex;
	SceneInfo* sceneInfoAddress;
	float4x4 lightVP;

	int useAccFog = true;
	float3 compressionParams;
	//int forwardSampleCount;
	float cameraNearPlane;
	uint3 frustumGridSize;
	float cameraFarPlane;
	float tanCameraFov_2;	//fov / 2
	float aspectRatio;

	float3 cameraPos_lastTime;

	float jitterStrength0;
	float jitterStrength1;

	int useFogBlur = true;

#ifndef NDEBUG
	SceneInfo* showCameraInfoAddress;
#endif
};

struct GlobalInfo_VolumetricFog {
	float4x4 VPMatrix_lastFrame;

	int useGlobalHeightFog;
	float2 globalHeightFogY;
	HeightFogInfo globalHeightFogInfo;

	AABB fluidAABB[MAX_FLUID_FOG_COUNT];
	int fluidStartUp[MAX_FLUID_FOG_COUNT];
	uint visibleVolumetricFogCount;
	uint visibleVolumetricFogIndices[MAX_VOLUMETRIC_FOG_COUNT];
#ifdef USE_ENVFOG
	float envFogLightAttenuationEstimator;
	float envFogLightAttenuationLength;

	float3 envStartPos;
	uint3 envGridSize;
	float3 envVoxelSize;
#endif

#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	uint hasFogCount;
#endif
};

struct FogAccHasFogVoxelInfo {
	uint3 voxelIndex;
	float3 voxelCenter;
	float3 voxelCenter_lastVoxel;
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

	eGridFogInfoBuffer,
	eGridFogImages,

#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	eFogAccHasFogVoxelInfoBuffer,
#endif

#ifdef FOG_ACC_DIVIDE_PART
	eFogAccResultBuffer,
#elif defined(FOG_ACC_ONE_DISPATCH)
	eFogAccSyncBuffer,
#endif

	eFogAccResultImage,
	eFogAccResultImage_sample,
#ifdef BLUR_FOG_VOXEL
	eFogAccResultHistoryImage_sample,
#endif

	eEnvFogInfoImage,
	eEnvFogInfoImage_sample,

#ifdef BLUR_FOG
	eRenderedFogResultImage,
	eDepthGradientImage,
	eFilterImages,
#endif

	eShadowMap,
	eRenderedImage,
#ifndef NDEBUG

#endif
};

NAMESPACE_SHADERIO_END()
#endif