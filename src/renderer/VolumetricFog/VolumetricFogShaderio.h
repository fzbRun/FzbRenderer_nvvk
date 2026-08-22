#pragma once

#include <common/Shader/shaderStructType.h>
#include <renderer/VolumetricFog/VolumetricFogCommonShaderio.h>
#include <renderer/VolumetricFog/HeightFog/HeightFogShaderio.h>
#include <renderer/VolumetricFog/FluidFog/FluidFogShaderio.h>
#include <renderer/VolumetricFog/GridFog/GridFogShaderio.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
#define FZBRENDERER_VOLUMETRIC_FOG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define MAX_VOLUMETRIC_FOG_COUNT 10

#define FLUID_SIMULATION_OBJECT_SAVE_FOG

//#define FLUID_GBUFFER_INJECT
//#define SAMPLE_JITTER_TAA
//#define MULTI_SCATTERING

#define BLUR_FOG_THREADGROUP_SIZE 256

#define FINAL_PROJECT
#ifdef FINAL_PROJECT

struct TAAGlobalInfo {
	float4x4 projMatrix;
	float4x4 projInvMatrix;
};

struct FogGlobalInfo {
	float lightAttenuationStrength = 1.0f;
	uint volumetricFogCount;
	uint heightFogCount;
	uint gridFogCount;
	uint fluidFogCount;

	int useGlobalHeightFog;
	AABB globalHeightFogAABB;
	HeightFogInfo globalHeightFogInfo;
};

struct InstanceInfo {
	AABB aabb_local;
	AABB aabb_lastFrame;
	int instanceIndex;
};
struct InFluidInstanceInfo {
	AABB aabb;
	float3 velocity;
};

struct FrustumGlobalInfo {
	float3 compressionParams;
	float cameraNearPlane;
	uint3 frustumGridSize;
	float cameraFarPlane;
	float tanCameraFov_2;
    float aspectRatio;
    float sphereFactor;
};
struct FogAccHasFogVoxelInfo {
	uint3 voxelIndex;
	float3 voxelCenter;
	float3 voxelCenter_lastVoxel;
};
//---------------------------------------------
struct InitFluidPushConstant {
	float time;
	float dt;
	int fluidIndex;
	int fogIndex;
	float3 fluidStartPos;

	SceneInfo* sceneInfoAddress;
	FogGlobalInfo* fogGlobalInfoAddress;
	uint* dynamicMeshInjectedAddress[MAX_FLUID_FOG_COUNT];

	float4x4 padding0;
	float4x4 padding1;
	float4x4 padding2;
};
struct InjectFluidPushConstant {
	float dt;
	int fluidIndex;

	int instanceIndex;
	//float3x3 normalMatrix;

	float4x4 tansfromMatrix_lastFrame;

	SceneInfo* sceneInfoAddress;
	Mesh* meshes_lowPoly;
	float4x4* fluidVPMatrixAddress;
	uint* dynamicMeshInjectedAddress[MAX_FLUID_FOG_COUNT];
};
//---------------------------------------------
struct CreateGBuffersPushConstant {
	float4x4 projMatrix_taa;
	int useTAA;

	int fluidFogCount;

	int instanceIndex;
	float3x3 normalMatrix;
	float4x4 tansfromMatrix_lastFrame;

	float4x4 vpMatrix_lastFrame;

	SceneInfo* sceneInfoAddress;

	float dt;
};
//---------------------------------------------
struct FluidSimulationPushConstant {
	int fluidIndex;
	float time;
	float dt;
	int iteration;

	float3 fluidStartPos;
	int useP;

	FogGlobalInfo* fogGlobalInfoAddress;

	float4x4 padding0;
	float4x4 padding1;
	float4x4 padding2;
	float4 padding3;
	float2 padding4;
};
//---------------------------------------------
struct FogAccPushConstant {
	FrustumGlobalInfo frustumInfo;
	int blurVoxelFog;
	int sampleCount;
	int temporalFrameIndex;
	float dt;

	float3 jitterUVW;
	int isJitterUVW;

	float2 fluidMergeRatio = { 0.4f, 0.7f };

	float4x4 lightVP;
	float4x4 viewInvMatrix_lastFrame;

	SceneInfo* sceneInfoAddress;
	FogGlobalInfo* fogGlobalInfoAddress;
	uint* fogAccHasFogVoxelCountAddress;

	float4 padding0;
};
//---------------------------------------------
struct renderOpaquePushConstant {
	FrustumGlobalInfo frustumInfo;
	uint2 screenSize;

	int temporalFrameIndex;
	int sampleCount;

	int useFogAcc;

	int useTAA;
	float4x4 projInvMatrix_taa;

	float4x4 lightVP;

	float jitterStrength;

	int useFogBlur;

	SceneInfo* sceneInfoAddress;
	FogGlobalInfo* fogGlobalInfoAddress;

	float4 padding0;
	float4 padding1;
};
//---------------------------------------------
struct FogBlurPushConstant {
	int fogBlurCount;
	int filterIndex;
	uint2 screenSize;

	float4x4 padding0;
	float4x4 padding1;
	float4x4 padding2;
	float4 padding3;
	float4 padding4;
	float4 padding5;
};
//---------------------------------------------
struct renderTransparentPushConstant {
	FrustumGlobalInfo frustumInfo;
	uint2 screenSize;

	float jitterStrength;

	int temporalFrameIndex;
	int sampleCount;

	int instanceIndex;
	float3x3 normalMatrix;

	float4x4 projMatrix_taa;
	int useTAA;
	int useFogAcc;

	float4x4 lightVP;

	SceneInfo* sceneInfoAddress;
	FogGlobalInfo* fogGlobalInfoAddress;
};

struct renderCameraFrustumPushConstant {
	FrustumGlobalInfo frustumInfo;
	uint3 showVoxelIndexMin;
	uint3 showVoxelIndexMax;
	
	float4x4 vpMatrix;
	float4x4 viewInvMatrix_showFrustum;

	float4 padding0;
	float4 padding1;
	float4 padding2;
	float3 padding3;
};
struct renderInstanceAABBPushConstant {
	int instanceIndex;
	SceneInfo* sceneInfoAddress;
	InstanceInfo* instanceInfoAddress;

};

enum class StaticBindingPoints_VolumetricFog {
	eTextures = 0,
	//-------GBuffers----------
	eAlbedoImage,
	eNormalImage,
	eDepthImage,
	eEmissiveImage,
	eVelocityImage,
	eVertexInfoImage,
	eRenderedImage,
	//------shadowMap---------
	eShadowMap,

	//-------FogInfo--------
	eVolumetricFogInfosBuffer,
	eHeightFogInfosBuffer,
	eGridFogInfosBuffer,
	eGridFogImages,
	eFluidFogInfosBuffer,
	eFluidFogImages,
	eFluidFogImages_sample,

	//----FluidSimulation------
	eFluidFogVoxelInfoBuffer,
	eFluidFogVoxelVelocityImage,
	eFluidFogVoxelVelocityImage_sample,
#ifdef FLUID_A_MACCORMACK
	eFluidFogVoxelInfoImage_temp1,
	eFluidFogVoxelInfoImage_temp1_sample,
	eFluidFogVoxelInfoImage_temp2,
	eFluidFogVoxelInfoImage_temp2_sample,
#endif

	//------FogAcc---------
	eFogAccImage,
	eFogAccImage_sample,
	eFogAccHasFogVoxelInfosBuffer,
	eFogAccHistoryImage,

	//-----FogBlur---------
	eRenderedFogResultImage,
	eDepthGradientImage,
	eFilterImages
};

#else
#define USE_TAA
#define USE_SVGF

#define Jacobi_Iteration_Count 20u

#define FOG_ATTENUATION_ESTIMATION

#define FLUID_SIMPLIFY_VISCOSITY
//#define FLUID_SIMPLIFY_PRESSURE

#define FLUID_SIMULATION_OBJECT_SAVE_FOG

//#define USE_ENVFOG
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

// 仅用于A/B测量：打开后 getMarchLayerOffset 退回“全局同相位”偏移（修复前的行为）。
// 正式配置必须保持注释状态。
//#define FOG_LEGACY_LAYER_OFFSET

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
	//是否开启视锥体素的时域滤波。开启后视锥网格会做每帧统一的亚体素抖动，
	//由时域滤波把多帧积分起来，等价于超采样，用来消除相机移动时的体素锯齿
	int useFogBlurVoxel = false;

#ifndef NDEBUG
	SceneInfo* showCameraInfoAddress;
#endif
};

struct GlobalInfo_VolumetricFog {
	float4x4 VPMatrix_lastFrame;
	//上一帧的 viewInvMatrix（相机到世界），用于视锥体素的时域重投影。
	//同时补偿相机的平移和旋转；取用方式和 sceneInfo.viewInvMatrix 完全一致：
	//[0]=right, [1]=up, [2]=-forward, [3]=position
	float4x4 invViewMatrix_lastFrame;
	//一直递增的帧号。pushConst.frameIndex在相机移动时会被清零，无法用于时域抖动序列
	uint temporalFrameIndex;

	float4x4 projMatrix_taaJitter;
	float4x4 invProjMatrix_taaJitter;

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
#endif

NAMESPACE_SHADERIO_END()
#endif
