#pragma once

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include <memory>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>
#include <feature/TAA/TAA.h>
#include "HeightFog/HeightFog.h"
#include "FluidFog/FluidFog.h"
#include "GridFog/GridFog.h"

#ifdef FINAL_PROJECT
namespace FzbRenderer {
enum class GBuffers_VolumetricFog {
	eAlbedo = 0,
	eNormal,
	eEmissive,
	eVelocity,
	eVertexInfo,	//meshID, instanceID, etc
	eRendered,
	eTonemapping,
};

class VolumetricFog : public Renderer {
public:
	VolumetricFog() = default;
	~VolumetricFog() = default;

	VolumetricFog(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;
private:
	void createSourceData();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPushConstantsInfo pushInfo;
	//-------------------------------------------------------------------
	float time = 0.0f;
	float dt = 0.0f;
	int temporalFrameIndex = 0;

	FzbRenderer::Buffer inDirectDispatchBuffer;
	//-----------------------------shadowMap-------------------------------
	ShadowMap shadowMap;
	//--------------------------------TAA---------------------------------
	bool useTAA = true;
	TAA taa;
	//-------------------------------GBuffer-------------------------------
	shaderio::CreateGBuffersPushConstant createGBuffersPushConstant;
	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};
	void createGBuffers(VkCommandBuffer cmd);
	//---------------------------FogGlobalInfo------------------------------
	bool globalInfoModified = false;
	shaderio::FogGlobalInfo fogGlobalInfo;
	FzbRenderer::Buffer fogGlobalInfoBuffer;
	//------------------------VolumetricFogInfo------------------------------
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;
	FzbRenderer::Buffer volumetricFogInfosBuffer;
	std::vector<int> volumetricFogInfoModified;

	std::unique_ptr<HeightFogSet> heightFogSet;
	std::unique_ptr<FluidFogSet> fluidFogSet;
	std::unique_ptr<GridFogSet> gridFogSet;
	//--------------------------FluidSimulation-----------------------------
	VkShaderEXT computeShader_initFluid{};
	VkShaderEXT computeShader_getInFluidInstanceInfo{};
	VkShaderEXT computeShader_injectFluidInfo{};
	FzbRenderer::Buffer instanceInfoBuffer;
	FzbRenderer::Buffer inFluidInstanceInfoBuffer;
	FzbRenderer::Buffer inFluidInstanceCountBuffer;
	shaderio::InitFluidPushConstant initFluidPushConstant;
	void initFluid(VkCommandBuffer cmd);

	VkShaderEXT computeShader_fluidSimulation_A{};
	VkShaderEXT computeShader_fluidSimulation_D{};
	VkShaderEXT computeShader_fluidSimulation_D_Iteration{};
	VkShaderEXT computeShader_fluidSimulation_F{};
	VkShaderEXT computeShader_fluidSimulation_P{};
	VkShaderEXT computeShader_fluidSimulation_P_Iteration{};
	VkShaderEXT computeShader_fluidSimulation_S{};

	shaderio::FluidSimulationPushConstant fluidSimulationPushConstant;
	bool fluidSimulation_IterationD = false;
	int fluidSimulation_IterationD_Count = 20;
	bool fluidSimulation_IterationP = true;
	int fluidSimulation_IterationP_Count = 20;

	void fluidSimulation(VkCommandBuffer cmd);
	//-----------------------------FogAcc-----------------------------------
	bool useFogAcc = false;
	shaderio::FrustumGlobalInfo frustumInfo;
	bool blurVoxelFog = true;
	int sampleCount_fogAcc = 20;

	void createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear);
	FzbRenderer::Image fogAccImage;

	FzbRenderer::Buffer fogAccHasFogVoxelCountBuffer;
	FzbRenderer::Buffer fogAccHasFogVoxelInfosBuffer;
	FzbRenderer::Image fogAccHistoryImage;

	shaderio::FogAccPushConstant fogAccPushConstant;
	VkShaderEXT computeShader_getHasFogVoxels{};
	VkShaderEXT computeShader_voxelGetFog{};
	VkShaderEXT computeShader_voxelBlurFog{};
	VkShaderEXT computeShader_voxelAccFog{};

	void fogAcc(VkCommandBuffer cmd);
	//---------------------------renderOpaque-------------------------------
	shaderio::renderOpaquePushConstant renderOpaquePushConstant;
	VkShaderEXT computeShader_renderOpaqueMaterial{};
	void renderOpaqueMaterial(VkCommandBuffer cmd);

	int sampleCount_renderOpaque = 20;
	float jitterStrength_randerOpaque = 1.0f;
	//-----------------------renderTransparent-------------------------------
	shaderio::renderTransparentPushConstant renderTransparentPushConstant;
	VkShaderEXT vertexShader_renderTransparentMaterial{};
	VkShaderEXT fragmentShader_renderTransparentMaterial{};
	void renderTransparentMaterial(VkCommandBuffer cmd);
	//--------------------------Debug----------------------------------------
	bool showCameraFrustum = false;
	shaderio::renderCameraFrustumPushConstant renderCameraFrustumPushConstant;
	VkShaderEXT vertexShader_renderCameraFrustum{};
	VkShaderEXT fragmentShader_renderCameraFrustum{};
	void renderCameraFrustum(VkCommandBuffer cmd);

	bool showInstanceAABB = false;
	int showInstanceIndex = 0;
	shaderio::renderInstanceAABBPushConstant renderInstanceAABBPushConstant;
	VkShaderEXT vertexShader_renderInstanceAABB{};
	VkShaderEXT fragmentShader_renderInstanceAABB{};
	void renderInstanceAABB(VkCommandBuffer cmd);
};

}

#else
namespace FzbRenderer {
	enum class GBuffers_VolumetricFog {
		eAlbedo = 0,
		eNormal,
		eEmissive,
		eVelocity,
		eVertexInfo,	//meshID, instanceID, etc
		eRendered,
#ifdef BLUR_FOG
		eRenderedResultFog,
		eDepthGradient,
		eFilter0,
		eFilter1,
#endif
		eTonemapping,
	};

class VolumetricFog : public Renderer {
public:
	VolumetricFog() = default;
	~VolumetricFog() = default;

	VolumetricFog(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;

private:
	void saveParams();
	void loadParams();

	void createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear = true);
	void createVolumetricFogData();

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void getVisibleFog(VkCommandBuffer cmd);
	void initFluidFog(VkCommandBuffer cmd);
	void createGBuffers(VkCommandBuffer cmd);
	void fluidSimulation(VkCommandBuffer cmd);
	void createEnvFog(VkCommandBuffer cmd);
	void envFogLightAttenuationEstimate(VkCommandBuffer cmd);
	void createFrustumAccFog(VkCommandBuffer cmd);
	void fogBlur_Voxel(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);
	void fogBlur(VkCommandBuffer cmd);
	void renderTransparentMaterial(VkCommandBuffer cmd);

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivFeatures = {};

	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_getVisibleVolumetricFog{};

	VkShaderEXT computeShader_initFluidFog{};

	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};

	VkShaderEXT computeShader_createVolumetricFog_Fluid_A{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_D{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_F{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_P{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_S{};

	VkShaderEXT computeShader_createEnvionmentFog{};
	VkShaderEXT computeShader_createLightAttenuationEstimator{};

	VkShaderEXT computeShader_createFrustumAccFog{};
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	VkShaderEXT computeShader_getHasFogVoxelInfo{};
#endif
#ifdef FOG_ACC_TWO_PASS
	VkShaderEXT computeShader_createFrustumAccFog_Pass1{};
	VkShaderEXT computeShader_createFrustumAccFog_Pass2{};
#endif

	VkShaderEXT computeShader_blurFog_voxel{};

	VkShaderEXT computeShader_deferredRenderring{};

	VkShaderEXT computeShader_getDepthGradient{};
	VkShaderEXT computeShader_varianceConvolution{};
	VkShaderEXT computeShader_blurFog_X{};
	VkShaderEXT computeShader_blurFog_Y{};
	VkShaderEXT computeShader_addFog{};

	VkShaderEXT vertexShader_renderTransparentMaterial{};
	VkShaderEXT fragmentShader_renderTransparentMaterial{};

	shaderio::VolumetricFogPushConstant pushConstant;

	bool useTAA = true;
	shaderio::float4x4 invProjMatrix_taaJitter;
	bool useSVGF = false;
	SVGF svgf;
	TAA taa;
	ShadowMap shadowMap;

	shaderio::float4x4 VPMatrix_lastFrame;
	shaderio::float4x4 invViewMatrix_lastFrame;
	uint32_t temporalFrameIndex = 0;

	FzbRenderer::Buffer GlobalInfoBuffer;
	//--------------------------FogInfo-----------------------------------
	bool globalHeightFogModified = false;
	int useGlobalHeightFog = 1;
	shaderio::float2 globalHeightFogY;
	shaderio::HeightFogInfo globalHeightFogInfo;
	//bool showGlobalHeightFog = false;

	uint32_t volumetricFogCount = 1;
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;
	FzbRenderer::Buffer volumetricFogInfosBuffer;
	std::vector<int> volumetricFogInfoModified;

	std::unique_ptr<HeightFogSet> heightFogSet;
	std::unique_ptr<FluidFogSet> fluidFogSet;
	std::unique_ptr<GridFogSet> gridFogSet;
	//-------------------------Frustum------------------------------------
	VkExtent3D frustumGridSize;
	//-------------------------Environment--------------------------------
	FzbRenderer::Image envVolumetricFogInfoImage;

	bool envChange = false;
	shaderio::float3 envStartPos = { 5083.0f, -77.5f, -4480.0f };
	shaderio::uint3 envGridSize = { 128, 128, 128 };
	shaderio::float3 envVoxelSize = { 1.6, 1, 1 };
	bool showEnvGrid = false;

	int sampleCount_env = 10;
	float jitterStrength_env = 0.0f;
	//-------------------------FogAcc-------------------------------------
#ifdef FOG_ACC_DELETE_NOFOGVOXEL
	FzbRenderer::Buffer fogAccHasFogVoxelInfoBuffer;
#endif

#ifdef FOG_ACC_DIVIDE_PART
	FzbRenderer::Buffer fogAccResultBuffer;
#elif defined(FOG_ACC_ONE_DISPATCH)
	FzbRenderer::Buffer fogAccSyncBuffer;
#endif
	FzbRenderer::Image volumetricFogAccResultImage;

#ifdef BLUR_FOG_VOXEL
	FzbRenderer::Image volumetricFogAccResultHistoryImage;
#endif
	//默认开启：视锥体素的时域滤波是在不改变160x160x80分辨率的前提下
	//消除相机移动锯齿的唯一手段（每帧亚体素抖动 + 多帧积分 = 超采样）
	bool useFogBlurVoxel = true;

	uint32_t rmSampleCountSampleCount_fogAcc = 8;
	int randomStepping_fogAcc = true;
	float accJitterStrength = 0.0f;
	float interpolationJitterStrength_fogAcc = 0.0f;
	//-------------------------Opaque------------------------------------
	uint32_t forwardSampleCount = 0;
	uint32_t rmSampleCount_opaque_noFogAcc = 50;
	uint32_t rmSampleCount_opaque_FogAcc = 1;
	uint32_t randomStepping_opaque = true;
	float interpolationJitterStrength_attenuation = 1.0f;
	float interpolationJitterStrength_L = 0.15f;
	//----------------------transparent------------------------------------
	float interpolationJitterStrength_attenuation_transparent = 0.0f;
	float interpolationJitterStrength_L_transparent = 0.0f;
	//-------------------------fogBlur------------------------------------
	bool useFogBlur = false;
	int FogFilterCount = 4;

#ifndef NDEBUG
	void renderVolumetricFogVoxelGrid(VkCommandBuffer cmd);
	void renderCameraFrustum(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_renderVoxelGrid{};
	VkShaderEXT fragmentShader_renderVoxelGrid{};

	VkShaderEXT vertexShader_renderCameraFrustum{};
	VkShaderEXT fragmentShader_renderCameraFrustum{};

	std::vector<int> showVolumetricFogVoxelGrids;

	bool showCameraFrustum = false;
	shaderio::SceneInfo showCameraInfo;
	nvvk::Buffer bShowCameraInfo;

	VkShaderEXT computeShader_test{};
	void test(VkCommandBuffer cmd);

	shaderio::int3 frustumVoxelShowMin = { 0, 0, 0 };
	shaderio::int3 frustumVoxelShowMax = { 160, 160, 80 };
#endif
};
}
#endif
#endif