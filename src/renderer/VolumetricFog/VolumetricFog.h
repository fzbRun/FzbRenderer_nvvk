#pragma once

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>
#include <feature/TAA/TAA.h>
#include <feature/SVGF/SVGF.h>
#include "HeightFog/HeightFog.h"
#include "FluidFog/FluidFog.h"

namespace FzbRenderer {
enum class GBuffers_VolumetricFog{
	eAlbedo = 0,
	eNormal,
	eEmissive,
	eVelocity,
	eVertexInfo,	//meshID, instanceID, etc
	eRendered,
#ifdef BLUR_FOG
	eRenderedFog,
	eDepthGradient,
	eFogVariance0,
	eFogVariance1,
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

	VkShaderEXT computeShader_smoothFog{};

	VkShaderEXT computeShader_deferredRenderring{};

	VkShaderEXT computeShader_getDepthGradient{};
	VkShaderEXT computeShader_varianceConvolution{};
	VkShaderEXT computeShader_blurFog_X{};
	VkShaderEXT computeShader_blurFog_Y{};
	VkShaderEXT computeShader_addFog{};

	VkShaderEXT vertexShader_renderTransparentMaterial{};
	VkShaderEXT fragmentShader_renderTransparentMaterial{};

	shaderio::VolumetricFogPushConstant pushConstant;

	bool useSVGF = false;
	SVGF svgf;
	TAA taa;
	ShadowMap shadowMap;

	glm::vec3 m_lastCameraPos = glm::vec3(0.0f);
	shaderio::float4x4 VPMatrix_lastFrame;

	FzbRenderer::Buffer GlobalInfoBuffer;				
	//--------------------------FogInfo-----------------------------------
	uint32_t volumetricFogCount = 1;										
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;			
	FzbRenderer::Buffer volumetricFogInfosBuffer;							
	std::vector<int> volumetricFogInfoModified;								

	HeightFogSet heightFogSet;
	FluidFogSet fluidFogSet;

	uint32_t volumetricFogNoiseCount = 0;
	std::map<int, int> volumetricFogNoiseIndexMap;							
	std::vector<shaderio::NoiseFogInfo> volumetricFogNoiseInfos;			
	FzbRenderer::Buffer volumetricFogNoiseInfoBuffer;	
	//-------------------------Frustum------------------------------------
	VkExtent3D frustumGridSize;
	//-------------------------Environment--------------------------------
	FzbRenderer::Image envVolumetricFogInfoImage;

	bool envChange = false;
	shaderio::float3 envStartPos = { 5083.0f, -77.5f, -4480.0f };
	shaderio::uint3 envGridSize = { 128, 128, 128 };
	shaderio::float3 envVoxelSize = {1.6, 1, 1};
	bool showEnvGrid = false;

	int sampleCount_env = 1;
	float jitterStrength_env = 0.0f;
	//-------------------------FogAcc-------------------------------------
	FzbRenderer::Image volumetricFogAttenuationImage;
	FzbRenderer::Image volumetricFogLImage;

	uint32_t rmSampleCountSampleCount_fogAcc = 4;
	int randomStepping_fogAcc = true;
	float accJitterStrength = 0.0f;
	float interpolationJitterStrength_fogAcc = 0.0f;
	//-------------------------Opaque------------------------------------
	uint32_t forwardSampleCount = 0;
	uint32_t rmSampleCount_opaque_noFogAcc = 50;
	uint32_t rmSampleCount_opaque_FogAcc = 1;
	uint32_t randomStepping_opaque = true;
	float interpolationJitterStrength_attenuation = 1.0f;
	float interpolationJitterStrength_L = 30.0f;
	//-------------------------fogBlur------------------------------------
	//bool useFogBlur = false;
	int FogFilterCount = 4;

#ifndef NDEBUG
	void renderVolumetricFogVoxelGrid(VkCommandBuffer cmd);
	void renderCameraFrustum(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_renderVoxelGrid{};
	VkShaderEXT fragmentShader_renderVoxelGrid{};

	VkShaderEXT vertexShader_renderCameraFrustum{};
	VkShaderEXT fragmentShader_renderCameraFrustum{};

	bool showVolumetricFogVoxelGrid = false;
	std::vector<int> showVolumetricFogVoxelGrids;

	bool showCameraFrustum = false;
	shaderio::SceneInfo showCameraInfo;
	nvvk::Buffer bShowCameraInfo;

	VkShaderEXT computeShader_test{};
	void test(VkCommandBuffer cmd);
#endif
};
}

#endif