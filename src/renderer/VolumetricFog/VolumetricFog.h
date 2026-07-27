#pragma once

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>
#include <feature/TAA/TAA.h>

namespace FzbRenderer {
enum class GBuffers_VolumetricFog{
	eAlbedo = 0,
	eNormal,
	eEmissive,
	eVelocity,
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
	void createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear = true);
	void createVolumetricFogData();

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void getVisibleFog(VkCommandBuffer cmd);
	void initVolumetricFogFluid(VkCommandBuffer cmd);
	void createGBuffers(VkCommandBuffer cmd);
	void fluidSimulation(VkCommandBuffer cmd);
	void createEnvFog(VkCommandBuffer cmd);
	void envFogLightAttenuationEstimate(VkCommandBuffer cmd);
	void createFrustumAccFog(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);
	void renderTransparentMaterial(VkCommandBuffer cmd);

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivFeatures = {};

	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_getVisibleVolumetricFog{};

	VkShaderEXT computeShader_initVolumetricFogFluid{};

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

	VkShaderEXT computeShader_deferredRenderring{};

	VkShaderEXT vertexShader_renderTransparentMaterial{};
	VkShaderEXT fragmentShader_renderTransparentMaterial{};

	shaderio::VolumetricFogPushConstant pushConstant;

	TAA taa;
	ShadowMap shadowMap;

	// mouse force
	glm::vec3 mouseForcePosition = glm::vec3(0.0f);
	float mouseForceStrength = 0.0f;
	float mouseForceRadius = 3.0f;

	glm::vec3 m_lastCameraPos = glm::vec3(0.0f);
	shaderio::float4x4 VPMatrix_lastFrame;

	FzbRenderer::Buffer GlobalInfoBuffer;				
	//--------------------------FogInfo-----------------------------------
	uint32_t volumetricFogCount = 1;										
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;			
	FzbRenderer::Buffer volumetricFogInfosBuffer;							
	std::vector<int> volumetricFogInfoModified;								

	uint32_t volumetricFogHeightCount = 0;									
	std::map<int, int> volumetricFogHeightIndexMap;							
	std::vector<shaderio::HeightFogInfo> volumetricFogHeightInfos;			
	FzbRenderer::Buffer volumetricFogHeightInfoBuffer;						

	uint32_t volumetricFogFluidCount = 0;								
	std::map<int, int> volumetricFogFluidIndexMap;							
	std::vector<shaderio::FluidFogInfo> volumetricFogFluidInfos;			
	FzbRenderer::Buffer volumetricFogFluidInfoBuffer;						
	std::vector<FzbRenderer::Buffer> volumetricFogFluidVoxelInfoBuffers;	
	std::vector<FzbRenderer::Image> volumetricFogFluidVoxelVelocityImages;	
	std::vector<FzbRenderer::Image> volumetricFogFluidVoxelInfoImages;	

	shaderio::float3 fluidLocalStartPos;

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

	int sampleCount_env = 20;
	//-------------------------FogAcc-------------------------------------
	FzbRenderer::Image volumetricFogAttenuationImage;
	FzbRenderer::Image volumetricFogLImage;

	uint32_t rmSampleCountSampleCount_fogAcc = 20;
	int randomStepping_fogAcc = false;
	float accJitterStrength = 0.0f;
	//-------------------------Opaque------------------------------------
	uint32_t forwardSampleCount = 0;
	uint32_t rmSampleCount_opaque_noFogAcc = 10;
	uint32_t rmSampleCount_opaque_FogAcc = 1;
	uint32_t randomStepping_opaque = true;
	float interpolationJitterStrength = 0.0f;

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
#endif
};
}

#endif