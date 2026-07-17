#pragma once

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>

namespace FzbRenderer {
enum class GBuffers_VolumetricFog{
	eAlbedo = 0,
	eNormal,
	eEmissive,
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
	void createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size);
	void createVolumetricFogData();

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initVolumetricFogFluid(VkCommandBuffer cmd);
	void createGBuffers(VkCommandBuffer cmd);
	void createVolumetricFog(VkCommandBuffer cmd);
	void createAttenuationImage(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);
	void renderTransparentMaterial(VkCommandBuffer cmd);

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivFeatures = {};

	VkPushConstantsInfo pushInfo;

	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};

	VkShaderEXT computeShader_getVisibleVolumetricFog{};

	VkShaderEXT computeShader_initVolumetricFogFluid{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_A{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_D{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_F{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_P{};
	VkShaderEXT computeShader_createVolumetricFog_Fluid_S{};

	VkShaderEXT computeShader_createLightAttenuationEstimator{};

	VkShaderEXT computeShader_deferredRenderring{};
	VkShaderEXT vertexShader_renderTransparentMaterial{};
	VkShaderEXT fragmentShader_renderTransparentMaterial{};

	shaderio::VolumetricFogPushConstant pushConstant;

	ShadowMap shadowMap;

	// mouse force
	glm::vec3 mouseForcePosition = glm::vec3(0.0f);
	float mouseForceStrength = 0.0f;
	float mouseForceRadius = 3.0f;

	FzbRenderer::Buffer GlobalInfoBuffer;				

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

	VkShaderEXT computeShader_createAttenuationImage{};
	VkExtent3D attenuationImageSize;
	FzbRenderer::Image volumetricFogAttenuationImage;
	FzbRenderer::Image volumetricFogAttenuation2Image;
	FzbRenderer::Image volumetricFogLImage;

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