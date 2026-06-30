#pragma once

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

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

	void createGBuffers(VkCommandBuffer cmd);
	void createVolumetricFog(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);

	VkPhysicalDeviceComputeShaderDerivativesFeaturesKHR derivFeatures = {};

	VkPushConstantsInfo pushInfo;

	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};

	VkShaderEXT computeShader_getVisibleVolumetricFog{};
	VkShaderEXT computeShader_createVolumetricFog{};
	VkShaderEXT computeShader_createLightAttenuationEstimator{};

	VkShaderEXT computeShader_deferredRenderring{};

	shaderio::VolumetricFogPushConstant pushConstant;

	FzbRenderer::Buffer GlobalInfoBuffer;

	uint32_t volumetricFogCount = 1;
	std::vector<FzbRenderer::Image> volumetricFogImages;
	std::vector<shaderio::VolumetricFogInfo> volumetricFogInfos;
	FzbRenderer::Buffer volumetricFogInfosBuffer;
	std::vector<int> volumetricFogInfoModified;

	FzbRenderer::Buffer visibleVolumetricFogIndexBuffer;

	ShadowMap shadowMap;

#ifndef NDEBUG
	void renderVolumetricFogVoxelGrid(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_renderVoxelGrid{};
	VkShaderEXT fragmentShader_renderVoxelGrid{};

	bool showVolumetricFogVoxelGrid = false;
	std::vector<int> showVolumetricFogVoxelGrids;
#endif
};
}

#endif