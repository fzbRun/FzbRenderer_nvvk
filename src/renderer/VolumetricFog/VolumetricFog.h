#pragma once

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./VolumetricFogShaderio.h"

#ifndef FZBRENDERER_VOLUMETRIC_FOG_H
#define FZBRENDERER_VOLUMETRIC_FOG_H

namespace FzbRenderer {
enum class GBuffers_VolumetricFog{
	eAlbedo = 0,
	eNormal,
	eRendered,
	eTonemapping,

	eVoxelGridDebug,
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
	void createVolumetricFogImage();

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void createGBuffers(VkCommandBuffer cmd);
	void createVolumetricFog(VkCommandBuffer cmd);
	void deferredRenderring(VkCommandBuffer cmd);

	VkPushConstantsInfo pushInfo;

	VkShaderEXT vertexShader_createGBuffer;
	VkShaderEXT fragmentShader_createGBuffer;

	VkShaderEXT computeShader_createVolumetricFog;

	VkShaderEXT computeShader_deferredRenderring;

	shaderio::VolumetricFogPushConstant pushConstant;
	FzbRenderer::Image volumetricFogImage;
};
}

#endif