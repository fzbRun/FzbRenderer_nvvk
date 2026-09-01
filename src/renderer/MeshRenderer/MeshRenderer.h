#pragma once

#ifndef FZBRENDERER_MESH_RENDERER_H
#define FZBRENDERER_MESH_RENDERER_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include <memory>
#include "./MeshRendererShaderio.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <common/Buffer/Buffer.h>
#include <feature/TAA/TAA.h>

namespace FzbRenderer {
class MeshRenderer : public Renderer {
public:
	MeshRenderer() = default;
	~MeshRenderer() = default;

	MeshRenderer(pugi::xml_node& rendererNode);

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

	VkPhysicalDeviceMeshShaderFeaturesEXT meshShaderFeatures{};
	VkPhysicalDeviceFragmentShadingRateFeaturesKHR fragmentShadingRateFeatures{};

	VkShaderEXT meshShader{};
	VkShaderEXT fragmentShader{};

	shaderio::MeshRendererPushConstant pushConstant;
};
}


#endif