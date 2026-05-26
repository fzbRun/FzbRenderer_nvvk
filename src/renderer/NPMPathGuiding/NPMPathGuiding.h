#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "common/Image/Image.h"
#include "common/Semaphore/Semaphore.h"
#include "test.cuh"
#include "./NPMPathGuidingShaderio.h"

#ifndef FZBRENDERER_NPM_PATHGUIDING_H
#define FZBRENDERER_NPM_PATHGUIDING_H

namespace FzbRenderer {
class NPMPathGuiding : public PathTracingRenderer {
public:
	NPMPathGuiding() = default;
	~NPMPathGuiding() = default;

	NPMPathGuiding(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void pathGuiding(VkCommandBuffer cmd);

private:
	shaderio::NPMPathGuidingPushConstant pushConstant{};
	VkShaderEXT computeShader_NPMPathGuiding{};

	FzbRenderer::Image flowerImage;
	FzbRenderer::Semaphore vulkanToCudaSemaphore;
	FzbRenderer::Semaphore cudaToVulkanSemaphore;

	Image_yReversal cudaPrograme;
};
}

#endif