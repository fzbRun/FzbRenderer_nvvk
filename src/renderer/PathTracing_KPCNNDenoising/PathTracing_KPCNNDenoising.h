#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "common/Image/Image.h"
#include "common/Semaphore/Semaphore.h"
#include <common/Buffer/Buffer.h>
#include "./PathTracing_KPCNNDenosingShaderio.h"

#ifndef FZBRENDERER_KPCNN_DENOSING_PATHTRACING_H
#define FZBRENDERER_KPCNN_DENOSING_PATHTRACING_H

namespace FzbRenderer {
class PathTracing_KPCNNDenoising : public PathTracingRenderer {
public:
	PathTracing_KPCNNDenoising() = default;
	~PathTracing_KPCNNDenoising() = default;

	PathTracing_KPCNNDenoising(pugi::xml_node& rendererNode);

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

	void pathTracing(VkCommandBuffer cmd);

private:
	shaderio::KPCNN_DenoisingPTPushConstant pushConstant{};
	VkShaderEXT computeShader_PathTracing{};
	VkShaderEXT computeShader_makeInputBuffer{};
	VkShaderEXT computeShader_Denoising{};

	FzbRenderer::Buffer inputBuffer_diff;
	FzbRenderer::Buffer inputBuffer_spec;
	FzbRenderer::Semaphore vulkanToCudaSemaphore;
	FzbRenderer::Semaphore cudaToVulkanSemaphore;
};
}

#endif