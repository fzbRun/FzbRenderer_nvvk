#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "common/Image/Image.h"
#include "common/Semaphore/Semaphore.h"
#include <common/Buffer/Buffer.h>
#include "./ZhiHuCodeShaderio.h"
#include "./models/VulkanCudaOpTest.cuh"
#include "./models/UseModel.cuh"

#ifndef FZBRENDERER_ZHIHUCODE_H
#define FZBRENDERER_ZHIHUCODE_H

namespace FzbRenderer {
class ZhiHuCode : public PathTracingRenderer {
public:
	ZhiHuCode() = default;
	~ZhiHuCode() = default;

	ZhiHuCode(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;

private:
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void renderFunction(VkCommandBuffer cmd);

	shaderio::ZhiHuCodePushConstant pushConstant{};

	FzbRenderer::Image flowerImage;
	FzbRenderer::Semaphore vulkanToCudaSemaphore;
	FzbRenderer::Semaphore cudaToVulkanSemaphore;

	VkShaderEXT computeShader_copyImage{};

#ifdef Step1_VulkanCudaOp
	Image_yReversal cudaPrograme;
#elif defined(Step2_UseModel)
	void createInputBuffer(VkCommandBuffer cmd);

	FzbRenderer::Buffer inputBuffer;

	VkShaderEXT computeShader_createInputBuffer{};

	ImageRecognition cudaPrograme;
#endif
};
}

#endif