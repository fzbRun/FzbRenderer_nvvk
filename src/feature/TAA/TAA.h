#pragma once

#ifndef FZBRENDERER_TAA_H
#define FZBRENDERER_TAA_H

#include "feature/Feature.h"
#include "common/Image/Image.h"
#include "./TAAShaderio.h"

namespace FzbRenderer {
struct TAACreateInfo{
	float mergeRatio = 0.05f;
	nvvk::Image depthImage;
	nvvk::Image velocityImage;
	nvvk::Image renderTarget;
};
class TAA : public Feature {
public:
	TAA();
	virtual ~TAA() = default;

	void init(TAACreateInfo createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image images[3]);
	void preRender();

	void mergeResult(VkCommandBuffer cmd);

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();

	shaderio::float4x4 projMatrix_taaJitter;

private:
	TAACreateInfo setting;
	shaderio::float2 Halton_2_3[8];
	uint32_t frameIndex = 0;
	shaderio::TAAPushConstant pushConstant;

	VkShaderEXT computeShader_mergeRenderTarget{};
};
}

#endif
