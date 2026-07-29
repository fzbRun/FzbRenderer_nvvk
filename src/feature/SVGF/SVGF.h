#pragma once

#ifndef FZBRENDERER_SVGF_H
#define FZBRENDERER_SVGF_H

#include "feature/Feature.h"
#include "common/Image/Image.h"
#include "./SVGFShaderio.h"

namespace FzbRenderer {
enum class GBuffers_SVGF {
	DepthGradient = 0,
	Irradiance,
	moment1,
	moment2,
	Variance0,
	Variance1,
	Filter0,
	Filter1,

	historyDepth,
	historyNormal,
	historyVertexInfo,

	bufferCount,
};

struct SVGFCreateInfo {
	nvvk::Image albedoImage;
	nvvk::Image depthImage;
	nvvk::Image normalImage;
	nvvk::Image velocityImage;
	nvvk::Image vertexInfoImage;
	nvvk::Image renderTarget;
};
class SVGF : public Feature {
public:
	SVGF();
	virtual ~SVGF() = default;

	void init(SVGFCreateInfo createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::Image images[6]);
	void preRender();
	void render(VkCommandBuffer cmd);

	void getFeatures(VkCommandBuffer cmd);
	void filter(VkCommandBuffer cmd);

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();

private:
	SVGFCreateInfo setting;
	shaderio::SVGFPushConstant pushConstant;
	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_getDepthGradient{};
	VkShaderEXT computeShader_getEffectiveIrradiance{};
	VkShaderEXT computeShader_varianceEstimate{};
	VkShaderEXT computeShader_varianceConvolution{};
	VkShaderEXT computeShader_Filter_X{};
	VkShaderEXT computeShader_Filter_Y{};
};
}

#endif