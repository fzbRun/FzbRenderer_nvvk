#pragma once
#include <vulkan/vulkan_core.h>
#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/descriptors.hpp>
#include <common/Shader/shaderStructType.h>
#include "common/Scene/Scene.h"

#ifndef FZBRENDERER_FEATURE_H
#define FZBRENDERER_FEATURE_H

namespace FzbRenderer {
class Feature{
public:
	Feature() = default;
	virtual ~Feature() = default;
	
	virtual void init() = 0;
	virtual void clean();
	virtual void uiRender();
	virtual void resize(VkCommandBuffer cmd, const VkExtent2D& size);
	virtual void render(VkCommandBuffer cmd) = 0;

	virtual void createGBuffer(bool useDepth = true);
	virtual void createGraphicsDescriptorSetLayout();
	virtual void createGraphicsPipelineLayout(uint32_t pushConstantSize = sizeof(shaderio::DefaultPushConstant));
	virtual void addTextureArrayDescriptor();

	virtual void compileAndCreateShaders();
	virtual void updateDataPerFrame(VkCommandBuffer cmd);

	nvvk::GBuffer gBuffers{};	//GBuffer实际上可以认为是所有的需要的纹理数据
	nvvk::DescriptorPack descPack;
	VkPipelineLayout graphicPipelineLayout{};
	nvvk::GraphicsPipelineState dynamicPipeline;

	FzbRenderer::Scene scene;
};
}

#endif