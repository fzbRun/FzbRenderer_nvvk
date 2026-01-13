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
	
	virtual void init();
	virtual void clean();
	virtual void uiRender();
	virtual void resize(VkCommandBuffer cmd, const VkExtent2D& size);
	virtual void preRender();
	virtual void render(VkCommandBuffer cmd);		//render其实可以分为在renderer的render之前还是之后
	virtual void postProcess(VkCommandBuffer cmd);		//同理，可以分为之前还是之后

	virtual void createGBuffer(bool useDepth = true, bool postProcess = true, uint32_t colorAttachmentCount = 1, VkExtent2D resolution = {0, 0});
	virtual void createDescriptorSetLayout();
	virtual void createPipelineLayout(uint32_t pushConstantSize = sizeof(shaderio::DefaultPushConstant));
	virtual void addTextureArrayDescriptor(uint32_t textureBinding = shaderio::eTextures, nvvk::DescriptorPack* descPackPtr = nullptr);

	virtual void compileAndCreateShaders();
	virtual void updateDataPerFrame(VkCommandBuffer cmd);

	nvvk::GBuffer gBuffers{};	//GBuffer实际上可以认为是所有的需要的纹理数据
	nvvk::DescriptorPack staticDescPack;
	nvvk::DescriptorPack dynamicDescPack;	//每帧更新的描述符集合
	VkPipelineLayout pipelineLayout{};
	nvvk::GraphicsPipelineState graphicsDynamicPipeline;

	FzbRenderer::Scene scene;
};
}

#endif