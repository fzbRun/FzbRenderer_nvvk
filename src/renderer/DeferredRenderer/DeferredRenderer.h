#pragma once

#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/descriptors.hpp>

#include "renderer/Renderer.h"
#include <glm/ext/vector_float2.hpp>

#ifndef FZB_DEFERRED_RENDERER_H
#define FZB_DEFERRED_RENDERER_H

namespace FzbRenderer {
class DeferredRenderer : public FzbRenderer::Renderer {
	enum
	{
		eImgRendered,
		eImgTonemapped
	};

public:
	DeferredRenderer() = default;
	~DeferredRenderer() = default;

	DeferredRenderer(RendererCreateInfo& createInfo);
	void init() override;
	void compileAndCreateGraphicsShaders();
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void render(VkCommandBuffer cmd) override;
	void onLastHeadlessFrame() override;
private:
	void createImage();
	void createGraphicsDescriptorSetLayout();
	void createGraphicsPipelineLayout();
	VkShaderModuleCreateInfo compileSlangShader(const std::filesystem::path& filename, const std::span<const uint32_t>& spirv);
	void updateTextures();
	void postProcess(VkCommandBuffer cmd) override;

	nvvk::GBuffer            gBuffers{};
	nvvk::GraphicsPipelineState dynamicPipeline;
	nvvk::DescriptorPack        descPack;
	VkPipelineLayout            graphicPipelineLayout{};

	VkShaderEXT vertexShader{};
	VkShaderEXT fragmentShader{};

	glm::vec2 metallicRoughnessOverride{ -0.01f, -0.01f };
};
}

#endif