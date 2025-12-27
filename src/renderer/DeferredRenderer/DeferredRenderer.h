#pragma once

#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/descriptors.hpp>

#include "renderer/Renderer.h"
#include <glm/ext/vector_float2.hpp>

#ifndef FZB_DEFERRED_RENDERER_H
#define FZB_DEFERRED_RENDERER_H

namespace FzbRenderer {
class DeferredRenderer : public FzbRenderer::Renderer {
public:
	DeferredRenderer() = default;
	~DeferredRenderer() = default;

	DeferredRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void render(VkCommandBuffer cmd) override;

	void compileAndCreateShaders() override;
private:
	VkShaderEXT vertexShader{};
	VkShaderEXT fragmentShader{};

	glm::vec2 metallicRoughnessOverride{ -0.01f, -0.01f };
};
}

#endif