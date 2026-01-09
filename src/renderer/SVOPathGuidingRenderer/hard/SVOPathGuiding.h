#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./shaderio.h"
#include <feature/LightInject/LightInject.h>

#ifndef FZBRENDERER_SVO_PATHGUIDING_H
#define FZBRENDERER_SVO_PATHGUIDING_H

namespace FzbRenderer {
class SVOPathGuidingRenderer : public PathTracingRenderer {
public:
	SVOPathGuidingRenderer() = default;
	~SVOPathGuidingRenderer() = default;

	SVOPathGuidingRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd) override;

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	shaderio::SVOPathGuidingPushConstant pushValues{};
	std::shared_ptr<FzbRenderer::RasterVoxelization> rasterVoxelization;
	std::shared_ptr<FzbRenderer::LightInject> lightInject;
};
}

#endif