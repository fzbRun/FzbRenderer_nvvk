#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./FzbPathGuidingShaderio.h"
#include "RasterVoxelization/RasterVoxelization_FzbPG.h"
#include "LightInject/LightInject_FzbPG.h"
#include "Octree/Octree_FzbPG.h"

#ifndef FZBRENDERER_FZB_PATHGUIDING_H
#define FZBRENDERER_FZB_PATHGUIDING_H

namespace FzbRenderer {
class FzbPathGuidingRenderer : public PathTracingRenderer {
public:
	FzbPathGuidingRenderer() = default;
	~FzbPathGuidingRenderer() = default;

	FzbPathGuidingRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer cmd) override;
	
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void pathGuiding(VkCommandBuffer cmd);
private:
	std::shared_ptr<RasterVoxelization_FzbPG> rasterVoxelization;
	std::shared_ptr<LightInject_FzbPG> lightInject;
	std::shared_ptr<Octree_FzbPG> octree;

	shaderio::FzbPathGuidingPushConstant pushConstant{};
	VkShaderEXT computeShader_FzbPathGuiding{};
};
}

#endif