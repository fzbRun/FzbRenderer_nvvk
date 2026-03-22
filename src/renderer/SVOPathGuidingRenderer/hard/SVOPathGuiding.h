#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./shaderio.h"
#include <feature/LightInject/LightInject.h>
#include <feature/SceneDivision/Octree/Octree.h>
#include <feature/SceneDivision/SparseVoxelOctree/SparseVoxelOctree.h>
#include "SVOWeight.h"
#include "RasterVoxelizationSVOPG.h"

#ifndef FZBRENDERER_SVO_PATHGUIDING_H
#define FZBRENDERER_SVO_PATHGUIDING_H

//#define USE_RAYQUERY_SVOPG

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
	void createPipelineLayout();
	void createShader();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void pathGuiding(VkCommandBuffer cmd);
	void pathGuiding_rayQuery(VkCommandBuffer cmd);

	shaderio::SVOPathGuidingPushConstant pushConstant{};
	std::shared_ptr<FzbRenderer::RasterVoxelization_SVOPG> rasterVoxelization;
	std::shared_ptr<FzbRenderer::LightInject> lightInject;
	std::shared_ptr<FzbRenderer::Octree> octree;
	std::shared_ptr<FzbRenderer::SparseVoxelOctree> svo;
	std::shared_ptr<FzbRenderer::SVOWeight> svoWeight;

private:
	VkShaderEXT computeShader_SVOPathGuiding{};
};
}

#endif