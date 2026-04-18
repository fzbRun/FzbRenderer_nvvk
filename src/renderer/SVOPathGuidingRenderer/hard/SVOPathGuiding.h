#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./shaderio.h"
#include "./RasterVoxelization/RasterVoxelizationSVOPG.h"
#include "./LightInject/LightInjectSVOPG.h"
#include "./Octree/OctreeSVOPG.h"
#include "./SVO/SVO.h"
#include "Weight/SVOWeight.h"

#ifndef FZBRENDERER_SVO_PATHGUIDING_H
#define FZBRENDERER_SVO_PATHGUIDING_H

#define USE_RAYQUERY_SVOPG

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
	std::shared_ptr<RasterVoxelization_SVOPG> rasterVoxelization;
	std::shared_ptr<LightInject_SVOPG> lightInject;
	std::shared_ptr<Octree_SVOPG> octree;
	#ifdef USE_SVO
	std::shared_ptr<SVO_SVOPG> svo;
	#endif
	std::shared_ptr<SVOWeight> svoWeight;

private:
	VkShaderEXT computeShader_SVOPathGuiding{};
};
}

#endif