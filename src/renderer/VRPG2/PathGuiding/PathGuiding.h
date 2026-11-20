#pragma once

#include "feature/Feature.h"
#include "renderer/VRPG2/PathGuiding/PathGuidingShaderio.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_VRPG2_PATHGUIDING_H
#define FZBRENDERER_VRPG2_PATHGUIDING_H
namespace FzbRenderer {
struct PathGuidingCreateInfo_VRPG2 {
	nvvk::GBuffer* gBuffers = nullptr;	// render target (eImgRendered / eImgTonemapped)
	AccelerationStructureManager* asManager = nullptr;
};

// Path tracing with guided sampling: every bounce mixes a BSDF sample and a sample drawn from the
// RIS candidate tables of the current geometric cluster (balance heuristic MIS).
class PathGuiding_VRPG : public Feature {
public:
	PathGuiding_VRPG() = default;
	virtual ~PathGuiding_VRPG() = default;

	PathGuiding_VRPG(pugi::xml_node& featureNode);

	void init(PathGuidingCreateInfo_VRPG2 createInfo);
	void clean();
	void uiRender();

	void preRender();
	void render(VkCommandBuffer cmd);

private:
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();
	void updateDataPerFrame(VkCommandBuffer cmd);

	shaderio::PathGuidingPushConstant_VRPG2 pushConstant{};
	VkPushConstantsInfo pushInfo{};

	nvvk::GBuffer* gBuffers = nullptr;
	AccelerationStructureManager* asManager = nullptr;

	VkPipelineLayout pipelineLayout{};

	VkShaderEXT computeShader_pathGuiding{};
};
}
#endif
