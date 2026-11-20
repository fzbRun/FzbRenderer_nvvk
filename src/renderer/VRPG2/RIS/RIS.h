#pragma once

#include "feature/Feature.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_VRPG2_RIS_H
#define FZBRENDERER_VRPG2_RIS_H
namespace FzbRenderer {
// Resampled importance sampling over the light tree. Fills the candidate tables that the path
// guiding pass consumes. Two candidate-selection strategies are compiled in, selected by
// VRPG_Candidate_Samples_SampleMode in RISShaderio.h (no spatiotemporal reuse).
class RIS_VRPG : public Feature {
public:
	RIS_VRPG() = default;
	virtual ~RIS_VRPG() = default;

	RIS_VRPG(pugi::xml_node& featureNode);

	void init();
	void clean();

	void preRender();
	void updateDataPerFrame(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd);
	void uiRender();

private:
	void createSource();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();

	void initCandidateNodes(VkCommandBuffer cmd);
	void selectCandidates(VkCommandBuffer cmd);
	void getCandidateLeafNodes(VkCommandBuffer cmd);
	void clusterPairHitTest(VkCommandBuffer cmd);
	void getCandidateNodeWeight(VkCommandBuffer cmd);
	void createCandidatePrefixWSum(VkCommandBuffer cmd);

#ifndef NDEBUG
	// label -> (layer, node) -> candidate chain can be compared with the guide lookup.
#endif

	shaderio::RISPushConstant_VRPG2 pushConstant{};
	VkPushConstantsInfo pushInfo{};


	VkShaderEXT computeShader_initCandidateNodes{};
	VkShaderEXT computeShader_selectCandidates{};
	VkShaderEXT computeShader_emitClusterPairs{};
	VkShaderEXT computeShader_getCandidateLeafNodes{};
	VkShaderEXT computeShader_clusterPairHitTest{};
	VkShaderEXT computeShader_getCandidateNodeWeight{};
	VkShaderEXT computeShader_createCandidatePrefixWSum{};

	bool dirty = true;
};
}
#endif
