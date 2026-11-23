#pragma once

#include "feature/Feature.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_VRPG2_RIS_H
#define FZBRENDERER_VRPG2_RIS_H
namespace FzbRenderer {
enum class GBuffers_RIS_VRPG2 : uint32_t {
	eClusterPairResult = 0,
	eBufferCount,
};

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
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void postProcess(VkCommandBuffer cmd) override;

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

	// Ported from VRPG Octree_VRPG::clusterPairDebug: draws the candidate chain of the geometric
	// cluster picked from the pixel under the cursor (right click / VRPG2_AUTO_PICK).
	

	shaderio::RISPushConstant_VRPG2 pushConstant{};
	VkPushConstantsInfo pushInfo{};

	shaderio::ClusterPairDebugPushConstant_VRPG2 debugPushConstant2{};
	VkPushConstantsInfo debugPushInfo{};

	nvvk::Buffer RISGlobalInfoBuffer;

	VkShaderEXT computeShader_init{};

	VkShaderEXT computeShader_getCandidateNodes_1{};
	VkShaderEXT computeShader_getCandidateNodes_2{};
	VkShaderEXT computeShader_getCandidateNodes_3{};
	VkShaderEXT computeShader_getCandidateNodes_4{};
	VkShaderEXT computeShader_getCandidateNodes_5{};

	VkShaderEXT computeShader_getCandidateNodes{};
	
	VkShaderEXT computeShader_dispatchGetCandidateLeafNodes{};
	VkShaderEXT computeShader_getCandidateLeafNodes{};
	VkShaderEXT computeShader_clusterPairHitTest{};
	VkShaderEXT computeShader_getCandidateNodeWeight{};
	VkShaderEXT computeShader_createCandidatePrefixWSum{};

	nvvk::Buffer clusterPairDebugGlobalInfoBuffer;
	VkShaderEXT computeShader_GetDebugClusterG{};
	VkShaderEXT vertexShader_ClusterPairDebug{};
	VkShaderEXT fragmentShader_ClusterPairDebug{};
	bool showClusterPairDebugMap = false;
	bool showNearbyClusters = false;
	int  debugClusterIndex = -1;
	bool changeClusterG = true;
	void clusterPairDebug(VkCommandBuffer cmd);
};
}
#endif
