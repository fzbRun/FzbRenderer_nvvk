#pragma once

#include "feature/Feature.h"
#include "./LightClusteringShaderio.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_Light_Clustering_VRPG_H
#define FZBRENDERER_Light_Clustering_VRPG_H
namespace FzbRenderer {
class LightClustering_VRPG : public Feature {
public:
	LightClustering_VRPG() = default;
	virtual ~LightClustering_VRPG() = default;

	LightClustering_VRPG(pugi::xml_node& featureNode);

	void init();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void uiRender();
	void updateDataPerFrame(VkCommandBuffer cmd) override;
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);
	void clean();
private:
	void createSource();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();

	nvvk::Buffer clusteringGlobalInfoBuffer;

	nvvk::Buffer lightNodeBlockInfoBuffer;
	nvvk::Buffer hasDataLightNodeBlockIndexBuffer;
	VkShaderEXT computeShader_init{};
	VkShaderEXT computeShader_GetHasDataLightNodeBlockInfo{};
	VkShaderEXT computeShader_DispatchClustering{};
	VkShaderEXT computeShader_Clustering1{};
	VkShaderEXT computeShader_Clustering2{};
	void createLightNodeOctree(VkCommandBuffer cmd);

	shaderio::LightClusteringPushConstant_VRPG pushConstant{};
	VkPushConstantsInfo pushInfo;

	// debug view of the light tree (VRPG: Octree_VRPG::lightTreeDebug)
	void lightTreeDebug(VkCommandBuffer cmd);
	bool showLightTree = false;
	shaderio::LightClusteringDebugPushConstant_VRPG debugPushConstant;
	VkShaderEXT vertexShader_TreeDebug{};
	VkShaderEXT fragmentShader_TreeDebug{};
	VkPushConstantsInfo debugPushInfo{};
};
}
#endif