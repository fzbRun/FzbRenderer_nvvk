#pragma once

#include "../../../feature/Feature.h"
#include "../../../feature/PathTracing/PathTracing.h"
#include "./shaderio.h"
#include <feature/SceneDivision/SparseVoxelOctree/SparseVoxelOctree.h>

#ifndef FZBRENDERER_SVO_WEIGHT_H
#define FZBRENDERER_SVO_WEIGHT_H

namespace FzbRenderer {
struct SVOWeightSetting{
	std::shared_ptr<SparseVoxelOctree> svo;
	AccelerationStructureManager* asManager;
};

class SVOWeight : public Feature {
public:
	SVOWeight() = default;
	virtual ~SVOWeight() = default;

	SVOWeight(pugi::xml_node& featureNode);

	void init(SVOWeightSetting settingInfo);
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd) override;
	void postProcess(VkCommandBuffer cmd) override;

	void createWeightArray();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	SVOWeightSetting setting;
	AccelerationStructureManager asManager;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

	nvvk::Buffer GlobalInfoBuffer;
	nvvk::Buffer indivisibleNodeInfosBuffer_G;
	nvvk::Buffer indivisibleNodeInfosBuffer_E;
	nvvk::Buffer weightBuffer;
	nvvk::Buffer weightSumsBuffer;

	VkShaderEXT computeShader_getIndivisibleNode{};
	VkShaderEXT computeShader_initWeights{};
	VkShaderEXT computeShader_getWeights{};
	VkShaderEXT computeShader_getProbability{};
private:
	shaderio::SVOWeightPushConstant pushConstant;
	VkPushConstantsInfo pushInfo;
	VkShaderModuleCreateInfo shaderCode;

	VkPhysicalDeviceRayQueryFeaturesKHR rayqueryFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR };

	void getIndivisibleNode(VkCommandBuffer cmd);
	void initWeights(VkCommandBuffer cmd);
	void getWeights(VkCommandBuffer cmd);
	void getProbability(VkCommandBuffer cmd);

#ifndef NDEBUG
	void debugPrepare();
	void debug_visualization(VkCommandBuffer cmd);

	glm::vec3 samplePoint = glm::vec3(0.8f, 1.99f, 0.5f);
	glm::vec3 outgoing = glm::vec3(0.0f, 1.0f, -1.0f);

	VkShaderEXT computeShader_getSampleNodeInfo{};
	VkShaderEXT vertexShader_visualization{};
	VkShaderEXT fragmentShader_visualization{};

	bool show = false;
#endif
};
}

#endif