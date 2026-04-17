#pragma once

#include "feature/Feature.h"
#include "feature/PathTracing/PathTracing.h"
#include "./shaderio.h"
#include "../RasterVoxelization/RasterVoxelizationSVOPG.h"
#include "../Octree/OctreeSVOPG.h"

#ifdef USE_SVO
#include "../SVO/SVO.h"
#endif

#ifndef FZBRENDERER_SVO_WEIGHT_H
#define FZBRENDERER_SVO_WEIGHT_H

namespace FzbRenderer {
struct SVOWeightSetting {
	#ifdef USE_SVO
	std::shared_ptr<SVO_SVOPG> svo;
	#endif
	std::shared_ptr<Octree_SVOPG> octree;
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

	shaderio::float3x3 randomRotateMatrix;

	nvvk::Buffer GlobalInfoBuffer;
	nvvk::Buffer weightBuffer;
	nvvk::Buffer nearbyNodeInfosBuffer;

	VkShaderEXT computeShader_initWeights{};
	VkShaderEXT computeShader_getWeights{};
	VkShaderEXT computeShader_getProbability{};

	VkShaderEXT computeShader_getNearbyNodes{};
	VkShaderEXT computeShader_getNearbyNodes2{};
private:

	shaderio::SVOWeightPushConstant pushConstant;
	VkPushConstantsInfo pushInfo;
	VkShaderModuleCreateInfo shaderCode;

	void initWeights(VkCommandBuffer cmd);
	void getNearbyNodeInfos(VkCommandBuffer cmd);
	void getWeights(VkCommandBuffer cmd);
	void getProbability(VkCommandBuffer cmd);

#ifndef NDEBUG
public:
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
private:
	void debugPrepare();

	void debug_visualization(VkCommandBuffer cmd);
	void debug_nearby(VkCommandBuffer cmd);

	//glm::vec3 samplePoint = glm::vec3(-1.5f, 1.0f, 0.5f);
	//glm::vec3 samplePoint = glm::vec3(1.0f, 1.7f, 0.5f);
	glm::vec3 samplePoint = glm::vec3(-2.2f, 1.0f, -4.5f);
	glm::vec3 outgoing = glm::vec3(0.0f, 1.0f, -1.0f);

	VkShaderEXT computeShader_getSampleNodeInfo{};
	VkShaderEXT vertexShader_visualization{};
	VkShaderEXT fragmentShader_visualization{};

	VkShaderEXT vertexShader_nearby{};
	VkShaderEXT fragmentShader_nearby{};

	bool showWeightMap = false;
	bool showNearbyMap = false;

	VkImageView depthImageView;
#endif
};
}

#endif