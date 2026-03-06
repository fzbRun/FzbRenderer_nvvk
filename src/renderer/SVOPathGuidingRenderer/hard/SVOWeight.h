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
	PathTracingContext* ptContext;
};

class SVOWeight : public PathTracing {
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

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	SVOWeightSetting setting;
	AccelerationStructureManager asManager;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

	nvvk::Buffer weightBuffer;
private:
	shaderio::SVOWeightPushConstant pushConstant;
	VkShaderModuleCreateInfo shaderCode;
};
}

#endif