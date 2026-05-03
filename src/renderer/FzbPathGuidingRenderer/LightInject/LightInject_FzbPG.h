#pragma once

#include "feature/Feature.h"
#include "./LightInjectShaderio_FzbPG.h"
#include <feature/PathTracing/PathTracing.h>

#ifndef FZBRENDERER_LIGHTINJECT_FZBPATHGUIDING_H
#define FZBRENDERER_LIGHTINJECT_FZBPATHGUIDING_H

namespace FzbRenderer {
struct LightInjectCreateInfo_FzbPG {
	std::vector<nvvk::Buffer> VGBs;
	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;
	PathTracingContext* ptContext;
	AccelerationStructureManager* asManager;
};
class LightInject_FzbPG : public PathTracing {
public:
	LightInject_FzbPG() = default;
	virtual ~LightInject_FzbPG() = default;

	LightInject_FzbPG(pugi::xml_node& featureNode);

	void init(LightInjectCreateInfo_FzbPG settingInfo);
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd) override;
	void postProcess(VkCommandBuffer cmd) override;

	void createBuffer();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void getHasGeometryVoxels(VkCommandBuffer cmd);
	void lightInject(VkCommandBuffer cmd);

	nvvk::Buffer hasGeometryVoxelInfoBuffer;
	nvvk::Buffer globalInfoBuffer;

	VkShaderEXT computeShader_getHasGeometryVoxels{};
	VkShaderEXT computeShader_setDispatchIndirectCommand{};
	VkShaderEXT computeShader_LightInject{};

#ifndef NDEBUG
	void debug_Cube(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};
	bool showCubeMap = false;
#endif

	LightInjectCreateInfo_FzbPG setting;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

private:
	shaderio::LightInjectPushConstant_FzbPG pushConstant;
};
}

#endif

