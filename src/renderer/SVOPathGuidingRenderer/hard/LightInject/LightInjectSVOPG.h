#pragma once

#include "feature/PathTracing/PathTracing.h"
#include "./shaderio.h"

#ifndef FZBRENDERER_FEATURE_LIGHTINJECT_H
#define FZBRENDERER_FEATURE_LIGHTINJECT_H

namespace FzbRenderer {
struct LightInjectSetting_SVOPG{
	std::vector<nvvk::Buffer> VGBs;
	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;
	shaderio::float3 sceneStartPos;
	shaderio::float3 sceneSize;
	PathTracingContext* ptContext;
	AccelerationStructureManager* asManager;
};

enum LightInjectGBuffer {
	LightInjectResult = 0,
	CubeMap_LightInject,
};

class LightInject_SVOPG : public PathTracing {
public:
	LightInject_SVOPG() = default;
	virtual ~LightInject_SVOPG() = default;

	LightInject_SVOPG(pugi::xml_node& featureNode);

	void init(LightInjectSetting_SVOPG settingInfo);
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

#ifndef NDEBUG
	void debug_Cube(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};
	bool showLightInjectResult = false;
	bool showCubeMap = false;
#endif

	LightInjectSetting_SVOPG setting;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

private:
	shaderio::LightInjectPushConstant_SVOPG pushConstant;
	VkShaderModuleCreateInfo shaderCode;
};
}

#endif