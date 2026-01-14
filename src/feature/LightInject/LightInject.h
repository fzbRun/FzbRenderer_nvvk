#pragma once

#include "feature/PathTracing/PathTracing.h"
#include "./shaderio.h"

#ifndef FZBRENDERER_FEATURE_LIGHTINJECT_H
#define FZBRENDERER_FEATURE_LIGHTINJECT_H

namespace FzbRenderer {
struct LightInjectSetting{
	nvvk::Buffer VGB;
	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;
	PathTracingContext* ptContext;
	AccelerationStructureManager* asManager;
};

enum LightInjectGBuffer {
	LightInjectResult = 0,
	CubeMap_LightInject,
};

class LightInject : public PathTracing {
public:
	LightInject() = default;
	virtual ~LightInject() = default;

	LightInject(pugi::xml_node& featureNode);

	void init(LightInjectSetting settingInfo);
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

	LightInjectSetting setting;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

private:
	shaderio::LightInjectPushConstant pushConstant;
	VkShaderModuleCreateInfo shaderCode;
};
}

#endif