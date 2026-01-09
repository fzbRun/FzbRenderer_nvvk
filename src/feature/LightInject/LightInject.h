#pragma once

#include "feature/PathTracing/PathTracing.h"
#include "./shaderio.h"

#ifndef FZBRENDERER_FEATURE_LIGHTINJECT_H
#define FZBRENDERER_FEATURE_LIGHTINJECT_H

namespace FzbRenderer {
struct LightInjectSetting{
	nvvk::Buffer VGB;
	glm::vec4 VGBStartPos_Size;
	PathTracingContext* ptContext;
	AccelerationStructureManager* asManager;
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

	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void debug_Cube(VkCommandBuffer cmd);

	LightInjectSetting setting;
	VkPipeline rtPipeline{};

private:
	shaderio::LightInjectPushConstant pushConstant;
};
}

#endif