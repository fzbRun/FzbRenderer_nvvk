#pragma once

#include "feature/Feature.h"
#include <pugixml.hpp>
#include "./GeometryInjectionShaderio.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_Geometry_Injection_VRPG
#define FZBRENDERER_Geometry_Injection_VRPG

namespace FzbRenderer {
class GeometryInjection_VRPG : public Feature {
public:
	GeometryInjection_VRPG() = default;
	virtual ~GeometryInjection_VRPG() = default;

	GeometryInjection_VRPG(pugi::xml_node& featureNode);

	void init() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void uiRender() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;
	void render(VkCommandBuffer cmd);
	void clean() override;

private:
	void createVGBs();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();

	void compileAndCreateShaders() override;


	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterFeature{};

	shaderio::GeometryInjectionPushConstant_VRPG pushConstant;
	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_initVGB{};
	void initVGB(VkCommandBuffer cmd);

	VkShaderEXT vertexShader{};
	VkShaderEXT geometryShader{};
	VkShaderEXT fragmentShader{};
	void injection(VkCommandBuffer cmd);

	bool showThreeView = false;
	VkShaderEXT geometryShader_ThreeView{};
	VkShaderEXT fragmentShader_ThreeView{};
	void injection_ThreeView(VkCommandBuffer cmd);
};
}

#endif