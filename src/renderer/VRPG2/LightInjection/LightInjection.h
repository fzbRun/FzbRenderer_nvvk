#pragma once

#include "feature/Feature.h"
#include <pugixml.hpp>
#include "./LightInjectionShaderio.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_Light_Injection_VRPG
#define FZBRENDERER_Light_Injection_VRPG

namespace FzbRenderer {
class LightInjection_VRPG : public Feature {
public:
	LightInjection_VRPG() = default;
	virtual ~LightInjection_VRPG() = default;

	LightInjection_VRPG(pugi::xml_node& featureNode);

	void init() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void uiRender() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);
	void clean() override;

private:
	void createSource();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();

	void compileAndCreateShaders() override;


	shaderio::LightInjectPushConstant_VRPG pushConstant;
	VkPushConstantsInfo pushInfo;

	nvvk::Buffer hasGeometryNodeInfoBuffer;
	nvvk::Buffer globalInfoBuffer;
	VkShaderEXT computeShader_init{};
	VkShaderEXT computeShader_dispatchLightInjection{};
	VkShaderEXT computeShader_lightInjection{};
	void injection(VkCommandBuffer cmd);

	// debug view of the injected voxels (VRPG: LightInject::debug_Cube)
	void injectionCube(VkCommandBuffer cmd);
	bool showInjectionResult = false;
	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};
	VkPushConstantsInfo debugPushInfo{};
};
}

#endif