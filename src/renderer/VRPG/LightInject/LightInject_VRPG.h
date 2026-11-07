#pragma once

#include "feature/Feature.h"
#include "./LightInjectShaderio_VRPG.h"
#include <feature/PathTracing/PathTracing.h>

#ifndef FZBRENDERER_LIGHTINJECT_VRPG_H
#define FZBRENDERER_LIGHTINJECT_VRPG_H

namespace FzbRenderer {
struct LightInjectCreateInfo_VRPG {
	std::vector<nvvk::Buffer> VGBs;          // light VGB
	std::vector<nvvk::Buffer> geometryVGBs;  // geometry VGB, used to cluster light voxels
	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;

	glm::vec3 geometryVGBStartPos;
	glm::vec3 geometryVGBVoxelSize;
	float geometryVGBSize;

	PathTracingContext* ptContext;
	AccelerationStructureManager* asManager;
};
class LightInject_VRPG : public PathTracing {
public:
	LightInject_VRPG() = default;
	virtual ~LightInject_VRPG() = default;

	LightInject_VRPG(pugi::xml_node& featureNode);

	void init(LightInjectCreateInfo_VRPG settingInfo);
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd) override;

	void createBuffer();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void clusterGeometryVGB(VkCommandBuffer cmd);
	void getHasGeometryVoxels(VkCommandBuffer cmd);
	void lightInject(VkCommandBuffer cmd);

	nvvk::Buffer hasGeometryVoxelInfoBuffer;
	nvvk::Buffer globalInfoBuffer;

	VkShaderEXT computeShader_clusterGeometryVGB{};
	VkShaderEXT computeShader_getHasGeometryVoxels{};
	VkShaderEXT computeShader_setDispatchIndirectCommand{};
	VkShaderEXT computeShader_LightInject{};

#ifndef NDEBUG
	void debug_Cube(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};
	bool showCubeMap = false;
#endif

	LightInjectCreateInfo_VRPG setting;
	// Per-frame light injection must not use the expensive static-only sample count.
	bool perFrameRebuild = false;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

private:
	shaderio::LightInjectPushConstant_VRPG pushConstant;
};
}

#endif

