#pragma once

#include "feature/Feature.h"
#include <pugixml.hpp>
#include "./RasterVoxelizationShaderio_VRPG.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_VRPG_RASTER_VOXELIZATION
#define FZBRENDERER_VRPG_RASTER_VOXELIZATION

namespace FzbRenderer {
struct RasterVoxelizationSetting_VRPG {
	VkExtent2D resolution;
	shaderio::RasterVoxelizationPushConstant pushConstant;
	shaderio::float3 sceneStartPos;
	shaderio::float3 sceneSize;
};
class RasterVoxelization_VRPG : public Feature {
public:
	RasterVoxelization_VRPG() = default;
	virtual ~RasterVoxelization_VRPG() = default;

	RasterVoxelization_VRPG(pugi::xml_node& featureNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createVGBs();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void clearVGB(VkCommandBuffer cmd);

	RasterVoxelizationSetting_VRPG setting;
	// Static scenes historically forced 4096x4096 here. That is wasteful when
	// rebuilding every frame, especially for the low-resolution light VGB.
	bool forceStaticHighResolution = true;
	std::vector<nvvk::Buffer> VGBs;

	VkShaderEXT computeShader_clearVGB{};
	VkShaderEXT vertexShader{};
	VkShaderEXT geometryShader{};
	VkShaderEXT fragmentShader{};

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPipelineRasterizationConservativeStateCreateInfoEXT conservativeRasterFeature{};
private:
	void createVGB(VkCommandBuffer cmd);

	VkShaderModuleCreateInfo shaderCode;
	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

#ifndef NDBUG
public:
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);

	VkShaderEXT geometryShader_ThreeView{};
	VkShaderEXT fragmentShader_ThreeView{};

	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};

	VkShaderEXT fragmentShader_Wireframe{};

	VkShaderEXT computeShader_postProcess{};

	inline static int normalIndex = 0;
private:
	void resetFragmentCount(VkCommandBuffer cmd);
	void createVGB_ThreeView(VkCommandBuffer cmd);
	void debug_Cube(VkCommandBuffer cmd);
	void debug_Wireframe(VkCommandBuffer cmd);
	void debug_MergeWireframe(VkCommandBuffer cmd);

	VkImageView depthImageView;

	uint32_t fragmentCount_host = 0;
	nvvk::Buffer fragmentCountBuffer;
	nvvk::Buffer fragmentCountStageBuffer;
	bool showThreeViewMap = false;
	bool showCubeMap = false;
	bool showWireframeMap = false;
#endif
};
}

#endif