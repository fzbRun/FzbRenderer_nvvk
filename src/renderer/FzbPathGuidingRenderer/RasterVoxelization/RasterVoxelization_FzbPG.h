#pragma once

#include "feature/Feature.h"
#include <pugixml.hpp>
#include "./RasterVoxelizationShaderio_FzbPG.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_FZBPG_RASTER_VOXELIZATION
#define FZBRENDERER_FZBPG_RASTER_VOXELIZATION

namespace FzbRenderer {
struct RasterVoxelizationCreateInfo_FzbPG {
	VkExtent2D resolution;
	shaderio::RasterVoxelizationPushConstant pushConstant;
	shaderio::float3 sceneStartPos;
	shaderio::float3 sceneSize;
};
class RasterVoxelization_FzbPG : public Feature {
public:
	RasterVoxelization_FzbPG() = default;
	virtual ~RasterVoxelization_FzbPG() = default;

	RasterVoxelization_FzbPG(pugi::xml_node& featureNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd) override;
	void postProcess(VkCommandBuffer cmd);

	void createVGBs();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void clearVGB(VkCommandBuffer cmd);

	RasterVoxelizationCreateInfo_FzbPG setting;
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