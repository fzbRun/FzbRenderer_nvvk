#pragma once

#include "feature/Feature.h"
#include <pugixml.hpp>
#include "./shaderio.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_SVOPG_RASTER_VOXELIZATION
#define FZBRENDERER_SVOPG_RASTER_VOXELIZATION

namespace FzbRenderer {
struct RasterVoxelizationSetting_SVOPG {
	VkExtent2D resolution;
	shaderio::RasterVoxelizationPushConstant pushConstant;
	shaderio::float3 sceneStartPos;
	shaderio::float3 sceneSize;
#ifndef NDEBUG
	float lineWidth = 2.0f;
#endif
};

enum RasterVoxelizationGBuffer_SVOPG {
	ThreeViewMap_SVOPG,
	CubeMap_SVOPG,
	WireframeMap_SVOPG
};

class RasterVoxelization_SVOPG : public Feature {
public:
	RasterVoxelization_SVOPG() = default;
	virtual ~RasterVoxelization_SVOPG() = default;

	RasterVoxelization_SVOPG(pugi::xml_node& featureNode);

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

	RasterVoxelizationSetting_SVOPG setting;
	std::vector<nvvk::Buffer> VGBs;
	#ifdef CLUSTER_WITH_MATERIAL
	std::vector<nvvk::Buffer> VGBMaterialInfos;
	#endif

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