#pragma once

#include "../../Feature.h"
#include <pugixml.hpp>
#include "./shaderio.h"
#include <nvvk/context.hpp>

#ifndef FZBRENDERER_RASTER_VOXELIZATION
#define FZBRENDERER_RASTER_VOXELIZATION

namespace FzbRenderer {
enum DebugMode{
	None,
	ThreeView,
	Cube,
	Wireframe,
};
enum RasterVoxelizationGBuffer {
	ThreeViewMap,
	CubeMap,
	WireframeMap
};

struct RasterVoxelizationSetting{
	VkExtent2D resolution;
	shaderio::RasterVoxelizationPushConstant pushConstant;
	DebugMode debugMode = DebugMode::None;
	float lineWidth = 1.0f;
};

class RasterVoxelization : public Feature {
public:
	RasterVoxelization() = default;
	virtual ~RasterVoxelization() = default;

	RasterVoxelization(pugi::xml_node& featureNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer cmd) override;

	void createDescriptorSetLayout() override;

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void clearVGB(VkCommandBuffer cmd);

	RasterVoxelizationSetting setting;
	nvvk::Buffer VGB;

	VkShaderEXT computeShader_clearVGB{};
	VkShaderEXT vertexShader{};
	VkShaderEXT geometryShader{};
	VkShaderEXT fragmentShader{};

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};

private:
	void createVGB(VkCommandBuffer cmd);

	VkShaderModuleCreateInfo shaderCode;
	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

#ifndef NDBUG
public:
	VkShaderEXT geometryShader_ThreeView{};
	VkShaderEXT fragmentShader_ThreeView{};

	VkShaderEXT vertexShader_Cube{};
	VkShaderEXT fragmentShader_Cube{};
private:
	void resetFragmentCount(VkCommandBuffer cmd);
	void createVGB_ThreeView(VkCommandBuffer cmd);
	void debug_Cube(VkCommandBuffer cmd);
	void debug_Wireframe(VkCommandBuffer cmd);

	uint32_t fragmentCount_host = 0;
	nvvk::Buffer fragmentCountBuffer;
	nvvk::Buffer fragmentCountStageBuffer;
	bool showThreeViewMap = false;
	bool showCubeMap = false;
#endif
};
}

#endif