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

	void createGraphicsDescriptorSetLayout() override;

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	RasterVoxelizationSetting setting;
	nvvk::Buffer VGB;

	VkShaderEXT vertexShader{};
	VkShaderEXT geometryShader{};
	VkShaderEXT fragmentShader{};

	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};

private:
	void copyFragmentCount(VkCommandBuffer cmd);

	void createVGB(VkCommandBuffer cmd, bool outputThreeView = false);
	void debug_Wireframe(VkCommandBuffer cmd);

	bool showThreeViewMap = false;

	uint32_t fragmentCount_host = 0;
	nvvk::Buffer fragmentCountBuffer;
	nvvk::Buffer fragmentCountStageBuffer;

	nvvk::Image wireframeImage;
};
}

#endif