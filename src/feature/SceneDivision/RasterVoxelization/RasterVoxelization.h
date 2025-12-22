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
	Cube,
	Wireframe,
};

struct RasterVoxelizationSetting{
	VkExtent2D resolution;
	shaderio::RasterVoxelizationPushConstant pushConstant;
	DebugMode debugMode;
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
};
}

#endif