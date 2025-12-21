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
	shaderio::RasterVoxelizationPushConstant pushConstant;
	DebugMode debugMode;
	float lineWidth = 1.0f;
};

class RasterVoxelization : public Feature{
	RasterVoxelization() = default;
	virtual ~RasterVoxelization() = default;

	RasterVoxelization(pugi::xml_node& featureNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void render(VkCommandBuffer cmd) override;

	void createGBuffer(bool useDepth = true) override;
	void createGraphicsDescriptorSetLayout() override;
	void createGraphicsPipelineLayout(uint32_t pushConstantSize = sizeof(shaderio::DefaultPushConstant)) override;

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	RasterVoxelizationSetting setting;
	nvvk::Buffer VGB;
};
}

#endif