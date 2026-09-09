#pragma once

#ifndef FZB_RESTIR_GI_H
#define FZB_RESTIR_GI_H

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./ReSTIR_GIShaderio.h"
#include <common/Buffer/Buffer.h>
#include <vector>

namespace FzbRenderer {
enum class ImageType_ReSTIR_GI {
	eImgRendered,
	eImgTonemapped,
};

enum class ReSTIR_GI_Mode {
	ePT = 0,
	eRIS,
	eRIS_Spatial_Reuse,
	eRIS_SpatialTemporal_Reuse,
};

class ReSTIR_GI : public PathTracingRenderer{
public:
	ReSTIR_GI() = default;
	~ReSTIR_GI() = default;

	ReSTIR_GI(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;

private:
	void createSourceData();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	inline static int frameIndex = 0;
	shaderio::ReSTIR_GIConstant pushConstant{};

	std::vector<shaderio::AreaLight_ReSTIR_GI> areaLights;
	FzbRenderer::Buffer areaLightsBuffer;

	FzbRenderer::Buffer pixelDataBuffer;

	VkShaderEXT computeShader_ReSTIR_GI{};
	VkShaderEXT computeShader_Spatial_Reuse_ReSTIR_GI{};
};
}

#endif