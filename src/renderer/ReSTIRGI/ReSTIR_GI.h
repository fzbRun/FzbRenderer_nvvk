#pragma once

#ifndef FZB_RESTIR_GI_H
#define FZB_RESTIR_GI_H

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./ReSTIR_GIShaderio.h"
#include <common/Buffer/Buffer.h>
#include <vector>

namespace FzbRenderer {
enum class GBuffers_ReSTIR_GI {
	eVelocity,
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

	void createGBuffers(VkCommandBuffer cmd);
	void RIS(VkCommandBuffer cmd);
	void SpatialReuse(VkCommandBuffer cmd);
	void TemporalReuse(VkCommandBuffer cmd);

	inline static int frameIndex = 0;
	shaderio::CreateGBufferPushConstant_ReSTIR_GI createGBuffersPushConstant{};
	shaderio::ReSTIR_GIConstant pushConstant{};

	std::vector<shaderio::AreaLight_ReSTIR_GI> areaLights;
	FzbRenderer::Buffer areaLightsBuffer;

	FzbRenderer::Buffer pixelDataBuffer;
	FzbRenderer::Buffer pixelDataBuffer_lastFrame;

	VkShaderEXT vertexShader_createGBuffer{};
	VkShaderEXT fragmentShader_createGBuffer{};

	VkShaderEXT computeShader_ReSTIR_GI{};
	VkShaderEXT computeShader_Spatial_Reuse_ReSTIR_GI{};
	VkShaderEXT computeShader_Temporal_Reuse_ReSTIR_GI{};

	VkPushConstantsInfo pushInfo;
};
}

#endif