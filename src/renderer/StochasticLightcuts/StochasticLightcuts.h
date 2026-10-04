#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "./StochasticLightcutsShaderio.h"
#include <common/Buffer/Buffer.h>

#ifndef FZBRENDERER_STOCHASTIC_LIGHTCUTS_H
#define FZBRENDERER_STOCHASTIC_LIGHTCUTS_H

namespace FzbRenderer {

enum class GBufferType_StochasticLightcuts {
	eNormal_MaterialIndex,
	eImgRendered,
	eLightTreeDebug,
	eImgTonemapped,
};

class StochasticLightcutsRenderer : public PathTracingRenderer {
public:
	StochasticLightcutsRenderer() = default;
	~StochasticLightcutsRenderer() = default;

	StochasticLightcutsRenderer(pugi::xml_node& rendererNode);

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

	void createLightTree(VkCommandBuffer cmd);
	void createGBuffers(VkCommandBuffer cmd);

	inline static int frameIndex = 0;
	shaderio::StochasticLightcutsPushConstant pushConstant{};
	shaderio::CreateGBufferPushConstant_StochasticLightcuts createGBufferPushConstant_StochasticLightcuts{};
	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_LightInfoInject{};
	VkShaderEXT computeShader_CreateLightcuts_1{};
	VkShaderEXT computeShader_CreateLightcuts_2{};
	VkShaderEXT computeShader_CreateLightcuts_3{};
	VkShaderEXT computeShader_CreateLightcuts_4{};
	VkShaderEXT vertexShader_createGBuffers{};
	VkShaderEXT fragmentShader_createGBuffers{};
	VkShaderEXT computeShader_Render{};

	std::vector<shaderio::AreaLight_StochasticLightcuts> areaLights;
	FzbRenderer::Buffer areaLightsBuffer;
	FzbRenderer::Buffer LightTreeBuffer;
	FzbRenderer::Buffer hasLightBlockCountBuffer;
	FzbRenderer::Buffer hasLightBlockArrayBuffer;
	FzbRenderer::Buffer hasLightBlockIndexBuffer;
	FzbRenderer::Buffer indirectCmd;

	void lightTreeDebug(VkCommandBuffer cmd);
	shaderio::LightTreeDebugPushConstant_StochasticLightcuts lightDebugPushConstant;
	VkShaderEXT vertexShader_lightTreeDebug{};
	VkShaderEXT fragmentShader_lightTreeDebug{};

	void lightTreeDebug2(VkCommandBuffer cmd);
	VkShaderEXT vertexShader_lightTreeDebug2{};
	VkShaderEXT fragmentShader_lightTreeDebug2{};
};

}

#endif