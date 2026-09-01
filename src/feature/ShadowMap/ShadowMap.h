#pragma once
/*
woc, multiview不能和shader object一起使用，也就是说shadowMap需要通过传统的pipeline模式
服了，先不搞了
*/

#include "feature/Feature.h"
#include "./ShadowMapShaderio.h"
#include "common/Image/Image.h"

#ifndef FZBRENDERER_SHADOWMAP_H
#define FZBRENDERER_SHADOWMAP_H

namespace FzbRenderer {
struct ShadowMapCreateInfo{
	VkExtent2D resolution;
};
class ShadowMap : public Feature{
public:
	ShadowMap();
	virtual ~ShadowMap() = default;

	void init(ShadowMapCreateInfo createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);
	
	VkResult createShadowMap();
	void createPipeline();
	void compileAndCreateShaders();

	std::vector<uint32_t> lightIndices;
	std::vector<FzbRenderer::Image> shadowMaps;
	shaderio::ShadowMapPushConstant pushConstant;
	VkPushConstantsInfo pushInfo;

	bool showShadowMap = false;
	bool showRestructResultMap = false;
private:
	ShadowMapCreateInfo setting;

	VkShaderEXT vertexShader_directionLight{};
	VkShaderEXT fragmentShader_directionLight{};

	VkShaderEXT vertexShader_pointLight{};
	VkShaderEXT fragmentShader_pointLight{};

	shaderio::AABB sceneAABB;

#ifndef NDEBUG
	void debug_prepare();
	void debug_Visualization(VkCommandBuffer cmd);
	VkShaderEXT computeShader_debug{};
#endif
};
}
#endif