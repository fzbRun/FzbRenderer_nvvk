#pragma once

#include "feature/Feature.h"
#include "./ShadowMapShaderio.h"

#ifndef FZBRENDERER_SHADOWMAP_H
#define FZBRENDERER_SHADOWMAP_H

namespace FzbRenderer {
struct ShadowMapCreateInfo{
	VkExtent2D resolution;
};
class ShadowMap : public Feature{
public:
	ShadowMap() = default;
	virtual ~ShadowMap() = default;

	void init(ShadowMapCreateInfo createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender(VkCommandBuffer cmd);
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);
	
	VkResult createShadowMap();
	void createPipeline();
	void compileAndCreateShaders();

	std::vector<uint32_t> lightIndices;
	std::vector<nvvk::Image> shadowMaps;
	shaderio::ShadowMapPushConstant pushConstant;
	VkPushConstantsInfo pushInfo;

	bool showShadowMap = false;
	bool showRestructResultMap = false;
private:
	ShadowMapCreateInfo setting;

	VkShaderEXT vertexShader_directionLight{};
	VkShaderEXT fragmentShader_directionLight{};

	VkShaderEXT vertexShader_pointLight{};
	VkShaderEXT geometryShader_pointLight{};
	VkShaderEXT fragmentShader_pointLight{};

#ifndef NDEBUG
	std::vector<VkImageView>     uiImageViews{};
	VkDescriptorPool descriptorPool = nullptr;
	VkDescriptorSetLayout descLayout{};
	std::vector<VkDescriptorSet> uiDescriptorSets{};

	void debug_prepare();
	void debug_Visualization(VkCommandBuffer cmd);
	VkShaderEXT computeShader_debug{};
#endif
};
}
#endif