#pragma once

#include "renderer/Renderer.h"
#include "./NeuralTextureCompressionShaderio.h"
#include <common/Image/Image.h>

#ifndef FZBRENDERER_Neural_Texture_Compression_H
#define FZBRENDERER_Neural_Texture_Compression_H

namespace FzbRenderer {
enum class ImageType_NTC {
	eImgRendered_Reduced,
	eImgRendered_Full,
	eLossImage,
	eImgTonemapped,
};
class NeuralTextureCompression : public Renderer {
public:
	NeuralTextureCompression() = default;
	~NeuralTextureCompression() = default;

	NeuralTextureCompression(pugi::xml_node& rendererNode);

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

	VkPhysicalDeviceCooperativeVectorFeaturesNV g_coopVecFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV };

	shaderio::NeuralTextureCompressionPushConstant pushConstant{};
	VkPushConstantsInfo pushInfo;

	VkShaderEXT computeShader_AutoDiff{};
	void AutoDiff(VkCommandBuffer cmd);

	std::vector<FzbRenderer::Image> sourceTextures;	//diffuse normal roughness 2048x2048
	std::vector<FzbRenderer::Image> mipmapSourceTempTextures;	//1024x1024
	std::vector<FzbRenderer::Image> mipmapSourceTextures;		//512x512
	FzbRenderer::Image mipmapFullSizeTexture;
	bool mipmapFullSize = false;
	VkShaderEXT computeShader_ReducedSizeMipmap{};
	VkShaderEXT computeShader_FullSizeMipmap{};
	bool showLossTexture = false;
	VkShaderEXT computeShader_getLossTexture{};
	VkShaderEXT computeShader_tonemapped{};
	void Mipmap(VkCommandBuffer cmd);
};
}

#endif