#pragma once

#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "common/Image/Image.h"
#include "common/Semaphore/Semaphore.h"
#include <common/Buffer/Buffer.h>
#include "./PathTracing_KPCNNDenosingShaderio.h"
#include "CUDA/KPCNN.cuh"

#ifndef FZBRENDERER_KPCNN_DENOSING_PATHTRACING_H
#define FZBRENDERER_KPCNN_DENOSING_PATHTRACING_H

namespace FzbRenderer {
enum class GBufferImageIndex_KPCNN {
#ifndef NDEBUG
	eColorDebugImage = 0,
	eDiffuseDebugImage,
	eSpecularDebugImage,
	eIrradianceDebugImage,
	eNormalDebugImage,
	eDepthDebugImage,
	eAlbedoDebugImage,

	eIrradianceVarianceDebugImage,
	eSpecularVariancDebugImage,
	eNormalVarianceDebugImage,
	eDepthVarianceDebugImage,
	eAlbedoVarianceDebugImage,
#endif
	eTonemapImage,
	eDebugImageCount,
};

class PathTracing_KPCNNDenoising : public PathTracingRenderer {
public:
	PathTracing_KPCNNDenoising() = default;
	~PathTracing_KPCNNDenoising() = default;

	PathTracing_KPCNNDenoising(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;
	
private:
	void createDataObject();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void pathTracing(VkCommandBuffer cmd);
	void createInputBuffers(VkCommandBuffer cmd);

	void saveSampleBuffers(std::string fileName);

	VkExtent2D screenSize{};

	VkPushConstantsInfo pushInfo{};
	shaderio::KPCNN_DenoisingPTPushConstant pushConstant{};
	VkShaderEXT computeShader_PathTracing{};
	VkShaderEXT computeShader_createGradBuffers{};

	FzbRenderer::Buffer inputBuffer_diff;
	FzbRenderer::Buffer inputBuffer_spec;

	FzbRenderer::Buffer normalBuffer;
	FzbRenderer::Buffer depthBuffer;
	FzbRenderer::Buffer maxDepthBuffer;
	FzbRenderer::Buffer albedoBuffer;

	FzbRenderer::Image colorImage;

	FzbRenderer::Semaphore vulkanToCudaSemaphore;
	FzbRenderer::Semaphore cudaToVulkanSemaphore;

	KPCNNDenoiser kpcnDenoiser;

#ifndef NDEBUG
	std::vector<bool> showImage;
#endif
};
}

#endif