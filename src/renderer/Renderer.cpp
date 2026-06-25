#include "Renderer.h"
#include <common/Application/Application.h>
#include "DeferredRenderer/DeferredRenderer.h"
#include "PathTracingRenderer/hard/PathTracingRenderer.h"
#include <nvvk/formats.hpp>
#include "FzbPathGuidingRenderer/FzbPathGuiding.h"
#include "NPMPathGuiding/NPMPathGuiding.h"
#include "PathTracing_KPCNNDenoising/PathTracing_KPCNNDenoising.h"
#include "ZhiHuCode/ZhiHuCode.h"
#include "VolumetricFog/VolumetricFog.h"

enum FzbRendererType {
	FZB_RENDERER_FORWARD,
	FZB_RENDERER_DEFERRED,
	FZB_RENDERER_PATH_TRACING,
	FZB_RENDERER_PATH_TRACING_SOFT,
	FZB_RENDERER_NPM_PATH_GUIDING,
	FZB_FEATURE_COMPONENT_SVO_DEBUG,
	FZB_FEATURE_COMPONENT_SVO_PG_DEBUG,
	FZB_RENDERER_FZB_PATH_GUIDING,
	FZB_RENDERER_PATH_TRACING_KPCNN_DENOISING,
	FZB_RENDERER_VOLUMETRIC_FOG,
	FZB_ZHIHU_CODE,
};
std::map<std::string, FzbRendererType> RendererTypeMap{
	{ "Forward", FZB_RENDERER_FORWARD },
	{ "Deferred", FZB_RENDERER_DEFERRED},
	{ "PathTracing", FZB_RENDERER_PATH_TRACING },
	{ "PathTracing_soft", FZB_RENDERER_PATH_TRACING_SOFT },
	{ "NPMPathGuiding", FZB_RENDERER_NPM_PATH_GUIDING },
	{ "FzbPathGuiding", FZB_RENDERER_FZB_PATH_GUIDING },
	{ "PathTracing_KPCNNDenoising", FZB_RENDERER_PATH_TRACING_KPCNN_DENOISING },
	{ "VolumetricFog", FZB_RENDERER_VOLUMETRIC_FOG },
	{ "ZhiHuCode", FZB_ZHIHU_CODE },
};

std::shared_ptr<FzbRenderer::Renderer> FzbRenderer::createRenderer(RendererCreateInfo& createInfo) {
	FzbRendererType rendererType;
	if (RendererTypeMap.count(createInfo.rendererTypeStr)) {
		rendererType = RendererTypeMap[createInfo.rendererTypeStr];
		switch (rendererType) {
			case FZB_RENDERER_DEFERRED: return std::make_shared<DeferredRenderer>(createInfo.rendererNode);
			case FZB_RENDERER_PATH_TRACING: return std::make_shared<PathTracingRenderer>(createInfo.rendererNode);
			case FZB_RENDERER_FZB_PATH_GUIDING: return std::make_shared<FzbPathGuidingRenderer>(createInfo.rendererNode);
			//case FZB_RENDERER_NPM_PATH_GUIDING: return std::make_shared<NPMPathGuiding>(createInfo.rendererNode);
			case FZB_RENDERER_PATH_TRACING_KPCNN_DENOISING: return std::make_shared<PathTracing_KPCNNDenoising>(createInfo.rendererNode);
			case FZB_RENDERER_VOLUMETRIC_FOG: return std::make_shared<VolumetricFog>(createInfo.rendererNode);
			case FZB_ZHIHU_CODE: return std::make_shared<ZhiHuCode>(createInfo.rendererNode);
		}
		return nullptr;
	}
	else throw std::runtime_error("目前暂未实现" + createInfo.rendererTypeStr);
	return nullptr;
}

void FzbRenderer::Renderer::init() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	Application::stagingUploader.cmdUploadAppended(cmd);
	Application::stagingUploaderExport.cmdUploadAppended(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void FzbRenderer::Renderer::clean() {
	Feature::clean();
}
void FzbRenderer::Renderer::onLastHeadlessFrame() {
	Application::app->saveImageToFile(gBuffers.getColorImage(eImgTonemapped), gBuffers.getSize(),
		nvutils::getExecutablePath().replace_extension(".jpg").string());
};

void FzbRenderer::Renderer::postProcess(VkCommandBuffer cmd, VkDescriptorImageInfo* inImage) {
	NVVK_DBG_SCOPE(cmd);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, inImage == nullptr ? gBuffers.getDescriptorImageInfo(eImgRendered) : *inImage,
		gBuffers.getDescriptorImageInfo(eImgTonemapped));
	//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}