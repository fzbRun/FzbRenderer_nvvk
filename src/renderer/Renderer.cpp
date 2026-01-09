#include "Renderer.h"
#include <common/Application/Application.h>
#include "DeferredRenderer/DeferredRenderer.h"
#include "PathTracingRenderer/hard/PathTracingRenderer.h"
#include "SVOPathGuidingRenderer/hard/SVOPathGuiding.h"
#include <nvvk/formats.hpp>

enum FzbRendererType {
	FZB_RENDERER_FORWARD,
	FZB_RENDERER_DEFERRED,
	FZB_RENDERER_PATH_TRACING,
	FZB_RENDERER_PATH_TRACING_SOFT,
	FZB_RENDERER_SVO_PATH_GUIDING,
	FZB_RENDERER_SVO_PATH_GUIDING_SOFT,
	FZB_FEATURE_COMPONENT_BVH_DEBUG,
	FZB_FEATURE_COMPONENT_SVO_DEBUG,
	FZB_FEATURE_COMPONENT_SVO_PG_DEBUG,
};
std::map<std::string, FzbRendererType> RendererTypeMap{
	{ "Forward", FZB_RENDERER_FORWARD },
	{ "Deferred", FZB_RENDERER_DEFERRED},
	{ "PathTracing", FZB_RENDERER_PATH_TRACING },
	{ "PathTracing_soft", FZB_RENDERER_PATH_TRACING_SOFT },
	{ "SVOPathGuiding", FZB_RENDERER_SVO_PATH_GUIDING },
	{ "SVOPathGuiding_soft", FZB_RENDERER_SVO_PATH_GUIDING_SOFT },
	{ "BVH_Debug", FZB_FEATURE_COMPONENT_BVH_DEBUG },
	{ "SVO_Debug", FZB_FEATURE_COMPONENT_SVO_DEBUG },
	{ "SVO_PG_Debug", FZB_FEATURE_COMPONENT_SVO_PG_DEBUG },
};

std::shared_ptr<FzbRenderer::Renderer> FzbRenderer::createRenderer(RendererCreateInfo& createInfo) {
	FzbRendererType rendererType;
	if (RendererTypeMap.count(createInfo.rendererTypeStr)) {
		rendererType = RendererTypeMap[createInfo.rendererTypeStr];
		switch (rendererType) {
			case FZB_RENDERER_DEFERRED: return std::make_shared<DeferredRenderer>(createInfo.rendererNode);
			case FZB_RENDERER_PATH_TRACING: return std::make_shared<PathTracingRenderer>(createInfo.rendererNode);
			case FZB_RENDERER_SVO_PATH_GUIDING: return std::make_shared<SVOPathGuidingRenderer>(createInfo.rendererNode);
		}
		return nullptr;
	}
	else throw std::runtime_error("目前暂未实现" + createInfo.rendererTypeStr);
	return nullptr;
}

void FzbRenderer::Renderer::init() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	Application::stagingUploader.cmdUploadAppended(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void FzbRenderer::Renderer::clean() {
	Feature::clean();
}
void FzbRenderer::Renderer::onLastHeadlessFrame() {
	Application::app->saveImageToFile(gBuffers.getColorImage(eImgTonemapped), gBuffers.getSize(),
		nvutils::getExecutablePath().replace_extension(".jpg").string());
};

void FzbRenderer::Renderer::postProcess(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo(eImgRendered),
		gBuffers.getDescriptorImageInfo(eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}