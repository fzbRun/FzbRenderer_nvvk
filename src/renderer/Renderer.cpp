#include "Renderer.h"
#include "DeferredRenderer/DeferredRenderer.h"
#include "PathTracing/hard/PathTracingRenderer.h"

enum FzbRendererType {
	FZB_RENDERER_FORWARD,
	FZB_RENDERER_DEFERRED,
	FZB_RENDERER_PATH_TRACING,
	FZB_RENDERER_PATH_TRACING_SOFT,
	FZB_RENDERER_SVO_PATH_GUIDING,
	FZB_FEATURE_COMPONENT_BVH_DEBUG,
	FZB_FEATURE_COMPONENT_SVO_DEBUG,
	FZB_FEATURE_COMPONENT_SVO_PG_DEBUG,
};
std::map<std::string, FzbRendererType> RendererTypeMap{
	{ "Forward", FZB_RENDERER_FORWARD },
	{ "Deferred", FZB_RENDERER_DEFERRED},
	{ "PathTracing", FZB_RENDERER_PATH_TRACING },
	{ "PathTracing_soft", FZB_RENDERER_PATH_TRACING_SOFT },
	{ "SVOPathGuiding_soft", FZB_RENDERER_SVO_PATH_GUIDING },
	{ "BVH_Debug", FZB_FEATURE_COMPONENT_BVH_DEBUG },
	{ "SVO_Debug", FZB_FEATURE_COMPONENT_SVO_DEBUG },
	{ "SVO_PG_Debug", FZB_FEATURE_COMPONENT_SVO_PG_DEBUG },
};

std::shared_ptr<FzbRenderer::Renderer> FzbRenderer::createRenderer(RendererCreateInfo& createInfo) {
	FzbRendererType rendererType;
	if (RendererTypeMap.count(createInfo.rendererTypeStr)) {
		rendererType = RendererTypeMap[createInfo.rendererTypeStr];
		switch (rendererType) {
			case FZB_RENDERER_DEFERRED: return std::make_shared<DeferredRenderer>(createInfo);
			case FZB_RENDERER_PATH_TRACING: return std::make_shared<PathTracingRenderer>(createInfo);
		}
		return nullptr;
	}
	else throw std::runtime_error("目前暂未实现" + createInfo.rendererTypeStr);
	return nullptr;
}

void FzbRenderer::Renderer::addExtensions() {};
void FzbRenderer::Renderer::compileAndCreateShaders() {};
void FzbRenderer::Renderer::onLastHeadlessFrame() {};
void FzbRenderer::Renderer::updateDataPerFrame(VkCommandBuffer cmd) {};