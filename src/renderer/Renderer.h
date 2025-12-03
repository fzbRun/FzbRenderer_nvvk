#pragma once

#include <map>
#include <pugixml.hpp>
#include <nvvk/context.hpp>

#ifndef FZB_RENDERER_H
#define FZB_RENDERER_H

namespace FzbRenderer {

struct RendererCreateInfo {
	std::string rendererTypeStr;
	pugi::xml_node& rendererNode;
	nvvk::ContextInitInfo& vkContextInfo;
};

class Renderer {
public:
	Renderer() = default;
	virtual ~Renderer() = default;
	virtual void init() = 0;

	virtual void compileAndCreateShaders();

	virtual void clean() = 0;
	virtual void uiRender() = 0;
	virtual void resize(VkCommandBuffer cmd, const VkExtent2D& size) = 0;
	virtual void updateDataPerFrame(VkCommandBuffer cmd);
	virtual void render(VkCommandBuffer cmd) = 0;
	virtual void onLastHeadlessFrame();
private:
	virtual void addExtensions();
	virtual void postProcess(VkCommandBuffer cmd) = 0;

};

std::shared_ptr<Renderer> createRenderer(RendererCreateInfo& createInfo);

}

#endif