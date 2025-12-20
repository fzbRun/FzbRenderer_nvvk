#pragma once

#include <map>
#include <pugixml.hpp>
#include "../feature/Feature.h"
#include <nvvk/context.hpp>

#ifndef FZB_RENDERER_H
#define FZB_RENDERER_H

namespace FzbRenderer {

struct RendererCreateInfo {
	std::string rendererTypeStr;
	pugi::xml_node& rendererNode;
	nvvk::ContextInitInfo& vkContextInfo;
};

class Renderer : public Feature {
public:
	enum
	{
		eImgRendered,
		eImgTonemapped
	};

	Renderer() = default;
	virtual ~Renderer() = default;

	void clean() override;
	virtual void onLastHeadlessFrame();

	virtual void postProcess(VkCommandBuffer cmd);

	std::vector<Feature> features;
};

std::shared_ptr<Renderer> createRenderer(RendererCreateInfo& createInfo);

}

#endif