#pragma once

#ifndef FZBRENDERER_HEIGHT_FOG_H
#define FZBRENDERER_HEIGHT_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./HeightFogShaderio.h"
#include <common/Buffer/Buffer.h>

namespace FzbRenderer {
struct HeightFogSetCreateInfo {
	std::vector<shaderio::VolumetricFogInfo>* fogInfos;
};

class HeightFogSet{
public:
	HeightFogSet() = default;
	~HeightFogSet() = default;

	HeightFogSet(HeightFogSetCreateInfo createInfo);

	void init();
	void clean();
	void uiRender();
	void preRender();
	void updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer);

	void addFog(shaderio::HeightFogInfo fogInfo, shaderio::AABB fogRange);

	int fogCount = 0;
	std::vector<int> fogIndexMap;
	std::vector<shaderio::HeightFogInfo> heightFogInfos;
	FzbRenderer::Buffer heightFogInfoBuffer;
	std::vector<int> fogInfoModified;
	std::vector<int> showFogGrid;

private:
	HeightFogSetCreateInfo setting;
};
}

#endif