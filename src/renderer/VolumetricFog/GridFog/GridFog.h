#pragma once

#ifndef FZBRENDERER_GRID_FOG_H
#define FZBRENDERER_GRID_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./GridFogShaderio.h"
#include <common/Buffer/Buffer.h>

namespace FzbRenderer {
struct GridFogCreateInfo {
	shaderio::GridFogInfo fogInfo;
	shaderio::AABB fogRange;
	shaderio::GridFogGenerationInfo generationInfo;
	int randomSeed = 0;
};
class GridFog {
public:
	GridFog(GridFogCreateInfo createInfo, int index, int indexMap, int randomSeed);

	void init();
	void clean();
	void uiRender(int index);

	void updateDataPerFrame(VkCommandBuffer cmd, shaderio::AABB fogRange);

	shaderio::GridFogPushConstant pushConstant;

	int fogIndexMap;
	shaderio::GridFogInfo gridFogInfo;

	shaderio::GridFogGenerationInfo generationInfo;
	FzbRenderer::Buffer gridFogGenerationDataBuffer;

	FzbRenderer::Image gridFogImage;

	bool fogInfoModified = false;
	bool reGeneration = false;
	int showFogGrid = false;
};

struct GridFogSetCreateInfo {
	std::vector<shaderio::VolumetricFogInfo>* fogInfos;
};
class GridFogSet : public Feature {
public:
	GridFogSet() = default;
	~GridFogSet() = default;

	GridFogSet(GridFogSetCreateInfo createInfo);

	void init();
	void clean();
	void uiRender();
	void preRender();
	void updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer);

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;

	void addFog(GridFogCreateInfo createInfo);

	inline int getFogIndexMap(int i) { return gridFogs[i].fogIndexMap; };
	inline bool getFogModified(int i) { return gridFogs[i].fogInfoModified; };
	inline shaderio::uint3 getFogGridSize(int i) { return gridFogs[i].gridFogInfo.gridSize; };
	inline bool getFogGridShow(int i) { return gridFogs[i].showFogGrid; };
	inline nvvk::Image* getFogGridImagesPtr() { return gridFogImages.data(); };

	VkShaderEXT computeShader_initGridFogImage{};

	int fogCount = 0;
	FzbRenderer::Buffer gridFogInfoBuffer;

private:
	GridFogSetCreateInfo setting;
	std::vector<GridFog> gridFogs;

	std::vector<nvvk::Image> gridFogImages;
};
}

#endif
