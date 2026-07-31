#pragma once

#ifndef FZBRENDERER_FLUID_FOG_H
#define FZBRENDERER_FLUID_FOG_H

#include "renderer/Renderer.h"
#include <common/Image/Image.h>
#include "./FluidFogShaderio.h"
#include <common/Buffer/Buffer.h>

namespace FzbRenderer {
struct FluidFogCreateInfo{
	shaderio::FluidFogInfo fogInfo;
	shaderio::AABB fogRange;
	bool follow = false;
	std::string followInstanceID;
};
class FluidFog {
public:
	void init();
	void clean();
	void uiRender(int index);
	void preRender(shaderio::AABB& fogAABB);

	//void createDescriptorSetLayout();
	//void createDescriptorSet();
	//void createPipelineLayout();
	//void compileAndCreateShaders();
	//void updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer);

	void initFluid(VkCommandBuffer cmd);
	//void fluidSimulation(VkCommandBuffer cmd);

	int fogIndexMap;
	shaderio::FluidFogInfo fluidFogInfo;
	FzbRenderer::Buffer fluidFogVoxelInfoBuffer;
	FzbRenderer::Image fluidFogVoxelVelocityImage;
	FzbRenderer::Image fluidFogVoxelInfoImage;

	bool follow = true;
	std::string followInstanceID = "mainCharacter";
	shaderio::float3 fluidLocalStartPos;

	bool fogInfoModified = false;
	int showFogGrid = false;
};

struct FluidFogSetCreateInfo {
	std::vector<shaderio::VolumetricFogInfo>* fogInfos;
};
class FluidFogSet {
public:
	FluidFogSet() = default;
	~FluidFogSet() = default;

	FluidFogSet(FluidFogSetCreateInfo createInfo);

	void init();
	void clean();
	void uiRender();
	void preRender();
	void updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer);

	void addFog(FluidFogCreateInfo createInfo);
	inline nvvk::Buffer* getfluidFogVoxelInfoBuffersPtr() { return fluidFogVoxelInfoBuffers.data(); };
	inline nvvk::Image* getfluidFogVoxelVelocityImagesPtr() { return fluidFogVoxelVelocityImages.data(); };
	inline nvvk::Image* getfluidFogVoxelInfoImagesPtr() { return fluidFogVoxelInfoImages.data(); };

	inline bool getFogStartUp(int i) { return fluidFogs[i].fluidFogInfo.startUp; };
	inline int getFogIndexMap(int i) { return fluidFogs[i].fogIndexMap;};
	inline bool getFogModified(int i) { return fluidFogs[i].fogInfoModified; };
	inline shaderio::uint3 getFogGridSize(int i) { return fluidFogs[i].fluidFogInfo.gridSize; };
	

	int fogCount = 0;
	FzbRenderer::Buffer fluidFogInfoBuffer;

private:
	FluidFogSetCreateInfo setting;
	std::vector<FluidFog> fluidFogs;

	std::vector<nvvk::Buffer> fluidFogVoxelInfoBuffers;
	std::vector<nvvk::Image> fluidFogVoxelVelocityImages;
	std::vector<nvvk::Image> fluidFogVoxelInfoImages;
};
}

#endif