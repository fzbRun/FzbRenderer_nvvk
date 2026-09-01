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

	int fogIndexMap;
	shaderio::FluidFogInfo fluidFogInfo;

	FzbRenderer::Buffer fluidFogVoxelInfoBuffer;
	FzbRenderer::Image fluidFogVoxelVelocityImage;
	FzbRenderer::Image fluidFogVoxelInfoImage;

#if defined(FLUID_A_MACCORMACK) || defined(FLUID_SIMULATION_CPF)
	FzbRenderer::Image fluidFogVoxelInfoImage_temp1;
	FzbRenderer::Image fluidFogVoxelInfoImage_temp2;
#endif

	bool follow = true;
	std::string followInstanceID = "mainCharacter";
	shaderio::float3 fluidLocalStartPos;

	shaderio::AABB fogAABB;
	shaderio::float4x4 VP[3];

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
	void updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer, FzbRenderer::Buffer fluidVPMatrixsBuffer);

	void addFog(FluidFogCreateInfo createInfo);
	inline nvvk::Buffer* getfluidFogVoxelInfoBuffersPtr() { return fluidFogVoxelInfoBuffers.data(); };
	inline nvvk::Image* getfluidFogVoxelVelocityImagesPtr() { return fluidFogVoxelVelocityImages.data(); };
	inline nvvk::Image* getfluidFogVoxelInfoImagesPtr() { return fluidFogVoxelInfoImages.data(); };
#ifdef FLUID_A_MACCORMACK
	inline nvvk::Image* getfluidFogVoxelInfoImages_temp1Ptr() { return fluidFogVoxelInfoImages_temp1.data(); };
	inline nvvk::Image* getfluidFogVoxelInfoImages_temp2Ptr() { return fluidFogVoxelInfoImages_temp2.data(); };
#endif

	inline bool getFogStartUp(int i) { return fluidFogs[i].fluidFogInfo.startUp; };
	inline int getFogIndexMap(int i) { return fluidFogs[i].fogIndexMap;};
	inline bool getFogModified(int i) { return fluidFogs[i].fogInfoModified; };
	inline shaderio::uint3 getFogGridSize(int i) { return fluidFogs[i].fluidFogInfo.gridSize; };
	inline bool getFogGridShow(int i) { return fluidFogs[i].showFogGrid; };

	inline shaderio::float4x4 getFogVPs(int i, int j) { return fluidFogs[i].VP[j]; };
	
	int fogCount = 0;
	FzbRenderer::Buffer fluidFogInfoBuffer;

private:
	FluidFogSetCreateInfo setting;
	std::vector<FluidFog> fluidFogs;

	std::vector<nvvk::Buffer> fluidFogVoxelInfoBuffers;
	std::vector<nvvk::Image> fluidFogVoxelVelocityImages;
	std::vector<nvvk::Image> fluidFogVoxelInfoImages;

#ifdef FLUID_A_MACCORMACK
	std::vector<nvvk::Image> fluidFogVoxelInfoImages_temp1;
	std::vector<nvvk::Image> fluidFogVoxelInfoImages_temp2;
#endif
};
}

#endif
