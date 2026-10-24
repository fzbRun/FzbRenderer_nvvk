#pragma once

#include "./VRPGShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "RasterVoxelization/RasterVoxelization_VRPG.h"
#include "LightInject/LightInject_VRPG.h"

#include "Octree/Octree_VRPG.h"
#include <feature/ShadowMap/ShadowMap.h>
#include <feature/FLIP/FLIP.h>

#ifndef FZBRENDERER_VRPG_H
#define FZBRENDERER_VRPG_H

namespace FzbRenderer {
enum class ImageType_VRPG{
	eImgRendered,
#ifndef NDEBUG
	eImgPGValue,
	eImgPGValue2,
	eImgPGVariance,
#endif
	eImgTonemapped,
};

class VRPGRenderer : public PathTracingRenderer {
public:
	VRPGRenderer() = default;
	~VRPGRenderer() = default;

	VRPGRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmd) override;
	
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void pathGuiding(VkCommandBuffer cmd);
private:
	bool renderStaticScene = true;
	bool rebuildPerFrame = false;
	float jitterRadiusScale = 1.0f;
	int nearbyNodeCount = NEARBY_NODE_COUNT_VRPG;
	inline static int frameIndex = 0;

	std::shared_ptr<ShadowMap> shadowMap;

	std::shared_ptr<RasterVoxelization_VRPG> rasterVoxelization;      // geometry VGB
	std::shared_ptr<RasterVoxelization_VRPG> lightRasterVoxelization; // light VGB
	std::shared_ptr<LightInject_VRPG> lightInject;
	std::shared_ptr<Octree_VRPG> octree;

	shaderio::VRPGPushConstant pushConstant{};
	VkShaderEXT computeShader_VRPG{};

#ifndef NDEBUG
	std::map<uint32_t, bool> showDebugImages;
#endif

	FzbRenderer::Image refImage;
	std::shared_ptr<FLIP> Flip;

	float lastFLIP = 0.0f;
	double flipSum = 0.0;
	int flipFrameCount = 0;
	bool flipReadbackPending = false;
	nvvk::Buffer flipReadbackBuffer{};
	VkExtent2D flipReadbackExtent{ 0, 0 };

	void createFlipReadbackBuffer();
	void getFLIP(VkCommandBuffer cmd);
	void readFLIP();
};
}

#endif