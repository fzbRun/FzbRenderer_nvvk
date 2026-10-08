#pragma once

#include "./FzbPathGuidingShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"
#include "RasterVoxelization/RasterVoxelization_FzbPG.h"
#include "LightInject/LightInject_FzbPG.h"

#ifdef StochasticLightcuts_RIS
#include "Octree2/Octree2_FzbPG.h"
#else
#include "Octree/Octree_FzbPG.h"
#endif
#include <feature/ShadowMap/ShadowMap.h>

#ifndef FZBRENDERER_FZB_PATHGUIDING_H
#define FZBRENDERER_FZB_PATHGUIDING_H

namespace FzbRenderer {
enum class ImageType_FzbPG{
	eImgRendered,
#ifndef NDEBUG
	eImgPGValue,
	eImgPGValue2,
	eImgPGVariance,
#endif
	eImgTonemapped,
};

class FzbPathGuidingRenderer : public PathTracingRenderer {
public:
	FzbPathGuidingRenderer() = default;
	~FzbPathGuidingRenderer() = default;

	FzbPathGuidingRenderer(pugi::xml_node& rendererNode);

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
	inline static int frameIndex = 0;

	std::shared_ptr<ShadowMap> shadowMap;

	std::shared_ptr<RasterVoxelization_FzbPG> rasterVoxelization;
	std::shared_ptr<LightInject_FzbPG> lightInject;
#ifdef StochasticLightcuts_RIS
	std::shared_ptr<Octree2_FzbPG> octree;
#else
	std::shared_ptr<Octree_FzbPG> octree;
#endif

	shaderio::FzbPathGuidingPushConstant pushConstant{};
	VkShaderEXT computeShader_FzbPathGuiding{};

#ifndef NDEBUG
	std::map<uint32_t, bool> showDebugImages;
#endif
};
}

#endif