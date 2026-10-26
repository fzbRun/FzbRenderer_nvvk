#pragma once

#include "./VXPGShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"

#ifndef FZBRENDERER_VXPG_H
#define FZBRENDERER_VXPG_H

namespace FzbRenderer {

enum class ImageType_VXPG
{
	eImgRendered,
	eImgTonemapped,
};

// Voxel path guiding renderer, port of SIByLEngine2023's "VXGuiding" basic pipeline
// (sample mode 4, "Guide + BSDF"). Everything it needs lives in this directory; see README.md.
class VXPGRenderer : public PathTracingRenderer
{
public:
	VXPGRenderer() = default;
	~VXPGRenderer() = default;

	explicit VXPGRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer* cmdPtr) override;

	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipelineLayout();
	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

private:
	// GPU resources
	void updateSceneBounds();
	void updateVoxelDomain();
	void computeResolution();
	void createResources();
	void destroyResources();
	void clearVoxelGrids(VkCommandBuffer cmd);
	void clearSpixelWeights(VkCommandBuffer cmd);
	void dumpGuidingState();

	// passes
	void bindCommonState(VkCommandBuffer cmd);
	void dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
	void runVBufferPass(VkCommandBuffer cmd);
	void runLightInjectionPass(VkCommandBuffer cmd);
	void runGeometryInjectionPass(VkCommandBuffer cmd);
	void runTreePasses(VkCommandBuffer cmd);
	void runSpixelPasses(VkCommandBuffer cmd);
	void runIntegratorPass(VkCommandBuffer cmd);

	// settings
	int  maxFrames = (MAX_FRAME) / 2;
	bool rebuildPerFrame = true;
	bool enableGuiding = true;
	int  debugMode = 0;

	shaderio::VXPGPushConstant pushConstant{};
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
	VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT dummyInterlock{};

	// GPU buffers
	nvvk::Buffer voxelIrradianceBuffer{};
	nvvk::Buffer voxelVplCountBuffer{};
	nvvk::Buffer voxelBoundMinBuffer{};
	nvvk::Buffer voxelBoundMaxBuffer{};
	nvvk::Buffer pixelDataBuffer{};
	nvvk::Buffer treeNodesBuffer{};
	nvvk::Buffer clusterRootsBuffer{};
	nvvk::Buffer topLevelTreeBuffer{};
	nvvk::Buffer spixelInfoBuffer{};
	nvvk::Buffer spixelClusterWeightBuffer{};
	nvvk::Buffer spixelClusterTotalBuffer{};

	// sizes
	VkExtent2D currentResolution{ 0, 0 };
	uint32_t   voxelCount = 0;
	uint32_t   leafCount = 0;
	uint32_t   nodeCount = 0;
	uint32_t   spixelGridWidth = 0;
	uint32_t   spixelGridHeight = 0;
	uint32_t   spixelCount = 0;

	// scene bounds (cubic voxel grid, same construction as the reference VoxelTexInfo)
	shaderio::float3 sceneStartPos{ 0.0f, 0.0f, 0.0f };
	shaderio::float3 sceneVoxelSize{ 1.0f, 1.0f, 1.0f };
	float sceneExtent = 0.5f;
	shaderio::float3 sceneCenter{ 0.0f, 0.0f, 0.0f };
	float autoClipExtent = 2.5f;
	float clipExtentSetting = 0.0f;
	glm::vec3 domainCenterBuild{ 0.0f };
	float domainExtentBuild = 0.0f;
	bool guidingDirty = true;

	// shader objects
	VkShaderEXT shaderVBuffer{};
	VkShaderEXT shaderLightInject{};
	VkShaderEXT shaderGeometryInject{};
	VkShaderEXT shaderTreeInit{};
	VkShaderEXT shaderTreeInternal{};
	VkShaderEXT shaderTreeAccumulate{};
	VkShaderEXT shaderClusterRoots{};
	VkShaderEXT shaderTreeMerge{};
	VkShaderEXT shaderSpixelCenters{};
	VkShaderEXT shaderSpixelAssign{};
	VkShaderEXT shaderSpixelAccum{};
	VkShaderEXT shaderTopLevel{};
	VkShaderEXT shaderIntegrator{};
};

}  // namespace FzbRenderer

#endif
