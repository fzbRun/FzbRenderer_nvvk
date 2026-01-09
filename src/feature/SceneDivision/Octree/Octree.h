#pragma once

#include "../../Feature.h"
#include <feature/SceneDivision/RasterVoxelization/RasterVoxelization.h>
#include "./shaderio.h"

#ifndef FZBRENDERER_OCTREE_H
#define FZBRENDERER_OCTREE_H

namespace FzbRenderer {
struct OctreeSetting{
	uint32_t OctreeDepth = 6;	//排除根节点，从第一层开始
	uint32_t clusteringLevel = 4;	//聚类到第二层停止，即8x8
};

class Octree : public Feature{
public:
	Octree() = default;
	virtual ~Octree() = default;

	Octree(pugi::xml_node& featureNode);

	void init();
	void clean();
	void uiRender();
#ifndef NDEBUG
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
#endif
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createDescriptorSetLayout() override;
	void createOctreeArray();
	void createDescriptorSet();

	void compileAndCreateShaders();
	
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initOctreeArray(VkCommandBuffer cmd);
	void createOctreeArray(VkCommandBuffer cmd);

	shaderio::OctreePushConstant pushConstant{};

	std::shared_ptr<FzbRenderer::RasterVoxelization> rasterVoxelization;
	OctreeSetting setting;
	std::vector<nvvk::Buffer> OctreeArray_G;
	std::vector<nvvk::Buffer> OctreeArray_E;

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;
};
}

#endif