#pragma once

#include "../../Feature.h"
#include <feature/SceneDivision/RasterVoxelization/RasterVoxelization.h>
#include "./shaderio.h"

#ifndef FZBRENDERER_OCTREE_H
#define FZBRENDERER_OCTREE_H

namespace FzbRenderer {
struct OctreeSetting{
	nvvk::Buffer VGB;
	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;

	uint32_t OctreeDepth = 6;	//排除根节点，从第一层开始
	uint32_t clusteringLevel = 4;	//聚类到第二层停止，即8x8

	float lineWidth = 2.0f;
};

class Octree : public Feature{
public:
	Octree() = default;
	virtual ~Octree() = default;

	Octree(pugi::xml_node& featureNode);

	void init(OctreeSetting setting);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createOctreeArray();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();

	void compileAndCreateShaders();
	
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initOctreeArray(VkCommandBuffer cmd);
	void createOctreeArray(VkCommandBuffer cmd);

	shaderio::OctreePushConstant pushConstant{};

	OctreeSetting setting;
	std::vector<nvvk::Buffer> OctreeArray_G;
	std::vector<nvvk::Buffer> OctreeArray_E;

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

#ifndef NDEBUG
	void debug_wirefame(VkCommandBuffer cmd);
	void debug_mergeResult(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_Wireframe{};
	VkShaderEXT fragmentShader_Wireframe{};
	VkShaderEXT computeShader_MergeResult{};

	VkImageView depthImageView;

	uint32_t clusterLevelCount = 1;
	bool showWireframeMap_G = false;
	bool showWireframeMap_E = false;
	int selectedWireframeMapIndex_G = 0;
	int selectedWireframeMapIndex_E = 0;
#endif
};
}

#endif