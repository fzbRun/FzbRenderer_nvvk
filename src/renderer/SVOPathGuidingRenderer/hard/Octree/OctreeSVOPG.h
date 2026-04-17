#pragma once

#include "feature/Feature.h"
#include "../RasterVoxelization/RasterVoxelizationSVOPG.h"
#include "./shaderio.h"

#ifndef FZBRENDERER_OCTREE_SVOPG_H
#define FZBRENDERER_OCTREE_SVOPG_H

namespace FzbRenderer {
struct OctreeSetting_SVOPG {
	std::vector<nvvk::Buffer> VGBs;

	#ifdef CLUSTER_WITH_MATERIAL
	std::vector<nvvk::Buffer> VGBMaterialInfos;
	#endif

	glm::vec3 VGBStartPos;
	glm::vec3 VGBVoxelSize;
	float VGBSize;
	
	uint32_t OctreeLayerCount = 6;
#ifndef NDEBUG
	float lineWidth = 2.0f;
	int normalIndex;
#endif
};

class Octree_SVOPG : public Feature {
public:
	Octree_SVOPG() = default;
	virtual ~Octree_SVOPG() = default;

	Octree_SVOPG(pugi::xml_node& featureNode);

	void init(OctreeSetting_SVOPG setting);
	void clean();
	void uiRender();
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

	shaderio::OctreePushConstant_SVOPG pushConstant{};

	OctreeSetting_SVOPG setting;

	std::vector<nvvk::Buffer> OctreeArray_G;	//layer0: 6  layer1： 48 ……
	std::vector<nvvk::Buffer> OctreeArray_E;
	nvvk::Buffer NodeData_E;
private:
	nvvk::Buffer blockInfoBuffer_G;
	nvvk::Buffer blockInfoBuffer_E;
	nvvk::Buffer hasDataBlockIndexBuffer_G;
	nvvk::Buffer hasDataBlockIndexBuffer_E;
	nvvk::Buffer hasDataBlockCountBuffer;

	nvvk::Buffer GlobalInfoBuffer;

#ifndef USE_SVO
	void getOctreeLabel(VkCommandBuffer cmd);

	nvvk::Buffer divisibleNodeInfos_G;		//每层可细分节点的索引
	nvvk::Buffer threadGroupInfos;
	nvvk::Buffer indivisibleNodeInfos_G;

	VkShaderEXT computeShader_getOctreeLabel1{};
	VkShaderEXT computeShader_getOctreeLabel2{};
	VkShaderEXT computeShader_getOctreeLabel3{};
#endif

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_initHasDataBlockInfo{};
	VkShaderEXT computeShader_getGlobalInfo{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

#ifndef NDEBUG
public:
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
private:
	void debugPrepare();

	void debug_wirefame(VkCommandBuffer cmd);
	void debug_mergeResult(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_Wireframe{};
	VkShaderEXT fragmentShader_Wireframe{};
	VkShaderEXT computeShader_MergeResult{};

	VkImageView depthImageView;

	uint32_t showLayerCount = 1;
	bool showWireframeMap_G = false;
	bool showWireframeMap_E = false;
	int selectedWireframeMapIndex_G = 0;
	int selectedWireframeMapIndex_E = 0;
#endif
};

}

#endif