#pragma once

#include "feature/Feature.h"
#include "./OctreeShaderio_FzbPG.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_OCTREE_FZBPG_H
#define FZBRENDERER_OCTREE_FZBPG_H
namespace FzbRenderer {
struct OctreeCreateInfo_FzbPG{
	std::vector<nvvk::Buffer> VGBs;

	shaderio::float3 VGBStartPos;
	shaderio::float3 VGBVoxelSize;
	float VGBSize;

	AccelerationStructureManager* asManager;
};

class Octree_FzbPG : public Feature {
public:
	Octree_FzbPG() = default;
	virtual ~Octree_FzbPG() = default;

	Octree_FzbPG(pugi::xml_node& featureNode);

	void init(OctreeCreateInfo_FzbPG createInfo);
	void clean();
	void uiRender();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createOctreeArray();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initOctreeArray(VkCommandBuffer cmd);
	void createOctreeArray(VkCommandBuffer cmd);
	void getOctreeLabel(VkCommandBuffer cmd);
	void getOctreeNodePairData(VkCommandBuffer cmd);
	void getNearbyNodeInfo(VkCommandBuffer cmd);

	shaderio::OctreePushConstant_FzbPG pushConstant{};

	uint32_t octreeMaxLayer = 6;

	std::vector<nvvk::Buffer> octreeClusterDataBuffer_G;
	std::vector<nvvk::Buffer> octreeDataBuffer_G;	//layer0: 6  layer1： 48 ……

	std::vector<nvvk::Buffer> octreeClusterDataBuffer_E;
	nvvk::Buffer clusterLayerDataBuffer_E;

	nvvk::Buffer globalInfoBuffer;
	nvvk::Buffer indivisibleNodeInfosBuffer_G;
	nvvk::Buffer indivisibleNodeInfosBuffer_E;

	nvvk::Buffer octreeNodePairVisibleDataBuffer;
	nvvk::Buffer octreeNodePairWeightBuffer;

	nvvk::Buffer nearbyNodeInfoBuffer;
private:
	OctreeCreateInfo_FzbPG setting;

	nvvk::Buffer blockInfoBuffer_G;
	nvvk::Buffer blockInfoBuffer_E;
	nvvk::Buffer hasDataBlockIndexBuffer_G;
	nvvk::Buffer hasDataBlockIndexBuffer_E;
	nvvk::Buffer hasDataBlockCountBuffer;

	nvvk::Buffer divisibleNodeInfoBuffer_G;		//每层可细分节点的索引
	nvvk::Buffer threadGroupInfoBuffer;

	nvvk::Buffer octreeNodePairEBuffer;

	nvvk::Buffer nearbyNodeTempInfoBuffer;

	VkShaderEXT computeShader_initOctreeArray{};
	VkShaderEXT computeShader_initHasDataBlockInfo{};
	VkShaderEXT computeShader_getGlobalInfo{};
	VkShaderEXT computeShader_createOctreeArray{};
	VkShaderEXT computeShader_createOctreeArray2{};

	VkShaderEXT computeShader_getOctreeLabel1{};
	VkShaderEXT computeShader_getOctreeLabel2{};
	VkShaderEXT computeShader_getOctreeLabel3{};
	VkShaderEXT computeShader_getOctreeLabel4{};

	VkShaderEXT computeShader_initWeights{};
	VkShaderEXT computeShader_octreeNodeHitTest{};
	VkShaderEXT computeShader_visibleAABBCluster{};
	VkShaderEXT computeShader_getProbability{};

	VkShaderEXT computeShader_getNearbyNodes1{};
	VkShaderEXT computeShader_getNearbyNodes2{};

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

#ifndef NDEBUG
public:
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
private:
	void debug_Prepare();
	void debug_OctreeLayer_Visualization(VkCommandBuffer cmd);
	void debug_OctreeIndivisibleNodes_Visualization(VkCommandBuffer cmd);
	void debug_OctreeNodePairHitTestResult_Visualization(VkCommandBuffer cmd);
	void debug_NearbyNodeInfoResult_Visualization(VkCommandBuffer cmd);
	void debug_NodePairVisibleAABB_Visualization(VkCommandBuffer cmd);

	VkShaderEXT vertexShader_OctreeLayer{};
	VkShaderEXT fragmentShader_OctreeLayer{};

	VkShaderEXT vertexShader_OctreeIndivisibleNodes{};
	VkShaderEXT fragmentShader_OctreeIndivisibleNodes{};

	VkShaderEXT vertexShader_OctreeNodePairHitTestResult{};
	VkShaderEXT fragmentShader_OctreeNodePairHitTestResult{};

	VkShaderEXT vertexShader_NearbyNodeInfoResult{};
	VkShaderEXT fragmentShader_NearbyNodeInfoResult{};

	VkImageView depthImageView;

	uint32_t showMapCount = 3;

	uint32_t showOctreeLayerMapCount = 1;
	bool showOctreeLayerMap_G = false;
	bool showOctreeLayerMap_E = false;
	int selectedOctreeLayerMapIndex_G = 0;
	int selectedOctreeLayerMapIndex_E = 0;

	uint32_t showOctreeIndivisibleNodeMapIndex = 2;
	bool showOctreeIndivisibleNodeMap_G = false;

	uint32_t showOctreeNodePairHitTestResultMapIndex = 3;
	bool showOctreeNodePairHitTestResultMap = false;

	uint32_t showNearbyNodeInfoResultMapIndex = 4;
	bool showNearbyNodeInfoResultMap = false;

	uint32_t showOctreeNodePairVisibleAabbMapIndex = 5;
	bool showOctreeNodePairVisibleAabbMap = false;
#endif
};
}
#endif
