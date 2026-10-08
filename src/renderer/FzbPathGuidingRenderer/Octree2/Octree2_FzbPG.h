#pragma once

#include "feature/Feature.h"
#include "./Octree2Shaderio_FzbPG.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_OCTREE_2_FZBPG_H
#define FZBRENDERER_OCTREE_2_FZBPG_H
namespace FzbRenderer {
struct Octree2CreateInfo_FzbPG {
	std::vector<nvvk::Buffer> VGBs;

	shaderio::float3 VGBStartPos;
	shaderio::float3 VGBVoxelSize;
	float VGBSize;

	AccelerationStructureManager* asManager;
};

class Octree2_FzbPG : public Feature {
public:
	Octree2_FzbPG() = default;
	virtual ~Octree2_FzbPG() = default;

	Octree2_FzbPG(pugi::xml_node& featureNode);

	void init(Octree2CreateInfo_FzbPG createInfo);
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

	shaderio::OctreePushConstant_FzbPG pushConstant{};

	uint32_t octreeMaxLayer = 6;

	std::vector<nvvk::Buffer> octreeNodeDataBuffer_G;
	std::vector<nvvk::Buffer> octreeNodeInfoBuffer_G;			//label_indivisible,  label: indivisible node在当前层是第几个indivisibleNode，从1开始；indivisible：判断node是不是indivisible的
	std::vector<nvvk::Buffer> octreeNodeDataBuffer_E;

	nvvk::Buffer globalInfoBuffer;
	nvvk::Buffer indivisibleNodeInfosBuffer_G;					//所有indivisibleNode按照label排序，存有uint2：layerIndex + nodeIndex

	nvvk::Buffer clusterPairInfoBuffer;
	nvvk::Buffer clusterPairGlobalInfoBuffer;
	nvvk::Buffer candidateNodeDataBuffer_E;

	nvvk::Buffer octreeNodePairDataBuffer;
private:
	Octree2CreateInfo_FzbPG setting;

	nvvk::Buffer blockInfoBuffer_G;
	nvvk::Buffer blockInfoBuffer_E;
	nvvk::Buffer hasDataBlockIndexBuffer_G;
	nvvk::Buffer hasDataBlockIndexBuffer_E;
	nvvk::Buffer hasDataBlockCountBuffer;

	nvvk::Buffer divisibleNodeInfoBuffer_G;
	nvvk::Buffer threadGroupInfoBuffer;

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
	VkShaderEXT computeShader_getCandidateNodes{};
	VkShaderEXT computeShader_dispatchGetWeight{};
	VkShaderEXT computeShader_getCandidateNodeWeights{};;

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo;
	VkPushConstantsInfo pushInfo;

	enum class GBuffers_Octree2_FzbPG {
		eGeometryClusterResult,
		eLightClusterResult,
		eGeometrySVO,
		eBufferCount,
	};
	shaderio::OctreeDebugPushConstant_FzbPG debugPushConstant;
	void geometryTreeDebug(VkCommandBuffer cmd);
	void lightTreeDebug(VkCommandBuffer cmd);
	void geometrySVODebug(VkCommandBuffer cmd);
	VkShaderEXT vertexShader_TreeDebug{};
	VkShaderEXT fragmentShader_TreeDebug{};
};
}
#endif
