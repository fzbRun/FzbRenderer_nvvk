#pragma once

#include "../../Feature.h"
#include "./shaderio.h"
#include <feature/SceneDivision/Octree/Octree.h>

#ifndef FZBRENDERER_SVO_H
#define FZBRENDERER_SVO_H

namespace FzbRenderer {
struct SVOSetting{
	std::shared_ptr<Octree> octree;
};

class SparseVoxelOctree : public Feature {
public:
	SparseVoxelOctree() = default;
	virtual ~SparseVoxelOctree() = default;

	SparseVoxelOctree(pugi::xml_node& featureNode);

	void init(SVOSetting setting);
	void clean();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void uiRender();
	void preRender();
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);

	void createSVOArray();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void compileAndCreateShaders() override;

	void updateDataPerFrame(VkCommandBuffer cmd) override;

	void initSVOArray(VkCommandBuffer cmd);
	void createSVOArray(VkCommandBuffer cmd);

	SVOSetting setting;
	shaderio::SVOPushConstant pushConstant{};
	VkPushConstantsInfo pushInfo;

	//后面可以将G的和E的initialSize分开来，一般E的会显著小于G的；并且我们可以根据场景的不同，设置不同的initialSize
	uint32_t SVOInitialSize_G[MAX_OCTREE_DEPTH] = { 1, SVONodeCount_G_Layer1, SVONodeCount_G_Layer2, SVONodeCount_G_Layer3, 1024, 1024, 1024, 2048 };
	uint32_t SVOInitialSize_E[MAX_OCTREE_DEPTH] = { 1, SVONodeCount_E_Layer1, SVONodeCount_E_Layer2, SVONodeCount_E_Layer3, 512, 512, 512, 1024 };
	std::vector<nvvk::Buffer> SVOArray_G;
	std::vector<nvvk::Buffer> SVOArray_E;
	nvvk::Buffer SVOIndivisibleNodes_G;

	nvvk::Buffer SVOLayerInfos_G;
	nvvk::Buffer SVOLayerInfos_E;

	VkShaderEXT computeShader_initSVOArray{};
	VkShaderEXT computeShader_createSVOArray{};
	VkShaderEXT computeShader_offsetLabelMultiBlock{};	//多个线程组需要跨线程组进行通信

private:
	nvvk::Buffer SVOGlobalInfo;

	nvvk::Buffer SVODivisibleNodeIndices_G;		//每层可细分节点的索引
	nvvk::Buffer SVODivisibleNodeIndices_E;

	nvvk::Buffer SVOThreadGroupInfos;

#ifndef NDEBUG
public:
	void debugPrepare();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex);
	void debug_wirefame(VkCommandBuffer cmd);

	VkImageView depthImageView;

	VkShaderEXT vertexShader_Wireframe{};
	VkShaderEXT fragmentShader_Wireframe{};

	bool showWireframeMap_G = false;
	bool showWireframeMap_E = false;
#endif
};
}

#endif