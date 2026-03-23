#pragma once

#include "feature/Feature.h"
#include "./shaderio.h"
#include <feature/SceneDivision/Octree/Octree.h>

#ifndef FZBRENDERER_SVOPATHGUIDING_SVO_H
#define FZBRENDERER_SVOPATHGUIDING_SVO_H

namespace FzbRenderer {
struct SVOSetting_SVOPG {
	std::shared_ptr<Octree> octree;
};

class SVO_SVOPG : public Feature {
public:
	SVO_SVOPG() = default;
	virtual ~SVO_SVOPG() = default;

	SVO_SVOPG(pugi::xml_node& featureNode);

	void init(SVOSetting_SVOPG setting);
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

	SVOSetting_SVOPG setting;
	shaderio::SVOPushConstant_SVOPG pushConstant{};
	VkPushConstantsInfo pushInfo;

	nvvk::Buffer SVO_G;		//buffer per normal
	nvvk::Buffer SVO_E;

	VkShaderEXT computeShader_initSVOArray{};
	VkShaderEXT computeShader_createSVOArray{};
	VkShaderEXT computeShader_offsetLabelMultiBlock{};	//多个线程组需要跨线程组进行通信
private:
	nvvk::Buffer SVOGlobalInfo;

	nvvk::Buffer SVODivisibleNodeInfos_G;		//每层可细分节点的索引
	nvvk::Buffer SVODivisibleNodeInfos_E;

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