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

	SVOSetting setting;

	uint32_t SVOInitialSize[8] = { 1, 8, 64, 512, 1024, 1024, 1024, 1024 };
	std::vector<nvvk::Buffer> SVOArray_G;
	std::vector<nvvk::Buffer> SVOArray_E;
	nvvk::Buffer SVOIndivisibleNodes_G;
};
}

#endif