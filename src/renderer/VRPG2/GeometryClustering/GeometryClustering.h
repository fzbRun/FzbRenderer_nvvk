#pragma once

#include "feature/Feature.h"
#include "./GeometryClusteringShaderio.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>

#ifndef FZBRENDERER_Geometry_Clustering_VRPG_H
#define FZBRENDERER_Geometry_Clustering_VRPG_H
namespace FzbRenderer {
class GeometryClustering_VRPG : public Feature {
public:
	GeometryClustering_VRPG() = default;
	virtual ~GeometryClustering_VRPG() = default;

	GeometryClustering_VRPG(pugi::xml_node& featureNode);

	void init();
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender();
	void uiRender();
	void updateDataPerFrame(VkCommandBuffer cmd) override;
	void render(VkCommandBuffer cmd);
	void postProcess(VkCommandBuffer cmd);
	void clean();
private:	void createSource();
	void createDescriptorSetLayout() override;
	void createDescriptorSet();
	void createPipeline();
	void compileAndCreateShaders();

	nvvk::Buffer clusteringGlobalInfoBuffer;

	VkShaderEXT computeShader_init{};
	void initClusterOctree(VkCommandBuffer cmd);

	nvvk::Buffer clusterBlockInfoBuffer;
	nvvk::Buffer hasDataClusterBlockIndexBuffer;
	VkShaderEXT computeShader_GetHasDataClusterBlockInfo{};
	VkShaderEXT computeShader_DispatchClustering{};
	VkShaderEXT computeShader_Clustering1{};
	VkShaderEXT computeShader_Clustering2{};
	void createClusterOctree(VkCommandBuffer cmd);

	std::vector<nvvk::Buffer> divisibleNodeInfoBuffer;
	nvvk::Buffer threadGroupInfoBuffer;
	VkShaderEXT computeShader_CreateSVO1{};
	VkShaderEXT computeShader_CreateSVO2{};
	VkShaderEXT computeShader_CreateSVO3{};
	VkShaderEXT computeShader_CreateSVO4{};
	void createClusterSVO(VkCommandBuffer cmd);

	nvvk::Buffer nearbyClusterTempInfoBuffer;
	VkShaderEXT computeShader_dispatchGetNearbyCluster{};
	VkShaderEXT computeShader_getNearbyCluster1{};
	VkShaderEXT computeShader_getNearbyCluster2{};
	void getNearbyClusterInfo(VkCommandBuffer cmd);

	shaderio::GeometryClusteringPushConstant_VRPG pushConstant{};
	VkPushConstantsInfo pushInfo;

	bool showOctree = false;
	bool showSVO = false;
	shaderio::GeometryClusteringDebugPushConstant_VRPG debugPushConstant;
	VkPushConstantsInfo debugPushInfo{};
	VkShaderEXT vertexShader_TreeDebug{};
	VkShaderEXT fragmentShader_TreeDebug{};
	void octreeDebug(VkCommandBuffer cmd);
	void svoDebug(VkCommandBuffer cmd);
};
}
#endif