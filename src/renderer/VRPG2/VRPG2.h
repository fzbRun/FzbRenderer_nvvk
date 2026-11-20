#pragma once

#include "./VRPG2Shaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"

#include <feature/ShadowMap/ShadowMap.h>
#include <feature/FLIP/FLIP.h>
#include "GeometryInjection/GeometryInjection.h"
#include "GeometryClustering/GeometryClustering.h"
#include "LightInjection/LightInjection.h"
#include "LightClustering/LightClustering.h"
#include "RIS/RIS.h"
#include "PathGuiding/PathGuiding.h"

#ifndef FZBRENDERER_VRPG_2_H
#define FZBRENDERER_VRPG_2_H

namespace FzbRenderer {
struct VRPGSetting {
	int frameIndex = 0;
	int randomFrameIndex = 0;

	int spp = 1;
	int maxBounces = 4;
	int maxFrameCount = 1;		//frames the image is accumulated over (renderer maxFrames)

	bool staticScene = true;
	bool rebuildPerFrame = false;

	//GeometryInjection
	bool geometryChange = false;
	shaderio::int2 rasterVoxelizatioResolution;
	int voxelGridResolution;
	shaderio::float3 voxelGridStartPos;
	shaderio::float3 voxelSize;

	//GeometryClustering
	bool clusteringChange = false;
	float clusteringThreshold = 0.75f;
	int geometryOctreeDepth;
	int geometryClusteringTopLayer = 3;

	//LightInjection
	bool lightChange = false;
	int spv_lightInjection = 32;		//spv: sample per voxel
	int bounces_lightInjection = 1;

	//LightClustering
	int lightOctreeDepth;

	//PathGuiding
	float jitterRadiusScale = 0.5f;
};
struct VRPGSource {
	AccelerationStructureManager* asManager;

	std::vector<nvvk::Buffer> VGBs;		//6 normal

	nvvk::Buffer globalInfoBuffer;

	std::vector<nvvk::Buffer> clusterDataBuffer_Octree;
	std::vector<nvvk::Buffer> clusterInfoBuffer_Octree;	//label_indivisible
	nvvk::Buffer clusterInfoBuffer_SVO;		//layerIndex, nodeIndex
	nvvk::Buffer nearbyClusterInfoBuffer;

	std::vector<nvvk::Buffer> lightNodeDataBuffer_Octree;

	// RIS candidate tables (written by RIS_VRPG, read by the path guiding pass)
	nvvk::Buffer clusterPairInfoBuffer;
	nvvk::Buffer candidatePoolInfoBuffer;
	nvvk::Buffer candidatePoolNodeInfoBuffer;
	nvvk::Buffer clusterPairHitInfoBuffer;
	nvvk::Buffer candidateSubTreeInfoBuffer;
	nvvk::Buffer candidateLeafNodeDataBuffer;
	nvvk::Buffer candidateSampleDataBuffer;
	nvvk::Buffer candidateSamplePrefixWSumBuffer;
};

class VRPG2Renderer : public PathTracingRenderer {
public:
	VRPG2Renderer() = default;
	~VRPG2Renderer() = default;

	VRPG2Renderer(pugi::xml_node& rendererNode);

	void init() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void uiRender() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;
	void render(VkCommandBuffer* cmd) override;
	void clean() override;


	static VRPGSetting setting;
	static VRPGSource source;
private:
	std::shared_ptr<GeometryInjection_VRPG> geometryInjection;
	std::shared_ptr<GeometryClustering_VRPG> geometryClustering;
	std::shared_ptr<LightInjection_VRPG> lightInjection;
	std::shared_ptr<LightClustering_VRPG> lightClustering;
	std::shared_ptr<RIS_VRPG> ris;
	std::shared_ptr<PathGuiding_VRPG> pathGuiding;
};
}

#endif