#pragma once

#include "./VXPGShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"

#ifndef FZBRENDERER_VXPG_H
#define FZBRENDERER_VXPG_H

namespace FzbRenderer {

	enum class ImageType_VXPG
	{
		eImgRendered,
		eImgTonemapped,
	};

	// Voxel path guiding renderer, port of SIByLEngine2023's "VXGuiding" basic pipeline
	// (sample mode 4, "Guide + BSDF"). Everything it needs lives in this directory; see README.md.
	class VXPGRenderer : public PathTracingRenderer
	{
	public:
		VXPGRenderer() = default;
		~VXPGRenderer() = default;

		explicit VXPGRenderer(pugi::xml_node& rendererNode);

		void init() override;
		void clean() override;
		void uiRender() override;
		void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
		void preRender() override;
		void render(VkCommandBuffer* cmdPtr) override;

		void createDescriptorSetLayout() override;
		void createDescriptorSet();
		void createPipelineLayout();
		void compileAndCreateShaders() override;
		void updateDataPerFrame(VkCommandBuffer cmd) override;

	private:
		// GPU resources
		void updateSceneBounds();
		void updateVoxelDomain();
		void computeResolution();
		void createResources();
		void destroyResources();
		void clearVoxelGrids(VkCommandBuffer cmd);
		void clearSpixelWeights(VkCommandBuffer cmd);
		void dumpGuidingState();

		// passes
		void bindCommonState(VkCommandBuffer cmd);
		void dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
	void dispatchComputeIndirect(VkCommandBuffer cmd, VkShaderEXT shader, VkBuffer buffer, VkDeviceSize offset);
		void runVBufferPass(VkCommandBuffer cmd);
		void runLightInjectionPass(VkCommandBuffer cmd);
		void runGeometryInjectionPass(VkCommandBuffer cmd);
		void runTreePasses(VkCommandBuffer cmd);
		void runClusterPasses(VkCommandBuffer cmd);
		void runSpixelPasses(VkCommandBuffer cmd);
		void runIntegratorPass(VkCommandBuffer cmd);
		void runDebugClusterPass(VkCommandBuffer cmd);
	void runSuperPixelDebugPass(VkCommandBuffer cmd);

		// settings
		int  maxFrames = (MAX_FRAME) / 2;
		bool rebuildPerFrame = true;
		bool enableGuiding = true;
		int  debugMode = 0;

		shaderio::VXPGPushConstant pushConstant{};
		VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
		VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT dummyInterlock{};

		// GPU buffers
		nvvk::Buffer voxelIrradianceBuffer{};
		nvvk::Buffer voxelVplCountBuffer{};
		nvvk::Buffer voxelBoundMinBuffer{};
		nvvk::Buffer voxelBoundMaxBuffer{};
		nvvk::Buffer pixelDataBuffer{};
		nvvk::Buffer treeNodesBuffer{};
		nvvk::Buffer clusterRootsBuffer{};
		nvvk::Buffer topLevelTreeBuffer{};
		nvvk::Buffer spixelInfoBuffer{};
		nvvk::Buffer spixelClusterWeightBuffer{};
		nvvk::Buffer spixelClusterTotalBuffer{};
		nvvk::Buffer leafVoxelBuffer{};      // N^3: leaf slot -> flat voxel index (indirection)
		nvvk::Buffer voxelLeafBuffer{};      // N^3: flat voxel index -> leaf slot
		nvvk::Buffer clusterRangeBuffer{};   // VXPG_CLUSTER_COUNT x uint2: leaf range per cluster
		nvvk::Buffer repSPBuffer{};            // VXPG_REP_SP_COUNT x float4 (stratified jittered SPs)
		nvvk::Buffer voxelSignatureBuffer{};   // N^3 x uint4 (128 bit visibility signatures)
		nvvk::Buffer litVoxelListBuffer{};     // VXPG_KMPP_MAX_VOXELS uint
		nvvk::Buffer litVoxelCountBuffer{};    // 1 uint
		nvvk::Buffer clusterCentroidBuffer{};
	nvvk::Buffer clusterCentroidEBuffer{};   // centroid irradiance (paper Eq. 4)  // VXPG_CLUSTER_COUNT x uint4
		nvvk::Buffer voxelClusterBuffer{};     // N^3 uint
		nvvk::Buffer spixelAccumBuffer{};
	nvvk::Buffer globalInfoBuffer{};        // spixelCount x VXPGSpixelAccum (SLIC sums)

		// sizes
		VkExtent2D currentResolution{ 0, 0 };
		uint32_t   voxelCount = 0;
		uint32_t   leafCount = 0;
		uint32_t   nodeCount = 0;
		uint32_t   spixelGridWidth = 0;
		uint32_t   spixelGridHeight = 0;
		uint32_t   spixelCount = 0;

		// scene bounds (cubic voxel grid, same construction as the reference VoxelTexInfo)
		shaderio::float3 sceneStartPos{ 0.0f, 0.0f, 0.0f };
		shaderio::float3 sceneVoxelSize{ 1.0f, 1.0f, 1.0f };
		float sceneExtent = 0.5f;
		shaderio::float3 sceneCenter{ 0.0f, 0.0f, 0.0f };
		float autoClipExtent = 2.5f;
		float clipExtentSetting = 0.0f;
		glm::vec3 domainCenterBuild{ 0.0f };
		float domainExtentBuild = 0.0f;
		bool guidingDirty = true;
		bool useVisibilityClustering = true;   // KMPP clustering (VXPG_KMPP=1 to enable)
		bool showClusterDebug = false;      // UI: colour the lit voxels by cluster
		bool showSuperPixelDebug = false;   // UI: colour the image by superpixel id
		bool fuzzySpixel = true;                // supplemental Eq. 5 (4 nearest superpixels)
		bool pdfTestEnabled = false;            // VXPG_PDFTEST=1: pdf consistency probe
		VkShaderEXT shaderPdfTest{};

		// shader objects
		VkShaderEXT shaderVBuffer{};
		VkShaderEXT shaderLightInject{};
		VkShaderEXT shaderGeometryInject{};
		VkShaderEXT shaderTreeInit{};
		VkShaderEXT shaderLeafMapInit{};
		VkShaderEXT shaderTreeInternal{};
		VkShaderEXT shaderTreeAccumulate{};
		VkShaderEXT shaderClusterRoots{};
		VkShaderEXT shaderRepSP{};
		VkShaderEXT shaderLitVoxelSort{};
	VkShaderEXT shaderSignature{};
		VkShaderEXT shaderClusterSeed{};
	VkShaderEXT shaderClusterSeed2{};
		VkShaderEXT shaderClusterAssign{};
		VkShaderEXT shaderClusterUpdate{};
		VkShaderEXT shaderClusterSort{};
		VkShaderEXT shaderTreeMerge{};
		VkShaderEXT shaderSpixelCenters{};
		VkShaderEXT shaderSpixelAssign{};
		VkShaderEXT shaderSpixelUpdate{};
		VkShaderEXT shaderSpixelAccum{};
		VkShaderEXT shaderTopLevel{};
		VkShaderEXT shaderIntegrator{};
		VkShaderEXT shaderSuperPixelDebug{};
	VkShaderEXT shaderDebugCluster_vert{};
		VkShaderEXT shaderDebugCluster_frag{};
	};

}  // namespace FzbRenderer

#endif
