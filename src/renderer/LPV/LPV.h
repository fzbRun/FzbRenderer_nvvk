#pragma once

#include "./LPVShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"

#include <array>
#include <string>
#include <vector>

#ifndef FZBRENDERER_LPV_H
#define FZBRENDERER_LPV_H

namespace FzbRenderer {

// Cascaded Light Propagation Volumes (Kaplanyan & Dachsbacher, I3D 2010).
//
// The renderer is a fully compute / ray query based implementation: the reflective shadow
// maps are traced instead of rasterised, the light propagation runs as a sequence of 6 face
// gather dispatches over the cascaded SH volumes and the final gather shades the primary hits.
// Everything it needs lives in this directory; see README.md for the data flow and for the
// differences to the paper.
class LPVRenderer : public PathTracingRenderer
{
public:
	LPVRenderer() = default;
	~LPVRenderer() = default;

	explicit LPVRenderer(pugi::xml_node& rendererNode);

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
	enum class ImageType_LPV
	{
		eImgRendered,
		eImgTonemapped,
	};

	// One nested grid, moved with the camera and snapped to whole cells.
	struct CascadeLayout
	{
		shaderio::float3  startPos{ 0.0f, 0.0f, 0.0f };
		float             cellSize = 0.25f;
	};

	// placement / bookkeeping
	void updateSceneBounds();
	void updateCascadeLayout();
	void updateFrameInfo();

	// GPU resources
	void createResources();
	void destroyResources();
	void clearVolumes(VkCommandBuffer cmd);
	VkDeviceSize volumeFloatCount() const;
	int          debugGridResolution() const;
	uint32_t     debugVoxelCount() const;
	void         logDebugViewCounts() const;

	// passes
	void runGBufferPass(VkCommandBuffer cmd);
	void runRsmInjectionPass(VkCommandBuffer cmd);
	void runGvMergePass(VkCommandBuffer cmd);
	void runPropagationPasses(VkCommandBuffer cmd);
	void runGatherPass(VkCommandBuffer cmd);
	void tonemapping(VkCommandBuffer cmd);

	// settings. Everything the paper specifies is fixed; only the grid/cascade sizes and the
	// iteration count are parameters (paper 3.3: "the number of required iterations depends on the
	// resolution of the grid", and paper 4 lets the cascade layout be chosen).
	int   cascadeCount = 2;
	int   baseResolution = 16;     // N of the coarsest cascade, doubled per cascade
	int   cascadeResolution[LPV_MAX_CASCADES] = {};
	int   gridResolution = 32;     // derived: finest cascade resolution (debug views, UI ranges)
	int   rsmResolution = 256;     // virtual RSM resolution per light
	int   iterationCount = 8;      // propagation iterations, 0 = injection only (no propagation)
	int   cascadeBlendCells = 4;
	// (the 5 exit-face decomposition of paper 3.3 is not a parameter: it is the scheme)
	float baseCellSize = 0.25f;    // world size of a cell of the finest cascade, <= 0 selects it
	bool  useOcclusion = true;
	bool  gatherOcclusionEnabled = true;
	float gatherOcclusion = 1.0f;
	bool  useDampening = true;
	bool  showDirectLight = true;    // comparison toggle: off drops the direct term from the image
	bool  useTrilinearSample = true; // gather: trilinear lookup of the accumulated volume, off = nearest cell
	bool  skipDirectShadow = false;  // debug: LPV_NO_SHADOW=1 disables the direct shadow ray
	bool  debugCascadeId = false;    // debug: LPV_DEBUG_CASCADE=1 tints the image by cascade index
	bool  debugTransmittance = false;  // debug: LPV_DEBUG_TRANSMIT=1 shows the gather transmittance
	bool  debugSH = false;             // debug: LPV_DEBUG_SH=1 shows the gathered SH magnitude
	int   saveFrame = -1;            // frame index at which the tonemapped image is written

	VkPushConstantsInfo pushInfo{};
	shaderio::LPVPushConstant pushConstant{};

	shaderio::LPVFrameInfo    frameInfo{};

	// GPU resources
	nvvk::Buffer pixelDataBuffer{};
	std::array<nvvk::Buffer, LPV_MAX_CASCADES> lpvVolumesBuffer{};
	std::array<nvvk::Buffer, LPV_MAX_CASCADES> lpvAccumBuffer{};
	std::array<nvvk::Buffer, LPV_MAX_CASCADES> gvMergedBuffer{};
	std::array<nvvk::Buffer, LPV_MAX_CASCADES> gvCameraBuffer{};
	std::array<nvvk::Buffer, LPV_MAX_CASCADES> gvRsmBuffer{};
	nvvk::Buffer frameInfoBuffer{};
	nvvk::Buffer instanceSizeBuffer{};  // paper 4.1: world space bounding box size per instance

	// per instance bounding box size, uploaded every frame (see LPV_ObjectSize in the shaders)
	std::vector<float> instanceSizes;
	bool   sceneBoundsValid        = false;
	size_t sceneBoundsInstanceCount = 0;
	bool   tlasBuilt         = false;
	size_t tlasInstanceCount = 0;
	std::vector<std::string> cascadeLabels;
	std::vector<const char*> cascadeLabelPtrs;

	// shader objects
	VkShaderEXT shaderGBuffer{};
	VkShaderEXT shaderRsmInject{};
	VkShaderEXT shaderGvMerge{};
	VkShaderEXT shaderPropagate{};
	VkShaderEXT shaderGather{};

	// device features required by the float atomics of the injection passes
	VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};

	// cascade / scene state
	VkExtent2D                                  currentResolution{ 0, 0 };
	std::array<CascadeLayout, LPV_MAX_CASCADES> cascades{};
	shaderio::float3                            sceneBoundsMin{ 0.0f };
	shaderio::float3                            sceneCenter{ 0.0f };
	shaderio::float3                            cameraForward{ 0.0f, 0.0f, -1.0f };
	float                                       sceneRadius = 1.0f;
	float                                       sceneHalfDiagonal = 1.0f;
	uint32_t                                    activeLightCount = 0;

	//-------------------------------------------------------------------
	shaderio::LPVDebugPushConstant debugPushConstant{};

	VkShaderEXT shaderCreateDepthImage{};
	void createDeptImage(VkCommandBuffer cmd);

	bool showGrid = false;
	nvvk::Buffer gridDataBuffer;
	VkShaderEXT shaderInjectGrid{};
	VkShaderEXT shaderShowGrid_vertex{};
	VkShaderEXT shaderShowGrid_fragment{};
	void showGridResult(VkCommandBuffer cmd);

	bool showSH = false;
	VkShaderEXT shaderShowSH_vertex{};
	VkShaderEXT shaderShowSH_fragment{};
	void showSHResult(VkCommandBuffer cmd);

	bool showGBuffer = false;
	VkShaderEXT shaderShowGBuffer{};
	void showGBufferResult(VkCommandBuffer cmd);

	bool showRSM = false;
	VkShaderEXT shaderShowRSM{};
	void showRSMResult(VkCommandBuffer cmd);

	bool printSH = false;
	VkShaderEXT shaderPrintSH{};
	void printSHResult(VkCommandBuffer cmd);

	// Parameter presets: LPVParams.md next to this renderer's sources.
	void saveParams();
	void loadParams();
	void rebuildVolumes();
};

}  // namespace FzbRenderer

#endif
