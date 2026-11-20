#pragma once

#include "./LPVShaderio.h"
#include "renderer/PathTracingRenderer/hard/PathTracingRenderer.h"

#include <array>
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
		shaderio::float3 origin{ 0.0f, 0.0f, 0.0f };
		float             cellSize = 0.25f;
		float             extent = 8.0f;
		float             startDistance = 0.0f;
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

	// passes
	void bindStaticState(VkCommandBuffer cmd);
	void dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
	void runGBufferPass(VkCommandBuffer cmd);
	void runRsmInjectionPass(VkCommandBuffer cmd);
	void runGvMergePass(VkCommandBuffer cmd);
	void runPropagationPasses(VkCommandBuffer cmd);
	void runGatherPass(VkCommandBuffer cmd);

	// settings. Everything the paper specifies is fixed; only the grid/cascade sizes and the
	// iteration count are parameters (paper 3.3: "the number of required iterations depends on the
	// resolution of the grid", and paper 4 lets the cascade layout be chosen).
	int   cascadeCount = 3;
	int   gridResolution = 32;     // N, cubic resolution of every cascade
	int   rsmResolution = 256;     // virtual RSM resolution per light
	int   iterationCount = 8;      // propagation iterations, 0 = injection only (no propagation)
	// (the 5 exit-face decomposition of paper 3.3 is not a parameter: it is the scheme)
	float baseCellSize = 0.25f;    // world size of a cell of the finest cascade, <= 0 selects it
	bool  useOcclusion = true;
	bool  useDampening = true;
	bool  showDirectLight = true;    // comparison toggle: off drops the direct term from the image
	int   debugMode = 0;             // LPVDebugView
	bool  skipDirectShadow = false;  // debug: LPV_NO_SHADOW=1 disables the direct shadow ray
	int   saveFrame = -1;            // frame index at which the tonemapped image is written

	shaderio::LPVPushConstant pushConstant{};
	shaderio::LPVFrameInfo    frameInfo{};

	// GPU resources
	nvvk::Buffer pixelDataBuffer{};
	nvvk::Buffer lpvVolumesBuffer{};
	nvvk::Buffer lpvAccumBuffer{};
	nvvk::Buffer gvMergedBuffer{};
	nvvk::Buffer gvCameraBuffer{};
	nvvk::Buffer gvRsmBuffer{};
	nvvk::Buffer frameInfoBuffer{};
	nvvk::Buffer instanceSizeBuffer{};  // paper 4.1: world space bounding box size per instance

	// per instance bounding box size, uploaded every frame (see LPV_ObjectSize in the shaders)
	std::vector<float> instanceSizes;

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
	shaderio::float3                            sceneBoundsMax{ 0.0f };
	shaderio::float3                            sceneCenter{ 0.0f };
	shaderio::float3                            cameraForward{ 0.0f, 0.0f, -1.0f };
	float                                       sceneRadius = 1.0f;
	uint32_t                                    activeLightCount = 0;
};

}  // namespace FzbRenderer

#endif
