#pragma once

#include "renderer/Renderer.h"
#include <glm/ext/vector_float2.hpp>
#include "common/Shader/nvvk/shaderio.h"
#include <nvvk/acceleration_structures.hpp>
#include <nvvk/sbt_generator.hpp>
#include <nvvk/staging.hpp>
#include "common/Shader/shaderStructType.h"
#include "./shaderio.h"
#include <feature/SceneDivision/RasterVoxelization/RasterVoxelization.h>
#include "common/Application/Application.h"
#include "AccelerationStructure.h"
#include <feature/PathTracing/PathTracing.h>

#ifndef FZB_PATH_TRACING_RENDERER_H
#define FZB_PATH_TRACING_RENDERER_H

namespace FzbRenderer {

class PathTracingRenderer : public FzbRenderer::Renderer {
public:
	PathTracingRenderer() = default;
	~PathTracingRenderer() = default;

	PathTracingRenderer(pugi::xml_node& rendererNode);

	void init() override;
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void preRender() override;
	void render(VkCommandBuffer cmd) override;

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	virtual void createRayTracingDescriptorLayout();
	virtual void createRayTracingDescriptor();
	virtual void createRayTracingPipeline();
	void rayTraceScene(VkCommandBuffer cmd);

	void resetFrame() { Application::frameIndex = 0; };

	int maxFrames = MAX_FRAME / 2;

	PathTracingContext ptContext;

	VkPipeline rtPipeline{};

	std::vector<VkAccelerationStructureInstanceKHR> staticTlasInstances;

	AccelerationStructureManager asManager;
	nvvk::SBTGenerator sbtGenerator;
	nvvk::Buffer sbtBuffer;
private:
	shaderio::PathTracingPushConstant pushValues{};
};

}

#endif