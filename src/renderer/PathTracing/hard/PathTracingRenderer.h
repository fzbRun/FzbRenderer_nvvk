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

#ifndef FZB_PATH_TRACING_RENDERER_H
#define FZB_PATH_TRACING_RENDERER_H

#define MAX_DEPTH 64U

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
	void preRender();
	void render(VkCommandBuffer cmd) override;

	void compileAndCreateShaders() override;
	void updateDataPerFrame(VkCommandBuffer cmd) override;

	nvvk::AccelerationStructureGeometryInfo primitiveToGeometry(const shaderio::Mesh& mesh);
	void createBottomLevelAS();
	void createTopLevelAS();
	void updateTopLevelAS(VkCommandBuffer cmd);
	void createRayTracingDescriptorLayout();
	void createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo);
	void createRayTracingPipeline();
	void rayTraceScene(VkCommandBuffer cmd);

	void resetFrame();

	void getRayTracingPropertiesAndFeature();

	int maxFrames = MAX_FRAME / 2;

	nvvk::DescriptorPack rtDescPack;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

	std::vector<VkAccelerationStructureInstanceKHR> staticTlasInstances;

	AccelerationStructureManager asManager;
	nvvk::SBTGenerator sbtGenerator;
	nvvk::Buffer sbtBuffer;

	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	//设置rtPosFetchFeature后，顶点位置数据以某种优化的形式与BLAS结构紧密关联；并且这些pos数据在创建blas时自动创建，因此无需顶点位置数据缓冲区
	VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rtPosFetchFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR };
	VkPhysicalDeviceRayTracingMotionBlurFeaturesNV rtMotionBlurFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV };

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

	shaderio::PathTracingPushConstant pushValues{};
};

}

#endif