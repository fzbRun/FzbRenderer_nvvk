#pragma once

#include "feature/Feature.h"
#include <renderer/PathTracingRenderer/hard/AccelerationStructure.h>
#include <nvvk/sbt_generator.hpp>

#ifndef FZBRENDERER_PATHTRACING_FEATURE_H
#define FZBRENDERER_PATHTRACING_FEATURE_H

#define MAX_DEPTH 64U

namespace FzbRenderer {

struct PathTracingContext {
	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };
	//设置rtPosFetchFeature后，顶点位置数据以某种优化的形式与BLAS结构紧密关联；并且这些pos数据在创建blas时自动创建，因此无需顶点位置数据缓冲区
	VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rtPosFetchFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR };
	VkPhysicalDeviceRayTracingMotionBlurFeaturesNV rtMotionBlurFeatures{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_MOTION_BLUR_FEATURES_NV };

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };

	void setContextInfo();
	void getRayTracingPropertiesAndFeature();
};

struct PathTracingSetting{
	AccelerationStructureManager* asManager;
	PathTracingContext* ptContext;
};
class PathTracing : public Feature{
public:
	PathTracing() = default;
	virtual ~PathTracing() = default;

	void init(PathTracingSetting setting);
	void clean() override;

	PathTracingSetting setting;
	nvvk::SBTGenerator sbtGenerator;
	nvvk::Buffer sbtBuffer;
};

void addPathTracingSlangMacro();
void createShaderBindingTable(
	const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo, VkPipeline& rtPipeline,
	nvvk::SBTGenerator& sbtGenerator, nvvk::Buffer& sbtBuffer);
}

#endif