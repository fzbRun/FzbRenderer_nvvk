#pragma once

#include <nvvk/gbuffers.hpp>
#include <nvvk/graphics_pipeline.hpp>
#include <nvvk/descriptors.hpp>

#include "renderer/Renderer.h"
#include <glm/ext/vector_float2.hpp>
#include "common/Shader/shaderio.h"

#ifndef FZB_PATH_TRACING_RENDERER_H
#define FZB_PATH_TRACING_RENDERER_H

namespace FzbRenderer {

class PathTracingRenderer : public FzbRenderer::Renderer {
	enum
	{
		eImgRendered,
		eImgTonemapped
	};

public:
	PathTracingRenderer() = default;
	~PathTracingRenderer() = default;

	PathTracingRenderer(RendererCreateInfo& createInfo);
	void init() override;
	void compileAndCreateGraphicsShaders();
	void clean() override;
	void uiRender() override;
	void resize(VkCommandBuffer cmd, const VkExtent2D& size) override;
	void render(VkCommandBuffer cmd) override;
	void onLastHeadlessFrame() override;
private:
	void primitiveToGeometry(const shaderio::GltfMesh& gltfMesh,
		VkAccelerationStructureGeometryKHR& geometry, VkAccelerationStructureBuildRangeInfoKHR& rangeInfo);
	void createAccelerationStructure(VkAccelerationStructureTypeKHR asType, nvvk::AccelerationStructure& accelStruct,
		VkAccelerationStructureGeometryKHR& asGeometry, VkAccelerationStructureBuildRangeInfoKHR& asBuildRangeInfo,
		VkBuildAccelerationStructureFlagsKHR flags);
	void createBottomLevelAS();
	void createToLevelAS();
	void createRayTracingDescriptorLayout();
	void createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo);
	void createRayTracingPipeline();

	void createImage();
	void createGraphicsDescriptorSetLayout();
	void createGraphicsPipelineLayout();
	VkShaderModuleCreateInfo compileSlangShader(const std::filesystem::path& filename, const std::span<const uint32_t>& spirv);
	void updateTextures();
	void postProcess(VkCommandBuffer cmd) override;

	nvvk::GraphicsPipelineState dynamicPipeline;
	nvvk::DescriptorPack        descPack;
	VkPipelineLayout            graphicPipelineLayout{};

	VkShaderEXT vertexShader{};
	VkShaderEXT fragmentShader{};

	glm::vec2 metallicRoughnessOverride{ -0.01f, -0.01f };

	nvvk::DescriptorPack rtDescPack;
	VkPipeline rtPipeline{};
	VkPipelineLayout rtPipelineLayout{};

	std::vector<nvvk::AccelerationStructure> blasAccel;
	nvvk::AccelerationStructure tlasAccel;

	nvvk::Buffer sbtBuffer;
	std::vector<uint8_t> shaderHandles;
	VkStridedDeviceAddressRegionKHR raygenRegion{};
	VkStridedDeviceAddressRegionKHR missRegion{};
	VkStridedDeviceAddressRegionKHR hitRegion{};
	VkStridedDeviceAddressRegionKHR callableRegion{};

	VkPhysicalDeviceAccelerationStructureFeaturesKHR accelFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR };
	VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeature{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR };

	VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR };
	VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };

};

}

#endif