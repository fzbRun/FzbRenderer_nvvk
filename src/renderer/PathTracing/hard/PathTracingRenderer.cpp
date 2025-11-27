#include "PathTracingRenderer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include "common/Shader/shaderio.h"
#include <common/utils.hpp>
#include <nvgui/sky.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/type_ptr.hpp>

FzbRenderer::PathTracingRenderer::PathTracingRenderer(RendererCreateInfo& createInfo) {
	createInfo.vkContextInfo.deviceExtensions.push_back( { VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature });
	createInfo.vkContextInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature });
	createInfo.vkContextInfo.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME });
}

void FzbRenderer::PathTracingRenderer::primitiveToGeometry(const shaderio::GltfMesh& gltfMesh,
	VkAccelerationStructureGeometryKHR& geometry, VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
	const shaderio::TriangleMesh triMesh = gltfMesh.triMesh;
	const auto triangleCount = static_cast<uint32_t>(triMesh.indices.count / 3U);

	VkAccelerationStructureGeometryTrianglesDataKHR triangles{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.positions.offset},
		.vertexStride = triMesh.positions.byteStride,
		.maxVertex = triMesh.positions.count - 1,
		.indexType = VkIndexType(gltfMesh.indexType),
		.indexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.indices.offset},
	};

	geometry = VkAccelerationStructureGeometryKHR{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {.triangles = triangles},
		.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
	};

	rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{ .primitiveCount = triangleCount };
}

void FzbRenderer::PathTracingRenderer::createImage() {};
void FzbRenderer::PathTracingRenderer::createGraphicsDescriptorSetLayout() {};
void FzbRenderer::PathTracingRenderer::createGraphicsPipelineLayout() {};
VkShaderModuleCreateInfo FzbRenderer::PathTracingRenderer::compileSlangShader(const std::filesystem::path& filename, const std::span<const uint32_t>& spirv) {
	return VkShaderModuleCreateInfo{};
};
void FzbRenderer::PathTracingRenderer::compileAndCreateGraphicsShaders() {};
void FzbRenderer::PathTracingRenderer::updateTextures() {};
void FzbRenderer::PathTracingRenderer::init() {
	VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	rtProperties.pNext = &asProperties;
	prop2.pNext = &rtProperties;
	vkGetPhysicalDeviceProperties2(Application::app->getPhysicalDevice(), &prop2);
}

void FzbRenderer::PathTracingRenderer::clean() {};
void FzbRenderer::PathTracingRenderer::uiRender() {};
void FzbRenderer::PathTracingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void FzbRenderer::PathTracingRenderer::render(VkCommandBuffer cmd) {};
void FzbRenderer::PathTracingRenderer::postProcess(VkCommandBuffer cmd) {};
void FzbRenderer::PathTracingRenderer::onLastHeadlessFrame() {};