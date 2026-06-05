#include "./PathTracing_KPCNNDenoising.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

#include <common/Semaphore/Semaphore.h>

using namespace FzbRenderer;

PathTracing_KPCNNDenoising::PathTracing_KPCNNDenoising(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();
	Application::cmdCount = 2;

	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME });
}
void PathTracing_KPCNNDenoising::init() {
	Feature::createGBuffer(true, true, 0);

	VkPhysicalDevice physicalDevice = Application::allocator.getPhysicalDevice();
	VkDevice device = Application::allocator.getDevice();

	vulkanToCudaSemaphore.init(true, 0, true);
	cudaToVulkanSemaphore.init(true, 0, true);

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();


}