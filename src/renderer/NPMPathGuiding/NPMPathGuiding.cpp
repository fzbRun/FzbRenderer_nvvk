#include "./NPMPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

#include <common/Image/Image.h>
#include <common/Semaphore/Semaphore.h>

using namespace FzbRenderer;

NPMPathGuiding::NPMPathGuiding(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();
	Application::cmdCount = 1;

	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME });
}
void NPMPathGuiding::init() {

	//测试一下cuda是否有问题，测试方案就是将一个image和一个buffer传给CUDA，然后写入数据，然后根据信号量进行同步，最后将数据读回来看看是否正确
	//ImageCreateInfo imageCreateInfo = createDefaultImageCreateInfo();
	flowerImage = FzbRenderer::Image("flowerImage", true);
	std::filesystem::path texturePath = FzbRenderer::getProjectRootDir() / "src/renderer/NPMPathGuiding/testImage/rose.jpg";
	flowerImage.init(texturePath);

	VkPhysicalDevice physicalDevice = Application::allocator.getPhysicalDevice();
	VkDevice device = Application::allocator.getDevice();

	vulkanToCudaSemaphore.init(true, 0, true);
	cudaToVulkanSemaphore.init(true, 0, true);

	//Renderer::init();
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	Application::stagingUploader.cmdUploadAppended(cmd);
	Application::stagingUploaderExport.cmdUploadAppended(cmd);
	// Submit and clean up
	vkEndCommandBuffer(cmd);

	// Create fence for synchronization
	const VkFenceCreateInfo fenceInfo{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
	std::array<VkFence, 1>  fence{};
	vkCreateFence(device, &fenceInfo, nullptr, fence.data());

	const VkCommandBufferSubmitInfo cmdBufferInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd };

	VkSemaphoreSubmitInfo signalSemaphoreInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = vulkanToCudaSemaphore.semaphoreState.getSemaphore(),
		.value = 1,
		.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
	};
	const std::array<VkSubmitInfo2, 1> submitInfo{
		{{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, 
		.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdBufferInfo,
		.signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signalSemaphoreInfo,}} };
	vkQueueSubmit2(Application::app->getQueue(0).queue, uint32_t(submitInfo.size()), submitInfo.data(), fence[0]);

	//VkSemaphoreSignalInfo signalInfo{};
	//signalInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
	//signalInfo.semaphore = vulkanToCudaSemaphore.semaphoreState.getSemaphore(); // 假设封装暴露了 VkSemaphore
	//signalInfo.value = 1;                              // 时间线值 +1
	//vkSignalSemaphore(device, &signalInfo);

	Image_yReversal_CreateInfo cudaCreateInfo = {
		.physicalDevice = physicalDevice,
		.image = flowerImage,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
	};
	cudaPrograme = Image_yReversal(cudaCreateInfo);
	cudaPrograme.reversal();

	vkWaitForFences(device, uint32_t(fence.size()), fence.data(), VK_TRUE, UINT64_MAX);

	// Cleanup
	vkDestroyFence(device, fence[0], nullptr);
	vkFreeCommandBuffers(device, Application::app->getCommandPool(), 1, &cmd);
}
void NPMPathGuiding::clean() {
	flowerImage.clean();
	vulkanToCudaSemaphore.clean();
	cudaToVulkanSemaphore.clean();

	cudaPrograme.clean();

	PathTracingRenderer::clean();
}
void NPMPathGuiding::uiRender() {
	Application::viewportImage = flowerImage.uiDescriptorSet;
}
void NPMPathGuiding::resize(VkCommandBuffer cmd, const VkExtent2D& size) {}
void NPMPathGuiding::preRender() {}
void NPMPathGuiding::render(VkCommandBuffer* cmdPtr) {}

void NPMPathGuiding::createDescriptorSetLayout() {}
void NPMPathGuiding::createDescriptorSet() {}
void NPMPathGuiding::createPipelineLayout() {}
void NPMPathGuiding::compileAndCreateShaders() {}
void NPMPathGuiding::updateDataPerFrame(VkCommandBuffer cmd) {}

void NPMPathGuiding::pathGuiding(VkCommandBuffer cmd) {}