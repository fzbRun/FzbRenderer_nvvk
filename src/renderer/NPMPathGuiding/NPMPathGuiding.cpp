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
	Application::cmdCount = 2;

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
	flowerImage = FzbRenderer::Image("flowerImage", false);
	std::filesystem::path texturePath = FzbRenderer::getProjectRootDir() / "src/renderer/NPMPathGuiding/testImage/rose.jpg";
	flowerImage.init(texturePath);

	colorImage = FzbRenderer::Image("colorImage", true);
	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R8G8B8A8_UNORM;
	colorImageCreateInfo.info.extent = { 512, 512, 1 };
	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImage.init(colorImageCreateInfo);

	Feature::createGBuffer(true, true, 1);

	VkPhysicalDevice physicalDevice = Application::allocator.getPhysicalDevice();
	VkDevice device = Application::allocator.getDevice();

	vulkanToCudaSemaphore.init(true, 0, true);
	cudaToVulkanSemaphore.init(true, 0, true);

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();

	Image_yReversal_CreateInfo cudaCreateInfo = {
		.physicalDevice = physicalDevice,
		.image = colorImage,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
	};
	cudaPrograme = Image_yReversal(cudaCreateInfo);

	/*
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

	cudaPrograme.reversal();

	vkWaitForFences(device, uint32_t(fence.size()), fence.data(), VK_TRUE, UINT64_MAX);

	// Cleanup
	vkDestroyFence(device, fence[0], nullptr);
	vkFreeCommandBuffers(device, Application::app->getCommandPool(), 1, &cmd);
	*/
}
void NPMPathGuiding::clean() {
	flowerImage.clean();
	colorImage.clean();
	vulkanToCudaSemaphore.clean();
	cudaToVulkanSemaphore.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_NPMPathGuiding, nullptr);

	cudaPrograme.clean();

	PathTracingRenderer::clean();
}
void NPMPathGuiding::uiRender() {
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);
}
void NPMPathGuiding::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NPMPG::eOutImage, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.screenSize = shaderio::uint2(size.width, size.height);
}
void NPMPathGuiding::preRender() {
	pushConstant.frameIndex = Application::frameIndex;
}
void NPMPathGuiding::render(VkCommandBuffer* cmdPtr) {
	static uint64_t timeline = 1;

	VkCommandBuffer cmd = cmdPtr[0];
	{ NVVK_DBG_SCOPE(cmd); }

	updateDataPerFrame(cmd);

	pushConstant.time = 0;
	pathGuiding(cmd);
	//--------------------------------------------------------------------------------------------------------------
	vkEndCommandBuffer(cmd);
	const VkCommandBufferSubmitInfo cmdBufferInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd };

	//GPU在每帧都会等待上一帧渲染完成后才开始下一帧的渲染，所以这里无需一个信号量来同步
	VkSemaphoreSubmitInfo signalSemaphoreInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = vulkanToCudaSemaphore.semaphoreState.getSemaphore(),
		.value = timeline,
		.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
	};
	const std::array<VkSubmitInfo2, 1> submitInfo{
		{{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
		.commandBufferInfoCount = 1, .pCommandBufferInfos = &cmdBufferInfo,
		.signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signalSemaphoreInfo,}} };
	vkQueueSubmit2(Application::app->getQueue(0).queue, uint32_t(submitInfo.size()), submitInfo.data(), nullptr);

	cudaPrograme.reversal(pushConstant.frameIndex, timeline);

	VkSemaphoreSubmitInfo waitSemaphoreInfo{
		.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
		.semaphore = cudaToVulkanSemaphore.semaphoreState.getSemaphore(),
		.value = timeline,
		.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
	};
	Application::app->addWaitSemaphore(waitSemaphoreInfo);

	//--------------------------------------------------------------------------------------------------------------
	cmd = cmdPtr[1];
	const VkCommandBufferBeginInfo beginInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
				 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
	NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

	{ NVVK_DBG_SCOPE(cmd); }

	pushConstant.time = 1;
	pathGuiding(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Renderer::postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

	++timeline;
}

void NPMPathGuiding::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NPMPG::eOutImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NPMPG::eFlowerImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NPMPG::eColorImageWrite,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NPMPG::eColorImageRead,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Fzb PathGuiding static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void NPMPathGuiding::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    flowerImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NPMPG::eFlowerImage, 0, 0, 1);
	write.append(flowerImageWrite, &flowerImage.image);

	VkWriteDescriptorSet    colorImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NPMPG::eColorImageWrite, 0, 0, 1);
	write.append(colorImageWrite, &colorImage.image);

	colorImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NPMPG::eColorImageRead, 0, 0, 1);
	write.append(colorImageWrite, &colorImage.image);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void NPMPathGuiding::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::NPMPathGuidingPushConstant)
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void NPMPathGuiding::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "NPMPathGuiding.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::NPMPathGuidingPushConstant),
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_NPMPathGuiding, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_reversalColor";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_NPMPathGuiding);
		NVVK_DBG_NAME(computeShader_NPMPathGuiding);
	}
}
void NPMPathGuiding::updateDataPerFrame(VkCommandBuffer cmd) {}

void NPMPathGuiding::pathGuiding(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_NPMPathGuiding);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::NPMPathGuidingPushConstant),
		.pValues = &pushConstant,
	};

	VkExtent2D sceneSize = Application::app->getViewportSize();
	if (pushConstant.time == 0) sceneSize = { colorImage.setting.info.extent.width, colorImage.setting.info.extent.height };
	pushConstant.screenSize = { sceneSize.width, sceneSize.height };
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}