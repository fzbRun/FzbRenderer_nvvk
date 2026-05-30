#include "./NPMPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

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
	flowerImage = FzbRenderer::Image("flowerImage", false);
	std::filesystem::path texturePath = FzbRenderer::getProjectRootDir() / "src/renderer/NPMPathGuiding/testImage/rose.jpg";
	flowerImage.init(texturePath);

	inputTensor = FzbRenderer::Buffer("inputTensor", true);
	uint32_t inputTensorSize = 1 * 3 * 224 * 224 * sizeof(float);
	inputTensor.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = inputTensorSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	
	//colorImage = FzbRenderer::Image("colorImage", true);
	//FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	//colorImageCreateInfo.info.format = VK_FORMAT_R8G8B8A8_UNORM;
	//colorImageCreateInfo.info.extent = { 512, 512, 1 };
	//colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	//colorImage.init(colorImageCreateInfo);

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

	ImageRecognition_CreateInfo cudaCreateInfo = {
		.physicalDevice = physicalDevice,
		.buffer = inputTensor,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
	};
	cudaPrograme = ImageRecognition(cudaCreateInfo);
}
void NPMPathGuiding::clean() {
	flowerImage.clean();
	inputTensor.clean();
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

	cudaPrograme.recognition(timeline);

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
		.binding = (uint32_t)shaderio::StaticBindingPoints_NPMPG::eInputTensor,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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

	VkWriteDescriptorSet    inputTensorWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NPMPG::eInputTensor, 0, 0, 1);
	write.append(inputTensorWrite, &inputTensor.buffer);

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
	if (pushConstant.time == 0) sceneSize = { 224, 224 };
	pushConstant.screenSize = { sceneSize.width, sceneSize.height };
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}