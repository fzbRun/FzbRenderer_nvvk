#include "./ZhiHuCode.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

#include <common/Semaphore/Semaphore.h>

using namespace FzbRenderer;

ZhiHuCode::ZhiHuCode(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();
	Application::cmdCount = 2;

	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME });
}
void ZhiHuCode::init() {
	flowerImage = FzbRenderer::Image("flowerImage", true);
	std::filesystem::path texturePath = FzbRenderer::getProjectRootDir() / "src/renderer/NPMPathGuiding/testImage/cat.jpg";
	flowerImage.init(texturePath);

#ifdef Step2_UseModel
	inputBuffer = FzbRenderer::Buffer("inputBuffer", true);
	uint32_t inputBufferSize = 1 * 3 * 224 * 224 * sizeof(float);
	inputBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = inputBufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
#endif

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

#ifdef Step1_VulkanCudaOp
	Image_yReversal_CreateInfo cudaCreateInfo = {
		.physicalDevice = physicalDevice,
		.image = flowerImage,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
	};
	cudaPrograme = Image_yReversal(cudaCreateInfo);
#elif defined(Step2_UseModel)
	ModelCreateInfo modelCreateInfo{
		.enginePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/ZhiHuCode/models/resnet34.engine",
		.precision = (int)nvinfer1::BuilderFlag::kFP16,
		.inputShape_min = {1, 3, 224, 224},
		.inputShape_opt = {1, 3, 224, 224},
		.inputShape_max = {1, 3, 224, 224},
		.outputShape = {1, 1000},
	};

	ImageRecognition_CreateInfo cudaCreateInfo = {
		.physicalDevice = physicalDevice,
		.buffer = inputBuffer,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
		.modelCreateInfo = modelCreateInfo,
	};
	cudaPrograme = ImageRecognition(cudaCreateInfo);
#endif
}
void ZhiHuCode::clean() {
	flowerImage.clean();
	vulkanToCudaSemaphore.clean();
	cudaToVulkanSemaphore.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_copyImage, nullptr);

	cudaPrograme.clean();

#ifdef Step2_UseModel
	inputBuffer.clean();
	vkDestroyShaderEXT(device, computeShader_createInputBuffer, nullptr);
#endif

	PathTracingRenderer::clean();
}
void ZhiHuCode::uiRender() {
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);
}
void ZhiHuCode::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eOutImage, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.screenSize = shaderio::uint2(size.width, size.height);
}
void ZhiHuCode::preRender() {
	pushConstant.frameIndex = Application::frameIndex;
}
void ZhiHuCode::render(VkCommandBuffer* cmdPtr) {
	static uint64_t timeline = 1;

	VkCommandBuffer cmd = cmdPtr[0];
	{ NVVK_DBG_SCOPE(cmd); }

	updateDataPerFrame(cmd);
#ifdef Step1_VulkanCudaOp
	renderFunction(cmd);
#elif defined(Step2_UseModel)
	createInputBuffer(cmd);
#endif
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

	cudaPrograme.infer(pushConstant.frameIndex, timeline);

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

	renderFunction(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Renderer::postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

	++timeline;
}

void ZhiHuCode::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eOutImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eFlowerImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eInputBuffer,
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
void ZhiHuCode::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    flowerImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eFlowerImage, 0, 0, 1);
	write.append(flowerImageWrite, &flowerImage.image);

#ifdef Step2_UseModel
	VkWriteDescriptorSet    inputTensorWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ZhiHuCode::eInputBuffer, 0, 0, 1);
	write.append(inputTensorWrite, &inputBuffer.buffer);
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void ZhiHuCode::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ZhiHuCodePushConstant)
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
void ZhiHuCode::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "copyImage.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::ZhiHuCodePushConstant),
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
		vkDestroyShaderEXT(device, computeShader_copyImage, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_copyImage";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_copyImage);
		NVVK_DBG_NAME(computeShader_copyImage);
	}
#ifdef Step2_UseModel
	//--------------------------------------------------------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "createInputBuffer.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_createInputBuffer, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createInputBuffer";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createInputBuffer);
		NVVK_DBG_NAME(computeShader_createInputBuffer);
	}
#endif
}
void ZhiHuCode::updateDataPerFrame(VkCommandBuffer cmd) {}

#ifdef Step2_UseModel
void ZhiHuCode::createInputBuffer(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createInputBuffer);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ZhiHuCodePushConstant),
		.pValues = &pushConstant,
	};

	pushConstant.screenSize = { 224, 224 };
	VkExtent2D groupSize = nvvk::getGroupCounts(VkExtent2D(224, 224), VkExtent2D{16, 16});

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
#endif
void ZhiHuCode::renderFunction(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_copyImage);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::ZhiHuCodePushConstant),
		.pValues = &pushConstant,
	};

	VkExtent2D sceneSize = Application::app->getViewportSize();
	pushConstant.screenSize = { sceneSize.width, sceneSize.height };
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}