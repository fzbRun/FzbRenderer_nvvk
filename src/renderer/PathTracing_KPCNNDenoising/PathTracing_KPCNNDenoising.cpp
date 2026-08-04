#include "./PathTracing_KPCNNDenoising.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

#include <common/Semaphore/Semaphore.h>

using namespace FzbRenderer;

std::vector<std::string> debugImageNames = {
	"ColorImage", "DiffuseImage", "SpecularImage",
	"IrradianceImage", "NormalImage", "DepthImage", "AlbedoImage",
	"IrradianceVarianceImage", "SpecularVarianceImage", "NormalVarianceImage", "DepthVarianceImage", "AlbedoVarianceImage",
};

PathTracing_KPCNNDenoising::PathTracing_KPCNNDenoising(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();
	Application::cmdCount = IF_TRAIN_SAMPLE(1, 2);

	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	Application::vkContextInitInfo.instanceExtensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME });

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		pushConstant.maxBounceCount = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node sppNode = rendererNode.child("spp"))
		pushConstant.spp = std::stoi(sppNode.attribute("value").value());

	IF_SAVE_SAMPLE(pushConstant.spp = 32, pushConstant.spp = 8192);
}
void PathTracing_KPCNNDenoising::init() {
	screenSize = { 512, 512 };

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	createDataObject();

	//IF_TRAIN_SAMPLE(loadSampleBuffers(), );

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	//IF_TRAIN_SAMPLE(bufferToImage(), );

	Renderer::init();

	KPCNNDenoiser_CreateInfo createInfo = {
		.physicalDevice = Application::allocator.getPhysicalDevice(),
		.inputBuffer_diff = inputBuffer_diff,
		.inputBuffer_spec = inputBuffer_spec,
		.albedoBuffer = albedoBuffer,
		.imageSize = screenSize,
		.outputImage = colorImage,
		.startSemaphoreHandle = vulkanToCudaSemaphore.handle,
		.endSemaphoreHandle = cudaToVulkanSemaphore.handle,
	};
	IF_TRAIN_SAMPLE(void(0), kpcnDenoiser = KPCNNDenoiser(createInfo));
}
void PathTracing_KPCNNDenoising::clean() {
	inputBuffer_diff.clean();
	inputBuffer_spec.clean();
	normalBuffer.clean();
	depthBuffer.clean();
	maxDepthBuffer.clean();
	albedoBuffer.clean();
	colorImage.clean();
	vulkanToCudaSemaphore.clean();
	cudaToVulkanSemaphore.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_PathTracing, nullptr);
	vkDestroyShaderEXT(device, computeShader_createGradBuffers, nullptr);
	vkDestroyShaderEXT(device, computeShader_bufferToImage, nullptr);

	IF_TRAIN_SAMPLE(void(), kpcnDenoiser.clean());

	PathTracingRenderer::clean();
}
void PathTracing_KPCNNDenoising::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBufferImageIndex_KPCNN::eTonemapImage);
	//for (int i = 0; i < showImage.size(); ++i) showImage[i] = false;

	if (ImGui::Begin("PathTracing")) {
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		PE::end();
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		ImGui::SeparatorText("Bounces");
		{
			PE::begin();
			PE::SliderInt("Bounces Depth", &pushConstant.maxBounceCount, 1, std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}
		ImGui::SeparatorText("SPP");
		{
			PE::begin();
			UIModified |= PE::SliderInt("SPP", &pushConstant.spp, 1, 8192, "%d", ImGuiSliderFlags_AlwaysClamp,
				"Sample Per Pixel");
			PE::end();
		}

		uint32_t imageIndex = (uint32_t)GBufferImageIndex_KPCNN::eColorDebugImage;
		if (PE::begin()) {
			if (PE::entry("PathTracing Result", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showImage[imageIndex] ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(imageIndex),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showImage[imageIndex] = !showImage[imageIndex];
				for (int i = 0; i < showImage.size(); ++i) {
					if (i == imageIndex) continue;
					showImage[i] = false;
				}
			}
		}
		PE::end();
	}
	ImGui::End();

	if (ImGui::Begin("KPCNN")) {
		for (int imageIndex = (int)GBufferImageIndex_KPCNN::eDiffuseDebugImage; imageIndex <= (int)GBufferImageIndex_KPCNN::eAlbedoVarianceDebugImage; ++imageIndex) {
			if (PE::begin()) {
				if (PE::entry(debugImageNames[imageIndex], [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showImage[imageIndex] ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(imageIndex),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showImage[imageIndex] = !showImage[imageIndex];
					for (int i = 0; i < showImage.size(); ++i) {
						if (i == imageIndex) continue;
						showImage[i] = false;
					}
				}
			}
			PE::end();
		}
	}
	ImGui::End();

	if (UIModified) resetFrame();

	for (int i = 0; i < showImage.size(); ++i) {
		if(showImage[i]) Application::viewportImage = gBuffers.getDescriptorSet(i);
	}
#endif
}
void PathTracing_KPCNNDenoising::resize(VkCommandBuffer cmd, const VkExtent2D& size) {}
void PathTracing_KPCNNDenoising::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) resetFrame();	//如果相机参数变化，则从新累计帧
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;

	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.maxFrameCount = maxFrames;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.screenSize = { screenSize.width, screenSize.height };
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	asManager.updateToplevelAS(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void PathTracing_KPCNNDenoising::render(VkCommandBuffer* cmdPtr) {
	static uint64_t timeline = 1;

	VkCommandBuffer cmd = cmdPtr[0];
	{ NVVK_DBG_SCOPE(cmd); }

	updateDataPerFrame(cmd);
	if (pushConstant.frameIndex >= maxFrames && maxFrames > 1) {
		cmd = cmdPtr[1];
		const VkCommandBufferBeginInfo beginInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
					 .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
		NVVK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

		++timeline;
		return;
	}

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::KPCNN_DenoisingPTPushConstant),
		.pValues = &pushConstant,
	};

#ifdef SAVE_TRAIN_BUFFERS
	if (timeline == 1) {
		pathTracing(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		IF_SAVE_SAMPLE(createInputBuffers(cmd), (void)0);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
	}
	else if (timeline == 2) {
		vkDeviceWaitIdle(Application::app->getDevice());	//after first frame end
		saveSampleBuffers("_7");
	}

	++timeline;
	return;
#else
	pathTracing(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	createInputBuffers(cmd);
#endif
	//--------------------------------------------------------------------------------------------------------------
	vkEndCommandBuffer(cmd);
	const VkCommandBufferSubmitInfo cmdBufferInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd };

	//nvpro_core2框架下GPU在每帧都会等待上一帧渲染完成后(m_frameTimelineSemaphore)才开始下一帧的渲染，所以这里无需一个信号量来同步
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

	kpcnDenoiser.denoising(timeline);

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

	//Renderer::postProcess(cmd, &colorImage.image.descriptor);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, colorImage.image.descriptor, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eTonemapImage));
	//Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, 
	//	gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eColorDebugImage), gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eTonemapImage));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

	++timeline;
}

void PathTracing_KPCNNDenoising::createDataObject() {
	uint32_t debugImageCount = IF_DEBUG((uint32_t)GBufferImageIndex_KPCNN::eDebugImageCount - 1, 0);
#ifndef NDEBUG
	showImage.resize(debugImageCount);
#endif
	Feature::createGBuffer(true, true, debugImageCount, screenSize);

	uint32_t imageSize = screenSize.width * screenSize.height;
	/*
		diffuse: 3, diffuseVariance: 1, gradDiffuse: 6
		normalVariance: 1, gradNormal: 6,
		depthVariance: 1, gradDepth: 2,
		albedoVariance: 1, gradAlbedo: 6
		sum: 27
	*/
	uint32_t inputBufferSize = (3 + 1 + 6 + 1 + 6 + 1 + 2 + 1 + 6) * imageSize * sizeof(float);
	IF_SAVE_SAMPLE(void(0), inputBufferSize = 3 * imageSize * sizeof(float));

	inputBuffer_diff = FzbRenderer::Buffer("inputBuffer_diff", true);
	inputBuffer_diff.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = inputBufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	inputBuffer_spec = FzbRenderer::Buffer("inputBuffer_spec", true);
	inputBuffer_spec.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = inputBufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	
	normalBuffer = FzbRenderer::Buffer("normalBuffer", false);
	normalBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = imageSize * sizeof(float3),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	
	depthBuffer = FzbRenderer::Buffer("depthBuffer", false);
	depthBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = imageSize * sizeof(float),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	maxDepthBuffer = FzbRenderer::Buffer("maxDepthBuffer", false);
	maxDepthBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(int),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	
	albedoBuffer = FzbRenderer::Buffer("albedoBuffer", true);
	albedoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = imageSize * sizeof(float3),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	colorImage = FzbRenderer::Image("colorImage", true);
	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorImageCreateInfo.info.extent = { screenSize.width, screenSize.height, 1 };
	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImage.init(colorImageCreateInfo);

	vulkanToCudaSemaphore.init(true, 0, true);
	cudaToVulkanSemaphore.init(true, 0, true);
}
void PathTracing_KPCNNDenoising::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	{
		bindings.addBinding({
			.binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({		//在我们程序中不需要，只需要占个位置
				.binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------Irradiance--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_irradiance,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		#ifndef SAVE_GROUNDTRUTH_BUFFERS
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_irradianceVariance,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradIrradiance,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------normalDiff--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------depthDiff--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------albedoDiff--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_diff,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		#endif
		//--------------------------------specular--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		#ifndef SAVE_GROUNDTRUTH_BUFFERS
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_specVariance,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradSpec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------normalSpec--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------depthSpec--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------albedoSpec--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_spec,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		//--------------------------------Normal, Depth, Albedo--------------------------------------
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eMaxDepthBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlbedoBuffer,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		#endif
	}
#ifndef NDEBUG
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eColorDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDiffuseDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eSpecularDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eIrradianceDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlebdoDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eIrradianceVarianceDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eSpecularVariancDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalVarianceDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthVarianceDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlbedoVarianceDebugImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
#endif

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Fzb PathGuiding static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));

	bindings.clear();
	bindings.addBinding({
			.binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
}
void PathTracing_KPCNNDenoising::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	uint32_t imageSize = screenSize.width * screenSize.height;
	uint32_t offset = 0;

	{
		//-------------------------------------------Irradiance---------------------------------------------------
		VkWriteDescriptorSet    inputBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_irradiance, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float) * 3);
		offset += imageSize * sizeof(float) * 3;

		#ifndef SAVE_GROUNDTRUTH_BUFFERS
		inputBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_irradianceVariance, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradIrradiance, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;
		//-------------------------------------------Normal_Diff---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;
		//-------------------------------------------Depth_Diff---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float) * 2);
		offset += imageSize * sizeof(float) * 2;
		//-------------------------------------------Albedo_Diff---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_diff, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_diff.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;
		#endif


		//-------------------------------------------Specular---------------------------------------------------
		offset = 0;

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float) * 3);
		offset += imageSize * sizeof(float3);

		#ifndef SAVE_GROUNDTRUTH_BUFFERS
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_specVariance, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradSpec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;
		//-------------------------------------------Normal_Spec---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_normalVariance_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradNormal_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;
		//-------------------------------------------Depth_Spec---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_depthVariance_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradDepth_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float) * 2);
		offset += imageSize * sizeof(float) * 2;
		//-------------------------------------------Albedo_Spec---------------------------------------------------
		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_albedoVariance_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float));
		offset += imageSize * sizeof(float);

		inputBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eInputBuffer_gradAlbedo_spec, 0, 0, 1);
		write.append(inputBufferWrite, inputBuffer_spec.buffer, offset, imageSize * sizeof(float3) * 2);
		offset += imageSize * sizeof(float3) * 2;

		//-------------------------------------------Normal, Depth, Albedo---------------------------------------------------
		VkWriteDescriptorSet    dataBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalBuffer, 0, 0, 1);
		write.append(dataBufferWrite, normalBuffer.buffer);

		dataBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthBuffer, 0, 0, 1);
		write.append(dataBufferWrite, depthBuffer.buffer);

		dataBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eMaxDepthBuffer, 0, 0, 1);
		write.append(dataBufferWrite, maxDepthBuffer.buffer);

		dataBufferWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlbedoBuffer, 0, 0, 1);
		write.append(dataBufferWrite, albedoBuffer.buffer);
		#endif
	}
#ifndef NDEBUG
	{
		VkWriteDescriptorSet    debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eColorDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eColorDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDiffuseDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eDiffuseDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eSpecularDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eSpecularDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eIrradianceDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eIrradianceDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eNormalDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eDepthDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlebdoDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eAlbedoDebugImage));


		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eIrradianceVarianceDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eIrradianceVarianceDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eSpecularVariancDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eSpecularVariancDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eNormalVarianceDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eNormalVarianceDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eDepthVarianceDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eDepthVarianceDebugImage));

		debugImageWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_KPCNNPT::eAlbedoVarianceDebugImage, 0, 0, 1);
		write.append(debugImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBufferImageIndex_KPCNN::eAlbedoVarianceDebugImage));
	}
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void PathTracing_KPCNNDenoising::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::KPCNN_DenoisingPTPushConstant)
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
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
void PathTracing_KPCNNDenoising::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "pathTracing_KPCNN.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::KPCNN_DenoisingPTPushConstant),
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
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
		vkDestroyShaderEXT(device, computeShader_PathTracing, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_PathTracing";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_PathTracing);
		NVVK_DBG_NAME(computeShader_PathTracing);
	}
	#ifndef SAVE_GROUNDTRUTH_BUFFERS
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_createGradBuffers, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createGradBuffers";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createGradBuffers);
		NVVK_DBG_NAME(computeShader_createGradBuffers);
	}
	#endif
//#ifndef NDEBUG
//	//--------------------------------------------------------------------------------------
//	{
//		vkDestroyShaderEXT(device, computeShader_bufferToImage, nullptr);
//
//		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
//		shaderInfo.nextStage = 0;
//		shaderInfo.pName = "computeMain_bufferToImage";
//		shaderInfo.codeSize = shaderCode.codeSize;
//		shaderInfo.pCode = shaderCode.pCode;
//		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_bufferToImage);
//		NVVK_DBG_NAME(computeShader_bufferToImage);
//	}
//#endif
}
void PathTracing_KPCNNDenoising::updateDataPerFrame(VkCommandBuffer cmd) {}

void PathTracing_KPCNNDenoising::pathTracing(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_PathTracing);

	//VkExtent2D sceneSize = Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ PATHTRACING_BLOCKSIZE_KPCNN, PATHTRACING_BLOCKSIZE_KPCNN });

	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdFillBuffer(cmd, maxDepthBuffer.buffer.buffer, 0, sizeof(int), 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void PathTracing_KPCNNDenoising::createInputBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createGradBuffers);

	pushConstant.screenSize = { screenSize.width, screenSize.height };
	VkExtent2D groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ 32, 32 });

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}

void PathTracing_KPCNNDenoising::saveSampleBuffers(std::string filename) {
	std::string samplePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models_libtorch/train/" + Application::sceneResource.name + "_" + std::to_string(pushConstant.spp) + filename;
	inputBuffer_diff.save(samplePath + "/diff.bin");
	inputBuffer_spec.save(samplePath + "/spec.bin");
	normalBuffer.save(samplePath + "/normal.bin");
	albedoBuffer.save(samplePath + "/albedo.bin");
	depthBuffer.save(samplePath + "/depth.bin");
}
void PathTracing_KPCNNDenoising::loadSampleBuffers() {
	std::string samplePath = FzbRenderer::getProjectRootDir().string() + "src/renderer/PathTracing_KPCNNDenoising/models_libtorch/train/" + Application::sceneResource.name + "_" + std::to_string(pushConstant.spp);
	inputBuffer_diff.load(samplePath + "/diff.bin");
	inputBuffer_spec.load(samplePath + "/spec.bin");
	normalBuffer.load(samplePath + "/normal.bin");
	albedoBuffer.load(samplePath + "/albedo.bin");
	depthBuffer.load(samplePath + "/depth.bin");
}
void PathTracing_KPCNNDenoising::bufferToImage() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::KPCNN_DenoisingPTPushConstant),
		.pValues = &pushConstant,
	};

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_bufferToImage);

	pushConstant.screenSize = { screenSize.width, screenSize.height };
	VkExtent2D groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ 32, 32 });

	vkCmdPushConstants2(cmd, &pushInfo);
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

	Application::app->submitAndWaitTempCmdBuffer(cmd);
}