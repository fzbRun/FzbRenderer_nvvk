#include "./SVOPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>

FzbRenderer::SVOPathGuidingRenderer::SVOPathGuidingRenderer(pugi::xml_node& rendererNode) {
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME });

	rtPosFetchFeature.rayTracingPositionFetch = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rtPosFetchFeature });

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		pushValues.maxDepth = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node useNEENode = rendererNode.child("useNEE"))
		pushValues.NEEShaderIndex = std::string(useNEENode.attribute("value").value()) == "true";

	if (pugi::xml_node rasterVoxelizationNode = rendererNode.child("RasterVoxelization"))
		rasterVoxelization = std::make_shared<RasterVoxelization>(rasterVoxelizationNode);
}
void FzbRenderer::SVOPathGuidingRenderer::init() {
	rasterVoxelization->init();
	
	PathTracingRenderer::getRayTracingPropertiesAndFeature();
	Renderer::createGBuffer(false, true, 2);
	createDescriptorSetLayout();
	Renderer::addTextureArrayDescriptor(shaderio::SVOPGBindingPoints::eTextures_SVOPG, &rtDescPack);

	asBuilder.init(&Application::allocator, &Application::stagingUploader, Application::app->getQueue(0));
	sbtGenerator.init(Application::app->getDevice(), rtProperties);

}
void FzbRenderer::SVOPathGuidingRenderer::clean() {
	rasterVoxelization->clean();
	PathTracingRenderer::clean();
};
void FzbRenderer::SVOPathGuidingRenderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);

	if (ImGui::Begin("SVOPathGuidingSettings"))
	{
		ImGui::SeparatorText("Jitter");
		UIModified |= ImGui::SliderInt("Max Frames", &maxFrames, 1, MAX_FRAME);
		ImGui::TextDisabled("Frame: %d", pushValues.frameIndex);

		ImGui::SeparatorText("Bounces");
		{
			PE::begin();
			PE::SliderInt("Bounces Depth", &pushValues.maxDepth, 1, std::min(MAX_DEPTH, rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}

		bool NEEChange = ImGui::Checkbox("USE NEE", (bool*)&pushValues.NEEShaderIndex);
		if (NEEChange) {
			vkQueueWaitIdle(Application::app->getQueue(0).queue);
			createRayTracingPipeline();

			UIModified |= NEEChange;
		}

		if (rtPosFetchFeature.rayTracingPositionFetch == VK_FALSE)
		{
			ImGui::TextColored({ 1, 0, 0, 1 }, "ERROR: Position Fetch not supported!");
			ImGui::Text("This hardware does not support");
			ImGui::Text("VK_KHR_ray_tracing_position_fetch");
			ImGui::Text("Please use RTX 20 series or newer GPU.");
		}
		else
		{
			ImGui::TextColored({ 0, 1, 0, 1 }, "Position Fetch: SUPPORTED");
			ImGui::Separator();
		}
	}
	ImGui::End();

	rasterVoxelization->uiRender();

	if (UIModified) resetFrame();
};
void FzbRenderer::SVOPathGuidingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	PathTracingRenderer::resize(cmd, size);
	rasterVoxelization->resize(cmd, size, gBuffers, eImgTonemapped);
};
void FzbRenderer::SVOPathGuidingRenderer::preRender() {
	PathTracingRenderer::preRender();
	rasterVoxelization->preRender();
}
void FzbRenderer::SVOPathGuidingRenderer::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	rasterVoxelization->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
	PathTracingRenderer::rayTraceScene(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	Renderer::postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	rasterVoxelization->postProcess(cmd);
};

void FzbRenderer::SVOPathGuidingRenderer::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::SVOPGBindingPoints::eTextures_SVOPG,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL }, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eTlas_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		}, 
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
		.binding = shaderio::SVOPGBindingPoints::eVGB_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL
		});
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eOutImage_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eOctree_G_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eOctree_E_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eSVO_G_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eSVO_E_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eSVOTlas_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eWeights_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		},
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT);
	bindings.addBinding({
			.binding = shaderio::SVOPGBindingPoints::eOutImage_MIS_SVOPG,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});

	rtDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Ray tracing descriptor layout created\n");
	NVVK_DBG_NAME(rtDescPack.getLayout());
	NVVK_DBG_NAME(rtDescPack.getPool());
	NVVK_DBG_NAME(rtDescPack.getSet(0));
}
void FzbRenderer::SVOPathGuidingRenderer::compileAndCreateShaders() {
	PathTracingRenderer::compileAndCreateShaders();
	rasterVoxelization->compileAndCreateShaders();


};
void FzbRenderer::SVOPathGuidingRenderer::updateDataPerFrame(VkCommandBuffer cmd) {
	rasterVoxelization->updateDataPerFrame(cmd);
}

void FzbRenderer::SVOPathGuidingRenderer::createOctree(VkCommandBuffer cmd) {

}