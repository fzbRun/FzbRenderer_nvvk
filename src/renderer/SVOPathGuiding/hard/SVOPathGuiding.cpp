#include "./SVOPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>

FzbRenderer::SVOPathGuidingRenderer::SVOPathGuidingRenderer(pugi::xml_node& rendererNode) {
	PathTracingRenderer::setContextInfo();

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
	createRayTracingDescriptorLayout();
	Renderer::addTextureArrayDescriptor(shaderio::BindingPoints::eTextures, &staticDescPack);

	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), rtProperties);

	createRayTracingPipeline();
	Renderer::init();
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

void FzbRenderer::SVOPathGuidingRenderer::createRayTracingDescriptorLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::BindingPoints::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL }, VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({
			.binding = shaderio::BindingPoints::eTlas,
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
			.binding = shaderio::BindingPoints::eOutImage,
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

	VkResult result;
	result = staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Ray tracing descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void FzbRenderer::SVOPathGuidingRenderer::createRayTracingPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI(" Creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), pipelineLayout, nullptr);

	enum StageIndices {
		eRaygen,
		eMiss,
		eClosestHit,

		eClosestHit_NEE,

		eCallable_DiffuseMaterial,
		eCallable_ConductorMaterial,
		eCallable_DielectricMaterial,
		eCallable_RoughConductorMaterial,
		eCallable_RoughDielectricMaterial,
		eShaderGroupCount
	};
	std::vector<VkPipelineShaderStageCreateInfo> stages(eShaderGroupCount);
	for (auto& s : stages)
		s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

	//VkShaderModuleCreateInfo shaderCode = compileSlangShader("pathTracingShaders.slang", {});
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVOPathTracingShaders.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	stages[eRaygen].pNext = &shaderCode;
	stages[eRaygen].pName = "raygenMain";
	stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[eMiss].pNext = &shaderCode;
	stages[eMiss].pName = "rayMissMain";
	stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;

	stages[eClosestHit].pNext = &shaderCode;
	stages[eClosestHit].pName = "rayClosestHitMain";
	stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	//stages[eAnyHit].pNext = &shaderCode;
	//stages[eAnyHit].pName = "rayAnyHitMain";
	//stages[eAnyHit].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;

	stages[eClosestHit_NEE].pNext = &shaderCode;
	stages[eClosestHit_NEE].pName = "NEEClosestHitMain";
	stages[eClosestHit_NEE].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	stages[eCallable_DiffuseMaterial].pNext = &shaderCode;
	stages[eCallable_DiffuseMaterial].pName = "diffuseMaterialMain";
	stages[eCallable_DiffuseMaterial].stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;

	stages[eCallable_ConductorMaterial].pNext = &shaderCode;
	stages[eCallable_ConductorMaterial].pName = "conductorMaterialMain";
	stages[eCallable_ConductorMaterial].stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;

	stages[eCallable_DielectricMaterial].pNext = &shaderCode;
	stages[eCallable_DielectricMaterial].pName = "dielectricMaterialMain";
	stages[eCallable_DielectricMaterial].stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;

	stages[eCallable_RoughConductorMaterial].pNext = &shaderCode;
	stages[eCallable_RoughConductorMaterial].pName = "roughConductorMaterialMain";
	stages[eCallable_RoughConductorMaterial].stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;

	stages[eCallable_RoughDielectricMaterial].pNext = &shaderCode;
	stages[eCallable_RoughDielectricMaterial].pName = "roughDielectricMaterialMain";
	stages[eCallable_RoughDielectricMaterial].stage = VK_SHADER_STAGE_CALLABLE_BIT_KHR;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;	//表示光线追踪pipeline有几个阶段，光纤生成->打中/没打中
	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	//光线生成shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	shader_groups.push_back(group);

	//光线没打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	shader_groups.push_back(group);

	//光线打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	//group.anyHitShader = eAnyHit;
	shader_groups.push_back(group);

	if (pushValues.NEEShaderIndex == 1) {		//使用NEE
		group.closestHitShader = eClosestHit_NEE;
		shader_groups.push_back(group);

		pushValues.NEEShaderIndex = 1;
	}

	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;

	group.generalShader = eCallable_DiffuseMaterial;
	shader_groups.push_back(group);

	group.generalShader = eCallable_ConductorMaterial;
	shader_groups.push_back(group);

	group.generalShader = eCallable_DielectricMaterial;
	shader_groups.push_back(group);

	group.generalShader = eCallable_RoughConductorMaterial;
	shader_groups.push_back(group);

	group.generalShader = eCallable_RoughDielectricMaterial;
	shader_groups.push_back(group);

	const VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::PathTracingPushConstant) };

	VkPipelineLayoutCreateInfo pipeline_layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipeline_layout_create_info.pushConstantRangeCount = 1;
	pipeline_layout_create_info.pPushConstantRanges = &push_constant;

	std::vector<VkDescriptorSetLayout> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout()} };	//二合一
	pipeline_layout_create_info.setLayoutCount = uint32_t(layouts.size());
	pipeline_layout_create_info.pSetLayouts = layouts.data();
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &pipelineLayout);
	NVVK_DBG_NAME(pipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::min(MAX_DEPTH, rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = pipelineLayout;
	rtPipelineInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, &rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	createShaderBindingTable(rtPipelineInfo);
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