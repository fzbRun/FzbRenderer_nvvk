#include "./SVOPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>

FzbRenderer::SVOPathGuidingRenderer::SVOPathGuidingRenderer(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		pushConstant.maxDepth = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node rasterVoxelizationNode = rendererNode.child("RasterVoxelization"))
		rasterVoxelization = std::make_shared<FzbRenderer::RasterVoxelization>(rasterVoxelizationNode);
	if (pugi::xml_node lightInjectNode = rendererNode.child("LightInject"))
		lightInject = std::make_shared<FzbRenderer::LightInject>(lightInjectNode);
	if (pugi::xml_node octreeNode = rendererNode.child("Octree"))
		octree = std::make_shared<FzbRenderer::Octree>(octreeNode);
}
void FzbRenderer::SVOPathGuidingRenderer::init() {
	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	rasterVoxelization->init();

	LightInjectSetting lightInjectSetting{
		.VGB = rasterVoxelization->VGB,
		.VGBStartPos = rasterVoxelization->setting.pushConstant.voxelGroupStartPos,
		.VGBVoxelSize = glm::vec3(rasterVoxelization->setting.pushConstant.voxelSize_Count),
		.VGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w,
		.ptContext = &ptContext,
		.asManager = &asManager
	};
	lightInject->init(lightInjectSetting);

	OctreeSetting octreeSetting{
		.VGB = rasterVoxelization->VGB,
		.VGBStartPos = lightInjectSetting.VGBStartPos,
		.VGBVoxelSize = lightInjectSetting.VGBVoxelSize,
		.VGBSize = lightInjectSetting.VGBSize
	};
	octree->init(octreeSetting);

	IF_DEBUG(Feature::createGBuffer(true, true, 1), Feature::createGBuffer(false, true, 1));
	createDescriptorSetLayout();
	Renderer::createPipelineLayout(sizeof(shaderio::SVOPathGuidingPushConstant));
	createDescriptorSet();
	createPipeline();

	Renderer::init();
}
void FzbRenderer::SVOPathGuidingRenderer::clean() {
	rasterVoxelization->clean();
	lightInject->clean();
	octree->clean();
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
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		ImGui::SeparatorText("Bounces");
		{
			PE::begin();
			PE::SliderInt("Bounces Depth", &pushConstant.maxDepth, 1, std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}

		if (ptContext.rtPosFetchFeature.rayTracingPositionFetch == VK_FALSE)
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
	lightInject->uiRender();
	octree->uiRender();

	if (UIModified) resetFrame();
};
void FzbRenderer::SVOPathGuidingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    depthImageWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOPG::eDepthImage_SVOPG, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	IF_DEBUG(rasterVoxelization->resize(cmd, size, gBuffers, eImgTonemapped), rasterVoxelization->resize(cmd, size));
	lightInject->resize(cmd, size);
	IF_DEBUG(octree->resize(cmd, size, gBuffers, eImgTonemapped), octree->resize(cmd, size));
	
};
void FzbRenderer::SVOPathGuidingRenderer::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) resetFrame();	//如果相机参数变化，则从新累计帧
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;
	pushConstant.frameIndex = std::min(Application::frameIndex, maxFrames - 1);
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	asManager.updateToplevelAS();

	rasterVoxelization->preRender();
	lightInject->preRender();
	octree->preRender();
}
void FzbRenderer::SVOPathGuidingRenderer::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	rasterVoxelization->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);
	lightInject->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	octree->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_NV);

	pathGuiding(cmd);
	//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	Renderer::postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	rasterVoxelization->postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	lightInject->postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT);
	octree->postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
};

void FzbRenderer::SVOPathGuidingRenderer::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
			.binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL});
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOPG::eVGB_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifndef NDEBUG
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOPG::eDepthImage_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("SVO PathGuiding static descriptor layout created\n");
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

	LOGI("SVO PathGuiding dynamic descriptor layout created\n");
}
void FzbRenderer::SVOPathGuidingRenderer::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOPG::eVGB_SVOPG, 0, 0, 1);
	write.append(VGBWrite, rasterVoxelization->VGB);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbRenderer::SVOPathGuidingRenderer::createPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI(" Creating SVO PathGuiding pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), pipelineLayout, nullptr);

	Application::slangCompiler.clearMacros();

	addPathTracingSlangMacro();
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVOPathGuidingShaders.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	enum StageIndices {
		eRaygen,
		eMiss,
		eClosestHit,
		
		eCallable_DiffuseMaterial,
		eCallable_ConductorMaterial,
		eCallable_DielectricMaterial,
		eCallable_RoughConductorMaterial,
		eCallable_RoughDielectricMaterial,

		eShaderGroupCount
	};
	std::vector<VkPipelineShaderStageCreateInfo> stages(eShaderGroupCount);
	{
		for (auto& s : stages)
			s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		//----------------------------------------rayGen----------------------------------------
		stages[eRaygen].pNext = &shaderCode;
		stages[eRaygen].pName = "raygenMain";
		stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
		//----------------------------------------miss----------------------------------------
		stages[eMiss].pNext = &shaderCode;
		stages[eMiss].pName = "rayMissMain";
		stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
		//----------------------------------------hit----------------------------------------
		stages[eClosestHit].pNext = &shaderCode;
		stages[eClosestHit].pName = "rayClosestHitMain";
		stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

		//stages[eAnyHit].pNext = &shaderCode;
		//stages[eAnyHit].pName = "rayAnyHitMain";
		//stages[eAnyHit].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
		//----------------------------------------callable----------------------------------------
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
	}
	
	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;	//表示光线追踪pipeline有几个阶段，光纤生成->打中/没打中
	{
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
	}
	
	const VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::SVOPathGuidingPushConstant) };

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
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = pipelineLayout;
#ifdef PathTracingMotionBlur
	rtPipelineInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
#endif
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, &rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	FzbRenderer::createShaderBindingTable(rtPipelineInfo, rtPipeline, sbtGenerator, sbtBuffer);
}
void FzbRenderer::SVOPathGuidingRenderer::compileAndCreateShaders() {
	createPipeline();
	rasterVoxelization->compileAndCreateShaders();
	lightInject->compileAndCreateShaders();
	octree->compileAndCreateShaders();
};
void FzbRenderer::SVOPathGuidingRenderer::updateDataPerFrame(VkCommandBuffer cmd) {
	rasterVoxelization->updateDataPerFrame(cmd);
	lightInject->updateDataPerFrame(cmd);
	octree->updateDataPerFrame(cmd);
}

void FzbRenderer::SVOPathGuidingRenderer::pathGuiding(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 1, write.size(), write.data());

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.size = sizeof(shaderio::SVOPathGuidingPushConstant),
		.pValues = &pushConstant
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	const nvvk::SBTGenerator::Regions& regions = sbtGenerator.getSBTRegions();
	const VkExtent2D& size = Application::app->getViewportSize();
	vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
}