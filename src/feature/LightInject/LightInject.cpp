#include "./LightInject.h"
#include "./shaderio.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

FzbRenderer::LightInject::LightInject(pugi::xml_node& featureNode) {
#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}
void FzbRenderer::LightInject::init(LightInjectSetting setting) {
	this->setting = setting;
	sbtGenerator.init(Application::app->getDevice(), setting.ptContext->rtProperties);

	createGBuffer(true, false, 2);
	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::LightInjectPushConstant));
	compileAndCreateShaders();

#ifndef NDEBUG
	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube(false, false);
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Cube", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif
}
void FzbRenderer::LightInject::clean() {
	PathTracing::clean();
	VkDevice device = Application::app->getDevice();
	vkDestroyPipeline(device, rtPipeline, nullptr);
	vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);
#endif
}
void FzbRenderer::LightInject::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("LightInject")) {
		if (PE::begin()) {
			if (PE::entry("LightInjectResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showLightInjectResult ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(LightInjectGBuffer::LightInjectResult),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showLightInjectResult = !showLightInjectResult;
				showCubeMap = false;
			}
		}
		PE::end();

		if (PE::begin()) {
			if (PE::entry("CubeMap", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showCubeMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(LightInjectGBuffer::CubeMap_LightInject),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showCubeMap = !showCubeMap;
				showLightInjectResult = false;
			}
		}
		PE::end();
	}
	ImGui::End();

	if (showLightInjectResult) Application::viewportImage = gBuffers.getDescriptorSet(LightInjectGBuffer::LightInjectResult);
	else if (showCubeMap) Application::viewportImage = gBuffers.getDescriptorSet(LightInjectGBuffer::CubeMap_LightInject);
#endif
}
void FzbRenderer::LightInject::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(LightInjectGBuffer::LightInjectResult), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
};
void FzbRenderer::LightInject::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;

	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void FzbRenderer::LightInject::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	{
		vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

		const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
			.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.layout = rtPipelineLayout,
			.firstSet = 0,
			.descriptorSetCount = 1,
			.pDescriptorSets = staticDescPack.getSetPtr()
		};
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 0, 1,
			staticDescPack.getSetPtr(), 0, nullptr);

		nvvk::WriteSetContainer write{};
		write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 1, write.size(), write.data());

		const VkPushConstantsInfo pushInfo{
			.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
			.layout = rtPipelineLayout,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.size = sizeof(shaderio::LightInjectPushConstant),
			.pValues = &pushConstant
		};
		vkCmdPushConstants2(cmd, &pushInfo);

		const nvvk::SBTGenerator::Regions& regions = sbtGenerator.getSBTRegions();
		const VkExtent2D& size = Application::app->getViewportSize();
		vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
	}
}
void FzbRenderer::LightInject::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_Cube(cmd);
#endif
}

void FzbRenderer::LightInject::createDescriptorSetLayout() {
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
			.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_LightInject::eVGB_LightInject,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("LightInject ray tracing static descriptor layout created\n");
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

	LOGI("LightInject ray tracing dynamic descriptor layout created\n");
}
void FzbRenderer::LightInject::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}
	//VkWriteDescriptorSet    OutImageWrite =
	//	staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	//write.append(OutImageWrite, gBuffers.getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_LightInject::eVGB_LightInject, 0, 0, 1);
	write.append(VGBWrite, setting.VGB, 0, setting.VGB.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbRenderer::LightInject::createPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI("LightInject creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), rtPipelineLayout, nullptr);

	addPathTracingSlangMacro();
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "lightInject.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

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

		stages[eClosestHit_NEE].pNext = &shaderCode;
		stages[eClosestHit_NEE].pName = "NEEClosestHitMain";
		stages[eClosestHit_NEE].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
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
		shader_groups.push_back(group);

		group.closestHitShader = eClosestHit_NEE;
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

	const VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::LightInjectPushConstant) };

	VkPipelineLayoutCreateInfo pipeline_layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipeline_layout_create_info.pushConstantRangeCount = 1;
	pipeline_layout_create_info.pPushConstantRanges = &push_constant;

	std::vector<VkDescriptorSetLayout> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout()} };	//二合一
	pipeline_layout_create_info.setLayoutCount = uint32_t(layouts.size());
	pipeline_layout_create_info.pSetLayouts = layouts.data();
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &rtPipelineLayout);
	NVVK_DBG_NAME(rtPipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::min(MAX_DEPTH, setting.ptContext->rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = rtPipelineLayout;
#ifdef PathTracingMotionBlur
	rtPipelineInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
#endif
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, & rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	FzbRenderer::createShaderBindingTable(rtPipelineInfo, rtPipeline, sbtGenerator, sbtBuffer);
}

void FzbRenderer::LightInject::compileAndCreateShaders() {
	createPipeline();

#ifndef NDEBUG
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant),
	};

	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = 1,
		.pSetLayouts = staticDescPack.getLayoutPtr(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();

	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_Cube";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_Cube);
	NVVK_DBG_NAME(vertexShader_Cube);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_Cube";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_Cube);
	NVVK_DBG_NAME(fragmentShader_Cube);
#endif
}
void FzbRenderer::LightInject::updateDataPerFrame(VkCommandBuffer cmd) {
	scene.sceneInfo = Application::sceneResource.sceneInfo;
	nvvk::cmdBufferMemoryBarrier(cmd, { scene.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
								   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	vkCmdUpdateBuffer(cmd, scene.bSceneInfo.buffer, 0, sizeof(shaderio::SceneInfo), &scene.sceneInfo);
	nvvk::cmdBufferMemoryBarrier(cmd, { scene.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT });
}

#ifndef NDEBUG
void FzbRenderer::LightInject::debug_Cube(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(LightInjectGBuffer::CubeMap_LightInject), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(LightInjectGBuffer::CubeMap_LightInject);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Cube, .fragment = fragmentShader_Cube });

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant),
		.pValues = &pushConstant,
	};
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)scene.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pow(pushConstant.VGBStartPos_Size.w, 3), 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(LightInjectGBuffer::CubeMap_LightInject), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
#endif
