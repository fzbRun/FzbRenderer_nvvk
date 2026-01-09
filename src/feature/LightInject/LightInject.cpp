#include "./LightInject.h"
#include "./shaderio.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>

FzbRenderer::LightInject::LightInject(pugi::xml_node& featureNode) {}
void FzbRenderer::LightInject::init(LightInjectSetting setting) {
	this->setting = setting;
	sbtGenerator.init(Application::app->getDevice(), setting.ptContext->rtProperties);

	createGBuffer(false, false, 1);
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout(sizeof(shaderio::LightInjectPushConstant));
	createPipeline();
}
void FzbRenderer::LightInject::clean() {
	PathTracing::clean();
	VkDevice device = Application::app->getDevice();
	vkDestroyPipeline(device, rtPipeline, nullptr);
}
void FzbRenderer::LightInject::uiRender() {}
void FzbRenderer::LightInject::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(0), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
};
void FzbRenderer::LightInject::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;

	pushConstant.VGBStartPos_Size = setting.VGBStartPos_Size;
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void FzbRenderer::LightInject::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

	const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.layout = pipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = staticDescPack.getSetPtr()
	};
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipelineLayout, 1, write.size(), write.data());

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.size = sizeof(shaderio::LightInjectPushConstant),
		.pValues = &pushConstant
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	const nvvk::SBTGenerator::Regions& regions = sbtGenerator.getSBTRegions();
	const VkExtent2D& size = Application::app->getViewportSize();
	vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
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
	write.append(VGBWrite, setting.VGB);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbRenderer::LightInject::createPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI("LightInject creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), pipelineLayout, nullptr);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "lightInject.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

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
		//group.anyHitShader = eAnyHit;
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
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &pipelineLayout);
	NVVK_DBG_NAME(pipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::min(MAX_DEPTH, setting.ptContext->rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = pipelineLayout;
	rtPipelineInfo.flags = VK_PIPELINE_CREATE_RAY_TRACING_ALLOW_MOTION_BIT_NV;
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, & rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	FzbRenderer::createShaderBindingTable(rtPipelineInfo, rtPipeline, sbtGenerator, sbtBuffer);
}

void FzbRenderer::LightInject::compileAndCreateShaders() {
	createPipeline();
}
void FzbRenderer::LightInject::updateDataPerFrame(VkCommandBuffer cmd) {}
