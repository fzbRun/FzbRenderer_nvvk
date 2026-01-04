#include "PathTracingRenderer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include "common/Shader/nvvk/shaderio.h"
#include <nvgui/sky.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <common/Shader/Shader.h>

FzbRenderer::PathTracingRenderer::PathTracingRenderer(pugi::xml_node& rendererNode) {
	Application::vkContextInitInfo.deviceExtensions.push_back( { VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature });
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME });

	rtPosFetchFeature.rayTracingPositionFetch = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, &rtPosFetchFeature });

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth")) 
		pushValues.maxDepth = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node useNEENode = rendererNode.child("useNEE"))
		pushValues.NEEShaderIndex = std::string(useNEENode.attribute("value").value()) == "true";
}
//-----------------------------------------创造光追管线----------------------------------------------------------
/*
void FzbRenderer::PathTracingRenderer::createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo) {
	//STB，顾名思义，就是shader大的绑定表；
	//其将pipeline各个阶段的shader组进行绑定
	//每个实例只能用每个阶段的shader组中的一个条目
	//每个组和条目都需要对齐
SCOPED_TIMER(__FUNCTION__);
Application::allocator.destroyBuffer(sbtBuffer);

//STB中shader必须对齐
VkDevice device = Application::app->getDevice();
uint32_t handleSize = rtProperties.shaderGroupHandleSize;		//shader句柄的大小
uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;		//每个shader的对齐
uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;		//每个shader组的对齐
uint32_t groupCount = rtPipelineInfo.groupCount;

size_t dataSize = handleSize * groupCount;	//目前每组只有一个shader（条目），所以乘以组数
shaderHandles.resize(dataSize);
NVVK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0, groupCount, dataSize, shaderHandles.data()));

auto alignUp = [](uint32_t size, uint32_t alignment) {return (size + alignment - 1) & ~(alignment - 1); };
uint32_t raygenSize = alignUp(handleSize, handleAlignment);		//目前没有附加数据，所以都一样
uint32_t missSize = alignUp(handleSize, handleAlignment);
uint32_t hitSize = alignUp(handleSize, handleAlignment);
uint32_t callableSize = 0;

uint32_t raygenOffset = 0;
uint32_t missOffset = alignUp(raygenSize, baseAlignment);
uint32_t hitOffset = alignUp(missOffset + missSize, baseAlignment);
uint32_t callableOffset = alignUp(hitOffset + hitSize, baseAlignment);

size_t bufferSize = callableOffset + callableSize;
NVVK_CHECK(Application::allocator.createBuffer(
	sbtBuffer, bufferSize,
	VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
));
NVVK_DBG_NAME(sbtBuffer.buffer);
uint8_t* pData = static_cast<uint8_t*>(sbtBuffer.mapping);

memcpy(pData + raygenOffset, shaderHandles.data() + 0 * handleSize, handleSize);
raygenRegion.deviceAddress = sbtBuffer.address + raygenOffset;
raygenRegion.stride = raygenSize;	//raygen中每个shader的条目
raygenRegion.size = raygenSize;		//raygen组总大小

memcpy(pData + missOffset, shaderHandles.data() + 1 * handleSize, handleSize);	//shaderHandles是紧密的
missRegion.deviceAddress = sbtBuffer.address + missOffset;
missRegion.stride = missSize;
missRegion.size = missSize;

memcpy(pData + hitOffset, shaderHandles.data() + 2 * handleSize, handleSize);
hitRegion.deviceAddress = sbtBuffer.address + hitOffset;
hitRegion.stride = hitSize;
hitRegion.size = hitSize;

callableRegion.deviceAddress = 0;
callableRegion.stride = 0;
callableRegion.size = 0;

LOGI(" Shader binding table created and populated\n");
}
*/
//-----------------------------------------创造光追管线----------------------------------------------------------
/*
整个rtPipeline的逻辑应该是：
1. 确定rtPipelien有哪些阶段stages：raygen->miss/(anyhit->hit)
2. 为每个阶段创建group，表示这个阶段可能有多个不同的shader，如hit有两个，一个处理导体，一个处理电介质
3. 创建SBT，将每个阶段的shader（条目）绑定到SBT中（有严格的对齐要求）
4. 再创建tlas时可以指定instanceShaderBindingTableRecordOffset，表面其使用每个组中的那个shader（条目），如电介质可以使用专门处理电介质的hit
*/
void FzbRenderer::PathTracingRenderer::createRayTracingDescriptorLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
			.binding = shaderio::BindingPoints::eTlas,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	bindings.addBinding({
			.binding = shaderio::BindingPoints::eOutImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});

	rtDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);

	LOGI("Ray tracing descriptor layout created\n");
}
void FzbRenderer::PathTracingRenderer::createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo) {
	/*
	STB，顾名思义，就是shader大的绑定表；
	其将pipeline各个阶段的shader组进行绑定
	每个实例只能用每个阶段的shader组中的一个条目
	每个组和条目都需要对齐
	*/
	SCOPED_TIMER(__FUNCTION__);
	Application::allocator.destroyBuffer(sbtBuffer);

	size_t bufferSize = sbtGenerator.calculateSBTBufferSize(rtPipeline, rtPipelineInfo);
	NVVK_CHECK(Application::allocator.createBuffer(sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		sbtGenerator.getBufferAlignment()));
	NVVK_DBG_NAME(sbtBuffer.buffer);
	NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));

	LOGI(" Shader binding table created and populated\n");
}
void FzbRenderer::PathTracingRenderer::createRayTracingPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI(" Creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), rtPipelineLayout, nullptr);

	//这里的枚举是指为了自己识别，真正有用的是stage，如VK_SHADER_STAGE_RAYGEN_BIT_KHR这些
	//nvvk的addIndices函数会将所有相同stage的shader作为一个group的条目
	enum StageIndices {
		eRaygen,
		eMiss,
		eClosestHit,
		//eAnyHit,

		eClosestHit_NEE,

		eCallable_DiffuseMaterial,
		eCallable_ConductorMaterial,
		eCallable_DielectricMaterial,
		eCallable_RoughConductorMaterial,
		eCallable_RoughDielectricMaterial,
		eShaderGroupCount
	};
	std::vector<VkPipelineShaderStageCreateInfo> stages(eShaderGroupCount);
	for(auto& s : stages)
		s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

	//VkShaderModuleCreateInfo shaderCode = compileSlangShader("pathTracingShaders.slang", {});
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "pathTracingShaders.slang";
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

	std::vector<VkDescriptorSetLayout> layouts = { { descPack.getLayout(), rtDescPack.getLayout()} };	//二合一
	pipeline_layout_create_info.setLayoutCount = uint32_t(layouts.size());
	pipeline_layout_create_info.pSetLayouts = layouts.data();
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &rtPipelineLayout);
	NVVK_DBG_NAME(rtPipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::min(MAX_DEPTH, rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = rtPipelineLayout;
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, & rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	//hitShader条目在group中会同时存储shaderhandle和data，这个data包含三角形索引范围、AABB和自定义数据
	//如可以通过PrimitiveIndex函数获得三角形索引
	//sbtGenerator.addData(nvvk::SBTGenerator::eHit, 1, hitShaderRecord[0]);		//data会与group中的条目（shader描述符）放在一起，需要一起对齐
	//sbtGenerator.addData(nvvk::SBTGenerator::eHit, 2, hitShaderRecord[1]);
	createShaderBindingTable(rtPipelineInfo);
}
//-----------------------------------------光追函数----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::rayTraceScene(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

	const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.layout = rtPipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = descPack.getSetPtr()
	};
	//vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 0, 1,
		descPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(rtDescPack.makeWrite(shaderio::BindingPoints::eTlas), asManager.asBuilder.tlas);
	write.append(rtDescPack.makeWrite(shaderio::BindingPoints::eOutImage), gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 1, write.size(), write.data());

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = rtPipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.size = sizeof(shaderio::PathTracingPushConstant),
		.pValues = &pushValues
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	const nvvk::SBTGenerator::Regions& regions = sbtGenerator.getSBTRegions();
	const VkExtent2D& size = Application::app->getViewportSize();
	vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);
}

void FzbRenderer::PathTracingRenderer::resetFrame() {
	Application::frameIndex = 0;
}
void FzbRenderer::PathTracingRenderer::updateDataPerFrame(VkCommandBuffer cmd) {}
//-----------------------------------------渲染器行为----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::init() {
	getRayTracingPropertiesAndFeature();

	std::string shaderioPath = (std::filesystem::path(__FILE__).parent_path() / "shaderio.h").string();
	Application::slangCompiler.addOption({ .name = slang::CompilerOptionName::Include,
		.value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = shaderioPath.c_str()}
		});

	Renderer::createGBuffer(false);
	Renderer::createDescriptorSetLayout();
	Renderer::createPipelineLayout(sizeof(shaderio::PathTracingPushConstant));
	Renderer::addTextureArrayDescriptor();

	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), rtProperties);

	createRayTracingDescriptorLayout();
	createRayTracingPipeline();

	Renderer::init();
}
void FzbRenderer::PathTracingRenderer::clean() {
	Renderer::clean();
	VkDevice device = Application::app->getDevice();

	asManager.clean();
	sbtGenerator.deinit();

	vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
	vkDestroyPipeline(device, rtPipeline, nullptr);
	rtDescPack.deinit();
	Application::allocator.destroyBuffer(sbtBuffer);
};
void FzbRenderer::PathTracingRenderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);

	if (ImGui::Begin("PathTracingSettings"))
	{
		ImGui::SeparatorText("Jitter");
		UIModified |= ImGui::SliderInt("Max Acc Frames", &maxFrames, 1, MAX_FRAME);
		ImGui::TextDisabled("Current PathTracing Frame: %d", pushValues.frameIndex);
		ImGui::TextDisabled("Current Renderer Frame: %d", Application::frameIndex);

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

	if(UIModified) resetFrame();
};
void FzbRenderer::PathTracingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));
};
void FzbRenderer::PathTracingRenderer::preRender() {
	std::shared_ptr<nvutils::CameraManipulator> cameraManip = Application::sceneResource.cameraManip;

	static glm::mat4 refCamMatrix;
	static float refFov{ cameraManip->getFov() };

	const auto& m = cameraManip->getViewMatrix();
	const auto& fov = cameraManip->getFov();

	if (refCamMatrix != m || refFov != fov) {	//如果相机参数变化，则从新累计帧
		resetFrame();
		refCamMatrix = m;
		refFov = fov;
	}

	Scene& scene = Application::sceneResource;
	if (scene.dynamicInstances.size() > 0 || scene.hasDynamicLight) maxFrames = 1;
	pushValues.frameIndex = std::min(Application::frameIndex, maxFrames - 1);
	pushValues.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	asManager.updateTopLevelAS_nvvk();
}
void FzbRenderer::PathTracingRenderer::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//maxFrames等于1表示只要一帧，我们就每帧都替换
	if (pushValues.frameIndex == maxFrames - 1 && maxFrames > 1) return;

	updateDataPerFrame(cmd);
	rayTraceScene(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	Renderer::postProcess(cmd);
};

void FzbRenderer::PathTracingRenderer::compileAndCreateShaders() {
	createRayTracingPipeline();
};

void FzbRenderer::PathTracingRenderer::getRayTracingPropertiesAndFeature() {
	//查询是否支持rtPosFetchFeature
	VkPhysicalDeviceFeatures2 deviceFeatures2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
	deviceFeatures2.pNext = &rtPosFetchFeature;
	rtPosFetchFeature.pNext = nullptr;
	rtPosFetchFeature.rayTracingPositionFetch = VK_FALSE;
	vkGetPhysicalDeviceFeatures2(Application::app->getPhysicalDevice(), &deviceFeatures2);
	//查询设备参数
	VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	prop2.pNext = &rtProperties;
	vkGetPhysicalDeviceProperties2(Application::app->getPhysicalDevice(), &prop2);
}