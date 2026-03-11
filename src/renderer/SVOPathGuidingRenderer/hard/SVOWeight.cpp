#include "SVOWeight.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>
#include "feature/PathTracing/shaderio.h"

using namespace FzbRenderer;

SVOWeight::SVOWeight(pugi::xml_node& featureNode) {
	rayqueryFeature.rayQuery = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature });
}
void SVOWeight::init(SVOWeightSetting setting) {
	this->setting = setting;

	createWeightArray();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

#ifndef NDEBUG
	debugPrepare();
#endif
}
void SVOWeight::clean() {
	Feature::clean();
	VkDevice device = Application::app->getDevice();

	Application::allocator.destroyBuffer(GlobalInfoBuffer);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_G);
	Application::allocator.destroyBuffer(weightSampleCountsBuffer);
	Application::allocator.destroyBuffer(weightBuffer);
	Application::allocator.destroyBuffer(weightSumsBuffer);

	vkDestroyShaderEXT(device, computeShader_getIndivisibleNode_G, nullptr);
	vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, computeShader_getSampleNodeInfo, nullptr);
	vkDestroyShaderEXT(device, vertexShader_visualization, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_visualization, nullptr);
#endif
}
void SVOWeight::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("SVO Weight")) {
		if (PE::begin()) {
			if (PE::entry("SVO Weight Debug Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = show ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(0),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				show = !show;
			}
		}
		PE::end();

		PE::begin();
		UIModified |= PE::DragFloat3("samplePos", (float*)&samplePoint);
		UIModified |= PE::DragFloat3("outgoing", (float*)&outgoing);
		PE::end();
	}
	ImGui::End();

	if(show) Application::viewportImage = gBuffers.getDescriptorSet(0);
#endif
};
void SVOWeight::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	gBuffers.update(cmd, size);
#endif
};
void SVOWeight::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
#endif
}

void SVOWeight::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "SVO_render");

	updateDataPerFrame(cmd);

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo = {
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.layout = pipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = staticDescPack.getSetPtr(),
	};

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::SVOWeightPushConstant),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	getIndivisibleNode_E(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	initWeights(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	getWeights(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	getProbability(cmd);
}
void SVOWeight::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_visualization(cmd);
#endif
};

void SVOWeight::createWeightArray() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	allocator->createBuffer(GlobalInfoBuffer, sizeof(shaderio::SVOWeightGlobalInfo),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(GlobalInfoBuffer.buffer);

	//由于我们事先不知道具体SVO聚类后有几层，因此我们按最大层来创建buffer，即octree的层数
	uint32_t OctreeDepth = setting.svo->setting.octree->setting.OctreeDepth;
	uint32_t SVONodeMaxCount = 0;
	for (int i = 1; i < OctreeDepth; ++i)
		SVONodeMaxCount += setting.svo->SVOInitialSize[i];

	uint32_t bufferSize = SVONodeMaxCount * sizeof(shaderio::SVOIndivisibleNodeInfo);
	allocator->createBuffer(indivisibleNodeInfosBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_G.buffer);

	uint32_t weightCount = OUTGOING_COUNT * SVONodeMaxCount * SVONodeMaxCount;
	bufferSize = weightCount * sizeof(uint32_t);
	allocator->createBuffer(weightSampleCountsBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightSampleCountsBuffer.buffer);

	bufferSize = weightCount * sizeof(float);
	allocator->createBuffer(weightBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightBuffer.buffer);

	weightCount = OUTGOING_COUNT * SVONodeMaxCount * setting.svo->SVOInitialSize[OctreeDepth - 1];
	bufferSize = weightCount * sizeof(float);
	allocator->createBuffer(weightSumsBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightSumsBuffer.buffer);

	for (int depth = 0; depth < 8; ++depth) pushConstant.sizes[depth] = setting.svo->SVOInitialSize[depth];
}
void SVOWeight::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.svo->SVOArray_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.svo->SVOArray_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSampleCounts_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSums_SVOWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

#ifndef NDEBUG
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
				 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				 .descriptorCount = 10,
				 .stageFlags = VK_SHADER_STAGE_ALL });
#endif

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

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
void SVOWeight::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet SVOArrayWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight, 0, 0, setting.svo->SVOArray_G.size());
	nvvk::Buffer* SVOArrayPtr = setting.svo->SVOArray_G.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	SVOArrayWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight, 0, 0, setting.svo->SVOArray_E.size());
	SVOArrayPtr = setting.svo->SVOArray_E.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	VkWriteDescriptorSet SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.svo->SVOLayerInfos_G, 0, setting.svo->SVOLayerInfos_G.bufferSize);

	SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.svo->SVOLayerInfos_E, 0, setting.svo->SVOLayerInfos_E.bufferSize);

	VkWriteDescriptorSet globalInfoWrite = 
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight, 0, 0, 1);
	write.append(globalInfoWrite, GlobalInfoBuffer, 0, GlobalInfoBuffer.bufferSize);

	VkWriteDescriptorSet IndivisibleNodeInfos_GWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight, 0, 0, 1);
	write.append(IndivisibleNodeInfos_GWrite, indivisibleNodeInfosBuffer_G, 0, indivisibleNodeInfosBuffer_G.bufferSize);

	VkWriteDescriptorSet weightSampleCountWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSampleCounts_SVOWeight, 0, 0, 1);
	write.append(weightSampleCountWrite, weightSampleCountsBuffer, 0, weightSampleCountsBuffer.bufferSize);

	VkWriteDescriptorSet weightsWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight, 0, 0, 1);
	write.append(weightsWrite, weightBuffer, 0, weightBuffer.bufferSize);

	VkWriteDescriptorSet weightSumsWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSums_SVOWeight, 0, 0, 1);
	write.append(weightSumsWrite, weightSumsBuffer, 0, weightSumsBuffer.bufferSize);

#ifndef NDEBUG
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void SVOWeight::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::SVOWeightPushConstant)
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
void SVOWeight::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVOWeightShaders2.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::SVOWeightPushConstant),
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
		vkDestroyShaderEXT(device, computeShader_getIndivisibleNode_G, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getIndivisibleNode_G";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getIndivisibleNode_G);
		NVVK_DBG_NAME(computeShader_getIndivisibleNode_G);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initWeights";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initWeights);
		NVVK_DBG_NAME(computeShader_initWeights);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_getWeights, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getWeights";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getWeights);
		NVVK_DBG_NAME(computeShader_getWeights);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getProbability";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getProbability);
		NVVK_DBG_NAME(computeShader_getProbability);
	}

#ifndef NDEBUG
	vkDestroyShaderEXT(device, computeShader_getSampleNodeInfo, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_getSampleNodeInfo";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getSampleNodeInfo);
	NVVK_DBG_NAME(computeShader_getSampleNodeInfo);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader_visualization, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_visualization, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_visualization);
	NVVK_DBG_NAME(vertexShader_visualization);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_visualization);
	NVVK_DBG_NAME(fragmentShader_visualization);
#endif
}
void SVOWeight::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}

void SVOWeight::getIndivisibleNode_E(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getIndivisibleNode_G);

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t OctreeDepth = setting.svo->setting.octree->setting.OctreeDepth;
	uint32_t SVONodeMaxCount = 0;
	for (int i = 1; i < OctreeDepth; ++i)
		SVONodeMaxCount += setting.svo->SVOInitialSize[i];
	uint32_t totalVoxelCount = SVONodeMaxCount;
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ THREADGROUP_SIZE2, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void SVOWeight::initWeights(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initWeights);

	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, 0);
}
void SVOWeight::getWeights(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getWeights);

	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, 0);
}
void SVOWeight::getProbability(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getProbability);

	for(int countdown = 0; countdown < setting.svo->setting.octree->setting.OctreeDepth; ++countdown) {
		pushConstant.countdown = countdown;
		vkCmdPushConstants2(cmd, &pushInfo);
	
		vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	}
}

#ifndef NDEBUG
void SVOWeight::debugPrepare() {
	Feature::createGBuffer(true, false, 1);
}

void SVOWeight::debug_visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getSampleNodeInfo);

	pushConstant.samplePos = samplePoint;
	pushConstant.outgoing = outgoing;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	nvvk::cmdImageMemoryBarrier(cmd,
		{ gBuffers.getColorImage(0),
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(0);
	colorAttachment.clearValue = { .color = {0.0f, 0.0f, 0.0f, 0.0f} };
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
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
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
	vkCmdSetDepthTestEnable(cmd, VK_TRUE);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_visualization, .fragment = fragmentShader_visualization });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0),
									  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
#endif