#include "SVOWeight.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>
#include "feature/PathTracing/shaderio.h"
#include <random>

using namespace FzbRenderer;

SVOWeight::SVOWeight(pugi::xml_node& featureNode) {}
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
	Application::allocator.destroyBuffer(weightBuffer);
	Application::allocator.destroyBuffer(nearbyNodeInfosBuffer);

	vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyNodes2, nullptr);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, computeShader_getSampleNodeInfo, nullptr);
	vkDestroyShaderEXT(device, vertexShader_visualization, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_visualization, nullptr);
	vkDestroyShaderEXT(device, vertexShader_nearby, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_nearby, nullptr);
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
				ImVec4 selectedColor = showWeightMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
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
				showWeightMap = !showWeightMap;
				showNearbyMap = false;
			}
		}
		PE::end();

		PE::begin();
		UIModified |= PE::DragFloat3("samplePos", (float*)&samplePoint);
		UIModified |= PE::DragFloat3("outgoing", (float*)&outgoing);
		PE::end();

		if (PE::begin()) {
			if (PE::entry("SVO NearbyNode Debug Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showNearbyMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(1),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showNearbyMap = !showNearbyMap;
				showWeightMap = false;
			}
		}
		PE::end();

		PE::begin();
		UIModified |= PE::DragInt("sampleNodeLabel", &pushConstant.sampleNodeLabel);
		PE::end();
	}
	ImGui::End();

	if (showWeightMap) Application::viewportImage = gBuffers.getDescriptorSet(0);
	if (showNearbyMap) Application::viewportImage = gBuffers.getDescriptorSet(1);
#endif
};
void SVOWeight::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	gBuffers.update(cmd, size);
#endif
};
void SVOWeight::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;

	float angle = FzbRenderer::rand(Application::frameIndex) * glm::four_over_pi<float>();
	randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));
	pushConstant.randomRotateMatrix = randomRotateMatrix;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.samplePos = samplePoint;
	pushConstant.outgoing = outgoing;
#endif
}
void SVOWeight::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "SVOWeight_render");

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

	initWeights(cmd);
	//getNearbyNodeInfos(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	getWeights(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	getProbability(cmd);
}
void SVOWeight::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_visualization(cmd);
	//debug_nearby(cmd);
#endif
};

void SVOWeight::createWeightArray() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	allocator->createBuffer(GlobalInfoBuffer, sizeof(shaderio::SVOWeightGlobalInfo),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(GlobalInfoBuffer.buffer);

	//由于我们事先不知道具体SVO聚类后有几层，因此我们按最大层来创建buffer，即octree的层数
	uint32_t OctreeDepth = setting.octree->setting.OctreeLayerCount;

	uint32_t weightCount = OUTGOING_COUNT * IndivisibleNodeCount_G * OCTREE_NODECOUNT_E;	//has layer 0
	uint32_t bufferSize = weightCount * sizeof(float);
	allocator->createBuffer(weightBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G * (IndivisibleNodeCount_G / GETNEARBYNODES_CS_THREADGROUP_SIZE) * sizeof(shaderio::IndivisibleNodeNearbyNodeInfo);
	allocator->createBuffer(nearbyNodeInfosBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(nearbyNodeInfosBuffer.buffer);
}
void SVOWeight::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		#ifdef USE_SVO
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eSVO_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		#else
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eOctreeArray_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.octree->OctreeArray_G.size(),
		#endif
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eNodeData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eTreeGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eSVOWeights,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeNearbyNodeInfos,
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

	#ifdef USE_SVO
	VkWriteDescriptorSet SVOArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eSVO_G, 0, 0, 1);
	write.append(SVOArrayWrite, setting.svo->SVO_G, 0, setting.svo->SVO_G.bufferSize);

	VkWriteDescriptorSet SVOLayerInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eTreeGlobalInfo, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.svo->SVOGlobalInfo, 0, setting.svo->SVOGlobalInfo.bufferSize);

	VkWriteDescriptorSet IndivisibleNodeInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_G, 0, 0, 1);
	write.append(IndivisibleNodeInfosWrite, setting.svo->indivisibleNodeInfosBuffer_G, 0, setting.svo->indivisibleNodeInfosBuffer_G.bufferSize);

	IndivisibleNodeInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_E, 0, 0, 1);
	write.append(IndivisibleNodeInfosWrite, setting.svo->indivisibleNodeInfosBuffer_E, 0, setting.svo->indivisibleNodeInfosBuffer_E.bufferSize);
	#else
	VkWriteDescriptorSet OctreeArray_GWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eOctreeArray_G, 0, 0, setting.octree->OctreeArray_G.size());
	nvvk::Buffer* OctreeArrayPtr = setting.octree->OctreeArray_G.data();
	write.append(OctreeArray_GWrite, OctreeArrayPtr);

	VkWriteDescriptorSet SVOLayerInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eTreeGlobalInfo, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.octree->GlobalInfoBuffer, 0, setting.octree->GlobalInfoBuffer.bufferSize);

	VkWriteDescriptorSet IndivisibleNodeInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_G, 0, 0, 1);
	write.append(IndivisibleNodeInfosWrite, setting.octree->indivisibleNodeInfosBuffer_G, 0, setting.octree->indivisibleNodeInfosBuffer_G.bufferSize);

	IndivisibleNodeInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_E, 0, 0, 1);
	write.append(IndivisibleNodeInfosWrite, setting.octree->indivisibleNodeInfosBuffer_E, 0, setting.octree->indivisibleNodeInfosBuffer_E.bufferSize);
	#endif

	VkWriteDescriptorSet Octree_EWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eNodeData_E, 0, 0, 1);
	write.append(Octree_EWrite, setting.octree->NodeData_E, 0, setting.octree->NodeData_E.bufferSize);

	VkWriteDescriptorSet globalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eGlobalInfo, 0, 0, 1);
	write.append(globalInfoWrite, GlobalInfoBuffer, 0, GlobalInfoBuffer.bufferSize);


	VkWriteDescriptorSet weightsWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eSVOWeights, 0, 0, 1);
	write.append(weightsWrite, weightBuffer, 0, weightBuffer.bufferSize);

	VkWriteDescriptorSet nearbyNodeInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_SVOWeight::eIndivisibleNodeNearbyNodeInfos, 0, 0, 1);
	write.append(nearbyNodeInfosWrite, nearbyNodeInfosBuffer, 0, nearbyNodeInfosBuffer.bufferSize);

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
	#ifdef USE_SVO
	std::filesystem::path shaderSource = shaderPath / "SVOWeight.slang";
	#else
	std::filesystem::path shaderSource = shaderPath / "OctreeWeight.slang";
	#endif
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
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_getNearbyNodes, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyNodes);
		NVVK_DBG_NAME(computeShader_getNearbyNodes);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_getNearbyNodes2, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyNodes2);
		NVVK_DBG_NAME(computeShader_getNearbyNodes2);
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
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader_nearby, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_nearby, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_nearby";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_nearby);
	NVVK_DBG_NAME(vertexShader_nearby);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_nearby";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_nearby);
	NVVK_DBG_NAME(fragmentShader_nearby);
#endif
}
void SVOWeight::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}

void SVOWeight::initWeights(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initWeights);

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t weightCount = OUTGOING_COUNT * IndivisibleNodeCount_G * OCTREE_NODECOUNT_E;
	VkExtent2D groupSize = nvvk::getGroupCounts({ weightCount, 1 }, VkExtent2D{ INITWEIGHT_CS_THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void SVOWeight::getNearbyNodeInfos(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyNodes);
	vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, offsetof(shaderio::SVOWeightGlobalInfo, cmd2));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyNodes2);
	vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, offsetof(shaderio::SVOWeightGlobalInfo, cmd2));
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

	for (int layerIndex = OCTREE_CLUSTER_LAYER; layerIndex >= 0; --layerIndex) {
		pushConstant.currentLayer_E = layerIndex;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	}
}

#ifndef NDEBUG
void SVOWeight::debugPrepare() {
	Feature::createGBuffer(true, false, 2);

	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube();
	FzbRenderer::MeshSet meshSet = FzbRenderer::MeshSet("Cube", primitive);
	scene.addMeshSet(meshSet);

	scene.createSceneInfoBuffer();

	pushConstant.sampleNodeLabel = 53;	// 39;354; 81;9
}
void SVOWeight::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	gBuffers.update(cmd, size);
	depthImageView = gBuffers_other.getDepthImageView();
}

void SVOWeight::debug_visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getSampleNodeInfo);

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
void SVOWeight::debug_nearby(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(1), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(1);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	depthAttachment.imageView = depthImageView;	// gBuffers.getDepthImageView();
	//depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

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
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_nearby, .fragment = fragmentShader_nearby });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	#ifdef USE_SVO
	vkCmdDrawIndexed(cmd, triMesh.indices.count, SVOSize_G, 0, 0, 0);
	#else
	vkCmdDrawIndexed(cmd, triMesh.indices.count, IndivisibleNodeCount_G, 0, 0, 0);
	#endif

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
#endif
