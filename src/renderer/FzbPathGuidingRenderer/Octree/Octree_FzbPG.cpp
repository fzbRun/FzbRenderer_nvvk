#include "./Octree_FzbPG.h"
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include "../RasterVoxelization/RasterVoxelization_FzbPG.h"
#include "feature/PathTracing/shaderio.h"

using namespace FzbRenderer;

Octree_FzbPG::Octree_FzbPG(pugi::xml_node& featureNode) {
#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}
void Octree_FzbPG::init(OctreeCreateInfo_FzbPG createInfo) {
	this->setting = createInfo;

	createOctreeArray();

#ifndef NDEBUG
	debug_Prepare();
#endif

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();
}
void Octree_FzbPG::clean() {
	Feature::clean();
	for (int i = 0; i < octreeClusterDataBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeClusterDataBuffer_G[i]);
	for (int i = 0; i < octreeDataBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeDataBuffer_G[i]);
	for (int i = 0; i < octreeClusterDataBuffer_E.size(); ++i) Application::allocator.destroyBuffer(octreeClusterDataBuffer_E[i]);
	Application::allocator.destroyBuffer(clusterLayerDataBuffer_E);

	Application::allocator.destroyBuffer(globalInfoBuffer);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_G);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_E);

	Application::allocator.destroyBuffer(blockInfoBuffer_G);
	Application::allocator.destroyBuffer(blockInfoBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_G);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockCountBuffer);

	Application::allocator.destroyBuffer(divisibleNodeInfoBuffer_G);
	Application::allocator.destroyBuffer(threadGroupInfoBuffer);

	Application::allocator.destroyBuffer(octreeNodePairWeightBuffer);

	Application::allocator.destroyBuffer(nearbyNodeTempInfoBuffer);
	Application::allocator.destroyBuffer(nearbyNodeInfoBuffer);

	Application::allocator.destroyBuffer(octreeNodePairDataBuffer);
	Application::allocator.destroyBuffer(partialHitNodePairCountBuffer);
	Application::allocator.destroyBuffer(partialHitNodePairTempDataBuffer);
	Application::allocator.destroyBuffer(hitTestNodePairCountBuffer);
	Application::allocator.destroyBuffer(hitTestNodePairInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);

	vkDestroyShaderEXT(device, computeShader_getOctreeLabel1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel2, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel3, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel4, nullptr);

	vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_octreeNodeHitTest, nullptr);
	vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);

	vkDestroyShaderEXT(device, computeShader_getNearbyNodes1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyNodes2, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_OctreeLayer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_OctreeLayer, nullptr);

	vkDestroyShaderEXT(device, vertexShader_OctreeIndivisibleNodes, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_OctreeIndivisibleNodes, nullptr);

	vkDestroyShaderEXT(device, vertexShader_OctreeNodePairHitTestResult, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_OctreeNodePairHitTestResult, nullptr);

	vkDestroyShaderEXT(device, vertexShader_NearbyNodeInfoResult, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_NearbyNodeInfoResult, nullptr);
#endif
}
void Octree_FzbPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	uint32_t octreeShowLayerCount = showOctreeLayerMapCount / 2;
	std::vector<std::string> wireframeMapNames(octreeShowLayerCount);
	for (int i = 0; i < octreeShowLayerCount; ++i) wireframeMapNames[i] = "octreeLayer" + std::to_string(i + OCTREE_CLUSTER_LAYER_FZBPG);

	std::vector<const char*> wireframeMapNames_pointers;
	for (const auto& wireframeMapName : wireframeMapNames)
		wireframeMapNames_pointers.push_back(wireframeMapName.c_str());

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("Octree")) {
		ImGui::Combo("GeometryOctree", &selectedOctreeLayerMapIndex_G, wireframeMapNames_pointers.data(), static_cast<int>(wireframeMapNames_pointers.size()));
		if (PE::begin()) {
			if (PE::entry("GeometryOctreeClusterResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showOctreeLayerMap_G ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedOctreeLayerMapIndex_G),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showOctreeLayerMap_G = !showOctreeLayerMap_G;
				showOctreeLayerMap_E = false;
				showOctreeIndivisibleNodeMap_G = false;
				showOctreeNodePairHitTestResultMap = false;
				showNearbyNodeInfoResultMap = false;
				showOctreeNodePairVisibleAabbMap = false;
			}
		}
		PE::end();

		ImGui::Combo("IrradianceOctree", &selectedOctreeLayerMapIndex_E, wireframeMapNames_pointers.data(), static_cast<int>(wireframeMapNames_pointers.size()));
		if (PE::begin()) {
			if (PE::entry("IrradianceOctreeClusterResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showOctreeLayerMap_E ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedOctreeLayerMapIndex_E + octreeShowLayerCount),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showOctreeLayerMap_E = !showOctreeLayerMap_E;
				showOctreeLayerMap_G = false;
				showOctreeIndivisibleNodeMap_G = false;
				showOctreeNodePairHitTestResultMap = false;
				showNearbyNodeInfoResultMap = false;
				showOctreeNodePairVisibleAabbMap = false;
			}
		}
		PE::end();

		if (PE::begin()) {
			if (PE::entry("GeometryOctree IndivisibleNode Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showOctreeIndivisibleNodeMap_G ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(showOctreeIndivisibleNodeMapIndex),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showOctreeIndivisibleNodeMap_G = !showOctreeIndivisibleNodeMap_G;
				showOctreeLayerMap_G = false;
				showOctreeLayerMap_E = false;
				showOctreeNodePairHitTestResultMap = false;
				showNearbyNodeInfoResultMap = false;
				showOctreeNodePairVisibleAabbMap = false;
			}
		}
		PE::end();

		if (PE::begin()) {
			if (PE::entry("OctreeNodePair Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showOctreeNodePairHitTestResultMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(showOctreeNodePairHitTestResultMapIndex),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showOctreeNodePairHitTestResultMap = !showOctreeNodePairHitTestResultMap;
				showOctreeLayerMap_G = false;
				showOctreeLayerMap_E = false;
				showOctreeIndivisibleNodeMap_G = false;
				showNearbyNodeInfoResultMap = false;
				showOctreeNodePairVisibleAabbMap = false;
			}
		}
		PE::end();

		if (PE::begin()) {
			if (PE::entry("Indivisible Nearby Node Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showNearbyNodeInfoResultMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(showNearbyNodeInfoResultMapIndex),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showNearbyNodeInfoResultMap = !showNearbyNodeInfoResultMap;
				showOctreeLayerMap_G = false;
				showOctreeLayerMap_E = false;
				showOctreeIndivisibleNodeMap_G = false;
				showOctreeNodePairHitTestResultMap = false;
				showOctreeNodePairVisibleAabbMap = false;
			}
		}
		PE::end();

		/*
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
		if (PE::begin()) {
			if (PE::entry("Node Pair Visible AABB Map", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showOctreeNodePairVisibleAabbMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(showOctreeNodePairVisibleAabbMapIndex),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showOctreeNodePairVisibleAabbMap = !showOctreeNodePairVisibleAabbMap;
				showOctreeLayerMap_G = false;
				showOctreeLayerMap_E = false;
				showOctreeIndivisibleNodeMap_G = false;
				showOctreeNodePairHitTestResultMap = false;
				showNearbyNodeInfoResultMap = false;
			}
		}
		PE::end();
#endif
		*/

		PE::begin();
		UIModified |= PE::DragInt("sampleNodeLabel_G", &pushConstant.sampleNodeLabel_G);
		UIModified |= PE::DragInt("sampleNodeLabel_E", &pushConstant.sampleNodeLabel_E);
		PE::end();
	}
	ImGui::End();

	if (showOctreeLayerMap_G) Application::viewportImage = gBuffers.getDescriptorSet(selectedOctreeLayerMapIndex_G);
	if (showOctreeLayerMap_E) Application::viewportImage = gBuffers.getDescriptorSet(selectedOctreeLayerMapIndex_E + octreeShowLayerCount);
	if (showOctreeIndivisibleNodeMap_G) Application::viewportImage = gBuffers.getDescriptorSet(showOctreeIndivisibleNodeMapIndex);
	if (showOctreeNodePairHitTestResultMap) Application::viewportImage = gBuffers.getDescriptorSet(showOctreeNodePairHitTestResultMapIndex);
	if (showNearbyNodeInfoResultMap) Application::viewportImage = gBuffers.getDescriptorSet(showNearbyNodeInfoResultMapIndex);
	if (showOctreeNodePairVisibleAabbMap) Application::viewportImage = gBuffers.getDescriptorSet(showOctreeNodePairVisibleAabbMapIndex);
#endif
}
void Octree_FzbPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	//gBuffers.update(cmd, size);
};
void Octree_FzbPG::preRender() {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.voxelVolume = setting.VGBVoxelSize.x * setting.VGBVoxelSize.y * setting.VGBVoxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);
	pushConstant.frameIndex = Application::frameIndex;

	float angle = FzbRenderer::rand(Application::frameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));
}
void Octree_FzbPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_FzbPG),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	vkCmdPushConstants2(cmd, &pushInfo);

	initOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	createOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getOctreeLabel(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getOctreeNodePairData(cmd);
	getNearbyNodeInfo(cmd);
}
void Octree_FzbPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_OctreeLayer_Visualization(cmd);
	debug_OctreeIndivisibleNodes_Visualization(cmd);
	debug_OctreeNodePairHitTestResult_Visualization(cmd);
	debug_NearbyNodeInfoResult_Visualization(cmd);
#endif
};

void Octree_FzbPG::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	octreeMaxLayer = std::countr_zero(VGBSize);	//start from 0
	if (octreeMaxLayer > MAX_OCTREE_LAYER_FZBPG) {
		throw std::runtime_error("八叉树最大深度为5，即32x32x32！");
	}

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	octreeDataBuffer_G.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_G_FzbPG);
		allocator->createBuffer(octreeDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeDataBuffer_G[layerIndex].buffer);

		layerNodeCount *= 8;
	}

	uint32_t octreeClusterLayerCount = octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG + 1;
	octreeClusterDataBuffer_G.resize(octreeClusterLayerCount);
	octreeClusterDataBuffer_E.resize(octreeClusterLayerCount);
	layerNodeCount = 6 * (1 << (3 * OCTREE_CLUSTER_LAYER_FZBPG));
	for (int layerIndex = 0; layerIndex < octreeClusterLayerCount; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeClusterData_G_FzbPG);
		allocator->createBuffer(octreeClusterDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeClusterDataBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeClusterData_E_FzbPG);
		allocator->createBuffer(octreeClusterDataBuffer_E[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeClusterDataBuffer_E[layerIndex].buffer);

		layerNodeCount *= 8;
	}

	uint32_t bufferSize = CLUSTER_LAYER_NODECOUNT_E_FZBPG * sizeof(shaderio::OctreeNodeData_E_FzbPG);
	allocator->createBuffer(clusterLayerDataBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(clusterLayerDataBuffer_E.buffer);

	bufferSize = uint32_t(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(uint32_t);
	allocator->createBuffer(blockInfoBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(blockInfoBuffer_G.buffer);

	allocator->createBuffer(blockInfoBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(blockInfoBuffer_E.buffer);

	bufferSize = int(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(uint32_t);
	allocator->createBuffer(hasDataBlockIndexBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasDataBlockIndexBuffer_G.buffer);

	allocator->createBuffer(hasDataBlockIndexBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasDataBlockIndexBuffer_E.buffer);

	bufferSize = sizeof(shaderio::HasDataOctreeBlockCount_FzbPG);
	allocator->createBuffer(hasDataBlockCountBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasDataBlockCountBuffer.buffer);

	bufferSize = sizeof(shaderio::OctreeGlobalInfo_FzbPG);
	allocator->createBuffer(globalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(globalInfoBuffer.buffer);

	bufferSize = sizeof(shaderio::uint2) * ((IndivisibleNodeCount_G_FZBPG + 7) / 8);
	allocator->createBuffer(divisibleNodeInfoBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(divisibleNodeInfoBuffer_G.buffer);

	uint32_t maxLayerNodeCount = (1 << (3 * octreeMaxLayer)) * 6;
	bufferSize = sizeof(shaderio::OctreeThreadGroupInfo_FzbPG) * ((maxLayerNodeCount + GETOCTREELABEL_CS_THREADGROUP_SIZE - 1) / GETOCTREELABEL_CS_THREADGROUP_SIZE);
	allocator->createBuffer(threadGroupInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(threadGroupInfoBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G_FZBPG * sizeof(shaderio::uint2);
	allocator->createBuffer(indivisibleNodeInfosBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_G.buffer);

	bufferSize = CLUSTER_LAYER_NODECOUNT_E_FZBPG * sizeof(uint32_t);
	allocator->createBuffer(indivisibleNodeInfosBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_E.buffer);

#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	bufferSize = IndivisibleNodeCount_G_FZBPG * CLUSTER_LAYER_NODECOUNT_E_FZBPG * sizeof(shaderio::OctreeNodePairData_FzbPG);
	allocator->createBuffer(octreeNodePairDataBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(octreeNodePairDataBuffer.buffer);

	uint32_t adaptiveImportantSampleLayerCount = ADAPTIVE_IMPORTANCE_SAMPLING_MAX_LAYER - OCTREE_CLUSTER_LAYER_FZBPG;
	bufferSize = adaptiveImportantSampleLayerCount * sizeof(uint32_t);
	allocator->createBuffer(partialHitNodePairCountBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(partialHitNodePairCountBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G_FZBPG * CLUSTER_LAYER_NODECOUNT_E_FZBPG * sizeof(shaderio::OctreePartialHiNodePairTempData_FzbPG);
	allocator->createBuffer(partialHitNodePairTempDataBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(partialHitNodePairTempDataBuffer.buffer);

	bufferSize = sizeof(uint32_t);
	allocator->createBuffer(hitTestNodePairCountBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hitTestNodePairCountBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G_FZBPG * CLUSTER_LAYER_NODECOUNT_E_FZBPG * sizeof(shaderio::OctreeHitTestNodePairInfo_FzbPG);
	allocator->createBuffer(hitTestNodePairInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hitTestNodePairInfoBuffer.buffer);
#endif

	bufferSize = OUTGOING_COUNT_FZBPG * IndivisibleNodeCount_G_FZBPG * OCTREE_NODECOUNT_E_FZBPG * sizeof(float);
	allocator->createBuffer(octreeNodePairWeightBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(octreeNodePairWeightBuffer.buffer);

	#ifdef NEARBYNODE_JITTER_FZBPG
	bufferSize = IndivisibleNodeCount_G_FZBPG * (IndivisibleNodeCount_G_FZBPG / GETNEARBYNODES_CS_THREADGROUP_SIZE) * sizeof(shaderio::IndivisibleNodeNearbyNodeTempInfo_FzbPG);
	allocator->createBuffer(nearbyNodeTempInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(nearbyNodeTempInfoBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G_FZBPG * sizeof(shaderio::OctreeNearbyNodeInfo_FzbPG);
	allocator->createBuffer(nearbyNodeInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(nearbyNodeInfoBuffer.buffer);
	#endif

	pushConstant.octreeMaxLayer = octreeMaxLayer;
	pushConstant.octreeNodeTotalCount = int(pow(8, octreeMaxLayer + 1) - 1) / 7 * 6;
	pushConstant.VGBVoxelTotalCount = VGBSize * VGBSize * VGBSize;
}
void Octree_FzbPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeClusterData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeClusterDataBuffer_G.size(),	//max 128x128x128
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeClusterData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeClusterDataBuffer_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterLayerData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eBlockInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eBlockInfos_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockIndices_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockIndices_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockCount,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eThreadGroupInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodePairData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::ePartialHitNodePairCount,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::ePartialHitNodePairTempData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHitTestNodePairCount,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHitTestNodePairInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodePairWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifdef NEARBYNODE_JITTER_FZBPG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eNearbyNodeTempInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eNearbyNodeInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
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
}
void Octree_FzbPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eVGB, 0, 0, setting.VGBs.size());
	nvvk::Buffer* VGBsPtr = setting.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	VkWriteDescriptorSet    OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeClusterData_G, 0, 0, octreeClusterDataBuffer_G.size());
	nvvk::Buffer* octreeArraysPtr = octreeClusterDataBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeData_G, 0, 0, octreeDataBuffer_G.size());
	octreeArraysPtr = octreeDataBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeClusterData_E, 0, 0, octreeClusterDataBuffer_E.size());
	octreeArraysPtr = octreeClusterDataBuffer_E.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterLayerData_E, 0, 0, 1);
	write.append(OctreeArrayWrite, clusterLayerDataBuffer_E, 0, clusterLayerDataBuffer_E.bufferSize);

	VkWriteDescriptorSet    HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eBlockInfos_G, 0, 0, 1);
	write.append(HasDataInfoWrite, blockInfoBuffer_G, 0, blockInfoBuffer_G.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eBlockInfos_E, 0, 0, 1);
	write.append(HasDataInfoWrite, blockInfoBuffer_E, 0, blockInfoBuffer_E.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockIndices_G, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_G, 0, hasDataBlockIndexBuffer_G.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockIndices_E, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_E, 0, hasDataBlockIndexBuffer_E.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHasDataBlockCount, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockCountBuffer, 0, hasDataBlockCountBuffer.bufferSize);

	VkWriteDescriptorSet    GlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eGlobalInfo, 0, 0, 1);
	write.append(GlobalInfoWrite, globalInfoBuffer, 0, globalInfoBuffer.bufferSize);

	VkWriteDescriptorSet    LabelInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G, 0, 0, 1);
	write.append(LabelInfoWrite, divisibleNodeInfoBuffer_G, 0, divisibleNodeInfoBuffer_G.bufferSize);

	LabelInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eThreadGroupInfos, 0, 0, 1);
	write.append(LabelInfoWrite, threadGroupInfoBuffer, 0, threadGroupInfoBuffer.bufferSize);

	VkWriteDescriptorSet    IndivisibleInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G, 0, 0, 1);
	write.append(IndivisibleInfoWrite, indivisibleNodeInfosBuffer_G, 0, indivisibleNodeInfosBuffer_G.bufferSize);

	IndivisibleInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E, 0, 0, 1);
	write.append(IndivisibleInfoWrite, indivisibleNodeInfosBuffer_E, 0, indivisibleNodeInfosBuffer_E.bufferSize);

	VkWriteDescriptorSet NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodePairWeight, 0, 0, 1);
	write.append(NodePairInfoWrite, octreeNodePairWeightBuffer, 0, octreeNodePairWeightBuffer.bufferSize);

#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodePairData, 0, 0, 1);
	write.append(NodePairInfoWrite, octreeNodePairDataBuffer, 0, octreeNodePairDataBuffer.bufferSize);

	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::ePartialHitNodePairCount, 0, 0, 1);
	write.append(NodePairInfoWrite, partialHitNodePairCountBuffer, 0, partialHitNodePairCountBuffer.bufferSize);

	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::ePartialHitNodePairTempData, 0, 0, 1);
	write.append(NodePairInfoWrite, partialHitNodePairTempDataBuffer, 0, partialHitNodePairTempDataBuffer.bufferSize);

	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHitTestNodePairCount, 0, 0, 1);
	write.append(NodePairInfoWrite, hitTestNodePairCountBuffer, 0, hitTestNodePairCountBuffer.bufferSize);

	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eHitTestNodePairInfo, 0, 0, 1);
	write.append(NodePairInfoWrite, hitTestNodePairInfoBuffer, 0, hitTestNodePairInfoBuffer.bufferSize);
#endif

#ifdef NEARBYNODE_JITTER_FZBPG
	VkWriteDescriptorSet    NearbyNodeInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eNearbyNodeTempInfos, 0, 0, 1);
	write.append(NearbyNodeInfoWrite, nearbyNodeTempInfoBuffer, 0, nearbyNodeTempInfoBuffer.bufferSize);

	NearbyNodeInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eNearbyNodeInfos, 0, 0, 1);
	write.append(NearbyNodeInfoWrite, nearbyNodeInfoBuffer, 0, nearbyNodeInfoBuffer.bufferSize);
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_FzbPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_FzbPG)
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
void Octree_FzbPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	#ifndef NDEBUG
	std::string octreeLayerMapCountName = "OctreeLayerMapCount";
	std::string octreeLayerMapCount = std::to_string(showOctreeLayerMapCount);
	Application::slangCompiler.addMacro({
			.name = octreeLayerMapCountName.c_str(),
			.value = octreeLayerMapCount.c_str()
		});
	#endif

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "Clustering.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	#ifndef NDEBUG
	Application::slangCompiler.clearMacros();
	#endif

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_FzbPG),
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
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_initOctreeArray";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initOctreeArray);
	NVVK_DBG_NAME(computeShader_initOctreeArray);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_initHasDataBlockInfo";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initHasDataBlockInfo);
	NVVK_DBG_NAME(computeShader_initHasDataBlockInfo);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_getGlobalInfo";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getGlobalInfo);
	NVVK_DBG_NAME(computeShader_getGlobalInfo);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createOctreeArray";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray);
	NVVK_DBG_NAME(computeShader_createOctreeArray);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createOctreeArray2";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray2);
	NVVK_DBG_NAME(computeShader_createOctreeArray2);
	
#ifndef NDEBUG
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader_OctreeLayer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_OctreeLayer, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_OctreeLayer";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_OctreeLayer);
	NVVK_DBG_NAME(vertexShader_OctreeLayer);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_OctreeLayer";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_OctreeLayer);
	NVVK_DBG_NAME(fragmentShader_OctreeLayer);
#endif

//---------------------------------getOctreeIndivisibleNodeLabel-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetOctreeIndivisibleNodeLabel.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel1, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel1);
		NVVK_DBG_NAME(computeShader_getOctreeLabel1);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel2, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel2);
		NVVK_DBG_NAME(computeShader_getOctreeLabel2);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel3, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel3";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel3);
		NVVK_DBG_NAME(computeShader_getOctreeLabel3);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getOctreeLabel4, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getOctreeLabel4";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getOctreeLabel4);
		NVVK_DBG_NAME(computeShader_getOctreeLabel4);
		#ifndef NDEBUG
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, vertexShader_OctreeIndivisibleNodes, nullptr);
		vkDestroyShaderEXT(device, fragmentShader_OctreeIndivisibleNodes, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain_OctreeIndivisibleNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_OctreeIndivisibleNodes);
		NVVK_DBG_NAME(vertexShader_OctreeIndivisibleNodes);

		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain_OctreeIndivisibleNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_OctreeIndivisibleNodes);
		NVVK_DBG_NAME(fragmentShader_OctreeIndivisibleNodes);
		#endif
	}

#ifdef NEARBYNODE_JITTER_FZBPG
//----------------------------------------getNearbyNodeInfo------------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetNearbyNodeInfo.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getNearbyNodes1, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyNodes1);
		NVVK_DBG_NAME(computeShader_getNearbyNodes1);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getNearbyNodes2, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyNodes2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyNodes2);
		NVVK_DBG_NAME(computeShader_getNearbyNodes2);
		#ifndef NDEBUG
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, vertexShader_NearbyNodeInfoResult, nullptr);
		vkDestroyShaderEXT(device, fragmentShader_NearbyNodeInfoResult, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain_nearby";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_NearbyNodeInfoResult);
		NVVK_DBG_NAME(vertexShader_NearbyNodeInfoResult);

		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain_nearby";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_NearbyNodeInfoResult);
		NVVK_DBG_NAME(fragmentShader_NearbyNodeInfoResult);
#endif
	}
#endif

//---------------------------------getOctreeNodePairData-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetNodePairData.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initWeights";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initWeights);
		NVVK_DBG_NAME(computeShader_initWeights);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_octreeNodeHitTest, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_octreeNodeHitTest";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_octreeNodeHitTest);
		NVVK_DBG_NAME(computeShader_octreeNodeHitTest);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getProbability";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getProbability);
		NVVK_DBG_NAME(computeShader_getProbability);
#ifndef NDEBUG
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, vertexShader_OctreeNodePairHitTestResult, nullptr);
		vkDestroyShaderEXT(device, fragmentShader_OctreeNodePairHitTestResult, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain_OctreeNodePairHitTestResult";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_OctreeNodePairHitTestResult);
		NVVK_DBG_NAME(vertexShader_OctreeNodePairHitTestResult);

		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain_OctreeNodePairHitTestResult";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_OctreeNodePairHitTestResult);
		NVVK_DBG_NAME(fragmentShader_OctreeNodePairHitTestResult);
#endif
	}
}
void Octree_FzbPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void Octree_FzbPG::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree_FzbPG::createOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	uint32_t layerBlockCount = uint32_t(pow(8, pushConstant.octreeMaxLayer)) * 6 / 8;
	for (int layerIndex = pushConstant.octreeMaxLayer; layerIndex > OCTREE_CLUSTER_LAYER_FZBPG; --layerIndex) {
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerBlockCount = layerBlockCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, hasDataBlockCountBuffer.buffer, 0, sizeof(shaderio::HasDataOctreeBlockCount_FzbPG), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initHasDataBlockInfo);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerBlockCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getGlobalInfo);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);

		layerBlockCount /= 8;
	}

	for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG; layerIndex > 0; --layerIndex) {
		uint32_t layerNodeCount = shaderio::OctreeLayerNodeCount_FzbPG[layerIndex];
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerNodeCount = layerNodeCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray2);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ CREATEOCTREE_CS_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}
void Octree_FzbPG::getOctreeLabel(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel1);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	for (int layerIndex = 2; layerIndex <= pushConstant.octreeMaxLayer; ++layerIndex) {
		pushConstant.currentLayer = layerIndex;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel2);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel3);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		if(layerIndex < pushConstant.octreeMaxLayer) 
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
		else nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel4);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void Octree_FzbPG::getOctreeNodePairData(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initWeights);
	uint32_t threadTotalCount = OUTGOING_COUNT_FZBPG * IndivisibleNodeCount_G_FZBPG * OCTREE_NODECOUNT_E_FZBPG;
	VkExtent2D groupSize = nvvk::getGroupCounts({ threadTotalCount, 1 }, VkExtent2D{ INITWEIGHT_CS_THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_octreeNodeHitTest);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getProbability);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
}
void Octree_FzbPG::getNearbyNodeInfo(VkCommandBuffer cmd) {
#ifdef NEARBYNODE_JITTER_FZBPG
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyNodes1);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, offsetof(shaderio::OctreeGlobalInfo_FzbPG, cmd2));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyNodes2);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, offsetof(shaderio::OctreeGlobalInfo_FzbPG, cmd2));
#endif
}

#ifndef NDEBUG
void Octree_FzbPG::debug_Prepare() {
	showOctreeLayerMapCount = octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG;
	showOctreeLayerMapCount = showOctreeLayerMapCount * 2;

	showOctreeIndivisibleNodeMapIndex = showOctreeLayerMapCount;
	showOctreeNodePairHitTestResultMapIndex = showOctreeIndivisibleNodeMapIndex + 1;
	showNearbyNodeInfoResultMapIndex = showOctreeIndivisibleNodeMapIndex + 2;

	showMapCount = showOctreeLayerMapCount + 1 + 1 + 1;
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	showOctreeNodePairVisibleAabbMapIndex = showOctreeIndivisibleNodeMapIndex + 3;
	++showMapCount;
#endif
	Feature::createGBuffer(true, false, showMapCount);

	pushConstant.sampleNodeLabel_G = 731;	// 188
	pushConstant.sampleNodeLabel_E = 15;

	for (int i = OCTREE_CLUSTER_LAYER_FZBPG; i < octreeMaxLayer; ++i)		//exclude maxlayer
		pushConstant.showOctreeNodeTotalCount += pow(8, i);		

	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	primitive = FzbRenderer::MeshSet::createCube(false, false);
	mesh = FzbRenderer::MeshSet("Cube", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
}
void Octree_FzbPG::resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex) {
	gBuffers.update(cmd, size);
	nvvk::WriteSetContainer write{};
	depthImageView = gBuffers_other.getDepthImageView();
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_FzbPG::debug_OctreeLayer_Visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	uint32_t wireframeMapCount = showOctreeLayerMapCount;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(wireframeMapCount);
	for (int i = 0; i < wireframeMapCount; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd,
			{ gBuffers.getColorImage(i),
			VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		colorAttachments[i] = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachments[i].imageView = gBuffers.getColorImageView(i);
		colorAttachments[i].clearValue = { .color = {0.0f, 0.0f, 0.0f, 0.0f} };
	}
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;		//使用PathGuiding的深度纹理
	//depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = depthImageView;	// gBuffers.getDepthImageView();	//depthImageView;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = colorAttachments.size();
	renderingInfo.pColorAttachments = colorAttachments.data();
	renderingInfo.pDepthAttachment = nullptr;	// &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	bool useWireframe = true;

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	if (useWireframe) {
		graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
		graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
		graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
		graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	}else graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.colorWriteMasks.resize(wireframeMapCount);
	graphicsDynamicPipeline.colorBlendEquations.resize(wireframeMapCount);
	graphicsDynamicPipeline.colorBlendEnables.resize(wireframeMapCount);
	for (int i = 0; i < wireframeMapCount; ++i) {
		graphicsDynamicPipeline.colorWriteMasks[i] = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		graphicsDynamicPipeline.colorBlendEquations[i] = {
		  .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
		  .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		  .colorBlendOp = VK_BLEND_OP_ADD,
		  .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
		  .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
		  .alphaBlendOp = VK_BLEND_OP_ADD,
		};
		graphicsDynamicPipeline.colorBlendEnables[i] = true;
	}

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_OctreeLayer, .fragment = fragmentShader_OctreeLayer });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.normalIndex = RasterVoxelization_FzbPG::normalIndex;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = pushConstant.showOctreeNodeTotalCount * 6 * 2;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	for (int i = 0; i < wireframeMapCount; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd,
			{ gBuffers.getColorImage(i),
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	}
}
void Octree_FzbPG::debug_OctreeIndivisibleNodes_Visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeIndivisibleNodeMapIndex), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(showOctreeIndivisibleNodeMapIndex);
	colorAttachment.clearValue = { .color = {0.0f, 0.0f, 0.0f, 0.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;		//使用PathGuiding的深度纹理
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();	// gBuffers.getDepthImageView();	//depthImageView;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;	// &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_OctreeIndivisibleNodes, .fragment = fragmentShader_OctreeIndivisibleNodes });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 0;
	uint32_t cubeMeshIndex = 1;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	vkCmdDrawIndexed(cmd, triMesh.indices.count, IndivisibleNodeCount_G_FZBPG, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeIndivisibleNodeMapIndex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void Octree_FzbPG::debug_OctreeNodePairHitTestResult_Visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeNodePairHitTestResultMapIndex), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(showOctreeNodePairHitTestResultMapIndex);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;		//使用PathGuiding的深度纹理
	//depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = depthImageView;	// gBuffers.getDepthImageView();	//depthImageView;

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
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_OctreeNodePairHitTestResult, .fragment = fragmentShader_OctreeNodePairHitTestResult });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	int cubeMeshIndex = 1;
	const shaderio::Mesh& mesh = scene.meshes[cubeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(cubeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];
	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	vkCmdDrawIndexed(cmd, triMesh.indices.count, CLUSTER_LAYER_NODECOUNT_E_FZBPG + 1, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeNodePairHitTestResultMapIndex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void Octree_FzbPG::debug_NearbyNodeInfoResult_Visualization(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showNearbyNodeInfoResultMapIndex), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(showNearbyNodeInfoResultMapIndex);
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
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_NearbyNodeInfoResult, .fragment = fragmentShader_NearbyNodeInfoResult });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[1];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(1);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, IndivisibleNodeCount_G_FZBPG, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showNearbyNodeInfoResultMapIndex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void Octree_FzbPG::debug_NodePairVisibleAABB_Visualization(VkCommandBuffer cmd) {
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	//NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeNodePairVisibleAabbMapIndex), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(showOctreeNodePairVisibleAabbMapIndex);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.imageView = gBuffers.getDepthImageView();	// gBuffers.getDepthImageView(); depthImageView
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr; // &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_OctreeNodePairHitTestResult, .fragment = fragmentShader_OctreeNodePairHitTestResult });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, CLUSTER_LAYER_NODECOUNT_E_FZBPG + 1, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(showOctreeNodePairVisibleAabbMapIndex), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
#endif
}
#endif