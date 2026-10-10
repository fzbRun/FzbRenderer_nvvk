#include "./Octree2_FzbPG.h"
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

Octree2_FzbPG::Octree2_FzbPG(pugi::xml_node& featureNode) {}

void Octree2_FzbPG::init(Octree2CreateInfo_FzbPG createInfo) {
	this->setting = createInfo;

	createOctreeArray();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	Feature::createGBuffer(true, true, (uint32_t)GBuffers_Octree2_FzbPG::eBufferCount);
	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
		scene.addMeshSet(mesh);

		primitive = FzbRenderer::MeshSet::createCube(false, false);
		mesh = FzbRenderer::MeshSet("Cube", primitive);
		scene.addMeshSet(mesh);

		scene.createSceneInfoBuffer();
	}
#endif
}
void Octree2_FzbPG::clean() {
	Feature::clean();
	for (int i = 0; i < octreeNodeDataBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeNodeDataBuffer_G[i]);
	for (int i = 0; i < octreeNodeInfoBuffer_G.size(); ++i) Application::allocator.destroyBuffer(octreeNodeInfoBuffer_G[i]);
	for (int i = 0; i < octreeNodeDataBuffer_E.size(); ++i) Application::allocator.destroyBuffer(octreeNodeDataBuffer_E[i]);

	Application::allocator.destroyBuffer(globalInfoBuffer);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_G);

	Application::allocator.destroyBuffer(divisibleNodeInfoBuffer_G);
	Application::allocator.destroyBuffer(threadGroupInfoBuffer);

	Application::allocator.destroyBuffer(blockInfoBuffer_G);
	Application::allocator.destroyBuffer(blockInfoBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_G);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockCountBuffer);

	Application::allocator.destroyBuffer(clusterPairInfoBuffer);
	Application::allocator.destroyBuffer(clusterPairGlobalInfoBuffer);
	Application::allocator.destroyBuffer(clusterPairHitInfoBuffer);
	Application::allocator.destroyBuffer(candidateNodeDataBuffer_E);

	Application::allocator.destroyBuffer(octreeNodePairDataBuffer);

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
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_dispatchGetWeight, nullptr);
	vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeights, nullptr);

	vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
}
void Octree2_FzbPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;

	std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward" };
	std::vector<const char*> normalIndexNames_pointers;
	for (const auto& normalIndexName : normalIndexNames)
		normalIndexNames_pointers.push_back(normalIndexName.c_str());

	{
		uint32_t geometryShowLayerCount = pushConstant.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG + 1;
		std::vector<std::string> layerMapNames(geometryShowLayerCount);
		for (int i = 0; i < geometryShowLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i + OCTREE_CLUSTER_LAYER_FZBPG);

		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());

		static bool showGeometryTreeDebugMap = false;
		static int debugLayer = 0;
		if (ImGui::Begin("Geometry Tree Debug")) {
			ImGui::Combo("normal", &debugPushConstant.normalIndex_G, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			ImGui::Combo("Geometry Tree Layer", &debugLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("GeometryTreeLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showGeometryTreeDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eGeometryClusterResult),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showGeometryTreeDebugMap = !showGeometryTreeDebugMap;
				}
			}
			PE::end();
		}
		ImGui::End();
		debugPushConstant.curLayer_G = debugLayer + OCTREE_CLUSTER_LAYER_FZBPG;

		if (showGeometryTreeDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eGeometryClusterResult);

		static bool showGeometrySVODebugMap = false;
		static int debugLayer2 = 0;
		if (ImGui::Begin("Geometry SVO Debug")) {
			ImGui::Combo("normal", &debugPushConstant.normalIndex_G_2, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			ImGui::Combo("Geometry SVO Layer", &debugLayer2, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("GeometrySVOLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showGeometrySVODebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eGeometrySVO),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showGeometrySVODebugMap = !showGeometrySVODebugMap;
				}
			}
			PE::end();
		}
		ImGui::End();
		debugPushConstant.curLayer_G_2 = debugLayer2 + OCTREE_CLUSTER_LAYER_FZBPG;

		if (showGeometrySVODebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eGeometrySVO);
	}

	{
		uint32_t lightShowLayerCount = pushConstant.octreeMaxLayer + 1;
		std::vector<std::string> layerMapNames(lightShowLayerCount);
		for (int i = 0; i < lightShowLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i);

		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());

		static bool showLightTreeDebugMap = false;
		static int debugLayer = 0;
		if (ImGui::Begin("Light Tree Debug")) {
			ImGui::Combo("normal", &debugPushConstant.normalIndex_E, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));

			ImGui::Combo("Light Tree Layer", &debugLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("LightTreeLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showLightTreeDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eLightClusterResult),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showLightTreeDebugMap = !showLightTreeDebugMap;
				}
			}
			PE::end();
		}
		ImGui::End();
		debugPushConstant.curLayer_E = debugLayer;

		if (showLightTreeDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_Octree2_FzbPG::eLightClusterResult);
	}
#endif
}
void Octree2_FzbPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	NVVK_CHECK(gBuffers.update(cmd, size));
	//清理深度纹理
	{
		VkSamplerCreateInfo samplerInfo = DEFAULT_VkSamplerCreateInfo;
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		Application::samplerPool.acquireSampler(gBuffers.m_res.gBufferDepth.descriptor.sampler, samplerInfo);

		const VkImageLayout layout{ VK_IMAGE_LAYOUT_GENERAL };
		VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier({ .image = gBuffers.m_res.gBufferDepth.image,
														.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
														.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
														.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									   .imageMemoryBarrierCount = 1,
									   .pImageMemoryBarriers = &barrier };

		vkCmdPipelineBarrier2(cmd, &depInfo);

		VkClearDepthStencilValue clearDepth = { 1.0f, 0 };
		VkImageSubresourceRange range = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
		vkCmdClearDepthStencilImage(cmd, gBuffers.m_res.gBufferDepth.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, 1, &range);

		// Setting the layout to the final one
		barrier = nvvk::makeImageMemoryBarrier(
			{ .image = gBuffers.m_res.gBufferDepth.image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = layout, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		gBuffers.m_res.gBufferDepth.descriptor.imageLayout = layout;
		vkCmdPipelineBarrier2(cmd, &depInfo);
	}
#endif
};
void Octree2_FzbPG::preRender() {
	pushConstant.frameIndex = Application::frameIndex;

	pushConstant.VGBVoxelTotalCount = setting.VGBSize * setting.VGBSize * setting.VGBSize;
	pushConstant.voxelVolume = setting.VGBVoxelSize.x * setting.VGBVoxelSize.y * setting.VGBVoxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);

	float angle = FzbRenderer::rand(Application::frameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void Octree2_FzbPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
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
}
void Octree2_FzbPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	pushInfo.pValues = &debugPushConstant;

	debugPushConstant.frameIndex = pushConstant.frameIndex;
	debugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;

	debugPushConstant.curLayerNodeCount_G = (1 << (3 * debugPushConstant.curLayer_G));
	debugPushConstant.curLayerNodeCount_G_2 = (1 << (3 * debugPushConstant.curLayer_G_2));
	debugPushConstant.curLayerNodeCount_E = (1 << (3 * debugPushConstant.curLayer_E));

	geometryTreeDebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	lightTreeDebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	geometrySVODebug(cmd);
#endif
};

void Octree2_FzbPG::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	octreeMaxLayer = std::countr_zero(VGBSize);	//start from 0
	if (octreeMaxLayer >= MAX_OCTREE_LAYER_FZBPG) throw std::runtime_error("八叉树最大深度为0-" + std::to_string(MAX_OCTREE_LAYER_FZBPG - 1));

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	octreeNodeDataBuffer_G.resize(octreeMaxLayer + 1);
	octreeNodeInfoBuffer_G.resize(octreeMaxLayer + 1);
	octreeNodeDataBuffer_E.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_G_FzbPG);
		allocator->createBuffer(octreeNodeDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeDataBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeInfo_G_FzbPG);
		allocator->createBuffer(octreeNodeInfoBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeInfoBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_E_FzbPG);
		allocator->createBuffer(octreeNodeDataBuffer_E[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeNodeDataBuffer_E[layerIndex].buffer);

		layerNodeCount *= 8;
	}

	uint32_t bufferSize = uint32_t(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(uint32_t);
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

	bufferSize = IndivisibleNodeCount_G_FZBPG * Candidate_Samples_Count * sizeof(shaderio::ClusterPairInfo);
	allocator->createBuffer(clusterPairInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(clusterPairInfoBuffer.buffer);

	bufferSize = sizeof(shaderio::ClusterPairGlobalInfo);
	allocator->createBuffer(clusterPairGlobalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(clusterPairGlobalInfoBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G_FZBPG * Candidate_Samples_Count * HITTEST_COUNT_FZBPG * sizeof(shaderio::ClusterPairHitInfo);
	allocator->createBuffer(clusterPairHitInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(clusterPairHitInfoBuffer.buffer);

	bufferSize = OUTGOING_COUNT_FZBPG * IndivisibleNodeCount_G_FZBPG * Candidate_Samples_Count * sizeof(shaderio::CandidateNodeData_E_FzbPG);
	allocator->createBuffer(candidateNodeDataBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(candidateNodeDataBuffer_E.buffer);

	pushConstant.octreeMaxLayer = octreeMaxLayer;
	pushConstant.octreeNodeTotalCount = int(pow(8, octreeMaxLayer + 1) - 1) / 7 * 6;
}
void Octree2_FzbPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeNodeDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeInfo_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeNodeInfoBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octreeNodeDataBuffer_E.size(),
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
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterPairInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterPairGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusetPairHitTestInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_FzbPG::eCandidateNodeData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

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
void Octree2_FzbPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eVGB, 0, 0, setting.VGBs.size());
	nvvk::Buffer* VGBsPtr = setting.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	VkWriteDescriptorSet    OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeData_G, 0, 0, octreeNodeDataBuffer_G.size());
	nvvk::Buffer* octreeArraysPtr = octreeNodeDataBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeInfo_G, 0, 0, octreeNodeInfoBuffer_G.size());
	octreeArraysPtr = octreeNodeInfoBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeNodeData_E, 0, 0, octreeNodeDataBuffer_E.size());
	octreeArraysPtr = octreeNodeDataBuffer_E.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

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

	VkWriteDescriptorSet 	candidateNodesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterPairInfo, 0, 0, 1);
	write.append(candidateNodesWrite, clusterPairInfoBuffer, 0, clusterPairInfoBuffer.bufferSize);

	candidateNodesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusterPairGlobalInfo, 0, 0, 1);
	write.append(candidateNodesWrite, clusterPairGlobalInfoBuffer, 0, clusterPairGlobalInfoBuffer.bufferSize);

	candidateNodesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eClusetPairHitTestInfo, 0, 0, 1);
	write.append(candidateNodesWrite, clusterPairHitInfoBuffer, 0, clusterPairHitInfoBuffer.bufferSize);
		
	candidateNodesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eCandidateNodeData_E, 0, 0, 1);
	write.append(candidateNodesWrite, candidateNodeDataBuffer_E, 0, candidateNodeDataBuffer_E.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree2_FzbPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
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
void Octree2_FzbPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);


	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

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
	{
		shaderSource = shaderPath / "Clustering.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initOctreeArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initOctreeArray);
		NVVK_DBG_NAME(computeShader_initOctreeArray);

		vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initHasDataBlockInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initHasDataBlockInfo);
		NVVK_DBG_NAME(computeShader_initHasDataBlockInfo);

		vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getGlobalInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getGlobalInfo);
		NVVK_DBG_NAME(computeShader_getGlobalInfo);

		vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createOctreeArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray);
		NVVK_DBG_NAME(computeShader_createOctreeArray);

		vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createOctreeArray2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createOctreeArray2);
		NVVK_DBG_NAME(computeShader_createOctreeArray2);
	}
	//---------------------------------getOctreeIndivisibleNodeLabel-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "CreateSVO.slang";
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
	}
	//---------------------------------getOctreeNodePairData-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "GetClusterWeight.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initCandidateNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initWeights);
		NVVK_DBG_NAME(computeShader_initWeights);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodes, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodes";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodes);
		NVVK_DBG_NAME(computeShader_getCandidateNodes);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_dispatchGetWeight, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_dispatchGetWeight";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_dispatchGetWeight);
		NVVK_DBG_NAME(computeShader_dispatchGetWeight);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_clusterPairHitTest";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_clusterPairHitTest);
		NVVK_DBG_NAME(computeShader_clusterPairHitTest);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeights, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getCandidateNodeWeight";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getCandidateNodeWeights);
		NVVK_DBG_NAME(computeShader_getCandidateNodeWeights);
	}
	//---------------------------------Debug-------------------------------------
	{
		shaderSource = shaderPath / "Debug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_TreeDebug);
		NVVK_DBG_NAME(vertexShader_TreeDebug);

		vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_TreeDebug);
		NVVK_DBG_NAME(fragmentShader_TreeDebug);
	}
}
void Octree2_FzbPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void Octree2_FzbPG::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree2_FzbPG::createOctreeArray(VkCommandBuffer cmd) {
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
void Octree2_FzbPG::getOctreeLabel(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel1);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG; layerIndex <= pushConstant.octreeMaxLayer; ++layerIndex) {
		pushConstant.currentLayer = layerIndex;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel2);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel3);
		vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
		if (layerIndex < pushConstant.octreeMaxLayer)
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
		else nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getOctreeLabel4);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void Octree2_FzbPG::getOctreeNodePairData(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initWeights);
	uint32_t threadTotalCount = OUTGOING_COUNT_FZBPG * IndivisibleNodeCount_G_FZBPG * Candidate_Samples_Count;
	VkExtent2D groupSize = nvvk::getGroupCounts({ threadTotalCount, 1 }, VkExtent2D{ INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchGetWeight);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clusterPairHitTest);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodeWeights);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

}

void Octree2_FzbPG::geometryTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eGeometryClusterResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree2_FzbPG::eGeometryClusterResult));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 0;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eGeometryClusterResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void Octree2_FzbPG::lightTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eLightClusterResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree2_FzbPG::eLightClusterResult));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 1;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_E;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eLightClusterResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void Octree2_FzbPG::geometrySVODebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eGeometrySVO)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_Octree2_FzbPG::eGeometrySVO));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 2;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G_2;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_Octree2_FzbPG::eGeometrySVO)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}