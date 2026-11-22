#include "./GeometryClustering.h"
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include "../VRPG2.h"
#include "feature/PathTracing/shaderio.h"
#include <cstdio>
#include <cstdlib>

using namespace FzbRenderer;

static_assert(sizeof(shaderio::GeometryClusteringPushConstant_VRPG) <= 256, "geometry clustering push constant exceeds 256 bytes");

GeometryClustering_VRPG::GeometryClustering_VRPG(pugi::xml_node& featureNode) {
	if (pugi::xml_node clusteringThresholdNode = featureNode.child("clusteringThreshold"))
		VRPG2Renderer::setting.clusteringThreshold = std::stof(clusteringThresholdNode.attribute("value").value());
	else VRPG2Renderer::setting.clusteringThreshold = 0.75f;

	if (pugi::xml_node clusteringTopLayerNode = featureNode.child("clusteringTopLayer"))
		VRPG2Renderer::setting.geometryClusteringTopLayer = std::stoi(clusteringTopLayerNode.attribute("value").value());
	else VRPG2Renderer::setting.geometryClusteringTopLayer = 3;
}

void GeometryClustering_VRPG::init() {
	createSource();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	Feature::createGBuffer(true, true, (uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eBufferCount);
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
void GeometryClustering_VRPG::clean() {
	Feature::clean();
	scene.clean();

	Application::allocator.destroyBuffer(clusteringGlobalInfoBuffer);

	Application::allocator.destroyBuffer(clusterBlockInfoBuffer);
	Application::allocator.destroyBuffer(hasDataClusterBlockIndexBuffer);

	for (int i = 0; i < 2; ++i) Application::allocator.destroyBuffer(divisibleNodeInfoBuffer[i]);
	Application::allocator.destroyBuffer(threadGroupInfoBuffer);

	Application::allocator.destroyBuffer(nearbyClusterTempInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_init, nullptr);

	vkDestroyShaderEXT(device, computeShader_GetHasDataClusterBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_DispatchClustering, nullptr);
	vkDestroyShaderEXT(device, computeShader_Clustering1, nullptr);
	vkDestroyShaderEXT(device, computeShader_Clustering2, nullptr);

	vkDestroyShaderEXT(device, computeShader_CreateSVO1, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateSVO2, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateSVO3, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateSVO4, nullptr);

	vkDestroyShaderEXT(device, computeShader_dispatchGetNearbyCluster, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyCluster1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getNearbyCluster2, nullptr);

	vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
}
void GeometryClustering_VRPG::uiRender() {

	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;

	std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward" };
	std::vector<const char*> normalIndexNames_pointers;
	for (const auto& normalIndexName : normalIndexNames)
		normalIndexNames_pointers.push_back(normalIndexName.c_str());

	if (ImGui::CollapsingHeader(std::string("Geometry Clustering ").c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::SeparatorText("Geometry Clustering");
		bool thresholdChange = ImGui::SliderFloat("clusteringThreshold", &VRPG2Renderer::setting.clusteringThreshold, 0.0f, 1.0f, "%.3f");
		UIModified |= thresholdChange;
		VRPG2Renderer::setting.clusteringChange |= thresholdChange;
#ifndef NDEBUG
		uint32_t geometryShowLayerCount = pushConstant.octreeMaxLayer - VRPG2Renderer::setting.geometryClusteringTopLayer + 1;
		std::vector<std::string> layerMapNames(geometryShowLayerCount);
		for (int i = 0; i < geometryShowLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i + VRPG2Renderer::setting.geometryClusteringTopLayer);

		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());

		if (ImGui::CollapsingHeader("Geometry Octree Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			static int debugLayer = 0;
			ImGui::Combo("normal_G", &debugPushConstant.normalIndex_G, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));
			ImGui::Combo("Geometry Octree Layer", &debugLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			ImGui::Checkbox("Show Geometry Octree", &showOctree);

			debugPushConstant.mode = 0;
			debugPushConstant.curLayer_G = debugLayer + VRPG2Renderer::setting.geometryClusteringTopLayer;

			ImGui::Unindent(20.0f);
		}

		if (ImGui::CollapsingHeader("Geometry SVO Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			std::vector<const char*> geometrySVONormalNames_pointers = normalIndexNames_pointers;
			geometrySVONormalNames_pointers.push_back("All");
			ImGui::Combo("normal-G2", &debugPushConstant.normalIndex_G_2,
				geometrySVONormalNames_pointers.data(), static_cast<int>(geometrySVONormalNames_pointers.size()));

			static int debugLayer2 = 0;
			std::vector<const char*> geometrySVOLayerNames_pointers = layerMapNames_pointers;
			geometrySVOLayerNames_pointers.push_back("All Layers");
			ImGui::Combo("Geometry SVO Layer", &debugLayer2, geometrySVOLayerNames_pointers.data(), static_cast<int>(geometrySVOLayerNames_pointers.size()));
			ImGui::Checkbox("Show Geometry SVO", &showSVO);

			debugPushConstant.curLayer_G_2 = debugLayer2 >= int(geometryShowLayerCount) ? pushConstant.octreeMaxLayer + 1 : debugLayer2 + VRPG2Renderer::setting.geometryClusteringTopLayer;

			ImGui::Unindent(20.0f);
		}

		if (showSVO) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eSVO);
		else if (showOctree) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eOctree);
#endif
	}
}
void GeometryClustering_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
	NVVK_CHECK(gBuffers.update(cmd, size));
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
void GeometryClustering_VRPG::preRender() {}
void GeometryClustering_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = (VRPG_RANDOM_FRAME_SOURCE != 0) ? int(VRPG2Renderer::setting.randomFrameIndex) : VRPG2Renderer::setting.frameIndex;

	pushConstant.voxelTotalCount = std::pow(VRPG2Renderer::setting.voxelGridResolution, 3);
	pushConstant.voxelVolume = VRPG2Renderer::setting.voxelSize.x * VRPG2Renderer::setting.voxelSize.y * VRPG2Renderer::setting.voxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(VRPG2Renderer::setting.voxelGridStartPos, VRPG2Renderer::setting.voxelGridResolution);
	pushConstant.voxelSize = glm::vec4(VRPG2Renderer::setting.voxelSize, 1.0f);

	pushConstant.clusteringThreshold = VRPG2Renderer::setting.clusteringThreshold;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	VRPG2Renderer::setting.clusteringChange |= VRPG2Renderer::setting.geometryChange;
}
void GeometryClustering_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "GeometryClustering_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::GeometryClusteringPushConstant_VRPG),
		.pValues = &pushConstant,
	};

	// the geometry clustering shaders never trace rays, so only set 0 is needed
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	initClusterOctree(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	createClusterOctree(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	createClusterSVO(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getNearbyClusterInfo(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}
void GeometryClustering_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	if (!showOctree && !showSVO) return;

	debugPushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::GeometryClusteringDebugPushConstant_VRPG),
		.pValues = &debugPushConstant,
	};

	debugPushConstant.frameIndex = pushConstant.frameIndex;
	debugPushConstant.maxLayer = pushConstant.octreeMaxLayer;
	debugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;

	debugPushConstant.curLayerNodeCount_G = (1 << (3 * debugPushConstant.curLayer_G));
	debugPushConstant.curLayerNodeCount_G_2 =
		debugPushConstant.curLayer_G_2 > int(pushConstant.octreeMaxLayer)
		? 0 : (1 << (3 * debugPushConstant.curLayer_G_2));

	if (showOctree) {
		octreeDebug(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	}
	if (showSVO) {
		svoDebug(cmd);
	}
#endif
}

void GeometryClustering_VRPG::createSource() {
	uint32_t VGBSize = uint32_t(VRPG2Renderer::setting.voxelGridResolution);
	VRPG2Renderer::setting.geometryOctreeDepth = std::countr_zero(VGBSize);	//start from 0
	int octreeMaxLayer = VRPG2Renderer::setting.geometryOctreeDepth;
	if (octreeMaxLayer >= MAX_OCTREE_LAYER_VRPG2)
		throw std::runtime_error("VRPG2: geometry voxel resolution is too large for MAX_OCTREE_LAYER_VRPG2");

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();
	uint32_t bufferSize = 0;

	// shared dispatch indirect command / cluster counters, consumed by every later stage
	if (VRPG2Renderer::source.globalInfoBuffer.buffer == VK_NULL_HANDLE) {
		bufferSize = sizeof(shaderio::GlobalInfo_VRPG2);
		allocator->createBuffer(VRPG2Renderer::source.globalInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.globalInfoBuffer.buffer);
	}

	//-----------------------------------------------------------Octree------------------------------------------------------------------------------
	VRPG2Renderer::source.clusterDataBuffer_Octree.resize(octreeMaxLayer + 1);
	VRPG2Renderer::source.clusterInfoBuffer_Octree.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		bufferSize = layerNodeCount * sizeof(shaderio::ClusterData_Octree_VRPG);
		allocator->createBuffer(VRPG2Renderer::source.clusterDataBuffer_Octree[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.clusterDataBuffer_Octree[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::ClusterInfo_Octree_VRPG);
		allocator->createBuffer(VRPG2Renderer::source.clusterInfoBuffer_Octree[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.clusterInfoBuffer_Octree[layerIndex].buffer);

		layerNodeCount *= 8;
	}
	//-----------------------------------------------------------Global------------------------------------------------------------------------------
	{
		bufferSize = sizeof(shaderio::ClusteringGlobalInfo_VRPG);
		allocator->createBuffer(clusteringGlobalInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
		NVVK_DBG_NAME(clusteringGlobalInfoBuffer.buffer);
	}
	//------------------------------------------------------------CreateOctree-----------------------------------------------------------------------------
	{
		bufferSize = uint32_t(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(shaderio::ClusterBlockInfo);
		allocator->createBuffer(clusterBlockInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(clusterBlockInfoBuffer.buffer);

		bufferSize = int(pow(8, octreeMaxLayer)) * 6 / 8 * sizeof(shaderio::ClusterBlockIndex);
		allocator->createBuffer(hasDataClusterBlockIndexBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(hasDataClusterBlockIndexBuffer.buffer);
	}
	//--------------------------------------------------------------SVO---------------------------------------------------------------------------
	{
		bufferSize = ClusterMaxCount_VRPG * sizeof(shaderio::ClusterInfo_SVO_VRPG);
		allocator->createBuffer(VRPG2Renderer::source.clusterInfoBuffer_SVO, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.clusterInfoBuffer_SVO.buffer);
	}
	//--------------------------------------------------------------CreateSVO---------------------------------------------------------------------------
	{
		bufferSize = sizeof(shaderio::SVODivisibleNodeInfo) * ((ClusterMaxCount_VRPG + 7) / 8);
		divisibleNodeInfoBuffer.resize(2);
		for (int i = 0; i < 2; ++i) {
			allocator->createBuffer(divisibleNodeInfoBuffer[i], bufferSize,
				VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
			NVVK_DBG_NAME(divisibleNodeInfoBuffer[i].buffer);
		}

		uint32_t maxLayerNodeCount = (1 << (3 * octreeMaxLayer)) * 6;
		bufferSize = sizeof(shaderio::SVOThreadGroupInfo_VRPG) * ((maxLayerNodeCount + CreateSVO_CS_THREADGROUP_SIZE - 1) / CreateSVO_CS_THREADGROUP_SIZE);
		allocator->createBuffer(threadGroupInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(threadGroupInfoBuffer.buffer);
	}
	//---------------------------------------------------------------NearbyCluster--------------------------------------------------------------------------
	{
		bufferSize = ClusterMaxCount_VRPG * (ClusterMaxCount_VRPG + GetNearbyCluster_CS_THREADGROUP_SIZE - 1) / GetNearbyCluster_CS_THREADGROUP_SIZE * sizeof(shaderio::NearbyClusterTempInfo_VRPG);
		allocator->createBuffer(nearbyClusterTempInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(nearbyClusterTempInfoBuffer.buffer);

		bufferSize = ClusterMaxCount_VRPG * sizeof(shaderio::NearbyClusterInfo_VRPG);
		allocator->createBuffer(VRPG2Renderer::source.nearbyClusterInfoBuffer, bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.nearbyClusterInfoBuffer.buffer);
	}

	pushConstant.octreeMaxLayer = octreeMaxLayer;
	pushConstant.octreeNodeTotalCount = int(pow(8, octreeMaxLayer + 1) - 1) / 7 * 6;
}
void GeometryClustering_VRPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eGlobalInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterData_Octree,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterDataBuffer_Octree.size(),
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterInfoBuffer_Octree.size(),
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterBlockInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eHasDataClusterBlockIndices,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterInfo_SVO,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eSVODivisibleNodeInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 2,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eSVOThreadGroupInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	{
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eNearbyClusterTempInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eNearbyClusterInfos,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void GeometryClustering_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	{
		VkWriteDescriptorSet    VGBWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eVGB, 0, 0, VRPG2Renderer::source.VGBs.size());
		nvvk::Buffer* VGBsPtr = VRPG2Renderer::source.VGBs.data();
		write.append(VGBWrite, VGBsPtr);
	}
	{
		VkWriteDescriptorSet    GlobalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eGlobalInfo, 0, 0, 1);
		write.append(GlobalInfoWrite, VRPG2Renderer::source.globalInfoBuffer, 0, VRPG2Renderer::source.globalInfoBuffer.bufferSize);

		GlobalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo, 0, 0, 1);
		write.append(GlobalInfoWrite, clusteringGlobalInfoBuffer, 0, clusteringGlobalInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet    OctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterData_Octree, 0, 0, VRPG2Renderer::source.clusterDataBuffer_Octree.size());
		nvvk::Buffer* octreeArraysPtr = VRPG2Renderer::source.clusterDataBuffer_Octree.data();
		write.append(OctreeWrite, octreeArraysPtr);

		OctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree, 0, 0, VRPG2Renderer::source.clusterInfoBuffer_Octree.size());
		octreeArraysPtr = VRPG2Renderer::source.clusterInfoBuffer_Octree.data();
		write.append(OctreeWrite, octreeArraysPtr);
	}
	{
		VkWriteDescriptorSet    createOctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterBlockInfos, 0, 0, 1);
		write.append(createOctreeWrite, clusterBlockInfoBuffer, 0, clusterBlockInfoBuffer.bufferSize);

		createOctreeWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eHasDataClusterBlockIndices, 0, 0, 1);
		write.append(createOctreeWrite, hasDataClusterBlockIndexBuffer, 0, hasDataClusterBlockIndexBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet    SVOWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eClusterInfo_SVO, 0, 0, 1);
		write.append(SVOWrite, VRPG2Renderer::source.clusterInfoBuffer_SVO, 0, VRPG2Renderer::source.clusterInfoBuffer_SVO.bufferSize);
	}
	{
		VkWriteDescriptorSet    createSVOWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eSVODivisibleNodeInfos, 0, 0, 2);
		nvvk::Buffer* divArraysPtr = divisibleNodeInfoBuffer.data();
		write.append(createSVOWrite, divArraysPtr);

		createSVOWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eSVOThreadGroupInfos, 0, 0, 1);
		write.append(createSVOWrite, threadGroupInfoBuffer, 0, threadGroupInfoBuffer.bufferSize);
	}
	{
		VkWriteDescriptorSet    nearbyWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eNearbyClusterTempInfos, 0, 0, 1);
		write.append(nearbyWrite, nearbyClusterTempInfoBuffer, 0, nearbyClusterTempInfoBuffer.bufferSize);

		nearbyWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::GeometryClusteringBindingPoints_VRPG::eNearbyClusterInfos, 0, 0, 1);
		write.append(nearbyWrite, VRPG2Renderer::source.nearbyClusterInfoBuffer, 0, VRPG2Renderer::source.nearbyClusterInfoBuffer.bufferSize);
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void GeometryClustering_VRPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
void GeometryClustering_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);


	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	// Slang rounds the constant-block size up to 256 bytes for this layout.
	// The declared range must cover the whole shader block, while the C++
	// pushConstant payload (sizeof) can stay at its natural size.
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = 256,
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
		shaderSource = shaderPath / "init.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_init, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_init";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_init);
		NVVK_DBG_NAME(computeShader_init);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "clustering.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_GetHasDataClusterBlockInfo, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_GetHasDataClusterBlockInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_GetHasDataClusterBlockInfo);
		NVVK_DBG_NAME(computeShader_GetHasDataClusterBlockInfo);

		vkDestroyShaderEXT(device, computeShader_DispatchClustering, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_DispatchClustering";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_DispatchClustering);
		NVVK_DBG_NAME(computeShader_DispatchClustering);

		vkDestroyShaderEXT(device, computeShader_Clustering1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_Clustering1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Clustering1);
		NVVK_DBG_NAME(computeShader_Clustering1);

		vkDestroyShaderEXT(device, computeShader_Clustering2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_Clustering2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Clustering2);
		NVVK_DBG_NAME(computeShader_Clustering2);
	}
	//---------------------------------getOctreeIndivisibleNodeLabel-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "createSVO.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_CreateSVO1, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createSVO1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateSVO1);
		NVVK_DBG_NAME(computeShader_CreateSVO1);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_CreateSVO2, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createSVO2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateSVO2);
		NVVK_DBG_NAME(computeShader_CreateSVO2);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_CreateSVO3, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createSVO3";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateSVO3);
		NVVK_DBG_NAME(computeShader_CreateSVO3);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_CreateSVO4, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createSVO4";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateSVO4);
		NVVK_DBG_NAME(computeShader_CreateSVO4);
	}
	//---------------------------------getNearbyNode-------------------------------------
	{
		shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
		shaderSource = shaderPath / "getNearbyCluster.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_dispatchGetNearbyCluster, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_dispatchGetNearbyCluster";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_dispatchGetNearbyCluster);
		NVVK_DBG_NAME(computeShader_dispatchGetNearbyCluster);

		vkDestroyShaderEXT(device, computeShader_getNearbyCluster1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyClusters1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyCluster1);
		NVVK_DBG_NAME(computeShader_getNearbyCluster1);

		vkDestroyShaderEXT(device, computeShader_getNearbyCluster2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getNearbyClusters2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getNearbyCluster2);
		NVVK_DBG_NAME(computeShader_getNearbyCluster2);
	}
	//----------------------------------------debug views----------------------------------------
#ifndef NDEBUG
	{
		shaderSource = shaderPath / "debug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_TreeDebug);
		NVVK_DBG_NAME(vertexShader_TreeDebug);

		vkDestroyShaderEXT(device, fragmentShader_TreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_TreeDebug);
		NVVK_DBG_NAME(fragmentShader_TreeDebug);
	}
#endif
}

void GeometryClustering_VRPG::initClusterOctree(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_init);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void GeometryClustering_VRPG::createClusterOctree(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	uint32_t layerBlockCount = uint32_t(pow(8, pushConstant.octreeMaxLayer)) * 6 / 8;
	for (int layerIndex = pushConstant.octreeMaxLayer; layerIndex > VRPG2Renderer::setting.geometryClusteringTopLayer; --layerIndex) {
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerBlockCount = layerBlockCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, clusteringGlobalInfoBuffer.buffer, 0, sizeof(shaderio::uint), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_GetHasDataClusterBlockInfo);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerBlockCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_DispatchClustering);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Clustering1);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

		layerBlockCount /= 8;
	}

	for (int layerIndex = VRPG2Renderer::setting.geometryClusteringTopLayer; layerIndex > 0; --layerIndex) {
		uint32_t layerNodeCount = shaderio::OctreeLayerNodeCount_VRPG2[layerIndex];
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerNodeCount = layerNodeCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Clustering2);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ Clustering_CS_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
	}
}
void GeometryClustering_VRPG::createClusterSVO(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateSVO1);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	int startLayer = 2;
	for (int layerIndex = startLayer; layerIndex <= pushConstant.octreeMaxLayer; ++layerIndex) {
		pushConstant.currentLayer = layerIndex;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateSVO2);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateSVO3);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateSVO4);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void GeometryClustering_VRPG::getNearbyClusterInfo(VkCommandBuffer cmd) {
	if (VRPG2Renderer::setting.jitterRadiusScale == 0.0f) return;
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchGetNearbyCluster);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyCluster1);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getNearbyCluster2);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
}

void GeometryClustering_VRPG::octreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eOctree), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eOctree);
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
	graphicsDynamicPipeline.rasterizationState.lineWidth = 1.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	// depth bounds is a dynamic state: it has to be set explicitly before drawing with shader objects
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t meshIndex = 0;		// wireframe cube built in init()
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	debugPushConstant.mode = 0;
	vkCmdPushConstants2(cmd, &debugPushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];
	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eOctree), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void GeometryClustering_VRPG::svoDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eSVO), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eSVO);
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
	graphicsDynamicPipeline.rasterizationState.lineWidth = 1.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	// depth bounds is a dynamic state: it has to be set explicitly before drawing with shader objects
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_TreeDebug, .fragment = fragmentShader_TreeDebug });

	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t meshIndex = 0;		// wireframe cube built in init()
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];
	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	// normalIndex == 6 means "all six normal partitions"
	const int selectedNormal = debugPushConstant.normalIndex_G_2;
	const int normalPartitionCount = 6;
	const int firstNormal = selectedNormal == normalPartitionCount ? 0 : selectedNormal;
	const int lastNormal = selectedNormal == normalPartitionCount ? normalPartitionCount : selectedNormal + 1;

	// curLayer_G_2 == octreeMaxLayer + 1 means "all layers"
	const int selectedLayer = debugPushConstant.curLayer_G_2;
	const int selectedLayerNodeCount = debugPushConstant.curLayerNodeCount_G_2;
	const int maxLayer = int(pushConstant.octreeMaxLayer);
	const int firstLayer = selectedLayer > maxLayer ? int(VRPG2Renderer::setting.geometryClusteringTopLayer) : selectedLayer;
	const int lastLayer = selectedLayer > maxLayer ? maxLayer + 1 : selectedLayer + 1;

	for (int layer = firstLayer; layer < lastLayer; ++layer) {
		debugPushConstant.curLayer_G_2 = layer;
		debugPushConstant.curLayerNodeCount_G_2 = (1 << (3 * layer));
		const uint32_t instanceCount = debugPushConstant.curLayerNodeCount_G_2;
		for (int normalIndex = firstNormal; normalIndex < lastNormal; ++normalIndex) {
			debugPushConstant.normalIndex_G_2 = normalIndex;
			debugPushConstant.mode = 2;
			vkCmdPushConstants2(cmd, &debugPushInfo);
			vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);
		}
	}
	debugPushConstant.normalIndex_G_2 = selectedNormal;
	debugPushConstant.curLayer_G_2 = selectedLayer;
	debugPushConstant.curLayerNodeCount_G_2 = selectedLayerNodeCount;

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::GBuffers_GeometryClustering_VRPG::eSVO), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}