#include "../../../renderer/VRPG2/VRPG2.h"
#include "./RIS.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <cstdio>
#include <cstring>
#include <cmath>

using namespace FzbRenderer;

namespace {
constexpr uint32_t kClusterMaxCount = ClusterMaxCount_VRPG;
constexpr uint32_t kCandidateSamples = Candidate_Samples_Count;
constexpr uint32_t kOutgoingCount = OUTGOING_COUNT_VRPG2;
constexpr uint32_t kHitTestCount = HITTEST_COUNT_VRPG2;
}

RIS_VRPG::RIS_VRPG(pugi::xml_node& featureNode) {}

void RIS_VRPG::init() {
	createSource();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	// Debug render target of clusterPairDebug (VRPG: Octree_VRPG::init).
	Feature::createGBuffer(true, true, (uint32_t)GBuffers_RIS_VRPG2::eBufferCount);
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
void RIS_VRPG::clean() {
	// Feature::clean release the descriptor packs, the pipeline layout, the debug GBuffer and the
	// local scene (wireframe cube) created by init().
	Feature::clean();

	Application::allocator.destroyBuffer(RISGlobalInfoBuffer);
	Application::allocator.destroyBuffer(clusterPairDebugGlobalInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_init, nullptr);

	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_2, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_3, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_4, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodes_5, nullptr);

	vkDestroyShaderEXT(device, computeShader_getCandidateNodes, nullptr);

	vkDestroyShaderEXT(device, computeShader_dispatchGetCandidateLeafNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateLeafNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeight, nullptr);
	vkDestroyShaderEXT(device, computeShader_createCandidatePrefixWSum, nullptr);

	vkDestroyShaderEXT(device, computeShader_GetDebugClusterG, nullptr);
	vkDestroyShaderEXT(device, vertexShader_ClusterPairDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_ClusterPairDebug, nullptr);
}
void RIS_VRPG::uiRender() {
#ifndef NDEBUG
	namespace PE = nvgui::PropertyEditor;

	std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward", "all"};
	std::vector<const char*> normalIndexNames_pointers;
	for (const auto& normalIndexName : normalIndexNames) normalIndexNames_pointers.push_back(normalIndexName.c_str());

	if (ImGui::CollapsingHeader("RIS")) {
		ImGui::TextDisabled("Sample mode: %d (compile time)", VRPG_Candidate_Samples_SampleMode);
		ImGui::TextDisabled("M = %d, K = %d, S = %d", kCandidateSamples, kOutgoingCount, kHitTestCount);

		if (ImGui::CollapsingHeader("Candidate Pool Debug", ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Indent(20.0f);

			std::vector<const char*> showModeNames = { "StochasticLightCut LeafNode", "Candidate Node Pool", "Iteration Result"};
			ImGui::Combo("Cluster Struct", &debugPushConstant2.mode2, showModeNames.data(), static_cast<int>(showModeNames.size()));

			ImGui::DragInt("OutogingIndex", (int*)&debugPushConstant2.outgoingIndex, 0, OUTGOING_COUNT_VRPG2, 1);

			ImGui::Checkbox("Show Nearby Clusters", &showNearbyClusters);

			ImGui::DragInt("iteration: ", (int*)&debugPushConstant2.iteration, 1, 0, 5);
			ImGui::Combo("normal", &debugPushConstant2.normalIndex,
				normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));
			ImGui::Checkbox("Show RIS Result", &showClusterPairDebugMap);
			if (showClusterPairDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_RIS_VRPG2::eClusterPairResult);

			ImGui::Unindent(20.0f);
		}
	}

	{
		const ImVec2 mousePos = ImGui::GetMousePos();
		const ImVec2 vpOrigin = Application::viewportScreenPos;
		const ImVec2 vpSize = Application::viewportContentSize;

		if (changeClusterG = ImGui::IsMouseClicked(ImGuiMouseButton_Right)
			&& vpSize.x > 0.0f && vpSize.y > 0.0f
			&& mousePos.x >= vpOrigin.x && mousePos.x < vpOrigin.x + vpSize.x
			&& mousePos.y >= vpOrigin.y && mousePos.y < vpOrigin.y + vpSize.y)
		{
			// 1) viewport-relative mouse position
			const float localX = mousePos.x - vpOrigin.x;
			const float localY = mousePos.y - vpOrigin.y;

			// 2) viewport pixel -> debug render target pixel
			const VkExtent2D renderSize = gBuffers.getSize();
			const float imageX = localX / vpSize.x * float(renderSize.width);
			const float imageY = localY / vpSize.y * float(renderSize.height);

			LOGI("[RightClick] screen = (%.1f, %.1f)  viewportLocal = (%.1f, %.1f)  imagePixel = (%.1f, %.1f)  size = %ux%u\n",
				mousePos.x, mousePos.y, localX, localY, imageX, imageY, renderSize.width, renderSize.height);

			debugPushConstant2.pixelIndex = shaderio::uint2(imageX, imageY);
		}
	}

	if (const char* autoPick = std::getenv("VRPG2_AUTO_PICK")) {
		int x = 0, y = 0;
		if (std::sscanf(autoPick, "%d,%d", &x, &y) == 2) {
			changeClusterG = true;
			debugPushConstant2.pixelIndex = shaderio::uint2(x, y);
		}
	}
	if (const char* debugCluster = std::getenv("VRPG2_DEBUG_CLUSTER")) {
		int cluster = -1;
		if (std::sscanf(debugCluster, "%d", &cluster) == 1 && cluster >= 0) {
			debugClusterIndex = cluster;
			changeClusterG = false;
		}
	}
#endif
}
void RIS_VRPG::preRender() {}
void RIS_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
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

	debugPushConstant2.pixelIndex = { size.width / 2, size.height / 2 };
#endif
}
void RIS_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = VRPG2Renderer::setting.randomFrameIndex;
	pushConstant.candidateSampleBufferIndex = 0;
	pushConstant.lightOctreeMaxLayer = uint32_t(VRPG2Renderer::setting.lightOctreeDepth);
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.iterationIndex = 0;
	pushConstant.globalInfoAddress = (shaderio::ClusterPairDebugGlobalInfo_VRPG2*)clusterPairDebugGlobalInfoBuffer.address;

	float angle = FzbRenderer::rand(pushConstant.randomFrameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::RISPushConstant_VRPG2),
		.pValues = &pushConstant,
	};
}
void RIS_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "RIS_render");

	updateDataPerFrame(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), VRPG2Renderer::source.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_init);
	uint32_t initThreadCount = kOutgoingCount * kClusterMaxCount * kCandidateSamples;
	vkCmdDispatch(cmd, (initThreadCount + 1023) / 1024, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

#if VRPG_Candidate_Samples_SampleMode == 1
	for (int pass = 0; pass < 4; ++pass) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_1);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		++pushConstant.iterationIndex;
		vkCmdPushConstants2(cmd, &pushInfo);
	}

	for (int pass = 0; pass < 2; ++pass) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_2);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_3);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_4);
		vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdFillBuffer(cmd, RISGlobalInfoBuffer.buffer, 0, sizeof(shaderio::RISGlobalInfo_VRPG2), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		++pushConstant.iterationIndex;
		vkCmdPushConstants2(cmd, &pushInfo);
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodes_5);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
#else
	// 2. greedy light cut driven by the relative error threshold
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_selectCandidates);
	vkCmdDispatch(cmd, (kClusterMaxCount + RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
#endif

	// 3. cut node -> light leaf, 8 lanes per candidate pair (VRPG: dispatchGetCandidateLeafNodes sets cmd)
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchGetCandidateLeafNodes);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateLeafNodes);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	// 4. visibility samples (only this pass traces rays, so it re-pushes the TLAS for its layout)
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clusterPairHitTest);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	// 5. target density and resampling weights
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodeWeight);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	// 6. prefix sum over the M candidates of every (outgoing, cluster)
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createCandidatePrefixWSum);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
}
void RIS_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	clusterPairDebug(cmd);
#endif
}

void RIS_VRPG::createSource() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();
	const VkBufferUsageFlags usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;

	VkDeviceSize size = VkDeviceSize(kClusterMaxCount) * kCandidateSamples * sizeof(shaderio::ClusterPairInfo_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.clusterPairInfoBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.clusterPairInfoBuffer.buffer);

	size = VkDeviceSize(kClusterMaxCount) * sizeof(shaderio::CandidatePoolInfo_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.candidatePoolInfoBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidatePoolInfoBuffer.buffer);

#if VRPG_Candidate_Samples_SampleMode == 1
	size = VkDeviceSize(kClusterMaxCount) * kCandidateSamples * sizeof(shaderio::CandidatePoolNodeInfo_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.candidatePoolNodeInfoBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidatePoolNodeInfoBuffer.buffer);

	// VRPG: ClusterPairGlobalInfo_VRPG, written by the compacting candidate passes
	size = sizeof(shaderio::RISGlobalInfo_VRPG2);
	allocator->createBuffer(RISGlobalInfoBuffer, size, usage | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT);
	NVVK_DBG_NAME(RISGlobalInfoBuffer.buffer);
#endif

	size = VkDeviceSize(kClusterMaxCount) * kCandidateSamples * kHitTestCount * sizeof(shaderio::ClusterPairHitInfo_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.clusterPairHitInfoBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.clusterPairHitInfoBuffer.buffer);

	size = VkDeviceSize(kClusterMaxCount) * kCandidateSamples * sizeof(shaderio::CandidateSubTreeInfo_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.candidateSubTreeInfoBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidateSubTreeInfoBuffer.buffer);

	size = VkDeviceSize(kClusterMaxCount) * kCandidateSamples * sizeof(shaderio::CandidateLeafNodeData_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.candidateLeafNodeDataBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidateLeafNodeDataBuffer.buffer);

	size = VkDeviceSize(kOutgoingCount) * kClusterMaxCount * kCandidateSamples * sizeof(shaderio::CandidateSampleData_VRPG2);
	allocator->createBuffer(VRPG2Renderer::source.candidateSampleDataBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidateSampleDataBuffer.buffer);

	size = VkDeviceSize(kOutgoingCount) * kClusterMaxCount * kCandidateSamples * sizeof(float);
	allocator->createBuffer(VRPG2Renderer::source.candidateSamplePrefixWSumBuffer, size, usage);
	NVVK_DBG_NAME(VRPG2Renderer::source.candidateSamplePrefixWSumBuffer.buffer);

#ifndef NDEBUG
	// VRPG: Octree_VRPG::createOctreeArray, the geometric cluster picked by the debug compute pass.
	size = sizeof(shaderio::ClusterPairDebugGlobalInfo_VRPG2);
	allocator->createBuffer(clusterPairDebugGlobalInfoBuffer, size, usage);
	NVVK_DBG_NAME(clusterPairDebugGlobalInfoBuffer.buffer);

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	vkCmdFillBuffer(cmd, clusterPairDebugGlobalInfoBuffer.buffer, 0, sizeof(shaderio::ClusterPairDebugGlobalInfo_VRPG2), 0);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
#endif
}
void RIS_VRPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	// the weight shader evaluates the BSDF, which samples the scene textures (set 0, binding 0)
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eClusterData_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterDataBuffer_Octree.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eClusterInfo_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterInfoBuffer_Octree.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eLightNodeData_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.lightNodeDataBuffer_Octree.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eClusterInfo_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eClusterPairInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eRISGlobalInfo,
						  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						  .descriptorCount = 1,
						  .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eClusterPairHitInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidatePoolInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
#if VRPG_Candidate_Samples_SampleMode == 1
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidatePoolNodeInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
#endif
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidateSubTreeInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidateLeafNodeData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidateSampleData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eCandidateSamplePrefixWSum,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::RISBindingPoints_VRPG2::eNearbyClusterInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));

	bindings.clear();
	bindings.addBinding({ .binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
		.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
}
void RIS_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet allTextures = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0,
			uint32_t(Application::sceneResource.textures.size()));
		write.append(allTextures, Application::sceneResource.textures.data());
	}

	auto writeArray = [&](shaderio::RISBindingPoints_VRPG2 binding, std::vector<nvvk::Buffer>& buffers) {
		VkWriteDescriptorSet w = staticDescPack.makeWrite((uint32_t)binding, 0, 0, (uint32_t)buffers.size());
		write.append(w, buffers.data());
	};
	auto writeOne = [&](shaderio::RISBindingPoints_VRPG2 binding, nvvk::Buffer& buffer) {
		VkWriteDescriptorSet w = staticDescPack.makeWrite((uint32_t)binding, 0, 0, 1);
		write.append(w, buffer, 0, buffer.bufferSize);
	};

	writeArray(shaderio::RISBindingPoints_VRPG2::eClusterData_Octree, VRPG2Renderer::source.clusterDataBuffer_Octree);
	writeArray(shaderio::RISBindingPoints_VRPG2::eClusterInfo_Octree, VRPG2Renderer::source.clusterInfoBuffer_Octree);
	writeArray(shaderio::RISBindingPoints_VRPG2::eLightNodeData_Octree, VRPG2Renderer::source.lightNodeDataBuffer_Octree);
	writeOne(shaderio::RISBindingPoints_VRPG2::eClusterInfo_SVO, VRPG2Renderer::source.clusterInfoBuffer_SVO);
	writeOne(shaderio::RISBindingPoints_VRPG2::eGlobalInfo, VRPG2Renderer::source.globalInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eRISGlobalInfo, RISGlobalInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eClusterPairInfo, VRPG2Renderer::source.clusterPairInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eClusterPairHitInfo, VRPG2Renderer::source.clusterPairHitInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidatePoolInfo, VRPG2Renderer::source.candidatePoolInfoBuffer);
#if VRPG_Candidate_Samples_SampleMode == 1
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidatePoolNodeInfo, VRPG2Renderer::source.candidatePoolNodeInfoBuffer);
#endif
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidateSubTreeInfo, VRPG2Renderer::source.candidateSubTreeInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidateLeafNodeData, VRPG2Renderer::source.candidateLeafNodeDataBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidateSampleData, VRPG2Renderer::source.candidateSampleDataBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eCandidateSamplePrefixWSum, VRPG2Renderer::source.candidateSamplePrefixWSumBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eNearbyClusterInfos, VRPG2Renderer::source.nearbyClusterInfoBuffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void RIS_VRPG::createPipeline() {
	// Slang rounds the constant-block size up to 256 bytes for this layout, and the cluster pair
	// debug graphics pass uses the full 256 byte block (VRPG: Octree_VRPG::createPipeline).
	// The declared range must cover the whole shader block, while the C++ pushConstant payload
	// (sizeof) can stay at its natural size.
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 256 };
	// set 0 = static buffers, set 1 = the push descriptor set holding the TLAS (hit tests trace rays)
	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = (uint32_t)layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void RIS_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	const std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	// Slang rounds the constant-block size up to 256 bytes for this layout, and the cluster pair
	// debug pass uses the full 256 byte block, so the declared range covers the whole block.
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = 256 };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = (uint32_t)layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();

	auto create = [&](const char* entry, VkShaderEXT& shader, std::filesystem::path source) {
		VkShaderModuleCreateInfo code = FzbRenderer::compileSlangShader(source, {});

		vkDestroyShaderEXT(device, shader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = entry;
		shaderInfo.codeSize = code.codeSize;
		shaderInfo.pCode = code.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader);
		NVVK_DBG_NAME(shader);
	};

	std::filesystem::path initSource = shaderPath / "init.slang";
	std::filesystem::path candidateSource = shaderPath / "getCandidateNodes.slang";
	std::filesystem::path weightSource = shaderPath / "getResampleWeights.slang";

	create("computeMain_init", computeShader_init, initSource);

#if VRPG_Candidate_Samples_SampleMode == 1
	create("computeMain_getCandidateNodes_1", computeShader_getCandidateNodes_1, candidateSource);
	create("computeMain_getCandidateNodes_2", computeShader_getCandidateNodes_2, candidateSource);
	create("computeMain_getCandidateNodes_3", computeShader_getCandidateNodes_3, candidateSource);
	create("computeMain_getCandidateNodes_4", computeShader_getCandidateNodes_4, candidateSource);
	create("computeMain_getCandidateNodes_5", computeShader_getCandidateNodes_5, candidateSource);
#else
	create("computeMain_getCandidateNodes", computeShader_getCandidateNodes, candidateSource);
#endif
	create("computeMain_dispatchGetCandidateLeafNodes", computeShader_dispatchGetCandidateLeafNodes, weightSource);
	create("computeMain_getCandidateLeafNodes", computeShader_getCandidateLeafNodes, weightSource);
	create("computeMain_clusterPairHitTest", computeShader_clusterPairHitTest, weightSource);
	create("computeMain_getCandidateNodeWeight", computeShader_getCandidateNodeWeight, weightSource);
	create("computeMain_createCandidatePrefixWSum", computeShader_createCandidatePrefixWSum, weightSource);

	//---------------------------------Debug-------------------------------------
	// VRPG: Octree_VRPG::compileAndCreateShaders, Debug_ClusterPair.slang
	{
		std::filesystem::path debugSource = shaderPath / "debug.slang";
		VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(debugSource, {});

		vkDestroyShaderEXT(device, computeShader_GetDebugClusterG, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_GetDebugClusterG";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_GetDebugClusterG);
		NVVK_DBG_NAME(computeShader_GetDebugClusterG);

		vkDestroyShaderEXT(device, vertexShader_ClusterPairDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_ClusterPairDebug);
		NVVK_DBG_NAME(vertexShader_ClusterPairDebug);

		vkDestroyShaderEXT(device, fragmentShader_ClusterPairDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_ClusterPairDebug);
		NVVK_DBG_NAME(fragmentShader_ClusterPairDebug);
	}
}

void RIS_VRPG::clusterPairDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	{
		debugPushConstant2.frameIndex = Application::frameIndex;
		debugPushConstant2.VGBVoxelTotalCount = uint32_t(std::pow(VRPG2Renderer::setting.voxelGridResolution, 3));
		debugPushConstant2.VGBStartPos_Size = glm::vec4(VRPG2Renderer::setting.voxelGridStartPos, VRPG2Renderer::setting.voxelGridResolution);
		debugPushConstant2.VGBVoxelSize = glm::vec4(VRPG2Renderer::setting.voxelSize, 1.0f);
		debugPushConstant2.octreeMaxLayer = uint32_t(VRPG2Renderer::setting.geometryOctreeDepth);
		debugPushConstant2.spatialReuse = VRPG_SPATIAL_REUSE;
		debugPushConstant2.screenSize = { gBuffers.getSize().width, gBuffers.getSize().height };
		debugPushConstant2.sceneInfoAddress = pushConstant.sceneInfoAddress;
		debugPushConstant2.globalInfoAddress = (shaderio::ClusterPairDebugGlobalInfo_VRPG2*)clusterPairDebugGlobalInfoBuffer.address;

		debugPushInfo = {
			.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
			.layout = pipelineLayout,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = sizeof(shaderio::ClusterPairDebugPushConstant_VRPG2),
			.pValues = &debugPushConstant2,
		};
		vkCmdPushConstants2(cmd, &debugPushInfo);
	}

	if (debugClusterIndex >= 0) {
		uint32_t clusterIndex = uint32_t(debugClusterIndex);
		vkCmdUpdateBuffer(cmd, clusterPairDebugGlobalInfoBuffer.buffer, 0, sizeof(clusterIndex), &clusterIndex);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	}

	//get clusterIndex_G by mouse
	if (changeClusterG) {
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		nvvk::WriteSetContainer write{};
		write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), VRPG2Renderer::source.asManager->asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

		vkCmdPushConstants2(cmd, &debugPushInfo);
		VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_GetDebugClusterG);
		vkCmdDispatch(cmd, 1, 1, 1);
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	shaderio::TriangleMesh triMesh;
	{
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_RIS_VRPG2::eClusterPairResult)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

		VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBuffers_RIS_VRPG2::eClusterPairResult));
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
		graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
		graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;

		graphicsDynamicPipeline.cmdApplyAllStates(cmd);
		// depth bounds is a dynamic state: it has to be set explicitly before drawing with shader objects
		vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
		graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
		graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_ClusterPairDebug, .fragment = fragmentShader_ClusterPairDebug });

		VkVertexInputBindingDescription2EXT bindingDescription{};
		VkVertexInputAttributeDescription2EXT attributeDescription = {};
		vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

		bool useWireframe = true;
		uint32_t meshIndex = useWireframe ? 0 : 1;
		const shaderio::Mesh& mesh = scene.meshes[meshIndex];
		triMesh = mesh.triMesh;

		uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = scene.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	}

	{
		debugPushConstant2.mode = 0;
		vkCmdPushConstants2(cmd, &debugPushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	if (showNearbyClusters) {
		debugPushConstant2.mode = 3;
		vkCmdPushConstants2(cmd, &debugPushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, NEARBY_NODE_COUNT_VRPG + 1, 0, 0, 0);
	}

	{
		debugPushConstant2.mode = 1;
		vkCmdPushConstants2(cmd, &debugPushInfo);
		vkCmdDraw(cmd, 2, 1, 0, 0);
	}

	{
		debugPushConstant2.mode = 2;
		vkCmdPushConstants2(cmd, &debugPushInfo);

		vkCmdDrawIndexed(cmd, triMesh.indices.count, Candidate_Samples_Count, 0, 0, 0);
	}

	{
		vkCmdEndRendering(cmd);

		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBuffers_RIS_VRPG2::eClusterPairResult)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	}
}


