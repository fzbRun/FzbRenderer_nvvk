#include "../../../renderer/VRPG2/VRPG2.h"
#include "./RIS.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <cstdio>
#include <cstring>

using namespace FzbRenderer;

namespace {
constexpr uint32_t kClusterMaxCount = ClusterMaxCount_VRPG;
constexpr uint32_t kCandidateSamples = Candidate_Samples_Count_VRPG2;
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
}

void RIS_VRPG::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initCandidateNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_selectCandidates, nullptr);
	vkDestroyShaderEXT(device, computeShader_emitClusterPairs, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateLeafNodes, nullptr);
	vkDestroyShaderEXT(device, computeShader_clusterPairHitTest, nullptr);
	vkDestroyShaderEXT(device, computeShader_getCandidateNodeWeight, nullptr);
	vkDestroyShaderEXT(device, computeShader_createCandidatePrefixWSum, nullptr);

	staticDescPack.deinit();
	dynamicDescPack.deinit();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

void RIS_VRPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;
	if (ImGui::CollapsingHeader("RIS")) {
		ImGui::TextDisabled("Sample mode: %d (compile time)", VRPG_Candidate_Samples_SampleMode);
		ImGui::TextDisabled("M = %d, K = %d, S = %d", kCandidateSamples, kOutgoingCount, kHitTestCount);
	}
#endif
}

void RIS_VRPG::preRender() {}

void RIS_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = VRPG2Renderer::setting.randomFrameIndex;
	pushConstant.candidateSampleBufferIndex = 0;
	pushConstant.lightOctreeMaxLayer = uint32_t(VRPG2Renderer::setting.lightOctreeDepth);
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

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

	// With shader objects the pipeline layout is established by binding the shader, so the shader
	// has to be bound before the push descriptor set: vkCmdPushDescriptorSetKHR targets whatever
	// layout is currently bound (the previous module only has one set, which would drop the TLAS).
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initCandidateNodes);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), VRPG2Renderer::source.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	vkCmdPushConstants2(cmd, &pushInfo);

	const uint32_t pairThreadCount = kClusterMaxCount * kCandidateSamples;

	// 1. reset the candidate tables
	vkCmdDispatch(cmd, (kClusterMaxCount + 255) / 256, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

#if VRPG_Candidate_Samples_SampleMode == 1
	// 2. bounded pool light cut: a fixed number of expansion passes, then emit the cut
	for (int pass = 0; pass < 6; ++pass) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_selectCandidates);
		vkCmdDispatch(cmd, (kClusterMaxCount + RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_emitClusterPairs);
	vkCmdDispatch(cmd, (kClusterMaxCount + RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#else
	// 2. greedy light cut driven by the relative error threshold
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_selectCandidates);
	vkCmdDispatch(cmd, (kClusterMaxCount + RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
#endif

	// 3. cut node -> light leaf
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateLeafNodes);
	vkCmdDispatch(cmd, (pairThreadCount + RIS_GET_CANDIDATE_LEAF_NODES_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_LEAF_NODES_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// 4. visibility samples (only this pass traces rays, so it re-pushes the TLAS for its layout)
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clusterPairHitTest);
	{
		nvvk::WriteSetContainer tlasWrite{};
		tlasWrite.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), VRPG2Renderer::source.asManager->asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, tlasWrite.size(), tlasWrite.data());
	}
	vkCmdDispatch(cmd, (pairThreadCount + RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE - 1) / RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// 5. target density and resampling weights
	const uint32_t weightThreadCount = pairThreadCount * kOutgoingCount;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getCandidateNodeWeight);
	vkCmdDispatch(cmd, (weightThreadCount + RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE - 1) / RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// 6. prefix sum over the M candidates of every (outgoing, cluster)
	const uint32_t prefixThreadCount = kOutgoingCount * kClusterMaxCount;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createCandidatePrefixWSum);
	vkCmdDispatch(cmd, (prefixThreadCount + RIS_PREFIX_WSUM_CS_THREADGROUP_SIZE - 1) / RIS_PREFIX_WSUM_CS_THREADGROUP_SIZE, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
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
	writeOne(shaderio::RISBindingPoints_VRPG2::eGlobalInfo, VRPG2Renderer::source.globalInfoBuffer);
	writeOne(shaderio::RISBindingPoints_VRPG2::eClusterInfo_SVO, VRPG2Renderer::source.clusterInfoBuffer_SVO);
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

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void RIS_VRPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::RISPushConstant_VRPG2) };
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
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::RISPushConstant_VRPG2) };
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

	std::filesystem::path candidateSource = shaderPath / "getCandidateNodes.slang";
	std::filesystem::path weightSource = shaderPath / "getResampleWeights.slang";

	create("computeMain_initCandidatePool", computeShader_initCandidateNodes, candidateSource);
#if VRPG_Candidate_Samples_SampleMode == 1
	create("computeMain_expandCandidatePool", computeShader_selectCandidates, candidateSource);
	create("computeMain_emitClusterPairs", computeShader_emitClusterPairs, candidateSource);
#else
	create("computeMain_lightCut", computeShader_selectCandidates, candidateSource);
#endif
	create("computeMain_getCandidateLeafNodes", computeShader_getCandidateLeafNodes, weightSource);
	create("computeMain_clusterPairHitTest", computeShader_clusterPairHitTest, weightSource);
	create("computeMain_getCandidateNodeWeight", computeShader_getCandidateNodeWeight, weightSource);
	create("computeMain_createCandidatePrefixWSum", computeShader_createCandidatePrefixWSum, weightSource);

	dirty = false;
}


