#include "../../../renderer/VRPG2/VRPG2.h"
#include "./PathGuiding.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

using namespace FzbRenderer;

PathGuiding_VRPG::PathGuiding_VRPG(pugi::xml_node& featureNode) {}

void PathGuiding_VRPG::init(PathGuidingCreateInfo_VRPG2 createInfo) {
	gBuffers = createInfo.gBuffers;
	asManager = createInfo.asManager;

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();
}

void PathGuiding_VRPG::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_pathGuiding, nullptr);
	staticDescPack.deinit();
	dynamicDescPack.deinit();
	vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
}

void PathGuiding_VRPG::uiRender() {}

void PathGuiding_VRPG::preRender() {}

void PathGuiding_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.maxFrameCount = VRPG2Renderer::setting.maxFrameCount;
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = (VRPG_RANDOM_FRAME_SOURCE != 0)
		? VRPG2Renderer::setting.randomFrameIndex
		: VRPG2Renderer::setting.frameIndex;
	pushConstant.staticScene = VRPG2Renderer::setting.staticScene ? 1 : 0;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.maxDepth = VRPG2Renderer::setting.maxBounces;
	pushConstant.spp = VRPG2Renderer::setting.spp;
	pushConstant.maxOctreeLayer = VRPG2Renderer::setting.geometryOctreeDepth;
	// the light tree is built on the coarse light voxels (VRPG's light VGB), not on the geometry VGB
	uint32_t lightVoxelCount = 1u << VRPG2Renderer::setting.lightOctreeDepth;
	glm::vec3 sceneExtent = VRPG2Renderer::setting.voxelSize * float(VRPG2Renderer::setting.voxelGridResolution);
	glm::vec3 lightVoxelSize = sceneExtent / float(lightVoxelCount);
	pushConstant.lightMaxOctreeLayer = int(VRPG2Renderer::setting.lightOctreeDepth);
	pushConstant.VGBVoxelSize = VRPG2Renderer::setting.voxelSize;
	pushConstant.VGBStartPos_Size = shaderio::float4(VRPG2Renderer::setting.voxelGridStartPos, VRPG2Renderer::setting.voxelGridResolution);
	pushConstant.lightVGBStartPos_Size = shaderio::float4(VRPG2Renderer::setting.voxelGridStartPos, float(lightVoxelCount));
	pushConstant.lightVGBVoxelSize = shaderio::float4(lightVoxelSize, 1.0f);
	pushConstant.jitterRadiusScale = VRPG2Renderer::setting.jitterRadiusScale;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	float angle = FzbRenderer::rand(pushConstant.randomFrameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));

	VkExtent2D sceneSize = Application::renderResolution;
	pushConstant.sceneSize = shaderio::uint2(sceneSize.width, sceneSize.height);
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y });
	pushConstant.threadGroupCount = shaderio::uint2(groupSize.width, groupSize.height);
	pushConstant.dispatchOffset = shaderio::uint2(0, 0);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::PathGuidingPushConstant_VRPG2),
		.pValues = &pushConstant,
	};
}

void PathGuiding_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "PathGuiding_render");

	updateDataPerFrame(cmd);

	// bind the shader first: it establishes the pipeline layout that the push descriptor set targets
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_pathGuiding);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	const VkExtent2D sceneSize = Application::renderResolution;
	const VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y });

	// long paths exceed the GPU watchdog in one dispatch, split the image into horizontal bands
	const uint32_t bandCount = pushConstant.maxDepth >= 8 ? 4u : 1u;
	const uint32_t bandGroupHeight = (groupSize.height + bandCount - 1u) / bandCount;
	for (uint32_t band = 0; band < bandCount; ++band) {
		const uint32_t groupY = band * bandGroupHeight;
		if (groupY >= groupSize.height) break;
		const uint32_t dispatchGroupHeight = std::min(bandGroupHeight, groupSize.height - groupY);

		pushConstant.dispatchOffset = shaderio::uint2(0, groupY * FZB_PATHGUIDING_THREADGROUP_SIZE_Y);
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDispatch(cmd, groupSize.width, dispatchGroupHeight, 1);
	}
}

void PathGuiding_VRPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eClusterInfo_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterInfoBuffer_Octree.size(), .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eLightNodeData_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.lightNodeDataBuffer_Octree.size(), .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSubTreeInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eCandidateLeafNodeData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSampleData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSamplePrefixWSum,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = (uint32_t)shaderio::PathGuidingBindingPoints_VRPG2::eClusterData_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)VRPG2Renderer::source.clusterDataBuffer_Octree.size(), .stageFlags = VK_SHADER_STAGE_ALL });

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

void PathGuiding_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet allTextures = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0,
			uint32_t(Application::sceneResource.textures.size()));
		write.append(allTextures, Application::sceneResource.textures.data());
	}
	VkWriteDescriptorSet outImageWrite = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(outImageWrite, gBuffers->getColorImageView(VRPG2Renderer::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	auto writeArray = [&](shaderio::PathGuidingBindingPoints_VRPG2 binding, std::vector<nvvk::Buffer>& buffers) {
		VkWriteDescriptorSet w = staticDescPack.makeWrite((uint32_t)binding, 0, 0, (uint32_t)buffers.size());
		write.append(w, buffers.data());
	};
	auto writeOne = [&](shaderio::PathGuidingBindingPoints_VRPG2 binding, nvvk::Buffer& buffer) {
		VkWriteDescriptorSet w = staticDescPack.makeWrite((uint32_t)binding, 0, 0, 1);
		write.append(w, buffer, 0, buffer.bufferSize);
	};

	writeArray(shaderio::PathGuidingBindingPoints_VRPG2::eClusterInfo_Octree, VRPG2Renderer::source.clusterInfoBuffer_Octree);
	writeArray(shaderio::PathGuidingBindingPoints_VRPG2::eLightNodeData_Octree, VRPG2Renderer::source.lightNodeDataBuffer_Octree);
	writeArray(shaderio::PathGuidingBindingPoints_VRPG2::eClusterData_Octree, VRPG2Renderer::source.clusterDataBuffer_Octree);
	writeOne(shaderio::PathGuidingBindingPoints_VRPG2::eGlobalInfo, VRPG2Renderer::source.globalInfoBuffer);
	writeOne(shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSubTreeInfo, VRPG2Renderer::source.candidateSubTreeInfoBuffer);
	writeOne(shaderio::PathGuidingBindingPoints_VRPG2::eCandidateLeafNodeData, VRPG2Renderer::source.candidateLeafNodeDataBuffer);
	writeOne(shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSampleData, VRPG2Renderer::source.candidateSampleDataBuffer);
	writeOne(shaderio::PathGuidingBindingPoints_VRPG2::eCandidateSamplePrefixWSum, VRPG2Renderer::source.candidateSamplePrefixWSumBuffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void PathGuiding_VRPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PathGuidingPushConstant_VRPG2) };
	// set 0 = static buffers, set 1 = the push descriptor set holding the TLAS (same as VRPG)
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

void PathGuiding_VRPG::compileAndCreateShaders() {
	const std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	const std::filesystem::path shaderSource = shaderPath / "pathGuiding.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::PathGuidingPushConstant_VRPG2) };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "computeMain_VRPG",
		.setLayoutCount = (uint32_t)layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_pathGuiding, nullptr);
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_pathGuiding);
	NVVK_DBG_NAME(computeShader_pathGuiding);
}
