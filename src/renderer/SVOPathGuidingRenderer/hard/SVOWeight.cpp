#include "SVOWeight.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

SVOWeight::SVOWeight(pugi::xml_node& featureNode) {
	rayqueryFeature.rayQuery = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_RAY_QUERY_EXTENSION_NAME, &rayqueryFeature });
}
void SVOWeight::init(SVOWeightSetting setting) {
	this->setting = setting;

	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::SVOWeightPushConstant));
	compileAndCreateShaders();
}
void SVOWeight::clean() {
	VkDevice device = Application::app->getDevice();

	Application::allocator.destroyBuffer(CSDispatchCommandBuffer);
	Application::allocator.destroyBuffer(indivisibleNodeInfosBuffer_G);
	Application::allocator.destroyBuffer(weightSampleCountsBuffer);
	Application::allocator.destroyBuffer(weightBuffer);
	Application::allocator.destroyBuffer(weightSumsBuffer);

	vkDestroyShaderEXT(device, computeShader_getIndivisibleNode_G, nullptr);
	vkDestroyShaderEXT(device, computeShader_initWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getWeights, nullptr);
	vkDestroyShaderEXT(device, computeShader_getProbability, nullptr);
}
void SVOWeight::uiRender() {};
void SVOWeight::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
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
		.size = sizeof(shaderio::SVOPushConstant),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	getIndivisibleNode_G(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

}
void SVOWeight::postProcess(VkCommandBuffer cmd) {};

void SVOWeight::createWeightArray() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	allocator->createBuffer(CSDispatchCommandBuffer, sizeof(VkDispatchIndirectCommand),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(CSDispatchCommandBuffer);

	//由于我们事先不知道具体SVO聚类后有几层，因此我们按最大层来创建buffer，即octree的层数
	uint32_t OctreeDepth = setting.svo->setting.octree->setting.OctreeDepth;
	uint32_t SVONodeMaxCount = 0;
	for (int i = 1; i < OctreeDepth; ++i)
		SVONodeMaxCount += setting.svo->SVOInitialSize[i];

	uint32_t bufferSize = SVONodeMaxCount * sizeof(shaderio::SVOIndivisibleNodeInfo);
	allocator->createBuffer(indivisibleNodeInfosBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_G);

	uint32_t weightCount = OUTGOING_COUNT * SVONodeMaxCount * SVONodeMaxCount;
	bufferSize = weightCount * sizeof(uint32_t);
	allocator->createBuffer(weightSampleCountsBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightSampleCountsBuffer);

	bufferSize = weightCount * sizeof(float);
	allocator->createBuffer(weightBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightBuffer);

	weightCount = OUTGOING_COUNT * SVONodeMaxCount * setting.svo->SVOInitialSize[OctreeDepth - 1];
	bufferSize = weightCount * sizeof(float);
	allocator->createBuffer(weightSumsBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(weightSumsBuffer);

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
		.binding = shaderio::StaticBindingPoints_SVOWeight::eDispatchIndirect_SVOWeight,
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

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void SVOWeight::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet SVOArrayWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight, 0, 0, setting.svo->SVOArray_G.size());
	nvvk::Buffer* SVOArrayPtr = setting.svo->SVOArray_G.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	SVOArrayWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight, 0, 0, setting.svo->SVOArray_E.size());
	nvvk::Buffer* SVOArrayPtr = setting.svo->SVOArray_G.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	VkWriteDescriptorSet SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.svo->SVOLayerInfos_G, 0, setting.svo->SVOLayerInfos_G.bufferSize);

	SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, setting.svo->SVOLayerInfos_E, 0, setting.svo->SVOLayerInfos_E.bufferSize);

	VkWriteDescriptorSet CSDispactchIndirectCommandWrite = 
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eDispatchIndirect_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, CSDispatchCommandBuffer, 0, CSDispatchCommandBuffer.bufferSize);

	VkWriteDescriptorSet IndivisibleNodeInfos_GWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, indivisibleNodeInfosBuffer_G, 0, indivisibleNodeInfosBuffer_G.bufferSize);

	VkWriteDescriptorSet weightSampleCountWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSampleCounts_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, weightSampleCountsBuffer, 0, weightSampleCountsBuffer.bufferSize);

	VkWriteDescriptorSet weightsWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, weightBuffer, 0, weightBuffer.bufferSize);

	VkWriteDescriptorSet weightsWrite =
		staticDescPack.makeWrite(shaderio::StaticBindingPoints_SVOWeight::eSVOWeightSums_SVOWeight, 0, 0, 1);
	write.append(SVOLayerInfosWrite, weightSumsBuffer, 0, weightSumsBuffer.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void SVOWeight::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVO.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::SVOWeightPushConstant),
	};

	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = 1,
		.pSetLayouts = staticDescPack.getLayoutPtr(),
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
}
void SVOWeight::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}

void SVOWeight::getIndivisibleNode_G(VkCommandBuffer cmd) {
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

	vkCmdDispatchIndirect(cmd, CSDispatchCommandBuffer.buffer, 0);
}
void SVOWeight::getWeights(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getWeights);

	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdDispatchIndirect(cmd, CSDispatchCommandBuffer.buffer, 0);
}
void SVOWeight::getProbability(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getWeights);

	for (int countdown = 0; countdown >= setting.svo->setting.octree->setting.OctreeDepth; ++countdown) {
		pushConstant.countdown = countdown;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdDispatchIndirect(cmd, CSDispatchCommandBuffer.buffer, 0);
	}
}