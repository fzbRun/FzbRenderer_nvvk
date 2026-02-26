#include "./SparseVoxelOctree.h"
#include "common/Application/Application.h"
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

SparseVoxelOctree::SparseVoxelOctree(pugi::xml_node& featureNode) {

}
void SparseVoxelOctree::init(SVOSetting setting) {
	this->setting = setting;

	createSVOArray();
	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::SVOPushConstant));
	compileAndCreateShaders();
}

void SparseVoxelOctree::clean() {
	Feature::clean();
	for(int i = 0; i < SVOArray_G.size(); ++i) Application::allocator.destroyBuffer(SVOArray_G[i]);
	for(int i = 0; i < SVOArray_E.size(); ++i) Application::allocator.destroyBuffer(SVOArray_E[i]);
	Application::allocator.destroyBuffer(SVOIndivisibleNodes_G);
	Application::allocator.destroyBuffer(SVOLayerInfos_G);
	Application::allocator.destroyBuffer(SVOLayerInfos_E);
	Application::allocator.destroyBuffer(SVODivisibleNodeIndices_G);
	Application::allocator.destroyBuffer(SVODivisibleNodeIndices_E);
	Application::allocator.destroyBuffer(SVOThreadGroupInfos);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initSVOArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createSVOArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_offsetLabelMultiBlock, nullptr);
}

void SparseVoxelOctree::resize(VkCommandBuffer cmd, const VkExtent2D& size) {

}
void SparseVoxelOctree::uiRender() {

}
void SparseVoxelOctree::preRender() {

}
void SparseVoxelOctree::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "SVO_render");

	VkBindDescriptorSetsInfo bindDescriptorSetsInfo = {
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
		.layout = pipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = staticDescPack.getSetPtr(),
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	initSVOArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	createSVOArray(cmd);
}
void SparseVoxelOctree::postProcess(VkCommandBuffer cmd) {

}

void SparseVoxelOctree::createSVOArray() {
	uint32_t OctreeDepth = setting.octree->setting.OctreeDepth;		//这个depth是从0开始的
	SVOArray_G.resize(OctreeDepth + 1);
	SVOArray_E.resize(OctreeDepth + 1);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	for (int depth = 0; depth <= OctreeDepth; ++depth) {
		uint32_t bufferSize = sizeof(shaderio::SVONodeData_G) * SVOInitialSize[depth];
		allocator->createBuffer(SVOArray_G[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(SVOArray_G[depth].buffer);

		bufferSize = sizeof(shaderio::SVONodeData_E) * SVOInitialSize[depth];
		allocator->createBuffer(SVOArray_E[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(SVOArray_E[depth].buffer);
	}

	//SVOIndivisibleNodes_G
	//……

	uint32_t bufferSize = sizeof(shaderio::SVOLayerInfo) * (OctreeDepth + 1);
	allocator->createBuffer(SVOLayerInfos_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVOLayerInfos_G.buffer);

	allocator->createBuffer(SVOLayerInfos_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVOLayerInfos_E.buffer);

	bufferSize = sizeof(uint32_t) * SVOInitialSize[7];
	allocator->createBuffer(SVODivisibleNodeIndices_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVODivisibleNodeIndices_G.buffer);

	allocator->createBuffer(SVODivisibleNodeIndices_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVODivisibleNodeIndices_E.buffer);

	bufferSize = sizeof(shaderio::SVOThreadGroupInfo) * (SVOInitialSize[7] / THREADGROUP_SIZE);
	allocator->createBuffer(SVOThreadGroupInfos, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVOThreadGroupInfos.buffer);

	for (int depth = 0; depth < 8; ++depth) pushConstant.sizes[depth] = SVOInitialSize[depth];
	pushConstant.maxDepth = OctreeDepth;
}
void SparseVoxelOctree::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eOctreeArray_G_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.octree->OctreeArray_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eOctreeArray_E_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.octree->OctreeArray_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVOArray_G_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)SVOArray_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVOArray_E_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)SVOArray_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVOLayerInfos_G_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVOLayerInfos_E_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVODivisibleNodeIndices_G_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVODivisibleNodeIndices_E_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_SVO::eSVOThreadGroupInfos_SVO,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void SparseVoxelOctree::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet OctreeArrayWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eOctreeArray_G_SVO, 0, 0, setting.octree->OctreeArray_G.size());
	nvvk::Buffer* OctreeArrayPtr = setting.octree->OctreeArray_G.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eOctreeArray_E_SVO, 0, 0, setting.octree->OctreeArray_E.size());
	OctreeArrayPtr = setting.octree->OctreeArray_E.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	VkWriteDescriptorSet SVOArrayWrite = 
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVOArray_G_SVO, 0, 0, SVOArray_G.size());
	nvvk::Buffer* SVOArrayPtr = SVOArray_G.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	SVOArrayWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVOArray_E_SVO, 0, 0, SVOArray_E.size());
	SVOArrayPtr = SVOArray_E.data();
	write.append(SVOArrayWrite, SVOArrayPtr);

	VkWriteDescriptorSet SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVOLayerInfos_G_SVO, 0, 0, 1);
	write.append(SVOLayerInfosWrite, SVOLayerInfos_G, 0, SVOLayerInfos_G.bufferSize);

	SVOLayerInfosWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVOLayerInfos_E_SVO, 0, 0, 1);
	write.append(SVOLayerInfosWrite, SVOLayerInfos_E, 0, SVOLayerInfos_E.bufferSize);

	VkWriteDescriptorSet SVODivisibleNodeIndicesWrite = 
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVODivisibleNodeIndices_G_SVO, 0, 0, 1);
	write.append(SVODivisibleNodeIndicesWrite, SVODivisibleNodeIndices_G, 0, SVODivisibleNodeIndices_G.bufferSize);

	SVODivisibleNodeIndicesWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVODivisibleNodeIndices_E_SVO, 0, 0, 1);
	write.append(SVODivisibleNodeIndicesWrite, SVODivisibleNodeIndices_E, 0, SVODivisibleNodeIndices_E.bufferSize);

	VkWriteDescriptorSet SVOThreadGroupInfosWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_SVO::eSVOThreadGroupInfos_SVO, 0, 0, 1);
	write.append(SVOThreadGroupInfosWrite, SVOThreadGroupInfos, 0, SVOThreadGroupInfos.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void SparseVoxelOctree::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVO.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::SVOPushConstant),
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
		vkDestroyShaderEXT(device, computeShader_initSVOArray, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initSVOArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initSVOArray);
		NVVK_DBG_NAME(computeShader_initSVOArray);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_createSVOArray, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createSVOArray";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createSVOArray);
		NVVK_DBG_NAME(computeShader_createSVOArray);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, computeShader_offsetLabelMultiBlock, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_offsetLabelMultiBlock";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_offsetLabelMultiBlock);
		NVVK_DBG_NAME(computeShader_offsetLabelMultiBlock);
	}
}

void SparseVoxelOctree::updateDataPerFrame(VkCommandBuffer cmd) {
	
}

void SparseVoxelOctree::initSVOArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initSVOArray);

	uint32_t totalVoxelCount = SVOInitialSize[pushConstant.maxDepth];
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void SparseVoxelOctree::createSVOArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::SVOPushConstant),
		.pValues = &pushConstant,
	};

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int depth = 1; depth <= pushConstant.maxDepth; ++depth) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createSVOArray);

		pushConstant.currentDepth = depth;
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t totalVoxelCount = SVOInitialSize[depth];
		VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1}, VkExtent2D{ THREADGROUP_SIZE, 1});
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		if (totalVoxelCount > THREADGROUP_SIZE) {
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_offsetLabelMultiBlock);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
	}
}