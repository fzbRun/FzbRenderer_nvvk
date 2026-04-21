/*
#include "./Octree_FzbPG.h"
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

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
	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::OctreePushConstant_FzbPG));
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

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);

	vkDestroyShaderEXT(device, computeShader_getOctreeLabel1, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel2, nullptr);
	vkDestroyShaderEXT(device, computeShader_getOctreeLabel3, nullptr);

	vkDestroyShaderEXT(device, computeShader_getIndivisibleNodeInfos, nullptr);
}
void Octree_FzbPG::uiRender() {}
void Octree_FzbPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	gBuffers.update(cmd, size);
};
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

	vkCmdPushConstants2(cmd, &pushInfo);

	initOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	createOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getOctreeLabel(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	getIndivisibleNodeInfos(cmd);
}
void Octree_FzbPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG

#endif
};

void Octree_FzbPG::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	octreeMaxLayer = std::countr_zero(VGBSize);	//start from 0

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	octreeClusterDataBuffer_G.resize(octreeMaxLayer + 1);
	octreeDataBuffer_G.resize(octreeMaxLayer + 1);
	uint32_t layerNodeCount = 6;
	for (int layerIndex = 0; layerIndex <= octreeMaxLayer; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeClusterData_G);
		allocator->createBuffer(octreeClusterDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeClusterDataBuffer_G[layerIndex].buffer);

		bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeData_G);
		allocator->createBuffer(octreeDataBuffer_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeDataBuffer_G[layerIndex].buffer);

		layerNodeCount *= 8;
	}

	uint32_t octreeLayerCount_E = octreeMaxLayer - OCTREE_CLUSTER_LAYER + 1;
	octreeClusterDataBuffer_E.resize(octreeLayerCount_E);
	layerNodeCount = 6 * (1 << (3 * OCTREE_CLUSTER_LAYER));
	for (int layerIndex = 0; layerIndex < octreeLayerCount_E; ++layerIndex) {
		uint32_t bufferSize = layerNodeCount * sizeof(shaderio::OctreeNodeClusterData_E);
		allocator->createBuffer(octreeClusterDataBuffer_E[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(octreeClusterDataBuffer_E[layerIndex].buffer);

		layerNodeCount *= 8;
	}

	uint32_t bufferSize = NODECOUNT_E * sizeof(shaderio::OctreeNodeData_E);
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

	bufferSize = sizeof(shaderio::HasDataOctreeBlockCount);
	allocator->createBuffer(hasDataBlockCountBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasDataBlockCountBuffer.buffer);

	bufferSize = sizeof(shaderio::OctreeGlobalInfo);
	allocator->createBuffer(globalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(globalInfoBuffer.buffer);

#ifndef USE_SVO
	bufferSize = sizeof(shaderio::uint2) * (1 << (3 * (octreeMaxLayer - 1)));
	allocator->createBuffer(divisibleNodeInfoBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(divisibleNodeInfoBuffer_G.buffer);

	uint32_t maxLayerNodeCount = (1 << (3 * octreeMaxLayer));
	bufferSize = sizeof(shaderio::OctreeThreadGroupInfo) * (maxLayerNodeCount / GETOCTREELABEL_CS_THREADGROUP_SIZE);
	allocator->createBuffer(threadGroupInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(threadGroupInfoBuffer.buffer);

	bufferSize = IndivisibleNodeCount_G * sizeof(shaderio::uint2);
	allocator->createBuffer(indivisibleNodeInfosBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_G.buffer);

	bufferSize = NODECOUNT_E * sizeof(uint32_t);
	allocator->createBuffer(indivisibleNodeInfosBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(indivisibleNodeInfosBuffer_E.buffer);
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
#ifndef USE_SVO
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
#endif

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
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
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_FzbPG::eOctreeData_G, 0, 0, octreeClusterDataBuffer_E.size());
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

#ifndef USE_SVO
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
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_FzbPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "Octree.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_FzbPG),
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
	vkDestroyShaderEXT(device, computeShader_getIndivisibleNodeInfos, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_getIndivisibleInfos";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getIndivisibleNodeInfos);
	NVVK_DBG_NAME(computeShader_getIndivisibleNodeInfos);
}
void Octree_FzbPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void Octree_FzbPG::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree_FzbPG::createOctreeArray(VkCommandBuffer cmd) {}
void Octree_FzbPG::getOctreeLabel(VkCommandBuffer cmd) {}
void Octree_FzbPG::getIndivisibleNodeInfos(VkCommandBuffer cmd) {}

*/
