#include "./Octree.h"
#include <nvutils/timers.hpp>
#include "./shaderio.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>

using namespace FzbRenderer;

Octree::Octree(pugi::xml_node& featureNode) {
	if (pugi::xml_node entropyThresholdNode = featureNode.child("entropyThreshold")) 
		pushConstant.entropyThreshold = getfloat2FromString(entropyThresholdNode.attribute("value").value());
	if (pugi::xml_node irradianceRelRatioThresholdNode = featureNode.child("irradianceRelRatioThreshold"))
		pushConstant.irradianceRelRatioThreshold = std::stof(irradianceRelRatioThresholdNode.attribute("value").value());
	if (pugi::xml_node clusteringLevelNode = featureNode.child("clusteringLevel"))
		setting.clusteringLevel = std::stoi(clusteringLevelNode.attribute("value").value());

	if (pugi::xml_node rasterVoxelizationNode = featureNode.child("RasterVoxelization"))
		rasterVoxelization = std::make_shared<RasterVoxelization>(rasterVoxelizationNode);
}
void Octree::init() {
	rasterVoxelization->init();
	createDescriptorSetLayout();
	createOctreeArray();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::OctreePushConstant));	//创建管线布局：pushConstant+描述符集合布局
	compileAndCreateShaders();
}
void Octree::clean() {
	rasterVoxelization->clean();

	Feature::clean();
	for(int i = 0; i < OctreeArray_G.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_G[i]);
	for(int i = 0; i < OctreeArray_E.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_E[i]);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
}
void Octree::uiRender() {
	rasterVoxelization->uiRender();
};
#ifndef NDEBUG
void Octree::resize(VkCommandBuffer cmd, const VkExtent2D& size, nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex) {
	rasterVoxelization->resize(cmd, size, gBuffers_other, baseMapIndex);
};
#endif
void FzbRenderer::RasterVoxelization::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void Octree::preRender() {
	rasterVoxelization->preRender();
}
void Octree::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

	updateDataPerFrame(cmd);
	rasterVoxelization->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR);

	bindDescriptorSetsInfo = {
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
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	//假设光照注入已经完成，那么要初始化OctreeArray
	initOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	createOctreeArray(cmd);

}
void Octree::postProcess(VkCommandBuffer cmd) {
	rasterVoxelization->postProcess(cmd);
};

void Octree::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eVGB_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL});
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eOctreeArray_G_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 8,	//VGB最大为128x128x128
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eOctreeArray_E_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 8,	//VGB最大为128x128x128
		.stageFlags = VK_SHADER_STAGE_ALL });
	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void Octree::createOctreeArray() {
	uint32_t VGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w;
	setting.OctreeDepth = std::countr_zero(VGBSize);
	OctreeArray_G.resize(setting.OctreeDepth + 1);
	OctreeArray_E.resize(setting.OctreeDepth + 1);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t bufferSize = sizeof(shaderio::VGBVoxelData);
	for (int depth = 0; depth < setting.OctreeDepth; ++depth) {
		allocator->createBuffer(OctreeArray_G[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_G[depth]);

		allocator->createBuffer(OctreeArray_E[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_E[depth]);

		bufferSize *= 8;
	}

	pushConstant.maxDepth = setting.OctreeDepth;

	//由renderer对创建buffer的命令进行提交
}
void Octree::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eVGB_Octree, 0, 0, 1);
	write.append(VGBWrite, rasterVoxelization->VGB, 0, rasterVoxelization->VGB.bufferSize);

	VkWriteDescriptorSet    OctreeArrayWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eOctreeArray_G_Octree, 0, 0, OctreeArray_G.size());
	nvvk::Buffer* OctreeArrayPtr = OctreeArray_G.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eOctreeArray_E_Octree, 0, 0, OctreeArray_E.size());
	OctreeArrayPtr = OctreeArray_E.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "Octree.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS | VK_SHADER_STAGE_COMPUTE_BIT ,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant),
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
}
void Octree::updateDataPerFrame(VkCommandBuffer cmd) {
	rasterVoxelization->updateDataPerFrame(cmd);
}

void Octree::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	uint32_t totalVoxelCount = pow(rasterVoxelization->setting.pushConstant.voxelSize_Count.w, 3);
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree::createOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray);

	uint32_t totalVoxelCount = pow(rasterVoxelization->setting.pushConstant.voxelSize_Count.w, 3);
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 256, 1 });
	for (int i = pushConstant.maxDepth; i > setting.clusteringLevel; --i) {
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		if(i > 1) nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		totalVoxelCount /= 8;
		groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 256, 1 });
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray2);

	totalVoxelCount = pow(8, setting.clusteringLevel);
	groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	for (int i = setting.clusteringLevel; i > 0; --i) {
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		if(i > 1) nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		totalVoxelCount /= 8;
		groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	}
}