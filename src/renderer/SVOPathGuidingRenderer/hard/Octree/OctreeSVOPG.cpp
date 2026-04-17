#include "./OctreeSVOPG.h"
#include <nvutils/timers.hpp>
#include "./shaderio.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

using namespace FzbRenderer;

Octree_SVOPG::Octree_SVOPG(pugi::xml_node& featureNode) {
#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}
void Octree_SVOPG::init(OctreeSetting_SVOPG setting) {
	this->setting = setting;
	createOctreeArray();

	#ifndef NDEBUG
	debugPrepare();
	#endif

	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::OctreePushConstant_SVOPG));
	compileAndCreateShaders();
}
void Octree_SVOPG::clean() {
	Feature::clean();
	for (int i = 0; i < OctreeArray_G.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_G[i]);
	for (int i = 0; i < OctreeArray_E.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_E[i]);
	Application::allocator.destroyBuffer(NodeData_E);
	//Application::allocator.destroyBuffer(SVO_G);
	Application::allocator.destroyBuffer(blockInfoBuffer_G);
	Application::allocator.destroyBuffer(blockInfoBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_G);
	Application::allocator.destroyBuffer(hasDataBlockIndexBuffer_E);
	Application::allocator.destroyBuffer(hasDataBlockCountBuffer);
	Application::allocator.destroyBuffer(GlobalInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_initHasDataBlockInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_getGlobalInfo, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, computeShader_MergeResult, nullptr);
#endif
}
void Octree_SVOPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	std::vector<std::string> wireframeMapNames(showLayerCount);
	for (int i = 0; i < showLayerCount; ++i) wireframeMapNames[i] = "octreeLayer" + std::to_string(i + OCTREE_CLUSTER_LAYER);

	std::vector<const char*> wireframeMapNames_pointers;
	for (const auto& wireframeMapName : wireframeMapNames)
		wireframeMapNames_pointers.push_back(wireframeMapName.c_str());

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("Octree")) {
		ImGui::Combo("GeometryOctree", &selectedWireframeMapIndex_G, wireframeMapNames_pointers.data(), static_cast<int>(wireframeMapNames_pointers.size()));
		if (PE::begin()) {
			if (PE::entry("GeometryOctreeClusterResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showWireframeMap_G ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedWireframeMapIndex_G),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showWireframeMap_G = !showWireframeMap_G;
				showWireframeMap_E = false;
			}
		}
		PE::end();

		ImGui::Combo("IrradianceOctree", &selectedWireframeMapIndex_E, wireframeMapNames_pointers.data(), static_cast<int>(wireframeMapNames_pointers.size()));
		if (PE::begin()) {
			if (PE::entry("IrradianceOctreeClusterResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showWireframeMap_E ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedWireframeMapIndex_E + showLayerCount),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showWireframeMap_E = !showWireframeMap_E;
				showWireframeMap_G = false;
			}
		}
		PE::end();
	}
	ImGui::End();

	if (showWireframeMap_G) Application::viewportImage = gBuffers.getDescriptorSet(selectedWireframeMapIndex_G);
	if (showWireframeMap_E) Application::viewportImage = gBuffers.getDescriptorSet(selectedWireframeMapIndex_E + showLayerCount);
#endif
};
void Octree_SVOPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	gBuffers.update(cmd, size);
};
void Octree_SVOPG::preRender() {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.voxelVolume = setting.VGBVoxelSize.x * setting.VGBVoxelSize.y * setting.VGBVoxelSize.z;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);
#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.normalIndex = RasterVoxelization_SVOPG::normalIndex;
#endif
}
void Octree_SVOPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_SVOPG),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	initOctreeArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	createOctreeArray(cmd);
}
void Octree_SVOPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_wirefame(cmd);
	//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	////debug_mergeResult(cmd);
#endif
};

void Octree_SVOPG::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	setting.OctreeLayerCount = std::countr_zero(VGBSize);	//start from 0

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	OctreeArray_G.resize(setting.OctreeLayerCount + 1);
	uint32_t bufferSize = 6 * sizeof(shaderio::OctreeNodeData_G);
	for (int layerIndex = 0; layerIndex <= setting.OctreeLayerCount; ++layerIndex) {
		allocator->createBuffer(OctreeArray_G[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_G[layerIndex].buffer);
		bufferSize = bufferSize * 8;
	}

	OctreeArray_E.resize(setting.OctreeLayerCount + 1);
	bufferSize = 6 * sizeof(shaderio::OctreeNodeData_E);
	for (int layerIndex = 0; layerIndex <= setting.OctreeLayerCount; ++layerIndex) {
		allocator->createBuffer(OctreeArray_E[layerIndex], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_E[layerIndex].buffer);
		bufferSize = bufferSize * 8;
	}

	bufferSize = NODECOUNT_E * sizeof(shaderio::OctreeNodeData_E);
	allocator->createBuffer(NodeData_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(NodeData_E.buffer);

	bufferSize = uint32_t(pow(8, setting.OctreeLayerCount)) * 6 / 8 * sizeof(uint32_t);
	allocator->createBuffer(blockInfoBuffer_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(blockInfoBuffer_G.buffer);

	allocator->createBuffer(blockInfoBuffer_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(blockInfoBuffer_E.buffer);

	bufferSize = int(pow(8, setting.OctreeLayerCount)) * 6 / 8 * sizeof(uint32_t);
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
	allocator->createBuffer(GlobalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(GlobalInfoBuffer.buffer);

	pushConstant.octreeMaxLayer = setting.OctreeLayerCount;
	pushConstant.octreeNodeTotalCount = int(pow(8, setting.OctreeLayerCount + 1) - 1) / 7 * 6;
	pushConstant.VGBVoxelTotalCount = VGBSize * VGBSize * VGBSize;
}
void Octree_SVOPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	#ifdef CLUSTER_WITH_MATERIAL
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eVGBMaterialInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBMaterialInfos.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	#endif

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eOctreeArray_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)OctreeArray_G.size(),	//max 128x128x128
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eOctreeArray_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)OctreeArray_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eNodeData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBlockInfos_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBlockInfos_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockIndices_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockIndices_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockCount,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifndef NDEBUG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eWireframeMap,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = showLayerCount * 2,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBaseMap,
		.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void Octree_SVOPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eVGB, 0, 0, setting.VGBs.size());
	nvvk::Buffer* VGBsPtr = setting.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	#ifdef CLUSTER_WITH_MATERIAL
	VkWriteDescriptorSet    VGBMaterialInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eVGBMaterialInfos, 0, 0, setting.VGBMaterialInfos.size());
	nvvk::Buffer* VGBMaterialInfosPtr = setting.VGBMaterialInfos.data();
	write.append(VGBMaterialInfoWrite, VGBMaterialInfosPtr);
	#endif

	VkWriteDescriptorSet    OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eOctreeArray_G, 0, 0, OctreeArray_G.size());
	nvvk::Buffer* octreeArraysPtr = OctreeArray_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eOctreeArray_E, 0, 0, OctreeArray_E.size());
	octreeArraysPtr = OctreeArray_E.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eNodeData_E, 0, 0, 1);
	write.append(OctreeArrayWrite, NodeData_E, 0, NodeData_E.bufferSize);

	VkWriteDescriptorSet    HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBlockInfos_G, 0, 0, 1);
	write.append(HasDataInfoWrite, blockInfoBuffer_G, 0, blockInfoBuffer_G.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBlockInfos_E, 0, 0, 1);
	write.append(HasDataInfoWrite, blockInfoBuffer_E, 0, blockInfoBuffer_E.bufferSize);
	
	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockIndices_G, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_G, 0, hasDataBlockIndexBuffer_G.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockIndices_E, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockIndexBuffer_E, 0, hasDataBlockIndexBuffer_E.bufferSize);

	HasDataInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eHasDataBlockCount, 0, 0, 1);
	write.append(HasDataInfoWrite, hasDataBlockCountBuffer, 0, hasDataBlockCountBuffer.bufferSize);

	VkWriteDescriptorSet    GlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eGlobalInfo, 0, 0, 1);
	write.append(GlobalInfoWrite, GlobalInfoBuffer, 0, GlobalInfoBuffer.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_SVOPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	#ifndef NDEBUG
	std::string wireframeMapCountName = "WireframeMapCount";
	std::string wireframeMapCount = std::to_string(showLayerCount * 2);
	Application::slangCompiler.addMacro({
		.name = wireframeMapCountName.c_str(),
		.value = wireframeMapCount.c_str()
		});
	#endif

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "Octree2.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	#ifndef NDEBUG
	Application::slangCompiler.clearMacros();
	#endif

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::OctreePushConstant_SVOPG),
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
#ifndef NDEBUG
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_Wireframe";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_Wireframe);
	NVVK_DBG_NAME(vertexShader_Wireframe);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_Wireframe";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_Wireframe);
	NVVK_DBG_NAME(fragmentShader_Wireframe);
	//---------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_MergeResult, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_MergeResult";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_MergeResult);
	NVVK_DBG_NAME(computeShader_MergeResult);
#endif
}
void Octree_SVOPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void Octree_SVOPG::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	VkExtent2D groupSize = nvvk::getGroupCounts({ pushConstant.octreeNodeTotalCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree_SVOPG::createOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	uint32_t layerBlockCount = uint32_t(pow(8, pushConstant.octreeMaxLayer)) * 6 / 8;
	for (int layerIndex = pushConstant.octreeMaxLayer; layerIndex > OCTREE_CLUSTER_LAYER; --layerIndex) {
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerBlockCount = layerBlockCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, hasDataBlockCountBuffer.buffer, 0, sizeof(shaderio::HasDataOctreeBlockCount), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initHasDataBlockInfo);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerBlockCount, 1 }, VkExtent2D{ 1024, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getGlobalInfo);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray);
		vkCmdDispatchIndirect(cmd, GlobalInfoBuffer.buffer, 0);

		layerBlockCount /= 8;
	}
	
	for (int layerIndex = OCTREE_CLUSTER_LAYER; layerIndex > 0; --layerIndex) {
		uint32_t layerNodeCount = shaderio::OctreeLayerInfo_E[layerIndex];
		pushConstant.currentLayer = layerIndex;
		pushConstant.currentLayerNodeCount = layerNodeCount;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray2);
		VkExtent2D groupSize = nvvk::getGroupCounts({ layerNodeCount, 1 }, VkExtent2D{ CREATEOCTREE_CS_THREADGROUP_SIZE, 1 });
		vkCmdDispatch(cmd, groupSize.width, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

#ifndef NDEBUG
void Octree_SVOPG::debugPrepare() {
	showLayerCount = setting.OctreeLayerCount - OCTREE_CLUSTER_LAYER;
	Feature::createGBuffer(true, false, showLayerCount * 2);

	for (int i = OCTREE_CLUSTER_LAYER; i < this->setting.OctreeLayerCount; ++i)
		pushConstant.showOctreeNodeTotalCount += pow(8, i);

	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
}
void Octree_SVOPG::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	gBuffers.update(cmd, size);

	nvvk::WriteSetContainer write{};

	depthImageView = gBuffers_other.getDepthImageView();

	for (int i = 0; i < showLayerCount * 2; ++i) {
		VkWriteDescriptorSet wireframeMapWrite = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eWireframeMap, 0, i, 1);
		write.append(wireframeMapWrite, gBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL);
	}

	VkWriteDescriptorSet baseMapWrite = staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_Octree_SVOPG::eBaseMap, 0, 0, 1);
	write.append(baseMapWrite, gBuffers_other.getColorImageView(baseMapIndex), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void Octree_SVOPG::debug_wirefame(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	uint32_t wireframeMapCount = showLayerCount * 2;
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
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = setting.lineWidth;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
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
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Wireframe, .fragment = fragmentShader_Wireframe });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 0;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
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
#endif