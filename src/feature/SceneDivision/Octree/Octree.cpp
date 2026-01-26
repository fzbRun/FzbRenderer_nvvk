#include "./Octree.h"
#include <nvutils/timers.hpp>
#include "./shaderio.h"
#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <bit>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

using namespace FzbRenderer;

Octree::Octree(pugi::xml_node& featureNode) {
	if (pugi::xml_node entropyThresholdNode = featureNode.child("entropyThreshold")) 
		pushConstant.entropyThreshold = getfloat2FromString(entropyThresholdNode.attribute("value").value());
	if (pugi::xml_node irradianceRelRatioThresholdNode = featureNode.child("irradianceRelRatioThreshold"))
		pushConstant.irradianceRelRatioThreshold = std::stof(irradianceRelRatioThresholdNode.attribute("value").value());
	if (pugi::xml_node clusteringLevelNode = featureNode.child("clusteringLevel"))
		pushConstant.clusteringLevel = std::stoi(clusteringLevelNode.attribute("value").value());

#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}
void Octree::init(OctreeSetting setting) {
	this->setting = setting;
	this->setting.clusteringLevel = pushConstant.clusteringLevel;
	createOctreeArray();

#ifndef NDEBUG
	clusterLevelCount = setting.OctreeDepth - this->setting.clusteringLevel;
	Feature::createGBuffer(false, false, clusterLevelCount * 2);

	for (int i = this->setting.clusteringLevel; i < setting.OctreeDepth; ++i)
		pushConstant.showOctreeNodeTotalCount += pow(8, i);

	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif

	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::OctreePushConstant));	//创建管线布局：pushConstant+描述符集合布局
	compileAndCreateShaders();
}
void Octree::clean() {
	Feature::clean();
	for(int i = 0; i < OctreeArray_G.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_G[i]);
	for(int i = 0; i < OctreeArray_E.size(); ++i) Application::allocator.destroyBuffer(OctreeArray_E[i]);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createOctreeArray2, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, computeShader_MergeResult, nullptr);
#endif
}
void Octree::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	std::vector<std::string> wireframeMapNames(clusterLevelCount);
	for (int i = 0; i < clusterLevelCount; ++i) wireframeMapNames[i] = "octreeLevel" + std::to_string(i + setting.clusteringLevel);

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

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(selectedWireframeMapIndex_E + clusterLevelCount),
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
	if (showWireframeMap_E) Application::viewportImage = gBuffers.getDescriptorSet(selectedWireframeMapIndex_E + clusterLevelCount);
#endif
};
void Octree::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	gBuffers.update(cmd, size);
};
void Octree::preRender() {
#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.clusteringLevel = setting.clusteringLevel;
	pushConstant.VGBStartPos_Size = glm::vec4(setting.VGBStartPos, setting.VGBSize);
#endif
}
void Octree::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "Octree_render");

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
		.stageFlags = VK_SHADER_STAGE_ALL,
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
	debug_wirefame(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	//debug_mergeResult(cmd);
};

void Octree::createOctreeArray() {
	uint32_t VGBSize = uint32_t(setting.VGBSize);
	setting.OctreeDepth = std::countr_zero(VGBSize);	//start from 0
	OctreeArray_G.resize(setting.OctreeDepth + 1);
	OctreeArray_E.resize(setting.OctreeDepth + 1);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t bufferSize = sizeof(shaderio::VGBVoxelData);
	for (int depth = 0; depth <= setting.OctreeDepth; ++depth) {
		allocator->createBuffer(OctreeArray_G[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_G[depth].buffer);

		allocator->createBuffer(OctreeArray_E[depth], bufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(OctreeArray_E[depth].buffer);

		bufferSize *= 8;
	}

	pushConstant.maxDepth = setting.OctreeDepth;

	//由renderer对创建buffer的命令进行提交
}
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
		.descriptorCount = 8,	//max 128x128x128
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eOctreeArray_E_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 8,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifndef NDEBUG
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eWireframeMap_Octree,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = clusterLevelCount * 2,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = shaderio::BindingPoints_Octree::eBaseMap_Octree,
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
void Octree::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eVGB_Octree, 0, 0, 1);
	write.append(VGBWrite, setting.VGB, 0, setting.VGB.bufferSize);

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

#ifndef NDEBUG
	std::string wireframeMapCountName = "WireframeMapCount";
	std::string wireframeMapCount = std::to_string(clusterLevelCount * 2);
	Application::slangCompiler.addMacro({
		.name = wireframeMapCountName.c_str(),
		.value = wireframeMapCount.c_str()
	});
#endif

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "Octree.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
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
void Octree::updateDataPerFrame(VkCommandBuffer cmd) {
	scene.sceneInfo = Application::sceneResource.sceneInfo;

	nvvk::cmdBufferMemoryBarrier(cmd, { scene.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
								   VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	vkCmdUpdateBuffer(cmd, scene.bSceneInfo.buffer, 0, sizeof(shaderio::SceneInfo), &scene.sceneInfo);
	nvvk::cmdBufferMemoryBarrier(cmd, { scene.bSceneInfo.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
									   VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT });
}

void Octree::initOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initOctreeArray);

	uint32_t totalVoxelCount = pow(setting.VGBSize, 3);
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void Octree::createOctreeArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray);

	uint32_t totalVoxelCount = pow(setting.VGBSize, 3);
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 256, 1 });
	for (int i = pushConstant.maxDepth; i > setting.clusteringLevel; --i) {
		pushConstant.currentDepth = i;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		if(i > 1) nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		totalVoxelCount /= 8;
		groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 256, 1 });
	}

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createOctreeArray2);

	totalVoxelCount = pow(8, setting.clusteringLevel);
	groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	for (int i = setting.clusteringLevel; i > 0; --i) {
		pushConstant.currentDepth = i;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		if(i > 1) nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		totalVoxelCount /= 8;
		groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 512, 1 });
	}
}

#ifndef NDEBUG
void FzbRenderer::Octree::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	gBuffers.update(cmd, size);

	nvvk::WriteSetContainer write{};

	depthImageView = gBuffers_other.getDepthImageView();

	for (int i = 0; i < clusterLevelCount * 2; ++i) {
		VkWriteDescriptorSet wireframeMapWrite = staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eWireframeMap_Octree, 0, i, 1);
		write.append(wireframeMapWrite, gBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL);
	}
		
	VkWriteDescriptorSet baseMapWrite = staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eBaseMap_Octree, 0, 0, 1);
	write.append(baseMapWrite, gBuffers_other.getColorImageView(baseMapIndex), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void Octree::debug_wirefame(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	uint32_t wireframeMapCount = clusterLevelCount * 2;
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
	depthAttachment.imageView = depthImageView;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize()};
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

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)scene.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.showOctreeNodeTotalCount * 2, 0, 0, 0);

	vkCmdEndRendering(cmd);

	for (int i = 0; i < wireframeMapCount; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd,
			{ gBuffers.getColorImage(i),
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	}
}
void Octree::debug_mergeResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//将wireframeMap与调用者的tonemapping后的结果进行结合
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_MergeResult);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D imageSize = gBuffers.getSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(imageSize, VkExtent2D{ 32, 32 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
#endif