#include "./SVO.h"
#include "common/Application/Application.h"
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>

using namespace FzbRenderer;

SVO_SVOPG::SVO_SVOPG(pugi::xml_node& featureNode) {
#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}

void SVO_SVOPG::init(SVOSetting_SVOPG setting) {
	this->setting = setting;

	createSVOArray();
	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::SVOPushConstant_SVOPG));
	compileAndCreateShaders();

	IF_DEBUG(debugPrepare(), );
}
void SVO_SVOPG::clean() {
	Feature::clean();
	Application::allocator.destroyBuffer(SVO_G);
	Application::allocator.destroyBuffer(SVO_E);
	Application::allocator.destroyBuffer(SVOGlobalInfo);
	Application::allocator.destroyBuffer(SVODivisibleNodeInfos_G);
	Application::allocator.destroyBuffer(SVODivisibleNodeInfos_E);
	Application::allocator.destroyBuffer(SVOThreadGroupInfos);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initSVOArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_createSVOArray, nullptr);
	vkDestroyShaderEXT(device, computeShader_offsetLabelMultiBlock, nullptr);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Wireframe, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);
#endif
}
void SVO_SVOPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {}
void SVO_SVOPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	std::vector<std::string> wireframeMapNames = { "GeometrySVO", "IlluminationSVO" };
	std::vector<const char*> wireframeMapNames_pointers;
	for (const auto& wireframeMapName : wireframeMapNames)
		wireframeMapNames_pointers.push_back(wireframeMapName.c_str());

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("SVO")) {
		if (PE::begin()) {
			if (PE::entry("GeometrySVOResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showWireframeMap_G ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(0),
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

		if (PE::begin()) {
			if (PE::entry("IlluminationSVOResult", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showWireframeMap_E ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(1),
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

	if (showWireframeMap_G) Application::viewportImage = gBuffers.getDescriptorSet(0);
	if (showWireframeMap_E) Application::viewportImage = gBuffers.getDescriptorSet(1);
#endif
}
void SVO_SVOPG::preRender() {
#ifndef NDEBUG
	pushConstant.frameIndex = Application::frameIndex;
#endif
}
void SVO_SVOPG::render(VkCommandBuffer cmd) {
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
		.size = sizeof(shaderio::SVOPushConstant_SVOPG),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	initSVOArray(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	createSVOArray(cmd);
}
void SVO_SVOPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	//debug_wirefame(cmd);
#endif
}

void SVO_SVOPG::createSVOArray() {
	uint32_t OctreeDepth = setting.octree->setting.OctreeDepth;		//这个depth是从0开始的

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t bufferSize = sizeof(shaderio::SVONodeData_G) * SVOSize_G;
	allocator->createBuffer(SVO_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVO_G.buffer);

	bufferSize = sizeof(shaderio::SVONodeData_E) * SVOSize_E;
	allocator->createBuffer(SVO_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVO_E.buffer);

	allocator->createBuffer(SVOGlobalInfo, sizeof(shaderio::SVOGlobalInfo_SVOPG),
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
		| VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(SVOGlobalInfo.buffer);

	bufferSize = sizeof(shaderio::uint2) * SVOSize_G;
	allocator->createBuffer(SVODivisibleNodeInfos_G, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVODivisibleNodeInfos_G.buffer);

	bufferSize = sizeof(uint32_t) * SVOSize_E;
	allocator->createBuffer(SVODivisibleNodeInfos_E, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVODivisibleNodeInfos_E.buffer);

	uint32_t maxLayerNodeCount = SVOSize;
	bufferSize = sizeof(shaderio::SVOThreadGroupInfo) * (maxLayerNodeCount / THREADGROUP_SIZE);
	allocator->createBuffer(SVOThreadGroupInfos, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(SVOThreadGroupInfos.buffer);

	pushConstant.maxDepth_Octree = OctreeDepth;
}
void SVO_SVOPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eOctreeArray_G_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.octree->OctreeArray_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eOctreeArray_E_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.octree->OctreeArray_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVO_G_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVO_E_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVOGlobalInfo_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVODivisibleNodeIndices_G_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVODivisibleNodeIndices_E_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::BindingPoints_SVOPG::eSVOThreadGroupInfos_SVOPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void SVO_SVOPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eOctreeArray_G_SVOPG, 0, 0, setting.octree->OctreeArray_G.size());
	nvvk::Buffer* OctreeArrayPtr = setting.octree->OctreeArray_G.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eOctreeArray_E_SVOPG, 0, 0, setting.octree->OctreeArray_E.size());
	OctreeArrayPtr = setting.octree->OctreeArray_E.data();
	write.append(OctreeArrayWrite, OctreeArrayPtr);

	VkWriteDescriptorSet SVOArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVO_G_SVOPG, 0, 0, 1);
	write.append(SVOArrayWrite, SVO_G, 0, SVO_G.bufferSize);;

	SVOArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVO_E_SVOPG, 0, 0, 1);
	write.append(SVOArrayWrite, SVO_E, 0, SVO_E.bufferSize);;

	VkWriteDescriptorSet SVOGlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVOGlobalInfo_SVOPG, 0, 0, 1);
	write.append(SVOGlobalInfoWrite, SVOGlobalInfo, 0, SVOGlobalInfo.bufferSize);

	VkWriteDescriptorSet SVODivisibleNodeIndicesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVODivisibleNodeIndices_G_SVOPG, 0, 0, 1);
	write.append(SVODivisibleNodeIndicesWrite, SVODivisibleNodeInfos_G, 0, SVODivisibleNodeInfos_G.bufferSize);

	SVODivisibleNodeIndicesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVODivisibleNodeIndices_E_SVOPG, 0, 0, 1);
	write.append(SVODivisibleNodeIndicesWrite, SVODivisibleNodeInfos_E, 0, SVODivisibleNodeInfos_E.bufferSize);

	VkWriteDescriptorSet SVOThreadGroupInfosWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::BindingPoints_SVOPG::eSVOThreadGroupInfos_SVOPG, 0, 0, 1);
	write.append(SVOThreadGroupInfosWrite, SVOThreadGroupInfos, 0, SVOThreadGroupInfos.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void SVO_SVOPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "SVOShaders.slang";
	std::vector<uint32_t> shaderBuffer;
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::SVOPushConstant_SVOPG),
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

#ifndef NDEBUG
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
#endif
}

void SVO_SVOPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void SVO_SVOPG::initSVOArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initSVOArray);

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t maxLayerNodeCount = SVOSize;
	VkExtent2D groupSize = nvvk::getGroupCounts({ maxLayerNodeCount, 1 }, VkExtent2D{ THREADGROUP_SIZE, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void SVO_SVOPG::createSVOArray(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int depth = 1; depth <= pushConstant.maxDepth_Octree; ++depth) {
		pushConstant.currentDepth_SVO_G = depth + 1;
		pushConstant.currentDepth_SVO_E = depth;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createSVOArray);
		vkCmdDispatchIndirect(cmd, SVOGlobalInfo.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_offsetLabelMultiBlock);
		vkCmdDispatchIndirect(cmd, SVOGlobalInfo.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	}
}

#ifndef NDEBUG
void SVO_SVOPG::debugPrepare() {
	Feature::createGBuffer(false, false, 2);

	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet meshSet = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(meshSet);

	scene.createSceneInfoBuffer();

	shaderio::SVOGloablInfo_SVO globalInfo;
	globalInfo.cmd = { 1, 1, 1 };

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;
	globalInfo.drawCmd = {
		.indexCount = triMesh.indices.count,
		.instanceCount = 0, .firstIndex = 0, .vertexOffset = 0, .firstInstance = 0,
	};

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	NVVK_CHECK(stagingUploader.appendBuffer(SVOGlobalInfo, 0, std::span<const shaderio::SVOGloablInfo_SVO>({ globalInfo })));
}

void SVO_SVOPG::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	gBuffers.update(cmd, size);
	depthImageView = gBuffers_other.getDepthImageView();

	//	nvvk::WriteSetContainer write{};
	//for (int i = 0; i < 2; ++i) {
	//	VkWriteDescriptorSet wireframeMapWrite = staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eWireframeMap_Octree, 0, i, 1);
	//	write.append(wireframeMapWrite, gBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL);
	//}
	//
	//VkWriteDescriptorSet baseMapWrite = staticDescPack.makeWrite(shaderio::BindingPoints_Octree::eBaseMap_Octree, 0, 0, 1);
	//write.append(baseMapWrite, gBuffers_other.getColorImageView(baseMapIndex), VK_IMAGE_LAYOUT_GENERAL);
	//
	//vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void SVO_SVOPG::debug_wirefame(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	uint32_t wireframeMapCount = 2;
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
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
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

	vkCmdDrawIndexedIndirect(cmd, SVOGlobalInfo.buffer, offsetof(shaderio::SVOGloablInfo_SVO, drawCmd), 1, 0);

	vkCmdEndRendering(cmd);

	for (int i = 0; i < wireframeMapCount; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd,
			{ gBuffers.getColorImage(i),
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	}
}
#endif