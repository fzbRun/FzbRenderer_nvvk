#include "./LightInject_FzbPG.h"
#include <common/Shader/Shader.h>
#include <nvutils/timers.hpp>
#include <common/Application/Application.h>
#include <nvvk/default_structs.hpp>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>
#include "../RasterVoxelization/RasterVoxelization_FzbPG.h"

using namespace FzbRenderer;

LightInject_FzbPG::LightInject_FzbPG(pugi::xml_node& featureNode) {
#ifndef NDEBUG
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}
void LightInject_FzbPG::init(LightInjectCreateInfo_FzbPG setting) {
	this->setting = setting;
	sbtGenerator.init(Application::app->getDevice(), setting.ptContext->rtProperties);

	createBuffer();
	createGBuffer(true, false, 2);
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipeline();
	compileAndCreateShaders();

#ifndef NDEBUG
	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube(false, false);
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Cube", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif
}
void LightInject_FzbPG::clean() {
	PathTracing::clean();
	VkDevice device = Application::app->getDevice();
	vkDestroyPipeline(device, rtPipeline, nullptr);
	vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);

	vkDestroyShaderEXT(device, computeShader_getHasGeometryVoxels, nullptr);
	vkDestroyShaderEXT(device, computeShader_setDispatchIndirectCommand, nullptr);
	vkDestroyShaderEXT(device, computeShader_LightInject, nullptr);

	Application::allocator.destroyBuffer(hasGeometryVoxelInfoBuffer);
	Application::allocator.destroyBuffer(globalInfoBuffer);

#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);
#endif
}
void LightInject_FzbPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("LightInject")) {
		if (PE::begin()) {
			if (PE::entry("CubeMap", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showCubeMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
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
				showCubeMap = !showCubeMap;
			}
		}
		PE::end();
	}
	ImGui::End();

	if (showCubeMap) Application::viewportImage = gBuffers.getDescriptorSet(0);
#endif
}
void LightInject_FzbPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));
};
void LightInject_FzbPG::preRender() {
	if (Application::sceneResource.cameraChange) Application::frameIndex = 0;

	pushConstant.VGBVoxelSize = glm::vec4(setting.VGBVoxelSize, 1.0f);
	pushConstant.VGBStartPos = glm::vec3(setting.VGBStartPos);
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.voxelCount = (uint32_t)pow(setting.VGBSize, 3);
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.sampleCount = Application::sceneResource.isStaticScene ? LIGHTINJECT_SAMPLE_COUNT_STATIC_SCENE : LIGHTINJECT_SAMPLE_COUNT;

	float angle = FzbRenderer::rand(Application::frameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));
}
void LightInject_FzbPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), setting.asManager->asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.size = sizeof(shaderio::LightInjectPushConstant_FzbPG),
		.pValues = &pushConstant
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	getHasGeometryVoxels(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);
	lightInject(cmd);
}
void LightInject_FzbPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_Cube(cmd);
#endif
}

void LightInject_FzbPG::createBuffer() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t voxelTotalCount = (uint32_t)pow(setting.VGBSize, 3) * 6;
	uint32_t bufferSize = voxelTotalCount * sizeof(shaderio::HasGeometryVoxelInfo);
	allocator->createBuffer(hasGeometryVoxelInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasGeometryVoxelInfoBuffer.buffer);
		
	bufferSize = sizeof(shaderio::LightInjectGlobalInfo);
	allocator->createBuffer(globalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(globalInfoBuffer.buffer);
}
void LightInject_FzbPG::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
			.binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eVGB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)setting.VGBs.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eHasGeometryVoxelInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("LightInject ray tracing static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));

	bindings.clear();
	bindings.addBinding({
			.binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);

	LOGI("LightInject ray tracing dynamic descriptor layout created\n");
}
void LightInject_FzbPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eVGB, 0, 0, setting.VGBs.size());
	nvvk::Buffer* VGBsPtr = setting.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	VkWriteDescriptorSet    hasGeometryVoxelInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eHasGeometryVoxelInfo, 0, 0, 1);
	write.append(hasGeometryVoxelInfoWrite, hasGeometryVoxelInfoBuffer, 0, hasGeometryVoxelInfoBuffer.bufferSize);

	VkWriteDescriptorSet    globalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_FzbPG::eGlobalInfo, 0, 0, 1);
	write.append(globalInfoWrite, globalInfoBuffer, 0, globalInfoBuffer.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void LightInject_FzbPG::createPipeline() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_FzbPG)
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void LightInject_FzbPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "LightInject.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_FzbPG),
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_getHasGeometryVoxels, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_getHasGeometryVoxels";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getHasGeometryVoxels);
	NVVK_DBG_NAME(computeShader_getHasGeometryVoxels);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_setDispatchIndirectCommand, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_setDispatchIndirectCommand";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_setDispatchIndirectCommand);
	NVVK_DBG_NAME(computeShader_setDispatchIndirectCommand);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, computeShader_LightInject, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_LightInject";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_LightInject);
	NVVK_DBG_NAME(computeShader_LightInject);
	//--------------------------------------------------------------------------------------
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain_Cube";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_Cube);
	NVVK_DBG_NAME(vertexShader_Cube);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_Cube";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_Cube);
	NVVK_DBG_NAME(fragmentShader_Cube);
#endif
}
void LightInject_FzbPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void LightInject_FzbPG::getHasGeometryVoxels(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdFillBuffer(cmd, globalInfoBuffer.buffer, 0, sizeof(shaderio::LightInjectGlobalInfo), 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getHasGeometryVoxels);

	uint32_t voxelTotalCount = pushConstant.voxelCount * 6;
	VkExtent2D groupSize = nvvk::getGroupCounts({ voxelTotalCount , 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_setDispatchIndirectCommand);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void LightInject_FzbPG::lightInject(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_LightInject);
	vkCmdDispatchIndirect(cmd, globalInfoBuffer.buffer, 0);
}

#ifndef NDEBUG
void LightInject_FzbPG::debug_Cube(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(0);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Cube, .fragment = fragmentShader_Cube });

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_FzbPG),
		.pValues = &pushConstant,
	};
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.normalIndex = RasterVoxelization_FzbPG::normalIndex;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.voxelCount * 6, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
#endif