#include "./VolumetricFog.h"

#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>

using namespace FzbRenderer;

VolumetricFog::VolumetricFog(pugi::xml_node& rendererNode) {}

void VolumetricFog::init() {
	Renderer::createGBuffer(true, true, (uint32_t)GBuffers_VolumetricFog::eTonemapping);

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void VolumetricFog::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);
	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	Renderer::clean();
};
void VolumetricFog::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);

	if (ImGui::Begin("Volumetric Fog Setting")) {
		UIModified |= ImGui::DragFloat3("Volumetric Fog Start Pos", (float*)&pushConstant.fogStartPos);

		bool change = ImGui::DragFloat3("Volumetric Fog Voxel Grid Size", (float*)&pushConstant.fogVoxelGridSize);
		if (change) {
			createVolumetricFogImage();
			UIModified = true;
		}

		UIModified |= ImGui::DragFloat3("Volumetric Fog Voxel Size", (float*)&pushConstant.fogVoxelSize);
	}
	ImGui::End();
}
void VolumetricFog::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	for (int i = 0; i <= (uint32_t)GBuffers_VolumetricFog::eVoxelGridDebug; ++i) {
		VkWriteDescriptorSet    OutImageWrite = staticDescPack.makeWrite(i, 0, 0, 1);
		write.append(OutImageWrite, gBuffers.getColorImageView(i), VK_IMAGE_LAYOUT_GENERAL);
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void VolumetricFog::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) Application::frameIndex = 0;
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
}
void VolumetricFog::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	createGBuffers(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	createVolumetricFog(cmd);
	//无需同步
	if (Application::sceneResource.sceneInfo.useSky)
	{
		const glm::mat4& viewMatrix = Application::sceneResource.cameraManip->getViewMatrix();
		const glm::mat4& projMatrix = Application::sceneResource.cameraManip->getPerspectiveMatrix();
		Application::skySimple.runCompute(cmd, Application::app->getViewportSize(), viewMatrix, projMatrix,
			Application::sceneResource.sceneInfo.skySimpleParam, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));
		//后续会有一个layout转换的barriar，所以这里无需同步
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	deferredRenderring(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Renderer::postProcess(cmd, &gBuffers.m_res.gBufferColor[((uint32_t)GBuffers_VolumetricFog::eTonemapping)].descriptor);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void VolumetricFog::createVolumetricFogImage() {
	volumetricFogImage.clean();

	volumetricFogImage = FzbRenderer::Image("volumetricFog3DTexture");
	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_3D,
	colorImageCreateInfo.info.extent = { pushConstant.fogVoxelGridSize.x, pushConstant.fogVoxelGridSize.y, pushConstant.fogVoxelGridSize.z };

	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;

	colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_LINEAR;
	colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_LINEAR;
	colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

	volumetricFogImage.init(colorImageCreateInfo);
}
void VolumetricFog::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Fzb PathGuiding static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void VolumetricFog::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet	gBuffersWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage, 0, 0, 1);
	write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eAlbedo));

	gBuffersWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage, 0, 0, 1);
	write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eNormal));

	gBuffersWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage, 0, 0, 1);
	write.append(gBuffersWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet	volumetricFogImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage, 0, 0, 1);
	write.append(volumetricFogImageWrite, volumetricFogImage.image);

	volumetricFogImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler, 0, 0, 1);
	write.append(volumetricFogImageWrite, volumetricFogImage.image);

	VkWriteDescriptorSet	renderedImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void VolumetricFog::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant)
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
void VolumetricFog::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "createGBuffers.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
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

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_createGBuffer);
	NVVK_DBG_NAME(vertexShader_createGBuffer);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_createGBuffer);
	NVVK_DBG_NAME(fragmentShader_createGBuffer);
	//--------------------------------------------------------------------------------------
	shaderSource = shaderPath / "createVolumetricFog.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createVolumetricFog";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog);
	NVVK_DBG_NAME(computeShader_createVolumetricFog);
	//--------------------------------------------------------------------------------------
	shaderSource = shaderPath / "deferredRendering.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_deferredRendering";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_deferredRenderring);
	NVVK_DBG_NAME(computeShader_deferredRenderring);
}
void VolumetricFog::updateDataPerFrame(VkCommandBuffer cmd) {}

void VolumetricFog::createGBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	std::vector<VkRenderingAttachmentInfo> colorAttachments((uint32_t)GBuffers_VolumetricFog::eRendered);
	for (int i = 0; i <= (uint32_t)GBuffers_VolumetricFog::eNormal; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		colorAttachments[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachments[i].imageView = gBuffers.getColorImageView(i);
	}
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	
	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = DEFAULT_VkRect2D(gBuffers.getSize());
	renderingInfo.colorAttachmentCount = colorAttachments.size();
	renderingInfo.pColorAttachments = colorAttachments.data();
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());
	vkCmdSetDepthTestEnable(cmd, VK_TRUE);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_createGBuffer, .fragment = fragmentShader_createGBuffer });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		pushConstant.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);
	
	for (int i = 0; i <= (uint32_t)GBuffers_VolumetricFog::eNormal; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void VolumetricFog::createVolumetricFog(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog);

	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{pushConstant.fogVoxelGridSize.x, pushConstant.fogVoxelGridSize.y, pushConstant.fogVoxelGridSize.z}, VkExtent3D{4, 4, 4});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
}
void VolumetricFog::deferredRenderring(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_deferredRenderring);

	VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{16, 16});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}