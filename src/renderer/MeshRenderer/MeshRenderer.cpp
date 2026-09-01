#include "./MeshRenderer.h"

#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

using namespace FzbRenderer;

MeshRenderer::MeshRenderer(pugi::xml_node& rendererNode) {
	{
		meshShaderFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT };
		meshShaderFeatures.taskShader = VK_TRUE; // Task/amplification shader
		meshShaderFeatures.meshShader = VK_TRUE; // Mesh shader
		meshShaderFeatures.meshShaderQueries = VK_TRUE; // For pipeline statistics
		Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_MESH_SHADER_EXTENSION_NAME, &meshShaderFeatures });
		Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME });

		fragmentShadingRateFeatures = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADING_RATE_FEATURES_KHR };
		fragmentShadingRateFeatures.primitiveFragmentShadingRate = VK_TRUE;
		Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_FRAGMENT_SHADING_RATE_EXTENSION_NAME, &fragmentShadingRateFeatures });
	}
}
void MeshRenderer::init() {
	createSourceData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void MeshRenderer::clean() {
	VkDevice device = Application::app->getDevice();

	vkDestroyShaderEXT(device, meshShader, nullptr);
	vkDestroyShaderEXT(device, fragmentShader, nullptr);

	Renderer::clean();
}
void MeshRenderer::uiRender() {
	Application::viewportImage = gBuffers.getDescriptorSet(1);
}
void MeshRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	{
		VkSamplerCreateInfo samplerInfo = DEFAULT_VkSamplerCreateInfo;
		samplerInfo.magFilter = VK_FILTER_NEAREST;
		samplerInfo.minFilter = VK_FILTER_NEAREST;
		samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
		Application::samplerPool.acquireSampler(gBuffers.m_res.gBufferDepth.descriptor.sampler, samplerInfo);

		const VkImageLayout layout{ VK_IMAGE_LAYOUT_GENERAL };
		VkImageMemoryBarrier2 barrier = nvvk::makeImageMemoryBarrier({ .image = gBuffers.m_res.gBufferDepth.image,
														.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
														.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
														.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		const VkDependencyInfo depInfo{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
									   .imageMemoryBarrierCount = 1,
									   .pImageMemoryBarriers = &barrier };

		vkCmdPipelineBarrier2(cmd, &depInfo);

		VkClearDepthStencilValue clearDepth = { 1.0f, 0 };
		VkImageSubresourceRange range = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
		vkCmdClearDepthStencilImage(cmd, gBuffers.m_res.gBufferDepth.image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearDepth, 1, &range);

		// Setting the layout to the final one
		barrier = nvvk::makeImageMemoryBarrier(
			{ .image = gBuffers.m_res.gBufferDepth.image, .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			.newLayout = layout, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
		gBuffers.m_res.gBufferDepth.descriptor.imageLayout = layout;
		vkCmdPipelineBarrier2(cmd, &depInfo);
	}
}
void MeshRenderer::preRender() {

}
void MeshRenderer::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::MeshRendererPushConstant),
	};
	pushInfo.pValues = &pushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t numColorAttachments = 1;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(numColorAttachments);
	for (int i = 0; i < numColorAttachments; ++i) {
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

		colorAttachments[i] = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachments[i].clearValue = { .color = {0, 0, 0, 1.0f} };
		colorAttachments[i].imageView = gBuffers.getColorImageView(i);
	}

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = DEFAULT_VkRect2D(gBuffers.getSize());
	renderingInfo.colorAttachmentCount = colorAttachments.size();
	renderingInfo.pColorAttachments = colorAttachments.data();
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());

	VkColorComponentFlags writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkBool32 blendEnable = VK_FALSE;
	for (uint32_t i = 0; i < numColorAttachments; ++i) {
		vkCmdSetColorWriteMaskEXT(cmd, i, 1, &writeMask);
		vkCmdSetColorBlendEnableEXT(cmd, i, 1, &blendEnable);
	}
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .fragment = fragmentShader, .mesh = meshShader }, true);

	const VkExtent2D fragmentSize{1, 1};
	const VkFragmentShadingRateCombinerOpKHR combinerOps[2] = {
		VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
		VK_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_KHR,
	};
	vkCmdSetFragmentShadingRateKHR(cmd, &fragmentSize, combinerOps);

	vkCmdDrawMeshTasksEXT(cmd, 1, 1, 1);

	vkCmdEndRendering(cmd);

	for (int i = 0; i < numColorAttachments; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo(0), gBuffers.getDescriptorImageInfo(1));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}

void MeshRenderer::createSourceData() {
	Renderer::createGBuffer(true, true, 1, { 1, 1 });
}
void MeshRenderer::createDescriptorSetLayout() {
	Renderer::createDescriptorSetLayout();
}
void MeshRenderer::createDescriptorSet() {

}
void MeshRenderer::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::MeshRendererPushConstant),
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
void MeshRenderer::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::MeshRendererPushConstant),
	};

	std::array<VkDescriptorSetLayout, 1> layouts = { {staticDescPack.getLayout()} };
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
	//---------------------------------------------------------------
	{
		shaderSource = shaderPath / "MeshShader.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, meshShader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_MESH_BIT_EXT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.flags = VK_SHADER_CREATE_NO_TASK_SHADER_BIT_EXT;
		shaderInfo.pName = "meshMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &meshShader);
		NVVK_DBG_NAME(meshShader);

		vkDestroyShaderEXT(device, fragmentShader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.flags = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader);
		NVVK_DBG_NAME(fragmentShader);
	}
}
void MeshRenderer::updateDataPerFrame(VkCommandBuffer cmd) {

}
