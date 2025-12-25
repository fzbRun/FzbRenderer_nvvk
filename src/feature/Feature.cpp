#include "Feature.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>

void FzbRenderer::Feature::clean() {
	VkDevice device = Application::app->getDevice();
	descPack.deinit();
	vkDestroyPipelineLayout(device, graphicPipelineLayout, nullptr);
	gBuffers.deinit();

	scene.clean();
}
void FzbRenderer::Feature::uiRender() {};
void FzbRenderer::Feature::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));
};
void FzbRenderer::Feature::preRender() {};

void FzbRenderer::Feature::createGBuffer(bool useDepth, bool postProcess, uint32_t colorAttachmentCount) {
	VkSampler linearSampler{};
	NVVK_CHECK(Application::samplerPool.acquireSampler(linearSampler));
	NVVK_DBG_NAME(linearSampler);

	std::vector<VkFormat> colorAttachmentFormat(colorAttachmentCount);
	for (int i = 0; i < colorAttachmentCount; ++i) colorAttachmentFormat[i] = VK_FORMAT_R32G32B32A32_SFLOAT;
	if (postProcess) colorAttachmentFormat.push_back(VK_FORMAT_R8G8B8A8_UNORM);
	nvvk::GBufferInitInfo gBufferInit{
		.allocator = &Application::allocator,
		.colorFormats = colorAttachmentFormat,
		.imageSampler = linearSampler,
		.descriptorPool = Application::app->getTextureDescriptorPool(),
	};
	if (useDepth) gBufferInit.depthFormat = nvvk::findDepthFormat(Application::app->getPhysicalDevice());

	gBuffers.init(gBufferInit);
};
void FzbRenderer::Feature::createGraphicsDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::BindingPoints::eTextures,
						 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						 .descriptorCount = 10,
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	descPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(descPack.getLayout());
	NVVK_DBG_NAME(descPack.getPool());
	NVVK_DBG_NAME(descPack.getSet(0));
}
void FzbRenderer::Feature::createGraphicsPipelineLayout(uint32_t pushConstantSize) {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
		.offset = 0,
		.size = pushConstantSize
	};

	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = descPack.getLayoutPtr(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &graphicPipelineLayout));
	NVVK_DBG_NAME(graphicPipelineLayout);
}
void FzbRenderer::Feature::addTextureArrayDescriptor(uint32_t textureBinding) {
	if (Application::sceneResource.textures.empty())
		return;

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    allTextures =
		descPack.makeWrite(textureBinding, 0, 0, uint32_t(Application::sceneResource.textures.size()));
	nvvk::Image* allImages = Application::sceneResource.textures.data();
	write.append(allTextures, allImages);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void FzbRenderer::Feature::compileAndCreateShaders() {};
void FzbRenderer::Feature::updateDataPerFrame(VkCommandBuffer cmd) {

};