#include "./NeuralTextureCompression.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <cstdlib>

using namespace FzbRenderer;

NeuralTextureCompression::NeuralTextureCompression(pugi::xml_node& rendererNode) {
    g_coopVecFeatures.cooperativeVector = VK_TRUE;
    g_coopVecFeatures.cooperativeVectorTraining = VK_FALSE;  // 只在真训练时开
    Application::vkContextInitInfo.deviceExtensions.push_back({ VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME, &g_coopVecFeatures, /*required=*/false });

	Application::renderResolution = { 512, 512 };
}

void NeuralTextureCompression::init() {
    const bool hasCoopVec = Application::vkContext->hasExtensionEnabled(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
    LOGI("NeuralTextureCompression: coopvec = %s\n", hasCoopVec ? "on" : "off (scalar fallback)");

	createSourceData();
    createDescriptorSetLayout();
    createDescriptorSet();
    createPipelineLayout();
    compileAndCreateShaders();

    Renderer::init();

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	{
		VkImageBlit2 blitRegion{};
		{
			blitRegion.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
			blitRegion.pNext = nullptr;

			// 定义源子资源：颜色，mip 0，数组层 0
			blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blitRegion.srcSubresource.mipLevel = 0;
			blitRegion.srcSubresource.baseArrayLayer = 0;
			blitRegion.srcSubresource.layerCount = 1;
			// 定义源区域：从 (0,0,0) 到 (width, height, 1)
			blitRegion.srcOffsets[0] = { 0, 0, 0 };
			blitRegion.srcOffsets[1] = { 2048, 2048, 1 };

			// 定义目标子资源（参数通常与源相同）
			blitRegion.dstSubresource = blitRegion.srcSubresource;
			// 定义目标区域：缩放到目标图像的完整尺寸
			blitRegion.dstOffsets[0] = { 0, 0, 0 };
			blitRegion.dstOffsets[1] = { 1024, 1024, 1 };
		}
		VkBlitImageInfo2 blitInfo{};
		{
			blitInfo.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
			blitInfo.pNext = nullptr;
			blitInfo.srcImage = nullptr;
			blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
			blitInfo.dstImage = nullptr;
			blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
			blitInfo.regionCount = 1;
			blitInfo.pRegions = &blitRegion;
			blitInfo.filter = VK_FILTER_LINEAR; // 使用线性过滤进行平滑缩放
		}
		{
			blitRegion.srcOffsets[1] = { 2048, 2048, 1 };
			blitRegion.dstOffsets[1] = { 1024, 1024, 1 };

			blitInfo.srcImage = sourceTextures[0].image.image;
			blitInfo.dstImage = mipmapSourceTempTextures[0].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);

			blitInfo.srcImage = sourceTextures[1].image.image;
			blitInfo.dstImage = mipmapSourceTempTextures[1].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);

			blitInfo.srcImage = sourceTextures[2].image.image;
			blitInfo.dstImage = mipmapSourceTempTextures[2].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);

			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		}
		{
			blitRegion.srcOffsets[1] = { 1024, 1024, 1 };
			blitRegion.dstOffsets[1] = { 512, 512, 1 };

			blitInfo.srcImage = mipmapSourceTempTextures[0].image.image;
			blitInfo.dstImage = mipmapSourceTextures[0].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);

			blitInfo.srcImage = mipmapSourceTempTextures[1].image.image;
			blitInfo.dstImage = mipmapSourceTextures[1].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);

			blitInfo.srcImage = mipmapSourceTempTextures[2].image.image;
			blitInfo.dstImage = mipmapSourceTextures[2].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);
		}
	}
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void NeuralTextureCompression::clean() {
    VkDevice device = Application::app->getDevice();
    vkDestroyShaderEXT(device, computeShader_AutoDiff, nullptr);
    vkDestroyShaderEXT(device, computeShader_ReducedSizeMipmap, nullptr);
    vkDestroyShaderEXT(device, computeShader_FullSizeMipmap, nullptr);
    vkDestroyShaderEXT(device, computeShader_getLossTexture, nullptr);
    vkDestroyShaderEXT(device, computeShader_tonemapped, nullptr);

	for (int i = 0; i < sourceTextures.size(); ++i) sourceTextures[i].clean();
	for (int i = 0; i < mipmapSourceTempTextures.size(); ++i) mipmapSourceTempTextures[i].clean();
	for (int i = 0; i < mipmapSourceTextures.size(); ++i) mipmapSourceTextures[i].clean();
	mipmapFullSizeTexture.clean();

    Renderer::clean();
}
void NeuralTextureCompression::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_NTC::eImgTonemapped);

	if (ImGui::Begin("NTCSettings")) {
		ImGui::Checkbox("mipmap full size", &mipmapFullSize);
		ImGui::Checkbox("loss texture", &showLossTexture);
	}
	ImGui::End();

	pushConstant.showTextureType = showLossTexture ? 2 : (mipmapFullSize ? 1 : 0);
}
void NeuralTextureCompression::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Reduced, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_NTC::eImgRendered_Reduced), VK_IMAGE_LAYOUT_GENERAL);

	OutImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Full_2, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_NTC::eImgRendered_Full), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    lossTextureWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eLossTexture, 0, 0, 1);
	write.append(lossTextureWrite, gBuffers.getColorImageView((uint32_t)ImageType_NTC::eLossImage), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    tonemappedTextureWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eTonemappedTexture, 0, 0, 1);
	write.append(tonemappedTextureWrite, gBuffers.getColorImageView((uint32_t)ImageType_NTC::eImgTonemapped), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.screenSize = shaderio::int2(size.width, size.height);
};
void NeuralTextureCompression::preRender() {
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
};
void NeuralTextureCompression::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};

	//AutoDiff(cmd);
	Mipmap(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	//VkDescriptorImageInfo imageInfo = mipmapFullSize ? gBuffers.getDescriptorImageInfo((uint32_t)ImageType_NTC::eImgRendered_Full) : gBuffers.getDescriptorImageInfo((uint32_t)ImageType_NTC::eImgRendered_Reduced);
	//imageInfo = showLossTexture ? gBuffers.getDescriptorImageInfo((uint32_t)ImageType_NTC::eLossImage) : imageInfo;
	//Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, imageInfo, gBuffers.getDescriptorImageInfo((uint32_t)ImageType_NTC::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}

void createImage_NTC(FzbRenderer::Image& image, std::filesystem::path imagePath, bool linearSample, bool sRGB, bool linearizeToFloat) {
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("texture" + std::to_string(imageCount));
	++imageCount;

	VkSamplerCreateInfo samplerCreateInfo = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
	if (linearSample) {
		samplerCreateInfo.magFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.minFilter = VK_FILTER_LINEAR;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
	else {
		samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
		samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
		samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	}

	image.init(imagePath, &samplerCreateInfo, sRGB, linearizeToFloat);
}
void createImage_NTC2(FzbRenderer::Image& image, shaderio::uint3 size, VkFormat format, bool linear) {
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("texture_2_" + std::to_string(imageCount));
	++imageCount;

	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = format;
	colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_2D;
	colorImageCreateInfo.info.extent = { size.x, size.y, size.z };

	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;

	if (linear) {
		colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_LINEAR;
		colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_LINEAR;
		colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	}
	else {
		colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_NEAREST;
		colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_NEAREST;
		colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	}
	colorImageCreateInfo.samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	image.init(colorImageCreateInfo);
}
void NeuralTextureCompression::createSourceData() {
	Feature::createGBuffer(false, true, (uint32_t)ImageType_NTC::eImgTonemapped, Application::renderResolution);

	{
		sourceTextures.resize(3);
		createImage_NTC(sourceTextures[0], FzbRenderer::getProjectRootDir() / "resources/NeuralTextureCompressSource/PavingStones070_2K.diffuse.jpg", true, true, true);
		createImage_NTC(sourceTextures[1], FzbRenderer::getProjectRootDir() / "resources/NeuralTextureCompressSource/PavingStones070_2K.normal.jpg", true, false, false);
		createImage_NTC(sourceTextures[2], FzbRenderer::getProjectRootDir() / "resources/NeuralTextureCompressSource/PavingStones070_2K.roughness.jpg", true, false, false);

		createImage_NTC2(mipmapFullSizeTexture, { 2048, 2048, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
	}
	{
		mipmapSourceTempTextures.resize(3);
		createImage_NTC2(mipmapSourceTempTextures[0], { 1024, 1024, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
		createImage_NTC2(mipmapSourceTempTextures[1], { 1024, 1024, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
		createImage_NTC2(mipmapSourceTempTextures[2], { 1024, 1024, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);

		mipmapSourceTextures.resize(3);
		createImage_NTC2(mipmapSourceTextures[0], { 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
		createImage_NTC2(mipmapSourceTextures[1], { 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
		createImage_NTC2(mipmapSourceTextures[2], { 512, 512, 1 }, VK_FORMAT_R32G32B32A32_SFLOAT, true);
	}
}
void NeuralTextureCompression::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eSourceTextures,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = (uint32_t)sourceTextures.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eMipmapSourceTextures,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = (uint32_t)mipmapSourceTextures.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Reduced,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Full,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Full_2,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eLossTexture,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_NTC::eTonemappedTexture,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void NeuralTextureCompression::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    sourceTexturesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eSourceTextures, 0, 0, sourceTextures.size());
	std::vector<nvvk::Image> textures(sourceTextures.size());
	for (size_t i = 0; i < sourceTextures.size(); ++i) textures[i] = sourceTextures[i].image;
	write.append(sourceTexturesWrite, textures.data());

	VkWriteDescriptorSet    mipmapSourceTexturesWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eMipmapSourceTextures, 0, 0, mipmapSourceTextures.size());
	for (size_t i = 0; i < mipmapSourceTextures.size(); ++i) textures[i] = mipmapSourceTextures[i].image;
	write.append(mipmapSourceTexturesWrite, textures.data());

	VkWriteDescriptorSet    fullSizeRnederedTextureWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_NTC::eRenderedTexture_Full, 0, 0, 1);
	write.append(fullSizeRnederedTextureWrite, mipmapFullSizeTexture.image);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void NeuralTextureCompression::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
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
void NeuralTextureCompression::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = 256,
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
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "AutoDiff.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_AutoDiff, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_AutoDiff";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_AutoDiff);
		NVVK_DBG_NAME(computeShader_AutoDiff);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "Mipmap.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_ReducedSizeMipmap, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_ReducedSizeMipmap";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_ReducedSizeMipmap);
		NVVK_DBG_NAME(computeShader_ReducedSizeMipmap);

		vkDestroyShaderEXT(device, computeShader_FullSizeMipmap, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_FullSizeMipmap";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_FullSizeMipmap);
		NVVK_DBG_NAME(computeShader_FullSizeMipmap);

		vkDestroyShaderEXT(device, computeShader_getLossTexture, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_getLossTexture";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getLossTexture);
		NVVK_DBG_NAME(computeShader_getLossTexture);

		vkDestroyShaderEXT(device, computeShader_tonemapped, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_tonemapped";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_tonemapped);
		NVVK_DBG_NAME(computeShader_tonemapped);
	}
}
void NeuralTextureCompression::updateDataPerFrame(VkCommandBuffer cmd) {}

void NeuralTextureCompression::AutoDiff(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_AutoDiff);
	vkCmdDispatch(cmd, 1, 1, 1);
}
void NeuralTextureCompression::Mipmap(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	VkExtent2D textureSize = { sourceTextures[0].setting.info.extent.width, sourceTextures[0].setting.info.extent.height };
	VkExtent2D screenSize = Application::renderResolution;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_ReducedSizeMipmap);
	VkExtent2D groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ 16, 16 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	//---------------------------------------------------------------------------------------------------------------
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_FullSizeMipmap);
	groupSize = nvvk::getGroupCounts(textureSize, VkExtent2D{ 16, 16 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	//---------------------------------------------------------------------------------------------------------------
	{
		VkImageBlit2 blitRegion{};
		{
			blitRegion.sType = VK_STRUCTURE_TYPE_IMAGE_BLIT_2;
			blitRegion.pNext = nullptr;

			// 定义源子资源：颜色，mip 0，数组层 0
			blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
			blitRegion.srcSubresource.mipLevel = 0;
			blitRegion.srcSubresource.baseArrayLayer = 0;
			blitRegion.srcSubresource.layerCount = 1;
			// 定义源区域：从 (0,0,0) 到 (width, height, 1)
			blitRegion.srcOffsets[0] = { 0, 0, 0 };
			blitRegion.srcOffsets[1] = { static_cast<int32_t>(screenSize.width), static_cast<int32_t>(screenSize.height), 1 };

			// 定义目标子资源（参数通常与源相同）
			blitRegion.dstSubresource = blitRegion.srcSubresource;
			// 定义目标区域：缩放到目标图像的完整尺寸
			blitRegion.dstOffsets[0] = { 0, 0, 0 };
			blitRegion.dstOffsets[1] = { (int32_t)gBuffers.getSize().width, (int32_t)gBuffers.getSize().height, 1 };
		}
		VkBlitImageInfo2 blitInfo{};
		{
			blitInfo.sType = VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2;
			blitInfo.pNext = nullptr;
			blitInfo.srcImage = mipmapFullSizeTexture.image.image;
			blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_GENERAL;
			blitInfo.dstImage = gBuffers.getColorImage((uint32_t)ImageType_NTC::eImgRendered_Full);
			blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_GENERAL;
			blitInfo.regionCount = 1;
			blitInfo.pRegions = &blitRegion;
			blitInfo.filter = VK_FILTER_LINEAR; // 使用线性过滤进行平滑缩放
		}
		{
			blitRegion.srcOffsets[1] = { 2048, 2048, 1 };
			blitRegion.dstOffsets[1] = { 1024, 1024, 1 };

			blitInfo.srcImage = mipmapFullSizeTexture.image.image;
			blitInfo.dstImage = mipmapSourceTempTextures[0].image.image;
			vkCmdBlitImage2(cmd, &blitInfo);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

			blitRegion.srcOffsets[1] = { 1024, 1024, 1 };
			blitRegion.dstOffsets[1] = { 512, 512, 1 };

			blitInfo.srcImage = mipmapSourceTempTextures[0].image.image;
			blitInfo.dstImage = gBuffers.getColorImage((uint32_t)ImageType_NTC::eImgRendered_Full);
			vkCmdBlitImage2(cmd, &blitInfo);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		}	
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	//---------------------------------------------------------------------------------------------------------------
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getLossTexture);
	screenSize = Application::renderResolution;
	groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ 16, 16 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	//---------------------------------------------------------------------------------------------------------------
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_tonemapped);
	screenSize = Application::renderResolution;
	groupSize = nvvk::getGroupCounts(screenSize, VkExtent2D{ 16, 16 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}