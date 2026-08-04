#include "./GridFog.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

GridFog::GridFog(GridFogCreateInfo createInfo, int index, int indexMap) {
	gridFogInfo = createInfo.fogInfo;
	fogIndexMap = indexMap;
	generationInfo = createInfo.generationInfo;

	pushConstant.mode = shaderio::GridFogGenerateMode::Cloud;
	pushConstant.index = index;

	pushConstant.fogRange = createInfo.fogRange;
	pushConstant.gridSize = gridFogInfo.gridSize;
	pushConstant.voxelSize = gridFogInfo.voxelSize;

	pushConstant.absorption = gridFogInfo.absorption;
	pushConstant.scattering = gridFogInfo.scattering;
	pushConstant.phase = gridFogInfo.phase;
	pushConstant.color = gridFogInfo.color;

	pushConstant.generationInfo = generationInfo;
}

void createGridFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear) {
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("gridFog3DTexture" + std::to_string(imageCount));
	++imageCount;

	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_3D;
	colorImageCreateInfo.info.extent = { size.x, size.y, size.z };

	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;

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
	//colorImageCreateInfo.samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
	//colorImageCreateInfo.samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	//colorImageCreateInfo.samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
	//colorImageCreateInfo.samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;

	image.init(colorImageCreateInfo);
}
void GridFog::init() {
	shaderio::uint3 gridSize = gridFogInfo.gridSize;
	createGridFogImage(gridFogImage, gridSize, true);
	{
		gridFogGenerationDataBuffer = FzbRenderer::Buffer("GridFogGenerationDataBuffer" + std::to_string(fogIndexMap), false);
		gridFogGenerationDataBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::GridFogGenerationInfo),
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT
		});
	}
}
void GridFog::clean() {
	gridFogGenerationDataBuffer.clean();
	gridFogImage.clean();
}
void GridFog::uiRender(int i) {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;

	int fogIndex = fogIndexMap;

	ImGui::BeginDisabled(true);
	fogInfoModified |= ImGui::DragInt3(std::string("Grid Fog Grid Size " + std::to_string(i)).c_str(), (int*)&gridFogInfo.gridSize);
	ImGui::EndDisabled();
	fogInfoModified |= ImGui::DragFloat3(std::string("Grid Fog Voxel Size " + std::to_string(i)).c_str(), (float*)&gridFogInfo.voxelSize);

	bool modified = false;
	modified |= ImGui::DragFloat3(std::string("Grid Fog Color " + std::to_string(i)).c_str(), (float*)&gridFogInfo.color);
	modified |= ImGui::DragFloat(std::string("Grid Fog Ambient Intensity " + std::to_string(i)).c_str(), (float*)&gridFogInfo.ambientIntensity, 0.1f, 0.0f, 10.0f);
	modified |= ImGui::DragFloat(std::string("Grid Fog Extinction Coefficient " + std::to_string(i)).c_str(), (float*)&gridFogInfo.absorption, 0.1f, 0.0f, 100.0f, "%.3f", ImGuiSliderFlags_ClampOnInput);
	modified |= ImGui::DragFloat(std::string("Grid Fog Scatter Coefficient " + std::to_string(i)).c_str(), (float*)&gridFogInfo.scattering, 0.1f, 0.0f, 1.0f);
	modified |= ImGui::DragFloat(std::string("Grid Fog Asymmetric Parameters " + std::to_string(i)).c_str(), (float*)&gridFogInfo.phase, 0.1f, -1.0f, 1.0f);

	fogInfoModified |= modified;
	reGeneration |= modified;

	const char* generationTypeItems[] = { "Cloud" };
	reGeneration |= ImGui::Combo(std::string("Type " + std::to_string(i)).c_str(), (int*)&pushConstant.mode, generationTypeItems, IM_ARRAYSIZE(generationTypeItems));
	if(pushConstant.mode == shaderio::GridFogGenerateMode::Cloud) {
		if (ImGui::CollapsingHeader(std::string("Cloud Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			reGeneration |= ImGui::DragFloat3(std::string("Cloud Scale " + std::to_string(i)).c_str(), (float*)&generationInfo.cloudInfo.cloudScale, 0.1f, 0.0f, 100.0f);
			reGeneration |= ImGui::DragFloat(std::string("Cloud Flow Speed " + std::to_string(i)).c_str(), (float*)&generationInfo.cloudInfo.cloudFlowSpeed, 0.1f, 0.0f, 10.0f);
			reGeneration |= ImGui::DragFloat2(std::string("Cloud Coverage " + std::to_string(i)).c_str(), (float*)&generationInfo.cloudInfo.cloudCoverage, 0.1f, 0.0f, 2.0f);
			reGeneration |= ImGui::DragFloat2(std::string("Cloud Type Preference " + std::to_string(i)).c_str(), (float*)&generationInfo.cloudInfo.cloudTypePreference, 0.1f, 0.0f, 1.0f);
			reGeneration |= ImGui::DragFloat(std::string("Weather Scale " + std::to_string(i)).c_str(), (float*)&generationInfo.cloudInfo.weatherScale, 0.01f, 0.0f, 2.0f);
		}
	}

	if (ImGui::SmallButton(std::string("Grid Fog Regenerate, seed: " + std::to_string(pushConstant.randomSeed)).c_str())) {
		reGeneration = true;
		pushConstant.randomSeed = Application::frameIndex;
	}
	
	fogInfoModified |= ImGui::Checkbox(std::string("Grid Fog show voxel grid " + std::to_string(i)).c_str(), (bool*)&showFogGrid);
}

void GridFog::updateDataPerFrame(VkCommandBuffer cmd, shaderio::AABB fogRange) {
	pushConstant.fogRange = fogRange;
	pushConstant.gridSize = gridFogInfo.gridSize;
	pushConstant.voxelSize = gridFogInfo.voxelSize;

	pushConstant.absorption = gridFogInfo.absorption;
	pushConstant.scattering = gridFogInfo.scattering;
	pushConstant.phase = gridFogInfo.phase;
	pushConstant.color = gridFogInfo.color;
	pushConstant.ambientIntensity = gridFogInfo.ambientIntensity;

	pushConstant.generationInfo = generationInfo;
}
//--------------------------------------------------------------------------------------------------------------------------------
GridFogSet::GridFogSet(GridFogSetCreateInfo createInfo) {
	this->setting = createInfo;
}
void GridFogSet::init() {
	if (fogCount == 0) return;
	gridFogInfoBuffer = FzbRenderer::Buffer("gridFogInfoBuffer", false);
	gridFogInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::GridFogInfo) * fogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	for (int i = 0; i < fogCount; ++i) {
		gridFogs[i].init();
		gridFogImages.push_back(gridFogs[i].gridFogImage.image);
	}

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Feature::init();

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	for (int i = 0; i < fogCount; ++i) {
		GridFog& gridFog = gridFogs[i];
		shaderio::uint3 gridSize = gridFog.gridFogInfo.gridSize;

		VkPushConstantsInfo pushInfo = {
			.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
			.layout = pipelineLayout,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = sizeof(shaderio::GridFogPushConstant),
			.pValues = &gridFog.pushConstant,
		};

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initGridFogImage);
		vkCmdPushConstants2(cmd, &pushInfo);
		VkExtent3D threadGroupSize = nvvk::getGroupCounts(VkExtent3D{ gridSize.x, gridSize.y, gridSize.z }, VkExtent3D{ 4, 4, 4 });
		vkCmdDispatch(cmd, threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);
	}
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void GridFogSet::clean() {
	gridFogInfoBuffer.clean();
	for (int i = 0; i < fogCount; ++i) gridFogs[i].clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initGridFogImage, nullptr);

	Feature::clean();
}
void GridFogSet::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	for (int i = 0; i < fogCount; ++i) {
		GridFog& gridFog = gridFogs[i];
		gridFog.fogInfoModified = false;
		gridFog.reGeneration = false;

		if (ImGui::CollapsingHeader(std::string("Grid Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text(std::string("Grid Fog " + std::to_string(i)).c_str());
			int fogIndex = gridFog.fogIndexMap;

			gridFog.fogInfoModified |= ImGui::DragFloat3(std::string("Grid Fog Start Pos " + std::to_string(i)).c_str(), (float*)&(*setting.fogInfos)[fogIndex].aabb.minimum);
			gridFogs[i].uiRender(i);
			(*setting.fogInfos)[fogIndex].aabb.maximum = (*setting.fogInfos)[fogIndex].aabb.minimum + gridFog.gridFogInfo.voxelSize * (shaderio::float3)gridFog.gridFogInfo.gridSize;
		}
	}
}
void GridFogSet::preRender() {}

void GridFogSet::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_GridFogSet::eGridFogImages,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = (uint32_t)fogCount,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void GridFogSet::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	VkWriteDescriptorSet	dsWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_GridFogSet::eGridFogImages, 0, 0, (uint32_t)fogCount);
	write.append(dsWrite, gridFogImages.data());

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void GridFogSet::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::GridFogPushConstant)
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
void GridFogSet::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::GridFogPushConstant),
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

	shaderSource = shaderPath / "initGridFogImage.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, computeShader_initGridFogImage, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_initGridFogImage";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initGridFogImage);
	NVVK_DBG_NAME(computeShader_initGridFogImage);
}

void GridFogSet::updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer) {
	if (fogCount == 0) return;
	nvvk::cmdBufferMemoryBarrier(cmd, { gridFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	for (int i = 0; i < fogCount; ++i) {
		GridFog& gridFog = gridFogs[i];
		gridFog.updateDataPerFrame(cmd, (*setting.fogInfos)[gridFog.fogIndexMap].aabb);
		if (gridFog.reGeneration) {
			nvvk::cmdImageMemoryBarrier(cmd, { gridFog.gridFogImage.image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL });

			vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
			VkPushConstantsInfo pushInfo = {
				.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
				.layout = pipelineLayout,
				.stageFlags = VK_SHADER_STAGE_ALL,
				.offset = 0,
				.size = sizeof(shaderio::GridFogPushConstant),
				.pValues = &gridFog.pushConstant,
			};

			VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initGridFogImage);
			vkCmdPushConstants2(cmd, &pushInfo);
			shaderio::uint3 gridSize = gridFog.gridFogInfo.gridSize;
			VkExtent3D threadGroupSize = nvvk::getGroupCounts(VkExtent3D{ gridSize.x, gridSize.y, gridSize.z }, VkExtent3D{ 4, 4, 4 });
			vkCmdDispatch(cmd, threadGroupSize.width, threadGroupSize.height, threadGroupSize.depth);

			nvvk::cmdImageMemoryBarrier(cmd, { gridFog.gridFogImage.image.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL });
		}

		if (!gridFog.fogInfoModified && !frist) continue;
		vkCmdUpdateBuffer(cmd, gridFogInfoBuffer.buffer.buffer, sizeof(shaderio::GridFogInfo) * i, sizeof(shaderio::GridFogInfo), &gridFog.gridFogInfo);

		int fogIndex = gridFog.fogIndexMap;
		vkCmdUpdateBuffer(cmd, fogInfoBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * fogIndex, sizeof(shaderio::VolumetricFogInfo), &(*setting.fogInfos)[fogIndex]);
	}
	nvvk::cmdBufferMemoryBarrier(cmd, { gridFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });


}
void GridFogSet::addFog(GridFogCreateInfo createInfo) {
	if (fogCount == MAX_GRID_FOG_COUNT) {
		printf("Grid fog count exceeds limit, please expand MAX_GRID_FOG_COUNT\n");
		return;
	}

	shaderio::VolumetricFogInfo volumetricFogInfo = {
		.aabb = createInfo.fogRange,
		.type = shaderio::VolumetricFogType::Grid,
		.typeFogIndex = fogCount,
	};
	setting.fogInfos->push_back(volumetricFogInfo);
	gridFogs.push_back(GridFog(createInfo, fogCount, setting.fogInfos->size() - 1));

	++fogCount;
}