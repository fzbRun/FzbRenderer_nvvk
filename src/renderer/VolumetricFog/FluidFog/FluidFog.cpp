#include "./FluidFog.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>

using namespace FzbRenderer;

void createFluidFogImage(FzbRenderer::Image& image, shaderio::uint3 size, bool linear) {
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("fluidFog3DTexture" + std::to_string(imageCount));
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

void FluidFog::init() {
	shaderio::uint3 gridSize = fluidFogInfo.gridSize;
	createFluidFogImage(fluidFogVoxelVelocityImage, gridSize, true);

	fluidFogVoxelInfoBuffer = FzbRenderer::Buffer("FluidFogVoxelInfoBuffer" + std::to_string(fogIndexMap), false);
	fluidFogVoxelInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::FluidFogVoxelInfo) * gridSize.x * gridSize.y * gridSize.z,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});

	createFluidFogImage(fluidFogVoxelInfoImage, gridSize, true);

	if (follow) {
		std::pair<uint32_t, uint32_t> instanceSetPair = Application::sceneResource.instanceIDToInstanceSet[followInstanceID];
		InstanceSet mainCharacter = Application::sceneResource.getInstanceSet((InstanceType)instanceSetPair.first, instanceSetPair.second);
		shaderio::AABB mainCharacterAABB;
		mainCharacterAABB.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		mainCharacterAABB.maximum = -mainCharacterAABB.minimum;
		for (int i = 0; i < mainCharacter.childInstances.size(); ++i) {
			shaderio::Instance childInstance = mainCharacter.childInstances[i];
			uint32_t meshIndex = childInstance.meshIndex;
			MeshInfo& meshInfo = Application::sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB childAABB = meshInfo.getAABB();
			{
				mainCharacterAABB.minimum.x = std::min(mainCharacterAABB.minimum.x, childAABB.minimum.x);
				mainCharacterAABB.minimum.y = std::min(mainCharacterAABB.minimum.y, childAABB.minimum.y);
				mainCharacterAABB.minimum.z = std::min(mainCharacterAABB.minimum.z, childAABB.minimum.z);
				mainCharacterAABB.maximum.x = std::max(mainCharacterAABB.maximum.x, childAABB.maximum.x);
				mainCharacterAABB.maximum.y = std::max(mainCharacterAABB.maximum.y, childAABB.maximum.y);
				mainCharacterAABB.maximum.z = std::max(mainCharacterAABB.maximum.z, childAABB.maximum.z);
			}
		}
		fluidLocalStartPos = (mainCharacterAABB.minimum + mainCharacterAABB.maximum) * 0.5f;
		fluidLocalStartPos.y = 0.0f;
	}
}
void FluidFog::clean() {
	fluidFogVoxelInfoBuffer.clean();
	fluidFogVoxelVelocityImage.clean();
	fluidFogVoxelInfoImage.clean();
}
void FluidFog::uiRender(int i) {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;

	int fogIndex = fogIndexMap;

	fogInfoModified |= ImGui::Checkbox(std::string("Fluid Fog Start Up " + std::to_string(i)).c_str(), (bool*)&fluidFogInfo.startUp);

	ImGui::BeginDisabled(true);
	fogInfoModified |= ImGui::DragInt3(std::string("Fluid Fog Grid Size " + std::to_string(i)).c_str(), (int*)&fluidFogInfo.gridSize);
	ImGui::EndDisabled();
	fogInfoModified |= ImGui::DragFloat3(std::string("Fluid Fog Voxel Size " + std::to_string(i)).c_str(), (float*)&fluidFogInfo.voxelSize);

	fogInfoModified |= ImGui::DragFloat3(std::string("Fluid Fog Magic Number " + std::to_string(i)).c_str(), (float*)&fluidFogInfo.magicNumber, 0.1f, 0.0f, 100.0f);
	fogInfoModified |= ImGui::DragFloat(std::string("Fluid Fog Viscosity " + std::to_string(i)).c_str(), (float*)&fluidFogInfo.viscosity, 0.1f, 0.0f, 1.0f);

	fogInfoModified |= ImGui::DragFloat(std::string("Fluid Fog F Intensity " + std::to_string(i)).c_str(), (float*)&fluidFogInfo.FIntensity, 1.0f, 0.0f, 100.0f);
	fogInfoModified |= ImGui::DragFloat(std::string("Fluid Fog Restore Speed " + std::to_string(i)).c_str(), (float*)&fluidFogInfo.restoreSpeed, 1.0f, 0.0f, 100.0f);

	fogInfoModified |= ImGui::Checkbox(std::string("Fluid Fog show voxel grid " + std::to_string(i)).c_str(), (bool*)&showFogGrid);
}
void FluidFog::preRender(shaderio::AABB& fogAABB) {
	if (follow) {
		std::pair<uint32_t, uint32_t> instanceSetPair = Application::sceneResource.instanceIDToInstanceSet[followInstanceID];
		InstanceSet mainCharacter = Application::sceneResource.getInstanceSet((InstanceType)instanceSetPair.first, instanceSetPair.second);
		shaderio::float3 fogStartPos = mainCharacter.transform * shaderio::float4(fluidLocalStartPos, 1.0f);
		shaderio::float3 fogStartPos_lastTime = mainCharacter.transform_lastTime * shaderio::float4(fluidLocalStartPos, 1.0f);

		uint32_t fogIndex = fogIndexMap;
		fogStartPos -= shaderio::float3(0.5f, 0.0f, 0.5f) * (shaderio::float3)fluidFogInfo.gridSize * fluidFogInfo.voxelSize;
		fogStartPos -= 3.0f * fluidFogInfo.voxelSize.y;
		fogAABB.minimum = fogStartPos;
		fogAABB.maximum = fogStartPos + (shaderio::float3)fluidFogInfo.gridSize * fluidFogInfo.voxelSize;

		fogStartPos_lastTime -= shaderio::float3(0.5f, 0.0f, 0.5f) * (shaderio::float3)fluidFogInfo.gridSize * fluidFogInfo.voxelSize;
		fogStartPos_lastTime -= 3.0f * fluidFogInfo.voxelSize.y;
		fluidFogInfo.fogStartPos_lastTime = fogStartPos_lastTime;
	}
}
//--------------------------------------------------------------------------------------------------------------------------------
FluidFogSet::FluidFogSet(FluidFogSetCreateInfo createInfo) {
	this->setting = createInfo;
}

void FluidFogSet::init() {
	if (fogCount == 0) return;
	fluidFogInfoBuffer = FzbRenderer::Buffer("FluidFogInfoBuffer", false);
	fluidFogInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::FluidFogInfo) * fogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});

	for (int i = 0; i < fogCount; ++i) {
		fluidFogs[i].init();

		fluidFogVoxelInfoBuffers.push_back(fluidFogs[i].fluidFogVoxelInfoBuffer.buffer);
		fluidFogVoxelVelocityImages.push_back(fluidFogs[i].fluidFogVoxelVelocityImage.image);
		fluidFogVoxelInfoImages.push_back(fluidFogs[i].fluidFogVoxelInfoImage.image);
	}
}
void FluidFogSet::clean() {
	fluidFogInfoBuffer.clean();
	for (int i = 0; i < fogCount; ++i) fluidFogs[i].clean();
}
void FluidFogSet::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	for (int i = 0; i < fogCount; ++i) {
		FluidFog& fluidFog = fluidFogs[i];

		fluidFog.fogInfoModified = false;
		if (ImGui::CollapsingHeader(std::string("Fluid Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text(std::string("Fluid Fog " + std::to_string(i)).c_str());
			int fogIndex = fluidFog.fogIndexMap;

			if (!fluidFog.follow) 
				fluidFog.fogInfoModified |= ImGui::DragFloat3(std::string("Fluid Fog Start Pos " + std::to_string(i)).c_str(), (float*)&(*setting.fogInfos)[fogIndex].aabb.minimum);

			fluidFogs[i].uiRender(i);

			if (!fluidFog.follow)
				(*setting.fogInfos)[fogIndex].aabb.maximum = (*setting.fogInfos)[fogIndex].aabb.minimum + fluidFog.fluidFogInfo.voxelSize * (shaderio::float3)fluidFog.fluidFogInfo.gridSize;
		}
	}
}
void FluidFogSet::preRender() {
	for (int i = 0; i < fogCount; ++i) {
		FluidFog& fluidFog = fluidFogs[i];
		fluidFog.preRender((*setting.fogInfos)[fluidFog.fogIndexMap].aabb);
	}
}

void FluidFogSet::updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer) {
	if (fogCount == 0) return;
	nvvk::cmdBufferMemoryBarrier(cmd, { fluidFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	for (int i = 0; i < fogCount; ++i) {
		FluidFog& fluidFog = fluidFogs[i];

		if (!fluidFog.follow && !fluidFog.fogInfoModified && !frist) continue;
		vkCmdUpdateBuffer(cmd, fluidFogInfoBuffer.buffer.buffer, sizeof(shaderio::FluidFogInfo) * i, sizeof(shaderio::FluidFogInfo), &fluidFog.fluidFogInfo);

		int fogIndex = fluidFog.fogIndexMap;
		vkCmdUpdateBuffer(cmd, fogInfoBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * fogIndex, sizeof(shaderio::VolumetricFogInfo), &(*setting.fogInfos)[fogIndex]);
	}
	nvvk::cmdBufferMemoryBarrier(cmd, { fluidFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
}
void FluidFogSet::addFog(FluidFogCreateInfo createInfo) {
	if (fogCount == MAX_FLUID_FOG_COUNT) {
		printf("Fluid fog count exceeds limit, please expand MAX_FLUID_FOG_COUNT\n");
		return;
	}

	shaderio::VolumetricFogInfo volumetricFogInfo = {
		.aabb = createInfo.fogRange,
		.type = shaderio::VolumetricFogType::Fluid,
		.typeFogIndex = fogCount,
	};
	setting.fogInfos->push_back(volumetricFogInfo);

	FluidFog fluidFog;
	fluidFog.fluidFogInfo = createInfo.fogInfo;
	fluidFog.fogIndexMap = setting.fogInfos->size() - 1;
	fluidFog.follow = createInfo.follow;
	fluidFog.followInstanceID = createInfo.followInstanceID;
	fluidFogs.push_back(fluidFog);

	++fogCount;
}
