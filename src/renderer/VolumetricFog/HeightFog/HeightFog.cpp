#include "./HeightFog.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>

using namespace FzbRenderer;

HeightFogSet::HeightFogSet(HeightFogSetCreateInfo createInfo) {
	this->setting = createInfo;
}

void HeightFogSet::init() {
	if (fogCount == 0) return;
	heightFogInfoBuffer = FzbRenderer::Buffer("HeightFogInfoBuffer", false);
	heightFogInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::HeightFogInfo) * fogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});
}
void HeightFogSet::clean() {
	heightFogInfoBuffer.clean();
}
void HeightFogSet::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	for (int i = 0; i < fogCount; ++i) {
		fogInfoModified[i] = false;
		if (ImGui::CollapsingHeader(std::string("Height Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
			ImGui::Text(std::string("Height Fog " + std::to_string(i)).c_str());
			int fogIndex = fogIndexMap[i];
			fogInfoModified[i] |= ImGui::DragFloat3(std::string("Height Fog Start Pos " + std::to_string(i)).c_str(), (float*)&(*setting.fogInfos)[fogIndex].aabb.minimum);
			fogInfoModified[i] |= ImGui::DragFloat3(std::string("Height Fog Range " + std::to_string(i)).c_str(), (float*)&(*setting.fogInfos)[fogIndex].aabb.maximum);

			fogInfoModified[i] |= ImGui::DragFloat3(std::string("Height Fog Color " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].color);
			fogInfoModified[i] |= ImGui::DragFloat(std::string("Height Fog Ambient Intensity " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].ambientIntensity, 0.1f, 0.0f, 10.0f);
			fogInfoModified[i] |= ImGui::DragFloat(std::string("Height Fog Extinction Coefficient " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].absorption, 0.1f, 0.0f);
			fogInfoModified[i] |= ImGui::DragFloat(std::string("Height Fog Scatter Coefficient " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].scattering, 0.1f, 0.0f, 1.0f);
			fogInfoModified[i] |= ImGui::DragFloat(std::string("Height Fog Asymmetric Parameters " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].phase, 0.1f, -1.0f, 1.0f);

			ImGui::DragFloat(std::string("Height Fog Attenuation " + std::to_string(i)).c_str(), (float*)&heightFogInfos[i].heightScale, 1.0f, 0.0f, 1000.0f);
			fogInfoModified[i] |= ImGui::Checkbox(std::string("Height Fog show voxel grid " + std::to_string(i)).c_str(), (bool*)&showFogGrid[i]);
		}
	}
}

void HeightFogSet::updateDataPerFrame(VkCommandBuffer cmd, bool frist, FzbRenderer::Buffer fogInfoBuffer) {
	nvvk::cmdBufferMemoryBarrier(cmd, { heightFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	for (int i = 0; i < fogCount; ++i) {
		if (!fogInfoModified[i] && !frist) continue;
		vkCmdUpdateBuffer(cmd, heightFogInfoBuffer.buffer.buffer, sizeof(shaderio::HeightFogInfo) * i, sizeof(shaderio::HeightFogInfo), &heightFogInfos[i]);
		
		int fogIndex = fogIndexMap[i];
		vkCmdUpdateBuffer(cmd, fogInfoBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * fogIndex, sizeof(shaderio::VolumetricFogInfo), &(*setting.fogInfos)[fogIndex]);
	}
	nvvk::cmdBufferMemoryBarrier(cmd, { heightFogInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
}

void HeightFogSet::addFog(shaderio::HeightFogInfo fogInfo, shaderio::AABB fogRange) {
	if (fogCount == MAX_HEIGHT_FOG_COUNT) {
		printf("Height fog count exceeds limit, please expand MAX_HEIGHT_FOG_COUNT\n");
		return;
	}

	this->heightFogInfos.push_back(fogInfo);
	shaderio::VolumetricFogInfo volumetricFogInfo = {
		.aabb = fogRange,
		.type = shaderio::VolumetricFogType::Height,
		.typeFogIndex = fogCount,
	};
	setting.fogInfos->push_back(volumetricFogInfo);
	fogIndexMap.push_back(setting.fogInfos->size() - 1);

	fogInfoModified.push_back(false);
	showFogGrid.push_back(0);

	++fogCount;
}