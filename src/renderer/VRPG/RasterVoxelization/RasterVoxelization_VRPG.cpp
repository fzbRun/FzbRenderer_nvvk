#include <common/Application/Application.h>
#include "./RasterVoxelization_VRPG.h"
#include "common/utils.hpp"
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <bit>
#include <cstdint>
#include <cstring>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>
#include <renderer/VRPG/VRPG.h>

using namespace FzbRenderer;

RasterVoxelization_VRPG::RasterVoxelization_VRPG(pugi::xml_node& featureNode) {
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().vertexPipelineStoresAndAtomics = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().shaderSampledImageArrayDynamicIndexing = VK_TRUE;

	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;

	// VRPG creates two RasterVoxelization instances (geometry and light VGB).
	// The device pNext chain may contain each feature structure only once.
	bool atomicFloatExtensionAdded = false;
	for (const auto& extension : Application::vkContextInitInfo.deviceExtensions) {
		if (extension.extensionName != nullptr &&
			std::strcmp(extension.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0) {
			atomicFloatExtensionAdded = true;
			break;
		}
	}
	if (!atomicFloatExtensionAdded)
		Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures });

	conservativeRasterFeature.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_CONSERVATIVE_STATE_CREATE_INFO_EXT;
	conservativeRasterFeature.conservativeRasterizationMode = VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT;
	conservativeRasterFeature.extraPrimitiveOverestimationSize = 0.5f;
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_CONSERVATIVE_RASTERIZATION_EXTENSION_NAME });

	if (pugi::xml_node resolutionNode = featureNode.child("resolution")) {
		glm::vec2 resolution = FzbRenderer::getfloat2FromString(resolutionNode.attribute("value").value());
		setting.resolution = VkExtent2D(resolution.x, resolution.y);
	}
	else setting.resolution = { 4096, 4096 };

	if (pugi::xml_node voxelCountNode = featureNode.child("voxelCount"))
		setting.pushConstant.voxelSize_Count.w = std::stoi(voxelCountNode.attribute("value").value());
	else setting.pushConstant.voxelSize_Count.w = 16;

#ifndef NDEBUG
	//-------------------------------------DebugSetting-------------------------------------------------
	Application::vkContext->getPhysicalDeviceFeatures11_notConst().multiview = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().multiViewport = VK_TRUE;

	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}

void RasterVoxelization_VRPG::init() {
	if (Application::sceneResource.isStaticScene && forceStaticHighResolution) setting.resolution = { 4096, 4096 };
#ifndef NDEBUG
	Feature::createGBuffer(true, true, 2, setting.resolution);		//缂備焦顨忛崗娑氱博閺夋埈鍤曢柣妯哄级缁傚牓鏌ㄥ☉娆樺姉hreeView闂佹寧绋戦懟顖炈囩捄銊﹀枂闁糕剝顨呯紞鎾绘煥濞戞瑧銆掓い鎴濇处缁傚秶浠﹂悾宀€鍊為梺鎼炲劜閹锋繄妲愰惂鍧瞓e闂佹寧绋掔粙鎾活敄閸ャ劎鈻斿璺猴功閻愬﹪鏌?闂佸憡鑹炬鎼佀囬埡鍛仩闁糕剝顨嗙粋?闂佹寧绋掗鐪巖eframe	//婵炴垶鎸哥粔鐟扳枔閵忋垻鐜绘俊銈傚亾鐟滅増鐩畷姘跺幢濮樿泛鎽甸梺?
	//---------------------------------------------cube----------------------------------------
	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube(false, false);
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Cube", primitive);
	scene.addMeshSet(mesh);
	//---------------------------------------------wireframe-----------------------------------
	primitive = FzbRenderer::MeshSet::createWireframe();
	mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif
	//---------------------------------------------------------------------------------------------
	createVGBs();
	createDescriptorSetLayout();	//闂佸憡甯楃粙鎴犵磽閹捐绠甸煫鍥ㄨ壘閻楁氨绱掑Δ瀣暤婵炲绠撳畷銉╁醇濠靛牜浼岄柣蹇曞亹閸?
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::RasterVoxelizationPushConstant));	//闂佸憡甯楃粙鎴犵磽閹惧墎涓嶉柨娑樺濞堢娀鎮介姘殭闁活亶鍓熼弫宥咁潰閻笩hConstant+闂佺顕х换妤呭醇椤忓棛绠旈柨鏇炲€归懖鐘绘煕濮橆剛澧曠紒鏃堫棑娴狅箓鍩€?
	compileAndCreateShaders();		//缂傚倸鍊归悧鐐烘儊椤ader婵炲濮伴崕鍗烆嚕閻戣棄绀嗘繛鎴烆焽缁憋箓姊婚崼鐔烘噭闁逞屽厸閸晱peline
}
void RasterVoxelization_VRPG::clean() {
	Feature::clean();
	for (int i = 0; i < 6; ++i) Application::allocator.destroyBuffer(VGBs[i]);
#ifdef CLUSTER_WITH_MATERIAL
	for (int i = 0; i < 6; ++i) Application::allocator.destroyBuffer(VGBMaterialInfos[i]);
#endif

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_clearVGB, nullptr);
	vkDestroyShaderEXT(device, vertexShader, nullptr);
	vkDestroyShaderEXT(device, geometryShader, nullptr);
	vkDestroyShaderEXT(device, fragmentShader, nullptr);
#ifndef NDEBUG
	Application::allocator.destroyBuffer(fragmentCountBuffer);
	Application::allocator.destroyBuffer(fragmentCountStageBuffer);

	vkDestroyShaderEXT(device, geometryShader_ThreeView, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_ThreeView, nullptr);

	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);

	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);

	vkDestroyShaderEXT(device, computeShader_postProcess, nullptr);
#endif
}
void RasterVoxelization_VRPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("RasterVoxelization")) {
		//---------------------------------------threeView-------------------------------------------
		std::string fragmentCountText = "fragment count: " + std::to_string(fragmentCount_host);
		ImGui::Text(fragmentCountText.c_str());
		if (PE::begin()) {
			if (PE::entry("ThreeViewMap", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showThreeViewMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(0),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));		//闁哄鏅滈悷鈺呭闯闁垮鍎熼柡鍐ㄦ祩閸ゅ鏌￠崼婵愭Х缂佹柨鐡ㄧ粙澶愵敂閸噥鈧牠姊洪幓鎺撳闁?

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showThreeViewMap = !showThreeViewMap;
				showCubeMap = false;
				showWireframeMap = false;
			}
		}
		PE::end();
		//---------------------------------------normalIndex--------------------------------------
		std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward" };

		std::vector<const char*> normalIndexNames_pointers;
		for (const auto& normalIndexName : normalIndexNames)
			normalIndexNames_pointers.push_back(normalIndexName.c_str());
		ImGui::Combo("normal", &normalIndex, normalIndexNames_pointers.data(), static_cast<int>(normalIndexNames_pointers.size()));
		//------------------------------------------cube---------------------------------------------
		if (PE::begin()) {
			if (PE::entry("CubeMap", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showCubeMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
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
				showCubeMap = !showCubeMap;
				showThreeViewMap = false;
				showWireframeMap = false;
			}
		}
		PE::end();
		//----------------------------------------wireframe------------------------------------------
		if (PE::begin()) {
			if (PE::entry("WireframeMap", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showWireframeMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(2),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showWireframeMap = !showWireframeMap;
				showThreeViewMap = false;
				showCubeMap = false;
			}
		}
		PE::end();
	}
	ImGui::End();

	//闂佺绻愰悧鎰崲濡吋鍋橀悗纰樺亾iRender闂佸憡鍔曠粔椋庢崲濡吋鍋橀悗鍦嚠nder闂佹寧绋戦張顒佹櫠瀹ュ棛顩烽柕澶堝妿缁犵懓鈽夐幘顖氫壕闁汇埄鍨庨崨顓濆寲闂佸憡鍨奸褔骞冨Δ鍛櫖閻忕偟鏅€瑰鎮归崶銉ュ姤妞も晪闄勭粙澶屸偓锝傛櫇椤忛亶鎮介鍡楀闂勫秹鏌￠崟顐⑩挃闁靛洦宀搁弫宥囦沪閹屾殹婵炲濮伴崕鏌ュ棘娓氣偓瀹曠兘濡搁妷銉瀫闂?
	if (showThreeViewMap) Application::viewportImage = gBuffers.getDescriptorSet(0);
	else if (showCubeMap) Application::viewportImage = gBuffers.getDescriptorSet(1);
	else if (showWireframeMap) Application::viewportImage = gBuffers.getDescriptorSet(2);
#endif
}
void RasterVoxelization_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void RasterVoxelization_VRPG::preRender(VkCommandBuffer cmd) {
#ifndef NDEBUG
	setting.pushConstant.frameIndex = Application::frameIndex;
	setting.pushConstant.normalIndex = normalIndex;

	if (Application::frameIndex != 1) return;
	VkBufferCopy2 copyRegionInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
		.srcOffset = 0,
		.dstOffset = 0,
		.size = sizeof(uint32_t),
	};
	VkCopyBufferInfo2 copyBufferInfo{
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
		.srcBuffer = fragmentCountBuffer.buffer,
		.dstBuffer = fragmentCountStageBuffer.buffer,
		.regionCount = 1,
		.pRegions = &copyRegionInfo,
	};
	vkCmdCopyBuffer2(cmd, &copyBufferInfo);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

	memcpy(&fragmentCount_host, fragmentCountStageBuffer.mapping, sizeof(uint32_t));
#endif
}
void RasterVoxelization_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "RasterVoxelization_render");

	updateDataPerFrame(cmd);

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
		.size = sizeof(shaderio::RasterVoxelizationPushConstant),
		.pValues = &setting.pushConstant,
	};
	setting.pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	clearVGB(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
#ifndef NDEBUG
	resetFragmentCount(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);
	createVGB_ThreeView(cmd);
#else
	createVGB(cmd);
#endif
}
void RasterVoxelization_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_Cube(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT);
	debug_Wireframe(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	debug_MergeWireframe(cmd);
#endif
}

void RasterVoxelization_VRPG::createVGBs() {
	{
		shaderio::AABB aabb;
		aabb.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		aabb.maximum = -aabb.minimum;
		FzbRenderer::Scene& sceneResource = Application::sceneResource;
		for (int i = 0; i < sceneResource.instances.size(); ++i) {
			uint32_t meshIndex = sceneResource.instances[i].meshIndex;
			MeshInfo meshInfo = sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB meshAABB = meshInfo.getAABB(sceneResource.instances[i].transform);	//闁诲海鏁婚埀顒佺〒閼归箖鏌涢弬璇插闁逞屽厸閼宠埖鏅堕幐搴㈠闁瑰灝瀚粈澶愬级閳哄倻鎳囬柛锝囧厴濡線鍩€椤掑倹鍟哄ù锝夋敱閸欏繘鏌￠埀?

			aabb.minimum.x = std::min(meshAABB.minimum.x, aabb.minimum.x);
			aabb.minimum.y = std::min(meshAABB.minimum.y, aabb.minimum.y);
			aabb.minimum.z = std::min(meshAABB.minimum.z, aabb.minimum.z);
			aabb.maximum.x = std::max(meshAABB.maximum.x, aabb.maximum.x);
			aabb.maximum.y = std::max(meshAABB.maximum.y, aabb.maximum.y);
			aabb.maximum.z = std::max(meshAABB.maximum.z, aabb.maximum.z);
		}

		setting.sceneSize = aabb.maximum - aabb.minimum;
		setting.sceneStartPos = aabb.minimum;

		//闂佽　鍋撻柟顖嗕椒娴锋繛鎴炴尨閸嬫捇鏌ｉ幇顔筋仧缂佽鲸绻堝濂稿矗婢舵ê澹橀柡澶婄墕濠€閬嶅疾椤愶絽绶為柛鏇ㄥ幗閻ｉ亶鏌℃担鍝勵暭鐎规挷绶氶弻銊モ枎閹烘繂娈╅梺鎸庣☉閻楀﹪鎯侀鈧顒傛兜閸滀礁娈?婵炴垶鎸惧畷涔穢el闂佹寧绋戦惌鍌炲磻閸涱喚鈻曢柛顐ｇ箓缁€渚€寮堕崼婵囪础闁哄拋鍋婇幆鍐礋椤撶偛浜堕悗娈垮枟濞叉ê危?婵炴垶鎸哥粔闈浳?;婵炴垶鎸哥粔鍫曞礉瑜庣€电厧顫濋澶婃闂?->0闂?voxel闂佸搫鐗滈崜姘舵儉閸涙潙瀚夊璺侯儐濞堝爼鏌熺拠鈥虫灈闁告ɑ鐩鍨緞婢跺苯鏅╅梺杞拌兌婢ф鐣垫笟鈧弫?
		glm::vec3 distance = setting.sceneSize * 1.1f;
		//float maxDistance = std::max(distance.x, std::max(distance.y, distance.z));
		glm::vec3 center = (aabb.maximum + aabb.minimum) * 0.5f;
		glm::vec3 minimum = center - distance * 0.5f;
		glm::vec3 maximum = center + distance * 0.5f;

		glm::mat4 VP[3];
		//闂佸憡鎸哥粔鐟邦焽?
		glm::vec3 viewPoint = glm::vec3(center.x, center.y, maximum.z + 0.1f);	//婵炴垶鎸婚悧婊堝疾椤愶箑閿ら柟閭﹀幘閸ㄥジ鏌涘▎鎺撶彧濠⒀勭矒閹藉爼鎮滃Ο娲绘闂佹寧绋戦懟顖氱暤?z闂佸搫鐗婄换鍌炲箖?
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.z + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[0] = orthoMatrix * viewMatrix;
		//閻庡綊娼荤粻鎺旂博?
		viewPoint = glm::vec3(minimum.x - 0.1f, center.y, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.z, 0.5f * distance.z, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.x + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[1] = orthoMatrix * viewMatrix;
		//婵炴垶鎸搁澶婎焽?
		viewPoint = glm::vec3(center.x, minimum.y - 0.1f, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.z, 0.5f * distance.z, 0.1f, distance.y + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[2] = orthoMatrix * viewMatrix;

		for (int i = 0; i < 3; ++i) setting.pushConstant.VP[i] = VP[i];
		setting.pushConstant.voxelGroupStartPos = minimum;
		setting.pushConstant.voxelSize_Count = glm::vec4(distance / setting.pushConstant.voxelSize_Count.w, setting.pushConstant.voxelSize_Count.w);
	}

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t voxelTotalCount = std::pow(setting.pushConstant.voxelSize_Count.w, 3);
	uint32_t VGBByteSize = voxelTotalCount * sizeof(shaderio::VGBVoxelData_VRPG);

	VGBs.resize(6);
	for (int i = 0; i < 6; ++i) {
		allocator->createBuffer(VGBs[i], VGBByteSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VGBs[i].buffer);
	}
}
void RasterVoxelization_VRPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = (uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({ .binding = (uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eVGB,
						 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						 .descriptorCount = (uint32_t)VGBs.size(),
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);


#ifndef NDEBUG
	bindings.addBinding({ .binding = (uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eFragmentCountBuffer,
					 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					 .descriptorCount = 1,
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	bindings.addBinding({ .binding = (uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eWireframeMap,
				 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				 .descriptorCount = 1,
				 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	bindings.addBinding({ .binding = (uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eBaseMap,
			 .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
			 .descriptorCount = 1,
			 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
#endif
	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void RasterVoxelization_VRPG::createDescriptorSet() {
	Feature::addTextureArrayDescriptor((uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eTextures);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eVGB, 0, 0, VGBs.size());
	nvvk::Buffer* VGBsPtr = VGBs.data();
	write.append(VGBWrite, VGBsPtr);

#ifndef NDEBUG
	//-------------------------------------------threeView----------------------------------------
	{
		nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
		nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();
		allocator->createBuffer(fragmentCountBuffer, sizeof(uint32_t),
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(fragmentCountBuffer.buffer);

		std::vector<uint32_t> fragmentCountData(1); fragmentCountData = { 0 };
		NVVK_CHECK(stagingUploader.appendBuffer(fragmentCountBuffer, 0, std::span<const uint32_t>(fragmentCountData)));

		allocator->createBuffer(fragmentCountStageBuffer, sizeof(uint32_t), VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_GPU_TO_CPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);
		NVVK_DBG_NAME(fragmentCountStageBuffer.buffer);
	}

	VkWriteDescriptorSet fcWrite = staticDescPack.makeWrite((uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eFragmentCountBuffer, 0, 0, 1);
	write.append(fcWrite, fragmentCountBuffer, 0, sizeof(uint32_t));

	VkWriteDescriptorSet wireframeMapWrite = staticDescPack.makeWrite((uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eWireframeMap, 0, 0, 1);
	write.append(wireframeMapWrite, gBuffers.getColorImageView(2), VK_IMAGE_LAYOUT_GENERAL);
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void RasterVoxelization_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	//缂傚倸鍊归悧鐐烘儊瑜斿畷銉︽償閿濆棛鏆犻梺杞拌兌婢ф鐣垫笟鈧銊╁箚瑜庨煬顒€霉濠婂啯鏅瀕angCompiler婵?
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "RasterVoxelization.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::RasterVoxelizationPushConstant),
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
	vkDestroyShaderEXT(device, computeShader_clearVGB, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_clearVGB";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_clearVGB);
	NVVK_DBG_NAME(computeShader_clearVGB);
	//--------------------------------------------------------------------------------------
	vkDestroyShaderEXT(device, vertexShader, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_GEOMETRY_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader);
	NVVK_DBG_NAME(vertexShader);
#ifndef NDEBUG
	//------------------------------------------ThreeView--------------------------------------
	vkDestroyShaderEXT(device, geometryShader_ThreeView, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_ThreeView, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "geometryMain_ThreeView";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &geometryShader_ThreeView);
	NVVK_DBG_NAME(geometryShader_ThreeView);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_ThreeView";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_ThreeView);
	NVVK_DBG_NAME(fragmentShader_ThreeView);
	//-------------------------------------------Cube--------------------------------------
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
	//-------------------------------------------Wireframe--------------------------------------
	vkDestroyShaderEXT(device, fragmentShader_Wireframe, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain_Wireframe";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_Wireframe);
	NVVK_DBG_NAME(fragmentShader_Wireframe);
	//-------------------------------------------PostProcess--------------------------------------
	vkDestroyShaderEXT(device, computeShader_postProcess, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_postProcess";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_postProcess);
	NVVK_DBG_NAME(computeShader_postProcess);
#else
	vkDestroyShaderEXT(device, geometryShader, nullptr);
	vkDestroyShaderEXT(device, fragmentShader, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "geometryMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &geometryShader);
	NVVK_DBG_NAME(geometryShader);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader);
	NVVK_DBG_NAME(fragmentShader);
#endif
}
void RasterVoxelization_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void RasterVoxelization_VRPG::clearVGB(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clearVGB);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	// clearVGB can be called without RasterVoxelization::render() (light VGB is
	// rebuilt from the geometry VGB now), so initialize its push info here.
	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::RasterVoxelizationPushConstant),
		.pValues = &setting.pushConstant,
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t totalVoxelCount = pow(setting.pushConstant.voxelSize_Count.w, 3) * 6;
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void RasterVoxelization_VRPG::createVGB(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, setting.resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	//婵炶揪缍€濞夋洟寮ˉ绯慱EXT_SHADER_OBJECT_EXTENSION_NAME闂佸憡鑹炬姝屻亹閸欏顩烽柕澶堝€楅悷婵嬫⒒閸ワ絽浜鹃柣鐔告磻閸晱peline闂佹寧绋戦惉鐓幟洪崸妤€绠抽柕澶嗘杹閸嬫挸顫濆畷鍥╃暫闂佸憡绋掗崹婵嬪箮閵堝洦濯奸柛鎾楀懏鐎┑鐐存尭瀵爼鎮＄€ｎ剚濯奸柛鎾楀懏鐎梺鍛婄矊閻厧顭囬崘顔藉殞闁圭粯甯楅悵?
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;		//depthBoundsTestEnable闂佸搫瀚烽崹閬嶆偋閹绢喖绠查柣銏㈣⒈n闂侀潧妫斿▽鐒寈 depth闁哄鏅滅粙鏍€侀幋婢濆湱鈧綆鍘惧Σ鎼佹煥濞戞鐏辨繛鏉戭槼閵囨劙骞橀懠顒冨闂佺厧顨庢禍婊堟偩閻愵剛鈻曞璺哄瘨濮婂潡鏌涢幙鍐х敖婵炲牊鍟囬梺姹囧妼鐎氼垳鈧灚鐗犲畷?
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, setting.resolution);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader, .geometry = geometryShader });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		setting.pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);
}

#ifndef NDEBUG
void RasterVoxelization_VRPG::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	//gBuffers.update(cmd, size);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet baseMapWrite = staticDescPack.makeWrite((uint32_t)shaderio::RasterVoxelizationBindingPoints_VRPG::eBaseMap, 0, 0, 1);
	write.append(baseMapWrite, gBuffers_other.getColorImageView(baseMapIndex), VK_IMAGE_LAYOUT_GENERAL);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	depthImageView = gBuffers_other.getDepthImageView();
}

void RasterVoxelization_VRPG::resetFragmentCount(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	int tempData = 0;
	memcpy(fragmentCountStageBuffer.mapping, &tempData, sizeof(uint32_t));

	VkBufferCopy2 copyRegionInfo{
		.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
		.srcOffset = 0,
		.dstOffset = 0,
		.size = sizeof(uint32_t),
	};
	VkCopyBufferInfo2 copyBufferInfo{
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
		.srcBuffer = fragmentCountStageBuffer.buffer,
		.dstBuffer = fragmentCountBuffer.buffer,
		.regionCount = 1,
		.pRegions = &copyRegionInfo,
	};
	vkCmdCopyBuffer2(cmd, &copyBufferInfo);
}
void RasterVoxelization_VRPG::createVGB_ThreeView(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "RasterVoxelization_createVGB_ThreeView");

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;		//闂佹椿浜為崰鎰邦敆濠婂啨鈧帡宕ｆ径灞藉脯闂傚倸娲犻崑鎾绘偡閺囨氨鍔嶉柣妤佹尦楠炴垿鎮炬俊鏀卻ky闂佸憡甯囬崐鏍蓟閸ヮ剙鍙婃い鏍ㄧ缁€鍫濃槈閹捐櫕鎯堥柕鍥ь槺缁岸鎼圭憴鍕紛婵炴潙鍚嬫穱娲€呴锔藉剮闁哄稄濡囬悷鎰槈閹绢垰浜鹃柣銏╁瀻閸愩劍鏆ラ柣搴℃贡濞呫垻妲愬┑瀣闁逞屽墯缁傛帡濡烽妶鍥╂喒闂傚倸娲犻崑鎾绘偡閺囨碍浼僱ear
	colorAttachment.imageView = gBuffers.getColorImageView(0);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, setting.resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	//婵炶揪缍€濞夋洟寮ˉ绯慱EXT_SHADER_OBJECT_EXTENSION_NAME闂佸憡鑹炬姝屻亹閸欏顩烽柕澶堝€楅悷婵嬫⒒閸ワ絽浜鹃柣鐔告磻閸晱peline闂佹寧绋戦惉鐓幟洪崸妤€绠抽柕澶嗘杹閸嬫挸顫濆畷鍥╃暫闂佸憡绋掗崹婵嬪箮閵堝洦濯奸柛鎾楀懏鐎┑鐐存尭瀵爼鎮＄€ｎ剚濯奸柛鎾楀懏鐎梺鍛婄矊閻厧顭囬崘顔藉殞闁圭粯甯楅悵?
	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	//graphicsDynamicPipeline.rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	//vkCmdSetConservativeRasterizationModeEXT(cmd, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
	//vkCmdSetExtraPrimitiveOverestimationSizeEXT(cmd, 0.75f);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);	//depthBoundsTestEnable闂佸搫瀚烽崹閬嶆偋閹绢喖绠查柣銏㈣⒈n闂侀潧妫斿▽鐒寈 depth闁哄鏅滅粙鏍€侀幋婢濆湱鈧綆鍘惧Σ鎼佹煥濞戞鐏辨繛鏉戭槼閵囨劙骞橀懠顒冨闂佺厧顨庢禍婊堟偩閻愵剛鈻曞璺哄瘨濮婂潡鏌涢幙鍐х敖婵炲牊鍟囬梺姹囧妼鐎氼垳鈧灚鐗犲畷?

	float width = float(setting.resolution.width) * 0.5f;
	float height = float(setting.resolution.height) * 0.5f;
	std::vector<VkViewport> viewports(3);
	viewports[0] = { 0.0F, 0.0F, width, height, 0.0f, 1.0f };
	viewports[1] = { width, 0.0F, width, height, 0.0f, 1.0f };
	viewports[2] = { 0.0F, height, width, height, 0.0f, 1.0f };
	std::vector<VkRect2D> scissors(3);
	scissors[0] = { {0, 0},{uint32_t(width),  uint32_t(height)} };
	scissors[1] = { {int(width), 0}, {uint32_t(width),  uint32_t(height)} };
	scissors[2] = { {0, int(height)}, { uint32_t(width),  uint32_t(height)} };
	vkCmdSetViewportWithCount(cmd, 3, viewports.data());
	vkCmdSetScissorWithCount(cmd, 3, scissors.data());

	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader_ThreeView, .geometry = geometryShader_ThreeView });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		setting.pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_VRPG::debug_Cube(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(1), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(1);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, setting.resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	//婵犵鈧啿鈧綊鎮樻径濞炬煢闁斥晛鍟粻鎺撶箾閿濆懎鑸归悗鐟板暱闇夐悗锝庡幘濡叉悂鏌ㄥ☉妯绘拱闁搞劍绮嶇粋鎺楊敆閳ь剟鎮块崟顒傜懝閻庯綆浜滈褔鏌￠崟闈涚仧缂?-n闁哄鏅滅粙鏍€侀幋婵勨偓鎺楀矗婢跺苯甯梺姹囧妼鐎氬摜妲愬┑瀣劷闁挎繂瀚悿鍕?闂備緡鍙忕徊鑲╃博閹绢喗顥堥柕蹇嬪灪缁犳瑩寮堕埡浣规悙妤犵偞鎸荤€电厧顫濋幇浣硅晧n闂備緡鍙忕徊鑲╃博閹绢喗顥堥柕蹇ョ秵濞差剟鏌ｉ埡鍌滃閻㈩垪鍋撻梺鎸庣⊕缁嬫劗寰婇悙顒傤浄鐎瑰憡甯￠弻鍥晲閸曨収浼囬梻鍌氱墑閸ㄨ锕㈤崨鏉戝強妞ゆ牗绋撳楣冩煛閸繍妯€婵☆偉鍩栭敍鎰熼崗鑲╂殸
	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, setting.resolution);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Cube, .fragment = fragmentShader_Cube });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	setting.pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pow(setting.pushConstant.voxelSize_Count.w, 3) * 6, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(1), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_VRPG::debug_Wireframe(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(2), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(2);
	colorAttachment.clearValue = { .color = {0.0f, 0.0f, 0.0f, 0.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, setting.resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//婵犵鈧啿鈧綊鎮樻径鎰畾闁稿繐鎳庨埢蹇涙煟椤剙濮€婵炲懎瀚惀顏堝垂椤旇壈顔夋繛瀵稿О閸庢挳顢欓弴鐘电＞闁伙絽鎽硈terizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;	//闂佺绻楀▔娑㈡偪閸曨垰绀冩繛鍡楃箰瀵娊鏌涢幇顓犳噧闁告瑥妫濋弻鍫ュ磹閻曚焦鏂€闂佹寧绋戦懟顖濄亹瑜忛幉鐗堟媴閸濆嫷鏆梺绋跨箰閸熸澘顭囬崘顔藉殞闁圭粯甯楅悵銈呪槈閹惧磭小闁逛究鍔戝銊ф喆閸曨厾鐣抽柟鑹版彧鐠侊絿妲愬┑鍫滅剨閺夊牜鍋勯崢鎾煙缂佹ê濮囧褏濮磋彁闁告鍋涢濠冪箾閺夋埈鍎撻柣?
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, setting.resolution);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Cube, .fragment = fragmentShader_Wireframe });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 1;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	setting.pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	vkCmdDrawIndexed(cmd, triMesh.indices.count, pow(setting.pushConstant.voxelSize_Count.w, 3) * 6, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(2), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_VRPG::debug_MergeWireframe(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//闁诲繐绻愰弫濯攔eframeMap婵炴垶鎸稿锕傛儍閻斿吋鍋ㄩ柕濞垮妷閸嬫捇宕ㄩ娑氭殸tonemapping闂佸憡鑹惧ù宄扳枔閹寸姷纾奸柟鎯ь嚟娴滎垶寮堕埡鍌滎灱妞ゃ垺鍨圭槐鎺楀箻閾忣偄鈧?
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_postProcess);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts({ setting.resolution.width, setting.resolution.height }, VkExtent2D{ 32, 32 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
#endif