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
	Feature::createGBuffer(true, true, 2, setting.resolution);		//缂佹鍏涚粩鏉戭嚕閻樺弶绂堥柨娑欘劗hreeView闁挎稑鑻ˇ璺ㄦ喆閸℃缍撻柨娑欑〒椤戝洦绂嶇仦鐣岀倞闁搞儲鎷濈槐鐧坲be闁挎稒绋撻鍥ㄧ▔婢跺﹦鐐婇柛?闁告艾楠搁ˇ鈺呮偠閸℃绂?闁挎稒顒眎reframe	//濞戞挸绉瑰▓銏㈢玻濡も偓瑜版盯宕氶崱姘跺摵闁?
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
	createDescriptorSetLayout();	//闁告帗绋戠紓鎾诲箵韫囨艾鐗氱紒妤嬬畵濞夛箓宕ラ崼婵堫伌閻忕偐鍋?
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::RasterVoxelizationPushConstant));	//闁告帗绋戠紓鎾剁不閿涘嫬娈犻悽顖氬暙閻剟鏁嶅鐪渟hConstant+闁硅绻楅崼顏嗙箔閿曞倹鑲犻柛姘墕缁旈浠﹂埀?
	compileAndCreateShaders();		//缂傚倹鐗為惁顪筯ader濞寸姰鍎卞鐑藉礆濞戞绱﹂梻鍫熺懄閳ь兛鍨peline
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
					ImVec2(100 * gBuffers.getAspectRatio(), 100));		//閺夆晜鐟╅崳閿嬫償閺冨浂鍤夐柡鍫濐槷缁斿瓨绋夐鍫燁€栭梺鎻掓处閻?

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

	//闁稿繐鐗愮换妯兼偘鐎碘偓iRender闁告劕绉风换妯兼偘鐎圭nder闁挎稑鏈晶宥嗙閵夈劎绠瑰☉鎾亾閻㈩垎鍛化闁告垼顕ч幃妤呮晬鐏炵晫瀹夐悹鍥ュ劥椤╋附绋夌€ｂ晝顏遍悽顖嗗嫭闄嶉柡鍕⒔閵囨岸鏁嶇仦鎯ь暡濞寸姰鍎查弬渚€宕烽妸銉ヮ枀闂?
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
			shaderio::AABB meshAABB = meshInfo.getAABB(sceneResource.instances[i].transform);	//閻庣敻鈧稓鑹鹃柛鏂诲妽閳ь兛鑳舵晶鎸庢媴閹垮嫮绀夐弶鈺傜懇閸ｇ兘妫侀埀顒傛啺娴ｉ攱鍙忛柡鈧?

			aabb.minimum.x = std::min(meshAABB.minimum.x, aabb.minimum.x);
			aabb.minimum.y = std::min(meshAABB.minimum.y, aabb.minimum.y);
			aabb.minimum.z = std::min(meshAABB.minimum.z, aabb.minimum.z);
			aabb.maximum.x = std::max(meshAABB.maximum.x, aabb.maximum.x);
			aabb.maximum.y = std::max(meshAABB.maximum.y, aabb.maximum.y);
			aabb.maximum.z = std::max(meshAABB.maximum.z, aabb.maximum.z);
		}

		setting.sceneSize = aabb.maximum - aabb.minimum;
		setting.sceneStartPos = aabb.minimum;

		//闁衡偓閹佷海濞戞挴鍋撻柣鎰缁辨繈姊奸崣澶樺壘閺夊牆婀遍弲顐ｅ緞閸曨厽鐣遍柡浣哄瀹撲線鏌ㄥ▎鎺濆殩闁挎稑鐗婇惁顕€寮涵鍜佸殯2濞戞挾宕乷xel闁挎稑鐭傞崑鍛▕閸繂绀侀弶鍫濇贡閺咁偊鎯冮崟顓炲亶鐎殿喗娲樺Σ?濞戞挸绉靛Σ?;濞戞挸绉堕崝褎瀵煎顒夊殼闁?->0闁?voxel闁哄牜鍓氶惀鍛村嫉婢跺娈堕柟璇″枛閸樻盯寮垫径澶屽晩闁轰胶澧楀畵渚€鏁?
		glm::vec3 distance = setting.sceneSize * 1.1f;
		//float maxDistance = std::max(distance.x, std::max(distance.y, distance.z));
		glm::vec3 center = (aabb.maximum + aabb.minimum) * 0.5f;
		glm::vec3 minimum = center - distance * 0.5f;
		glm::vec3 maximum = center + distance * 0.5f;

		glm::mat4 VP[3];
		//闁告挸绉瑰?
		glm::vec3 viewPoint = glm::vec3(center.x, center.y, maximum.z + 0.1f);	//濞戞挻鐗滈弲顐﹀锤閹邦厾鍨奸柛娆掓珪婢ф粓鎽堕悜妯活棆闁挎稑鑻畵?z闁哄牊绻傞幃?
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.z + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[0] = orthoMatrix * viewMatrix;
		//鐎归潻绠掔粩?
		viewPoint = glm::vec3(minimum.x - 0.1f, center.y, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.z, 0.5f * distance.z, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.x + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[1] = orthoMatrix * viewMatrix;
		//濞戞挸顑夊?
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

	//缂傚倹鐗為惁褔宕ユ惔锝嗙暠闁轰胶澧楀畵渚€寮ㄩ幆褎韬ù婊冩晞langCompiler濞?
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

	//濞达綀娉曢弫顥糑_EXT_SHADER_OBJECT_EXTENSION_NAME闁告艾楠歌ぐ鍙夌閵夈倗鐟濋梻鍥ｅ亾閻熸洑鍨peline闁挎稑鐬煎ú鍧楀箳閵夆斁鍋撳宕囩畺闁告稒鍨濋幎銈囨媼閸撗呮瀭婵炴挸寮堕悡瀣媼閸撗呮瀭闁告粌鐬煎鍐嚌閹绘帗鐝?
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;		//depthBoundsTestEnable闁哄嫷鍨遍悧鎾箲閻㈢袱n闁靛棔娌焌x depth閺夆晜绋栭、鎴澝圭€ｎ厾妲搁柨娑樼灱濞村銇愰幘鑼壘闁煎浜滈悾鐐▕婢跺寒姊块柛鎿冧簽濞堟啇闁汇劌瀚€垫牠宕?
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
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;		//闁活亞鍠愰婊冦€掗崣澶屽帬闂傚洠鍋撻悷鏇氱劍閻楁挳骞戦悾濡攅sky闁告帇鍊栭弻鍥及椤栨碍绀堝☉鎾虫惈閵囧绮氶搹瑙勭＊濞村吋淇洪々顐︽儎閺嶏妇鐟愬☉鎾亾閻㈩垎鍐ㄦ暥閻庡湱娅㈢槐婵嬪箥閳ь剚绂掗妷銈囩憹闂傚洠鍋撻悷鏇氭伃lear
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

	//濞达綀娉曢弫顥糑_EXT_SHADER_OBJECT_EXTENSION_NAME闁告艾楠歌ぐ鍙夌閵夈倗鐟濋梻鍥ｅ亾閻熸洑鍨peline闁挎稑鐬煎ú鍧楀箳閵夆斁鍋撳宕囩畺闁告稒鍨濋幎銈囨媼閸撗呮瀭婵炴挸寮堕悡瀣媼閸撗呮瀭闁告粌鐬煎鍐嚌閹绘帗鐝?
	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	//graphicsDynamicPipeline.rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	//vkCmdSetConservativeRasterizationModeEXT(cmd, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
	//vkCmdSetExtraPrimitiveOverestimationSizeEXT(cmd, 0.75f);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);	//depthBoundsTestEnable闁哄嫷鍨遍悧鎾箲閻㈢袱n闁靛棔娌焌x depth閺夆晜绋栭、鎴澝圭€ｎ厾妲搁柨娑樼灱濞村銇愰幘鑼壘闁煎浜滈悾鐐▕婢跺寒姊块柛鎿冧簽濞堟啇闁汇劌瀚€垫牠宕?

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
	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
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

	//濠碘€冲€归悘澶娾柦閳╁啯绠掓繛锝呭船鐎瑰啿霉鐎ｎ厾妲搁柨娑樻湰閸ㄦ粍绂掗鈧悿鍕瑹鐎ｎ亜顕ч柡鍕靛灟缁?-n閺夆晜绋栭、鎴濄€掗崣澶屽帬闁汇劌瀚哥槐婵嬫焽閿濆嫮鐤勫ù?闂侇叏绲肩粩鎾閵忋垺绠欓弶鈺佹搐楠炴挻瀵煎鎰佹蕉n闂侇叏绲肩粩鎾閵忥綆娲柣鈺傜墬鐢偓闁挎稒绋愮徊鐐瀹告帡鏌囬敐鍕伇闂傚牄鍨诲﹢鍛村及椤栨稓姊鹃柡鍫濐樀濡埖锛愬Ο鍏肩暠
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
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//濠碘€冲€归悘澶愬箚閸忓懎鈻忛柣顫姀濞呭嫮鐥崹顔艰濞寸姰鍎撮鏇犵磾閻ｅ摳sterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;	//闁稿繗娉涢悿鍕礃濞嗗繐寮抽柛鎰懃閸欏棝鏌堥崐鐕佹斀闁挎稑鑻ぐ褏鎲版担鍝勵暬闁稿繐鍟板鍐嚌閹绘帗鐝ゅ☉鎾崇С閹便劑寮ㄧ憴鍕畳閹艰揪璁ｇ槐婵堜焊鏉堫偄鍘撮柟缁樺姇婢х姴菐閸楃偛顔婃繛鏉戭儓閻?
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

	//閻忓繐鏁媔reframeMap濞戞挸姘﹂惃鐔兼偨閵娿劉鍋撻崨顖涚暠tonemapping闁告艾娴峰▓鎴犵磼閹惧浜弶鈺傜椤㈡垹绱掗幘铏€?
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_postProcess);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts({ setting.resolution.width, setting.resolution.height }, VkExtent2D{ 32, 32 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
#endif