#include <common/Application/Application.h>
#include "./RasterVoxelizationSVOPG.h"
#include "common/utils.hpp"
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <bit>
#include <cstdint>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

RasterVoxelization_SVOPG::RasterVoxelization_SVOPG(pugi::xml_node& featureNode) {
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().vertexPipelineStoresAndAtomics = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().shaderSampledImageArrayDynamicIndexing = VK_TRUE;

	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
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

void RasterVoxelization_SVOPG::init() {
#ifndef NDEBUG
	Feature::createGBuffer(true, true, 2, setting.resolution);		//第一张图：threeView，多视口；第二张图：Cube；第三张图(后处理图)：wireframe	//不随窗口分辨率
	//Feature::createGBuffer(true, true, 2, {1, 1});
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
	createDescriptorSetLayout();	//创建描述符集合布局
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::RasterVoxelizationPushConstant));	//创建管线布局：pushConstant+描述符集合布局
	compileAndCreateShaders();		//编译shader以及创建静态pipeline

}
void RasterVoxelization_SVOPG::clean() {
	Feature::clean();
	for(int i = 0; i < 6; ++i) Application::allocator.destroyBuffer(VGBs[i]);
	#ifdef CLUSTER_WITH_MATERIAL
	for(int i = 0; i < 6; ++i) Application::allocator.destroyBuffer(VGBMaterialInfos[i]);
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
void RasterVoxelization_SVOPG::uiRender() {
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

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::ThreeViewMap_SVOPG),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));		//这里应该有一个降采样

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
		std::vector<std::string> normalIndexNames = { "left", "right", "bottom", "up", "back", "forward"};

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

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG),
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

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG),
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

	//先进行uiRender再进行render，所以这一帧点击后，应该要下一帧来显示，所以放在前面
	if (showThreeViewMap) Application::viewportImage = gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::ThreeViewMap_SVOPG);
	else if (showCubeMap) Application::viewportImage = gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG);
	else if (showWireframeMap) Application::viewportImage = gBuffers.getDescriptorSet(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG);
#endif
}
void RasterVoxelization_SVOPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void RasterVoxelization_SVOPG::preRender(VkCommandBuffer cmd) {
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
void RasterVoxelization_SVOPG::render(VkCommandBuffer cmd) {
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
void RasterVoxelization_SVOPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	debug_Cube(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT, VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT);
	debug_Wireframe(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	debug_MergeWireframe(cmd);
#endif
}

void RasterVoxelization_SVOPG::createVGBs() {
	{
		shaderio::AABB aabb;
		aabb.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		aabb.maximum = -aabb.minimum;
		FzbRenderer::Scene& sceneResource = Application::sceneResource;
		for (int i = 0; i < sceneResource.instances.size(); ++i) {
			uint32_t meshIndex = sceneResource.instances[i].meshIndex;
			MeshInfo meshInfo = sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB meshAABB = meshInfo.getAABB(sceneResource.instances[i].transform);	//对于动态物体，这里需要修改

			aabb.minimum.x = std::min(meshAABB.minimum.x, aabb.minimum.x);
			aabb.minimum.y = std::min(meshAABB.minimum.y, aabb.minimum.y);
			aabb.minimum.z = std::min(meshAABB.minimum.z, aabb.minimum.z);
			aabb.maximum.x = std::max(meshAABB.maximum.x, aabb.maximum.x);
			aabb.maximum.y = std::max(meshAABB.maximum.y, aabb.maximum.y);
			aabb.maximum.z = std::max(meshAABB.maximum.z, aabb.maximum.z);
		}

		setting.sceneSize = aabb.maximum - aabb.minimum;
		setting.sceneStartPos = aabb.minimum;

		//放大一点，防止边界处的数据错误（比方说2个voxel，那么右边界的索引是2不是1;不然会导致2->0，0voxel本没有数据先有了数据）
		glm::vec3 distance = setting.sceneSize * 1.1f;
		float maxDistance = std::max(distance.x, std::max(distance.y, distance.z));
		glm::vec3 center = (aabb.maximum + aabb.minimum) * 0.5f;
		glm::vec3 minimum = center - distance * 0.5f;
		glm::vec3 maximum = center + distance * 0.5f;

		glm::mat4 VP[3];
		//前面
		glm::vec3 viewPoint = glm::vec3(center.x, center.y, maximum.z + 0.1f);	//世界坐标右手螺旋，即+z朝后
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.z + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[0] = orthoMatrix * viewMatrix;
		//左边
		viewPoint = glm::vec3(minimum.x - 0.1f, center.y, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.z, 0.5f * distance.z, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.x + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[1] = orthoMatrix * viewMatrix;
		//下面
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
	uint32_t VGBByteSize = voxelTotalCount * sizeof(shaderio::VGBVoxelData_SVOPG);

	#ifdef CLUSTER_WITH_MATERIAL
	uint32_t VGBMaterialInfosBufferSize = voxelTotalCount * sizeof(shaderio::VGBMaterialInfo_SVOPG);
	if (Application::sceneResource.materials.size() > MAX_MATERIAL_COUNT) {
		printf("Scene Material Count: %d, except maxCount: %d", (uint32_t)Application::sceneResource.materials.size(), MAX_MATERIAL_COUNT);
		throw std::runtime_error("");
	}
	VGBMaterialInfos.resize(6);
	for (int i = 0; i < 6; ++i) {
		allocator->createBuffer(VGBMaterialInfos[i], VGBMaterialInfosBufferSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VGBMaterialInfos[i].buffer);
	}
	#endif
		
	VGBs.resize(6); 
	for (int i = 0; i < 6; ++i) {
		allocator->createBuffer(VGBs[i], VGBByteSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VGBs[i].buffer);
	}
}
void RasterVoxelization_SVOPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eTextures_SVOPG,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eVGB_SVOPG,
						 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						 .descriptorCount = (uint32_t)VGBs.size(),
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	#ifdef CLUSTER_WITH_MATERIAL
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eVGBMaterialInfo_SVOPG,
					 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					 .descriptorCount = (uint32_t)VGBMaterialInfos.size(),
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	#endif

#ifndef NDEBUG
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eFragmentCountBuffer_SVOPG,
					 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					 .descriptorCount = 1,
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eWireframeMap_SVOPG,
				 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
				 .descriptorCount = 1,
				 .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints_SVOPG::eBaseMap_SVOPG,
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
void RasterVoxelization_SVOPG::createDescriptorSet() {
	Feature::addTextureArrayDescriptor(shaderio::RasterVoxelizationBindingPoints_SVOPG::eTextures_SVOPG);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite(shaderio::RasterVoxelizationBindingPoints_SVOPG::eVGB_SVOPG, 0, 0, VGBs.size());
	nvvk::Buffer* VGBsPtr = VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	#ifdef CLUSTER_WITH_MATERIAL
	VkWriteDescriptorSet    VGBMaterialInfoWrite =
		staticDescPack.makeWrite(shaderio::RasterVoxelizationBindingPoints_SVOPG::eVGBMaterialInfo_SVOPG, 0, 0, VGBMaterialInfos.size());
	nvvk::Buffer* VGBMaterialInfosPtr = VGBMaterialInfos.data();
	write.append(VGBMaterialInfoWrite, VGBMaterialInfosPtr);
	#endif

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

	VkWriteDescriptorSet fcWrite = staticDescPack.makeWrite(shaderio::RasterVoxelizationBindingPoints_SVOPG::eFragmentCountBuffer_SVOPG, 0, 0, 1);
	write.append(fcWrite, fragmentCountBuffer, 0, sizeof(uint32_t));

	VkWriteDescriptorSet wireframeMapWrite = staticDescPack.makeWrite(shaderio::RasterVoxelizationBindingPoints_SVOPG::eWireframeMap_SVOPG, 0, 0, 1);
	write.append(wireframeMapWrite, gBuffers.getColorImageView(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG), VK_IMAGE_LAYOUT_GENERAL);
	#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void RasterVoxelization_SVOPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	//编译后的数据放在了slangCompiler中
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "RasterVoxelization_SVOPG.slang";
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
void RasterVoxelization_SVOPG::updateDataPerFrame(VkCommandBuffer cmd) {}

void RasterVoxelization_SVOPG::clearVGB(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_clearVGB);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t totalVoxelCount = pow(setting.pushConstant.voxelSize_Count.w, 3) * 6;
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void RasterVoxelization_SVOPG::createVGB(VkCommandBuffer cmd) {
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

	//使用VK_EXT_SHADER_OBJECT_EXTENSION_NAME后可以不需要pipeline，直接通过命令设置渲染设置和着色器
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;		//depthBoundsTestEnable是根据min、max depth进行测试，相当于自定义裁剪的z的范围
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
void RasterVoxelization_SVOPG::resize(
	VkCommandBuffer cmd, const VkExtent2D& size,
	nvvk::GBuffer& gBuffers_other, uint32_t baseMapIndex
) {
	//gBuffers.update(cmd, size);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet baseMapWrite = staticDescPack.makeWrite(shaderio::RasterVoxelizationBindingPoints_SVOPG::eBaseMap_SVOPG, 0, 0, 1);
	write.append(baseMapWrite, gBuffers_other.getColorImageView(baseMapIndex), VK_IMAGE_LAYOUT_GENERAL);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	depthImageView = gBuffers_other.getDepthImageView();
}

void RasterVoxelization_SVOPG::resetFragmentCount(VkCommandBuffer cmd) {
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
void RasterVoxelization_SVOPG::createVGB_ThreeView(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "RasterVoxelization_createVGB_ThreeView");

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::ThreeViewMap_SVOPG), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;		//真正渲染需要根据usesky判断是因为天空盒会覆盖上一帧内容，所以不需要clear
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

	//使用VK_EXT_SHADER_OBJECT_EXTENSION_NAME后可以不需要pipeline，直接通过命令设置渲染设置和着色器
	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	//graphicsDynamicPipeline.rasterizationState.rasterizerDiscardEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	//vkCmdSetConservativeRasterizationModeEXT(cmd, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
	//vkCmdSetExtraPrimitiveOverestimationSizeEXT(cmd, 0.75f);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);	//depthBoundsTestEnable是根据min、max depth进行测试，相当于自定义裁剪的z的范围

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
	//vkCmdSetConservativeRasterizationModeEXT(cmd, VK_CONSERVATIVE_RASTERIZATION_MODE_DISABLED_EXT);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::ThreeViewMap_SVOPG), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_SVOPG::debug_Cube(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };
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

	//如果没有深度测试，我们实例化是从0-n进行渲染的，那么从0那一面看过去会被n那一面覆盖掉；但从n那一面看是没有问题的
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

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::CubeMap_SVOPG), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_SVOPG::debug_Wireframe(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG);
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
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = setting.lineWidth;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;	//其实写入写入都行，只要片元着色器不修改深度，就能提前深度测试
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

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(RasterVoxelizationGBuffer_SVOPG::WireframeMap_SVOPG), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}
void RasterVoxelization_SVOPG::debug_MergeWireframe(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//将wireframeMap与调用者的tonemapping后的结果进行结合
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_postProcess);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts({ setting.resolution.width, setting.resolution.height }, VkExtent2D{ 32, 32 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
#endif