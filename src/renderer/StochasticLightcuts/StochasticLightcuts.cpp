#include "./StochasticLightcuts.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>

using namespace FzbRenderer;

StochasticLightcutsRenderer::StochasticLightcutsRenderer(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();
}
void StochasticLightcutsRenderer::init() {
	createSourceData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void StochasticLightcutsRenderer::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_LightInfoInject, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateLightcuts_1, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateLightcuts_2, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateLightcuts_3, nullptr);
	vkDestroyShaderEXT(device, computeShader_CreateLightcuts_4, nullptr);
	vkDestroyShaderEXT(device, vertexShader_createGBuffers, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffers, nullptr);
	vkDestroyShaderEXT(device, computeShader_Render, nullptr);

	vkDestroyShaderEXT(device, vertexShader_lightTreeDebug, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_lightTreeDebug, nullptr);
	vkDestroyShaderEXT(device, vertexShader_lightTreeDebug2, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_lightTreeDebug2, nullptr);

	areaLightsBuffer.clean();
	LightTreeBuffer.clean();
	hasLightBlockCountBuffer.clean();
	hasLightBlockArrayBuffer.clean();
	hasLightBlockIndexBuffer.clean();
	indirectCmd.clean();

	PathTracingRenderer::clean();
}
void StochasticLightcutsRenderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBufferType_StochasticLightcuts::eImgTonemapped);

	{
		std::vector<const char*> modeNames_pointers = { "PT", "Stochastic Lightcuts" };
		if (ImGui::Begin("StochasticLightcuts Settings"))
		{
			ImGui::SeparatorText("Temporal Acc");
			PE::begin();
			UIModified |= PE::DragInt("Max Frames", &maxFrames);
			PE::end();
			ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

			UIModified |= ImGui::Combo("Mode", &pushConstant.mode, modeNames_pointers.data(), static_cast<int>(modeNames_pointers.size()));
		}
		ImGui::End();
	}

	{
		uint32_t showLayerCount = pushConstant.lightTreeDepth + 1;
		std::vector<std::string> layerMapNames(showLayerCount);
		for (int i = 0; i < showLayerCount; ++i) layerMapNames[i] = "Layer" + std::to_string(i);

		std::vector<const char*> layerMapNames_pointers;
		for (const auto& layerMapName : layerMapNames)
			layerMapNames_pointers.push_back(layerMapName.c_str());

		static bool showLightTreeDebugMap = false;
		if (ImGui::Begin("Light Tree Debug")) {
			ImGui::Combo("Light Tree Layer", &lightDebugPushConstant.curLayer, layerMapNames_pointers.data(), static_cast<int>(layerMapNames_pointers.size()));
			if (PE::begin()) {
				if (PE::entry("LightTreeLayerClusterResult", [&] {
					static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
					ImVec4 selectedColor = showLightTreeDebugMap ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
					ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
					ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
					ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
					ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

					bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)GBufferType_StochasticLightcuts::eLightTreeDebug),
						ImVec2(100 * gBuffers.getAspectRatio(), 100));

					ImGui::PopStyleColor(2);
					ImGui::PopStyleVar();
					return result;
					}))
				{
					showLightTreeDebugMap = !showLightTreeDebugMap;
				}
			}
			PE::end();
		}
		ImGui::End();

		if (showLightTreeDebugMap) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBufferType_StochasticLightcuts::eLightTreeDebug);
	}

	if (UIModified) resetFrame();
}
void StochasticLightcutsRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));
	//清理深度纹理
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

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)GBufferType_StochasticLightcuts::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    depthImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_StochasticLightcuts::eDepthImage, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    normal_MaterialIndexImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_StochasticLightcuts::eNormal_MaterialIndexImage, 0, 0, 1);
	write.append(normal_MaterialIndexImageWrite, gBuffers.getColorImageView((uint32_t)GBufferType_StochasticLightcuts::eNormal_MaterialIndex), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.screenSize = { size.width, size.height };
}
void StochasticLightcutsRenderer::preRender() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) resetFrame();	//如果相机参数变化，则从新累计帧
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.maxFrameCount = maxFrames;

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.areaLightsAddress = (shaderio::AreaLight_StochasticLightcuts*)areaLightsBuffer.buffer.address;
	pushConstant.lightTreeAddress = (shaderio::LightNode*)LightTreeBuffer.buffer.address;
	pushConstant.hasLightBlockCountAddress = (uint32_t*)hasLightBlockCountBuffer.buffer.address;
	pushConstant.hasLightBlockArrayAddress = (uint32_t*)hasLightBlockArrayBuffer.buffer.address;
	pushConstant.hasLightBlockIndexAddress = (uint32_t*)hasLightBlockIndexBuffer.buffer.address;
	pushConstant.cmd = (shaderio::DispatchIndirectCommand*)indirectCmd.buffer.address;
	asManager.updateToplevelAS(cmd);

	{
		lightDebugPushConstant.frameIndex = pushConstant.frameIndex;
		lightDebugPushConstant.lightTreeDepth = pushConstant.lightTreeDepth;
		lightDebugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;
		lightDebugPushConstant.lightTreeAddress = pushConstant.lightTreeAddress;

		int debugLayer = lightDebugPushConstant.curLayer;
		lightDebugPushConstant.curLayerNodeCount = (1 << (3 * debugLayer));
		lightDebugPushConstant.curLayerNodeStartIndex = ((1 << (3 * debugLayer)) - 1) / 7;
	}

	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void StochasticLightcutsRenderer::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	if (pushConstant.frameIndex >= maxFrames && maxFrames > 1) return;

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	createLightTree(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	{
		NVVK_DBG_SCOPE(cmd);

		VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Render);
		pushInfo.pValues = &pushConstant;
		vkCmdPushConstants2(cmd, &pushInfo);
		VkExtent2D sceneSize = Application::app->getViewportSize();
		VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);

	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	//lightTreeDebug(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT);
	lightTreeDebug2(cmd);

	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo((uint32_t)GBufferType_StochasticLightcuts::eImgRendered), gBuffers.getDescriptorImageInfo((uint32_t)GBufferType_StochasticLightcuts::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}

void StochasticLightcutsRenderer::createSourceData() {
	Feature::createGBuffer(true, true, (uint32_t)GBufferType_StochasticLightcuts::eImgTonemapped);

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t sceneVoxelCount = Scene_Resolution * Scene_Resolution * Scene_Resolution;
	uint32_t lightNodeTotalCount = (8 * sceneVoxelCount - 1) / 7;
	pushConstant.lightNodeTotalCount = lightNodeTotalCount;
	pushConstant.lightTreeDepth = std::countr_zero((uint32_t)Scene_Resolution);

	uint32_t bufferSize = lightNodeTotalCount * sizeof(shaderio::LightNode);
	LightTreeBuffer = FzbRenderer::Buffer("LightTreeBuffer", false);
	LightTreeBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	NVVK_DBG_NAME(LightTreeBuffer.buffer.buffer);

	{
		Scene& mainScene = Application::sceneResource;
		int lightCountPerAxis = 9;
		shaderio::float3 wallMin = { -2.85, 0, -7.59 };
		shaderio::float3 wallMax = { 7.42, 3.8, -7.59 };

		float lightSizeX = (wallMax.x - wallMin.x) / lightCountPerAxis;
		float lightSizeY = (wallMax.y - wallMin.y) / lightCountPerAxis;
		shaderio::float3 scaleValue = { lightSizeX, lightSizeY, 1.0f };

		std::string materialIDs[3][3] = {
			{ "LightBSDF_red", "LightBSDF_green", "LightBSDF_blue" },
			{ "LightBSDF_green", "LightBSDF_blue", "LightBSDF_red" },
			{ "LightBSDF_blue", "LightBSDF_red", "LightBSDF_green" }
		};

		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createPlane(1, 1.0f, 1.0f);
		std::string meshSetID = "customPlaneLight";
		MeshSet lightMeshSet = FzbRenderer::MeshSet(meshSetID, primitive);
		mainScene.addMeshSet(lightMeshSet);

		areaLights.resize(lightCountPerAxis * lightCountPerAxis);

		for (int x = 0; x < lightCountPerAxis; ++x) {
			for (int y = 0; y < lightCountPerAxis; ++y) {
				shaderio::float3 lightPos = { wallMin.x + lightSizeX * x, wallMin.y + lightSizeY * y, wallMin.z };
				std::string materialID = materialIDs[y % 3][x % 3];

				InstanceSet instanceSet;
				instanceSet.type = InstanceType::Static;
				instanceSet.instanceID = "light_" + std::to_string(x) + "_" + std::to_string(y);

				if (!mainScene.meshSetIDToIndex.count(meshSetID)) LOGW("实例没有对应的mesh：%s\n", meshSetID.c_str());
				instanceSet.meshSetIndex = mainScene.meshSetIDToIndex[meshSetID];
				FzbRenderer::MeshSet& meshSet = mainScene.meshSets[instanceSet.meshSetIndex];
				std::vector<MeshInfo>& childMeshInfos = meshSet.childMeshInfos;

				instanceSet.baseMatrix_translate = glm::translate(instanceSet.baseMatrix_translate, lightPos);
				instanceSet.baseMatrix_scale = glm::scale(instanceSet.baseMatrix_scale, scaleValue);
				instanceSet.baseMatrix = instanceSet.baseMatrix_translate * instanceSet.baseMatrix_rotate * instanceSet.baseMatrix_scale;
				instanceSet.transform = instanceSet.baseMatrix;
				instanceSet.transform_lastTime = instanceSet.transform;

				instanceSet.childInstances.resize(childMeshInfos.size());
				for (int i = 0; i < childMeshInfos.size(); i++) {
					MeshInfo& childMesh = childMeshInfos[i];
					shaderio::Instance& instance = instanceSet.childInstances[i];
					instance.meshIndex = childMesh.meshIndex;

					if (!mainScene.uniqueMaterialIDToIndex.count(materialID)) materialID = "defaultMaterial";
					instance.materialIndex = mainScene.uniqueMaterialIDToIndex[materialID];

					instance.transform = instanceSet.baseMatrix;
				}

				mainScene.addInstanceSet(instanceSet);

				shaderio::AreaLight_StochasticLightcuts areaLight = {
					.startPos = lightPos,
					.edge1 = shaderio::float3(scaleValue.x, 0, 0),
					.edge2 = shaderio::float3(0, scaleValue.y, 0),
					.normal = {0, 0, 1},
					.color = mainScene.materials[instanceSet.childInstances[0].materialIndex].emissive
				};
				areaLights[x * lightCountPerAxis + y] = areaLight;
			}
		}
		{
			uint32_t offset = 0;
			mainScene.instances.resize(mainScene.staticInstanceCount + mainScene.periodInstanceCount + mainScene.randomInstanceCount);
			for (int i = 0; i < mainScene.staticInstanceSets.size(); ++i) {
				mainScene.staticInstanceSets[i].getInstance(mainScene.instances, offset, 0);

				for (int j = 0; j < mainScene.staticInstanceSets[i].childInstances.size(); ++j)
					mainScene.staticInstanceIndexToInstanceSetIndex.insert({ offset + j, i });

				offset += mainScene.staticInstanceSets[i].childInstances.size();
			}
			for (int i = 0; i < mainScene.periodInstanceSets.size(); ++i) {
				InstanceSet& instanceSet = mainScene.periodInstanceSets[i];
				instanceSet.getInstance(mainScene.instances, offset, 0);

				for (int j = 0; j < mainScene.periodInstanceSets[i].childInstances.size(); ++j)
					mainScene.periodInstanceIndexToInstanceSetIndex.insert({ offset + j, i });

				offset += instanceSet.childInstances.size();
			}
			for (int i = 0; i < mainScene.randomInstanceSets.size(); ++i) {
				InstanceSet& instanceSet = mainScene.randomInstanceSets[i];
				instanceSet.getInstance(mainScene.instances, offset, 0);
				offset += instanceSet.childInstances.size();
			}

			mainScene.isStaticScene = mainScene.instances.size() == mainScene.staticInstanceCount;
		}

		mainScene.createSceneInfoBuffer();
	}
	pushConstant.areaLightCount = areaLights.size();
	bufferSize = pushConstant.areaLightCount * sizeof(shaderio::AreaLight_StochasticLightcuts);
	areaLightsBuffer = FzbRenderer::Buffer("areaLightsBuffer", false);
	areaLightsBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	NVVK_CHECK(Application::stagingUploader.appendBuffer(areaLightsBuffer.buffer, 0, std::span<const shaderio::AreaLight_StochasticLightcuts>(areaLights)));
	NVVK_DBG_NAME(areaLightsBuffer.buffer.buffer);

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

		pushConstant.sceneSize = aabb.maximum - aabb.minimum;
		pushConstant.sceneStartPos = aabb.minimum;
	}

	hasLightBlockCountBuffer = FzbRenderer::Buffer("hasLightBlockCountBuffer", false);
	hasLightBlockCountBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(uint32_t),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	NVVK_DBG_NAME(hasLightBlockCountBuffer.buffer.buffer);

	bufferSize = sceneVoxelCount / 8 * sizeof(uint32_t);
	hasLightBlockArrayBuffer = FzbRenderer::Buffer("hasLightBlockArrayBuffer", false);
	hasLightBlockArrayBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	NVVK_DBG_NAME(hasLightBlockArrayBuffer.buffer.buffer);

	hasLightBlockIndexBuffer = FzbRenderer::Buffer("hasLightBlockIndexBuffer", false);
	hasLightBlockIndexBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = bufferSize,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	NVVK_DBG_NAME(hasLightBlockIndexBuffer.buffer.buffer);

	indirectCmd = FzbRenderer::Buffer("indirectCmd", false);
	indirectCmd.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::DispatchIndirectCommand),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT,
		});
	NVVK_DBG_NAME(indirectCmd.buffer.buffer);

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
		scene.addMeshSet(mesh);

		primitive = FzbRenderer::MeshSet::createCube(false, false);
		mesh = FzbRenderer::MeshSet("Cube", primitive);
		scene.addMeshSet(mesh);

		scene.createSceneInfoBuffer();
	}
}
void StochasticLightcutsRenderer::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
			.binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_StochasticLightcuts::eNormal_MaterialIndexImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_StochasticLightcuts::eDepthImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Fzb PathGuiding static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));

	bindings.clear();
	bindings.addBinding({
			.binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);

	LOGI("Fzb PathGuiding dynamic descriptor layout created\n");
}
void StochasticLightcutsRenderer::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void StochasticLightcutsRenderer::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
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
void StochasticLightcutsRenderer::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = 256,
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
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "createLightcuts.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_LightInfoInject, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_injectLightInfo";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_LightInfoInject);
		NVVK_DBG_NAME(computeShader_LightInfoInject);

		vkDestroyShaderEXT(device, computeShader_CreateLightcuts_1, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_CreateLightcuts_1";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateLightcuts_1);
		NVVK_DBG_NAME(computeShader_CreateLightcuts_1);

		vkDestroyShaderEXT(device, computeShader_CreateLightcuts_2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_CreateLightcuts_2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateLightcuts_2);
		NVVK_DBG_NAME(computeShader_CreateLightcuts_2);

		vkDestroyShaderEXT(device, computeShader_CreateLightcuts_3, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_CreateLightcuts_3";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateLightcuts_3);
		NVVK_DBG_NAME(computeShader_CreateLightcuts_3);

		vkDestroyShaderEXT(device, computeShader_CreateLightcuts_4, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_CreateLightcuts_4";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_CreateLightcuts_4);
		NVVK_DBG_NAME(computeShader_CreateLightcuts_4);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "createGBuffers.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_createGBuffers, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_createGBuffers);
		NVVK_DBG_NAME(vertexShader_createGBuffers);

		vkDestroyShaderEXT(device, fragmentShader_createGBuffers, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_createGBuffers);
		NVVK_DBG_NAME(fragmentShader_createGBuffers);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "render.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_Render, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_render";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Render);
		NVVK_DBG_NAME(computeShader_Render);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "lightTreeDebug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_lightTreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_lightTreeDebug);
		NVVK_DBG_NAME(vertexShader_lightTreeDebug);

		vkDestroyShaderEXT(device, fragmentShader_lightTreeDebug, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_lightTreeDebug);
		NVVK_DBG_NAME(fragmentShader_lightTreeDebug);
	}
	//--------------------------------------------------------------------------------------
	{
		vkDestroyShaderEXT(device, vertexShader_lightTreeDebug2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_lightTreeDebug2);
		NVVK_DBG_NAME(vertexShader_lightTreeDebug2);

		vkDestroyShaderEXT(device, fragmentShader_lightTreeDebug2, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain2";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_lightTreeDebug2);
		NVVK_DBG_NAME(fragmentShader_lightTreeDebug2);
	}
}
void StochasticLightcutsRenderer::updateDataPerFrame(VkCommandBuffer cmd) {}

void StochasticLightcutsRenderer::createLightTree(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	pushInfo.pValues = &pushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_LightInfoInject);
	VkExtent3D sceneVoxelSize = { Scene_Resolution, Scene_Resolution, Scene_Resolution };
	VkExtent3D groupSize = nvvk::getGroupCounts(sceneVoxelSize, VkExtent3D{ 4, 4, 4 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateLightcuts_1);
	sceneVoxelSize = { (uint32_t)pushConstant.lightNodeTotalCount, 1, 1 };
	groupSize = nvvk::getGroupCounts(sceneVoxelSize, VkExtent3D{ 1024, 1, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	for (int layer = pushConstant.lightTreeDepth; layer > 0; --layer) {
		pushConstant.curLayer = layer;
		uint32_t curLayerNodeCount = (1 << (3 * layer));
		pushConstant.curLayerBlockCount = curLayerNodeCount / 8;
		pushConstant.curLayerNodeStartIndex = ((1 << (3 * layer)) - 1) / 7;
		pushConstant.fatherLayerNodeStartIndex = ((1 << (3 * layer - 3)) - 1) / 7;
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdFillBuffer(cmd, hasLightBlockCountBuffer.buffer.buffer, 0, sizeof(uint32_t), 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateLightcuts_2);
		sceneVoxelSize = { (uint32_t)pushConstant.curLayerBlockCount, 1, 1 };
		groupSize = nvvk::getGroupCounts(sceneVoxelSize, VkExtent3D{ 1024, 1, 1 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateLightcuts_3);
		vkCmdDispatch(cmd, 1, 1, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_CreateLightcuts_4);
		vkCmdDispatchIndirect(cmd, indirectCmd.buffer.buffer, 0);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
	}
}
void StochasticLightcutsRenderer::createGBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t numColorAttachments = (uint32_t)GBufferType_StochasticLightcuts::eImgRendered;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(numColorAttachments);
	for (int i = 0; i < (uint32_t)GBufferType_StochasticLightcuts::eImgRendered; ++i) {
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
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.stencilTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, Application::app->getViewportSize());

	VkColorComponentFlags writeMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
	VkBool32 blendEnable = VK_FALSE;
	for (uint32_t i = 0; i < numColorAttachments; ++i) {
		vkCmdSetColorWriteMaskEXT(cmd, i, 1, &writeMask);
		vkCmdSetColorBlendEnableEXT(cmd, i, 1, &blendEnable);
	}

	vkCmdSetDepthTestEnable(cmd, VK_TRUE);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_createGBuffers, .fragment = fragmentShader_createGBuffers });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	pushInfo.pValues = &createGBufferPushConstant_StochasticLightcuts;
	createGBufferPushConstant_StochasticLightcuts.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i) {
		createGBufferPushConstant_StochasticLightcuts.instanceIndex = int(i);

		createGBufferPushConstant_StochasticLightcuts.normalMatrix = glm::transpose(glm::inverse(Application::sceneResource.instances[i].transform));
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}
	vkCmdEndRendering(cmd);

	for (int i = 0; i < (uint32_t)GBufferType_StochasticLightcuts::eImgRendered; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

void StochasticLightcutsRenderer::lightTreeDebug(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	bool useWireframe = false;

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_lightTreeDebug, .fragment = fragmentShader_lightTreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	pushInfo.pValues = &lightDebugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = lightDebugPushConstant.curLayerNodeCount;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void StochasticLightcutsRenderer::lightTreeDebug2(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	//VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	//colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	//colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug));
	//
	//VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	//depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	//depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug));
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
	depthAttachment.imageView = gBuffers.getDepthImageView();

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_lightTreeDebug, .fragment = fragmentShader_lightTreeDebug });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	bool useWireframe = true;
	uint32_t meshIndex = useWireframe ? 0 : 1;
	const shaderio::Mesh& mesh = scene.meshes[meshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	pushInfo.pValues = &lightDebugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(meshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	uint32_t instanceCount = lightDebugPushConstant.curLayerNodeCount;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(GBufferType_StochasticLightcuts::eLightTreeDebug)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}