#include "./ReSTIR_GI.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>

using namespace FzbRenderer;

ReSTIR_GI::ReSTIR_GI(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();

	if (pugi::xml_node sppNode = rendererNode.child("spp"))
		pushConstant.spp = std::stoi(sppNode.attribute("value").value());
}

void ReSTIR_GI::init() {
	createSourceData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void ReSTIR_GI::clean() {
	areaLightsBuffer.clean();
	pixelDataBuffer.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_ReSTIR_GI, nullptr);
	vkDestroyShaderEXT(device, computeShader_Spatial_Reuse_ReSTIR_GI, nullptr);

	PathTracingRenderer::clean();
};
void ReSTIR_GI::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_ReSTIR_GI::eImgTonemapped);

	std::vector<const char*> modeNames_pointers = { "PT", "RIS", "eRIS_Spatial_Reuse", "RIS_SpatialTemporal_Reuse" };
	if (ImGui::Begin("SLPGSettings"))
	{
		ImGui::SeparatorText("Jitter");
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		PE::end();
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		ImGui::Combo("Mode", &pushConstant.mode, modeNames_pointers.data(), static_cast<int>(modeNames_pointers.size()));
	}
	ImGui::End();

	if (UIModified) resetFrame();
}
void ReSTIR_GI::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	pixelDataBuffer.clean();
	pixelDataBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::PixelData_ReSTIR_GI) * (size.width * size.height),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_ReSTIR_GI::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet	pixelDataBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::ePixelData, 0, 0, 1);
	write.append(pixelDataBufferWrite, pixelDataBuffer.buffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.sceneSize = { size.width, size.height };
}
void ReSTIR_GI::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) resetFrame();	//如果相机参数变化，则从新累计帧
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;

	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.maxFrameCount = maxFrames;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	asManager.updateToplevelAS(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void ReSTIR_GI::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	if (pushConstant.frameIndex >= maxFrames && maxFrames > 1) return;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};
	
	{
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_ReSTIR_GI);
		vkCmdPushConstants2(cmd, &pushInfo);
		VkExtent2D sceneSize = Application::app->getViewportSize();
		VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	if (pushConstant.mode == (uint32_t)ReSTIR_GI_Mode::eRIS_Spatial_Reuse) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Spatial_Reuse_ReSTIR_GI);
		pushConstant.spatialReuseRadius = 8;
		vkCmdPushConstants2(cmd, &pushInfo);
		VkExtent2D sceneSize = Application::app->getViewportSize();
		VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo((uint32_t)ImageType_ReSTIR_GI::eImgRendered), gBuffers.getDescriptorImageInfo((uint32_t)ImageType_ReSTIR_GI::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void ReSTIR_GI::createSourceData() {
	Feature::createGBuffer(false, true, 1);

	{
		Scene& mainScene= Application::sceneResource;
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

				shaderio::AreaLight_ReSTIR_GI areaLight = {
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

	areaLightsBuffer = FzbRenderer::Buffer("areaLightsBuffer", false);
	areaLightsBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::AreaLight_ReSTIR_GI) * areaLights.size(),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});
	NVVK_CHECK(Application::stagingUploader.appendBuffer(areaLightsBuffer.buffer, 0, std::span<const shaderio::AreaLight_ReSTIR_GI>(areaLights)));

	pixelDataBuffer = FzbRenderer::Buffer("pixelDataBuffer", false);

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);
}
void ReSTIR_GI::createDescriptorSetLayout() {
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
		.binding = (uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::eAreaLights,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::ePixelData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("ReSTIR GI static descriptor layout created\n");
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

	LOGI("ReSTIR GI dynamic descriptor layout created\n");
}
void ReSTIR_GI::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet	areaLightsBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::eAreaLights, 0, 0, 1);
	write.append(areaLightsBufferWrite, areaLightsBuffer.buffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void ReSTIR_GI::createPipelineLayout() {
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
void ReSTIR_GI::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "ReSTIR_GI.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

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
		vkDestroyShaderEXT(device, computeShader_ReSTIR_GI, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_ReSTIR_GI";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_ReSTIR_GI);
		NVVK_DBG_NAME(computeShader_ReSTIR_GI);
	}
	{
		vkDestroyShaderEXT(device, computeShader_Spatial_Reuse_ReSTIR_GI, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_ReSTIR_GI_SpatialReuse";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Spatial_Reuse_ReSTIR_GI);
		NVVK_DBG_NAME(computeShader_Spatial_Reuse_ReSTIR_GI);
	}
}
void ReSTIR_GI::updateDataPerFrame(VkCommandBuffer cmd) {}