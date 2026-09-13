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
	pixelDataBuffer_lastFrame.clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_ReSTIR_GI, nullptr);
	vkDestroyShaderEXT(device, computeShader_Spatial_Reuse_ReSTIR_GI, nullptr);
	vkDestroyShaderEXT(device, computeShader_Temporal_Reuse_ReSTIR_GI, nullptr);

	PathTracingRenderer::clean();
};
void ReSTIR_GI::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_ReSTIR_GI::eImgTonemapped);

	std::vector<const char*> modeNames_pointers = { "PT", "RIS", "eRIS_Spatial_Reuse", "RIS_SpatialTemporal_Reuse" };
	if (ImGui::Begin("SLPGSettings"))
	{
		ImGui::SeparatorText("Jitter");
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		PE::end();
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		UIModified |= ImGui::Combo("Mode", &pushConstant.mode, modeNames_pointers.data(), static_cast<int>(modeNames_pointers.size()));
		PE::begin();
		if(pushConstant.mode == (uint32_t)ReSTIR_GI_Mode::eRIS_Spatial_Reuse) 
			UIModified |= PE::DragInt("Spatial Reuse Radius", &pushConstant.spatialReuseRadius, 1, 1, 32);
		PE::end();
	}
	ImGui::End();

	if (UIModified) resetFrame();
}
void ReSTIR_GI::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
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

	pixelDataBuffer.clean();
	pixelDataBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::PixelData_ReSTIR_GI) * (size.width * size.height),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	pixelDataBuffer_lastFrame.clean();
	pixelDataBuffer_lastFrame.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::PixelData_ReSTIR_GI) * (size.width * size.height),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)GBuffers_ReSTIR_GI::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet	pixelDataBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::ePixelData, 0, 0, 1);
	write.append(pixelDataBufferWrite, pixelDataBuffer.buffer);

	VkWriteDescriptorSet	gBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::eVelocityImage, 0, 0, 1);
	write.append(gBufferWrite, gBuffers.getColorImageView((uint32_t)GBuffers_ReSTIR_GI::eVelocity), VK_IMAGE_LAYOUT_GENERAL);

	pixelDataBufferWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::ePixelData_lastFrame, 0, 0, 1);
	write.append(pixelDataBufferWrite, pixelDataBuffer_lastFrame.buffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	pushConstant.screenSize = { size.width, size.height };
	resetFrame();
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

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
		.pValues = &pushConstant,
	};

	//if (pushConstant.mode == (uint32_t)ReSTIR_GI_Mode::eRIS_SpatialTemporal_Reuse) {
	//	if (Application::sceneResource.cameraChange) pushConstant.mode = (uint32_t)ReSTIR_GI_Mode::eRIS_Spatial_Reuse;
	//}

	if (pushConstant.mode == (uint32_t)ReSTIR_GI_Mode::eRIS_SpatialTemporal_Reuse) {
		createGBuffers(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	
	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());
	
	RIS(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	if (pushConstant.mode >= (uint32_t)ReSTIR_GI_Mode::eRIS_Spatial_Reuse) {
		SpatialReuse(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	if (pushConstant.mode == (uint32_t)ReSTIR_GI_Mode::eRIS_SpatialTemporal_Reuse) {
		TemporalReuse(cmd);
		//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_ReSTIR_GI::eImgRendered), gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_ReSTIR_GI::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void ReSTIR_GI::createSourceData() {
	Feature::createGBuffer(true, true, (uint32_t)GBuffers_ReSTIR_GI::eImgTonemapped);

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
	pixelDataBuffer_lastFrame = FzbRenderer::Buffer("pixelDataBuffer_lastFrame", false);

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

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::eVelocityImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_ReSTIR_GI::ePixelData_lastFrame,
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
		shaderSource = shaderPath / "createGBuffers.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_createGBuffer);
		NVVK_DBG_NAME(vertexShader_createGBuffer);

		vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_createGBuffer);
		NVVK_DBG_NAME(fragmentShader_createGBuffer);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "ReSTIR_GI.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
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
	{
		vkDestroyShaderEXT(device, computeShader_Temporal_Reuse_ReSTIR_GI, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_ReSTIR_GI_TemporalReuse";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_Temporal_Reuse_ReSTIR_GI);
		NVVK_DBG_NAME(computeShader_Temporal_Reuse_ReSTIR_GI);
	}
}
void ReSTIR_GI::updateDataPerFrame(VkCommandBuffer cmd) {}

void ReSTIR_GI::createGBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t numColorAttachments = (uint32_t)GBuffers_ReSTIR_GI::eImgRendered;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(numColorAttachments);
	for (int i = 0; i < (uint32_t)GBuffers_ReSTIR_GI::eImgRendered; ++i) {
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
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_createGBuffer, .fragment = fragmentShader_createGBuffer });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	pushInfo.pValues = &createGBuffersPushConstant;
	createGBuffersPushConstant.vpMatrix_lastFrame = Application::sceneResource.cameraInfo_lastFrame.projMatrix * Application::sceneResource.cameraInfo_lastFrame.viewMatrix;
	createGBuffersPushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i){
		createGBuffersPushConstant.instanceIndex = int(i);

		const FzbRenderer::InstanceSet* instanceSet = nullptr;
		if (Application::sceneResource.periodInstanceIndexToInstanceSetIndex.count(i)) {
			uint32_t instanceSetIndex = Application::sceneResource.periodInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.periodInstanceSets[instanceSetIndex];
		}
		else if (Application::sceneResource.staticInstanceIndexToInstanceSetIndex.count(i)) {
			uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
			instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
		}
		createGBuffersPushConstant.tansfromMatrix_lastFrame = instanceSet ? instanceSet->transform_lastTime : Application::sceneResource.instances[i].transform;
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

	for (int i = 0; i < (uint32_t)GBuffers_ReSTIR_GI::eImgRendered; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void ReSTIR_GI::RIS(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_ReSTIR_GI);
	pushInfo.pValues = &pushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent2D sceneSize = Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ 16, 16 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void ReSTIR_GI::SpatialReuse(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Spatial_Reuse_ReSTIR_GI);
	pushInfo.pValues = &pushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent2D sceneSize = Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ Spatial_Reuse_GroupSize, Spatial_Reuse_GroupSize });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}
void ReSTIR_GI::TemporalReuse(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_Temporal_Reuse_ReSTIR_GI);
	pushInfo.pValues = &pushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent2D sceneSize = Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ Spatial_Reuse_GroupSize, Spatial_Reuse_GroupSize });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

	VkBufferCopy2 copyRegion = {
		.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
		.pNext = nullptr,
		.srcOffset = 0,
		.dstOffset = 0,
		.size = pixelDataBuffer.allocMemSize,
	};
	VkCopyBufferInfo2 copyInfo = {
		.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
		.pNext = nullptr,
		.srcBuffer = pixelDataBuffer.buffer.buffer,
		.dstBuffer = pixelDataBuffer_lastFrame.buffer.buffer,
		.regionCount = 1,
		.pRegions = &copyRegion
	};
	vkCmdCopyBuffer2(cmd, &copyInfo);
}