#include "./VolumetricFog.h"

#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <iostream>

using namespace FzbRenderer;

VolumetricFog::VolumetricFog(pugi::xml_node& rendererNode) {
	derivFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COMPUTE_SHADER_DERIVATIVES_FEATURES_KHR;
	derivFeatures.pNext = nullptr;
	derivFeatures.computeDerivativeGroupQuads = VK_TRUE;
	derivFeatures.computeDerivativeGroupLinear = VK_FALSE;
	//Application::vkContextInitInfo.deviceExtensions.push_back({ VK_KHR_COMPUTE_SHADER_DERIVATIVES_EXTENSION_NAME, &derivFeatures });

	//if (pugi::xml_node volumetricFogCountNode = rendererNode.child("volumetricFogCount_local"))
	//	volumetricFogCount = std::stoi(volumetricFogCountNode.attribute("value").value());
	volumetricFogCount = 2;
	pushConstant.volumetricFogCount = volumetricFogCount;
	volumetricFogInfos.resize(volumetricFogCount);
	volumetricFogInfoModified.resize(volumetricFogCount);
#ifndef NDEBUG
	showVolumetricFogVoxelGrids.resize(volumetricFogCount);
#endif

	volumetricFogHeightCount = 0;
	volumetricFogHeightInfos.resize(volumetricFogHeightCount);

	volumetricFogFluidCount = 1;
	volumetricFogFluidInfos.resize(volumetricFogFluidCount);
	volumetricFogFluidVoxelInfoBuffers.resize(volumetricFogFluidCount);
	volumetricFogFluidVoxelVelocityImages.resize(volumetricFogFluidCount);
	volumetricFogFluidExtinctionImages.resize(volumetricFogFluidCount);

	volumetricFogNoiseCount = 1;
	volumetricFogNoiseInfos.resize(volumetricFogNoiseCount);

	volumetricFogInfos[0] = {
		.fogStartPos = {5098.0f, -69.5f, -4470.0f},
		.fogVoxelGridSize = {16, 16, 16},
		.fogVoxelSize = {2.0f, 0.05f, 2.0f},
		.color = {1.0f, 1.0f, 1.0f},
		.ambientIntensity = 0.1f,
		.absorption = { 0.0, 3.0 },
		.scattering = 0.7f,
		.phase = 0.5,
		.type = shaderio::VolumetricFogType::Noise,
		.volumetricFogTypeIndex = 0,
	};
	volumetricFogNoiseInfos[0] = {
		.cloudScale = {0.5f, 0.5f, 0.5f},
		.cloudFlowSpeed = 0.05f,
		.cloudCoverage = {0.3f, 0.2f},
		.cloudTypePreference = {1.0f, 0.0f},
		.weatherScale = 0.01f,
	};
	volumetricFogNoiseIndexMap.insert({ 0, 0 });

	volumetricFogInfos[1] = {
		.fogStartPos = {5111.0f, -69.0f, -4453.0f},
		.fogVoxelGridSize = {32, 32, 32},
		.fogVoxelSize = { 0.2, 0.2, 0.2 },
		.color = {1.0f, 1.0f, 1.0f},
		.ambientIntensity = 0.0f,
		.absorption = { 0.1, 0.3 },
		.scattering = 0.7f,
		.phase = 0.5f,
		.type = shaderio::VolumetricFogType::Fluid,
		.volumetricFogTypeIndex = 0
	};
	volumetricFogFluidInfos[0] = {
		.viscosity = 0.01f,
		.FIntensity = 10.0f,
		.lightAttenuationEstimator = 1.0f,
	};
	volumetricFogFluidIndexMap.insert({ 0, 1 });
	pushConstant.ambientFogDensity = volumetricFogInfos[1].absorption.x + volumetricFogInfos[1].scattering;
}

void VolumetricFog::init() {
	VkSamplerCreateInfo samplerInfo = DEFAULT_VkSamplerCreateInfo;
	samplerInfo.magFilter = VK_FILTER_NEAREST;
	samplerInfo.minFilter = VK_FILTER_NEAREST;
	samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
	Renderer::createGBuffer(true, true, (uint32_t)GBuffers_VolumetricFog::eTonemapping, { 1, 1 }, samplerInfo);
	Application::samplerPool.acquireSampler(gBuffers.m_res.gBufferDepth.descriptor.sampler, samplerInfo);

#ifndef NDEBUG
	//---------------------------------------------wireframe-----------------------------------
	nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
	FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
	scene.addMeshSet(mesh);

	scene.createSceneInfoBuffer();
#endif

	shadowMap.init({ 2048, 2048 });

	createVolumetricFogData();
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();

	//for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i) {
	//	if (!Application::sceneResource.staticInstanceIndexToInstanceSetIndex.count(i)) continue;
	//
	//	uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
	//	FzbRenderer::InstanceSet* instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
	//
	//	uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
	//	MeshInfo& meshInfo = Application::sceneResource.getMeshInfo(meshIndex);
	//	instanceSet->aabb = meshInfo.getAABB(Application::sceneResource.instances[i].transform, false);
	//}
}
void VolumetricFog::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_getVisibleVolumetricFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);

	vkDestroyShaderEXT(device, computeShader_createVolumetricFogFluid, nullptr);
	vkDestroyShaderEXT(device, computeShader_initVolumetricFogFluid, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_A, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_D, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_F, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_P, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_S, nullptr);

	vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);
	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	GlobalInfoBuffer.clean();
	visibleVolumetricFogIndexBuffer.clean();

	volumetricFogInfosBuffer.clean();

	volumetricFogHeightInfoBuffer.clean();

	volumetricFogFluidInfoBuffer.clean();
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		volumetricFogFluidVoxelVelocityImages[i].clean();
		volumetricFogFluidVoxelInfoBuffers[i].clean();
		volumetricFogFluidExtinctionImages[i].clean();
	}

	volumetricFogNoiseInfoBuffer.clean();


	shadowMap.clean();

#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_renderVoxelGrid, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_renderVoxelGrid, nullptr);
#endif

	Renderer::clean();
};
void VolumetricFog::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)GBuffers_VolumetricFog::eTonemapping);

	const char* fogTypeItems[] = { "Height", "Fluid", "Noise"};

	uint32_t heightFogIndex = 0, fluidFogIndex = 0, noiseFogIndex = 0;
	if (ImGui::Begin("Volumetric Fog Setting")) {
		for (int i = 0; i < volumetricFogCount; ++i) {
			volumetricFogInfoModified[i] = false;
			if (ImGui::CollapsingHeader(std::string("Volumetric Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
				volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Start Pos " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].fogStartPos);

				ImGui::BeginDisabled(true);
				bool change = ImGui::DragInt3(std::string("Volumetric Fog Voxel Grid Size " + std::to_string(i)).c_str(), (int*)&volumetricFogInfos[i].fogVoxelGridSize);
				ImGui::EndDisabled();
				if (change) {
					//createVolumetricFogData();
					volumetricFogInfoModified[i] = true;
				}

				volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Voxel Size " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].fogVoxelSize);

				volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Color " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].color);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Ambient Intensity " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].ambientIntensity, 0.1f, 0.0f, 10.0f);

				volumetricFogInfoModified[i] |= ImGui::DragFloat2(std::string("Extinction Coefficient " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].absorption, 0.1f, 0.0f);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Scatter Coefficient " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].scattering, 0.1f, 0.0f, 1.0f);
				volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Asymmetric Parameters " + std::to_string(i)).c_str(), (float*)&volumetricFogInfos[i].phase, 0.1f, -1.0f, 1.0f);

				ImGui::BeginDisabled(true);
				volumetricFogInfoModified[i] |= ImGui::Combo(std::string("Type " + std::to_string(i)).c_str(), (int*)&volumetricFogInfos[i].type, fogTypeItems, IM_ARRAYSIZE(fogTypeItems));
				ImGui::EndDisabled();
				if (volumetricFogInfos[i].type == shaderio::VolumetricFogType::Height) {
					if (ImGui::CollapsingHeader(std::string("Height Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= 
							ImGui::DragFloat(std::string("Height Attenuation " + std::to_string(i)).c_str(), (float*)&volumetricFogHeightInfos[heightFogIndex].heightScale, 1.0f, 0.0f, 1000.0f);
					}
					++heightFogIndex;
				}
				else if(volumetricFogInfos[i].type == shaderio::VolumetricFogType::Fluid) {
					if (ImGui::CollapsingHeader(std::string("Fluid Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Viscosity " + std::to_string(i)).c_str(), (float*)&volumetricFogFluidInfos[fluidFogIndex].viscosity, 0.1f, 0.0f, 1.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("F Intensity " + std::to_string(i)).c_str(), (float*)&volumetricFogFluidInfos[fluidFogIndex].FIntensity, 1.0f, 0.0f, 100.0f);
					}
					++fluidFogIndex;
				}
				else if (volumetricFogInfos[i].type == shaderio::VolumetricFogType::Noise) {
					if (ImGui::CollapsingHeader(std::string("Noise Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
						volumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Cloud Scale " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudScale, 0.01f, 0.0f, 2.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Cloud Flow Speed " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudFlowSpeed, 0.1f, 0.0f, 10.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat2(std::string("Cloud Coverage " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudCoverage, 0.1f, 0.0f, 2.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Cloud Type Preference " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].cloudTypePreference, 0.1f, 0.0f, 1.0f);
						volumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Weather Scale " + std::to_string(i)).c_str(), (float*)&volumetricFogNoiseInfos[noiseFogIndex].weatherScale, 0.01f, 0.0f, 2.0f);
					}
					++noiseFogIndex;
				}

				volumetricFogInfoModified[i] |= ImGui::Checkbox(std::string("show voxel grid " + std::to_string(i)).c_str(), (bool*)&showVolumetricFogVoxelGrids[i]);
			}
			UIModified |= volumetricFogInfoModified[i];
		}
	}
	ImGui::End();

	shadowMap.uiRender();
}
void VolumetricFog::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));
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

	VkWriteDescriptorSet    imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eAlbedo]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eNormal]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferDepth);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eEmissive]);

	imageWrite = staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(imageWrite, gBuffers.m_res.gBufferColor[(uint32_t)GBuffers_VolumetricFog::eRendered]);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	shadowMap.resize(cmd, size);
}
void VolumetricFog::preRender() {
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) Application::frameIndex = 0;
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	shadowMap.preRender();
}
void VolumetricFog::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
		.pValues = &pushConstant,
	};

	static float time = 0.0f;
	pushConstant.time = time;
	pushConstant.dt = ImGui::GetIO().DeltaTime;
	time += pushConstant.dt;

	initVolumetricFogFluid(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

	createGBuffers(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	shadowMap.render(cmd);
	createVolumetricFog(cmd);
	if (Application::sceneResource.sceneInfo.useSky)
	{
		const glm::mat4& viewMatrix = Application::sceneResource.cameraManip->getViewMatrix();
		const glm::mat4& projMatrix = Application::sceneResource.cameraManip->getPerspectiveMatrix();
		Application::skySimple.runCompute(cmd, Application::app->getViewportSize(), viewMatrix, projMatrix,
			Application::sceneResource.sceneInfo.skySimpleParam, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	deferredRenderring(cmd);

#ifndef NDEBUG
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
	renderVolumetricFogVoxelGrid(cmd);
#endif
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	//Renderer::postProcess(cmd, &gBuffers.m_res.gBufferColor[((uint32_t)GBuffers_VolumetricFog::eTonemapping)].descriptor);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
		gBuffers.getDescriptorImageInfo(((uint32_t)GBuffers_VolumetricFog::eRendered)),
		gBuffers.getDescriptorImageInfo(((uint32_t)GBuffers_VolumetricFog::eTonemapping)));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

	firstFrame = false;
}

void VolumetricFog::createVolumetricFogImage(FzbRenderer::Image& image, shaderio::uint3 size){
	image.clean();

	static int imageCount = 0;
	image = FzbRenderer::Image("volumetricFog3DTexture" + std::to_string(imageCount));
	++imageCount;

	FzbRenderer::ImageCreateInfo colorImageCreateInfo = FzbRenderer::createDefaultImageCreateInfo();
	colorImageCreateInfo.info.format = VK_FORMAT_R32G32B32A32_SFLOAT;
	colorImageCreateInfo.info.imageType = VK_IMAGE_TYPE_3D;
	colorImageCreateInfo.info.extent = { size.x, size.y, size.z };

	colorImageCreateInfo.viewInfo.format = colorImageCreateInfo.info.format;
	colorImageCreateInfo.viewInfo.viewType = VK_IMAGE_VIEW_TYPE_3D;

	colorImageCreateInfo.samplerInfo.magFilter = VK_FILTER_LINEAR;
	colorImageCreateInfo.samplerInfo.minFilter = VK_FILTER_LINEAR;
	colorImageCreateInfo.samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
	colorImageCreateInfo.samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
	colorImageCreateInfo.samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

	image.init(colorImageCreateInfo);
}
void VolumetricFog::createVolumetricFogData() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	GlobalInfoBuffer = FzbRenderer::Buffer("GlobalInfoBuffer", false);
	GlobalInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::GlobalInfo_VolumetricFog),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	visibleVolumetricFogIndexBuffer = FzbRenderer::Buffer("visibleVolumetricFogIndexBuffer", false);
	visibleVolumetricFogIndexBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(uint32_t) * volumetricFogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	volumetricFogInfosBuffer = FzbRenderer::Buffer("volumetricFogInfosBuffer", false);
	volumetricFogInfosBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::VolumetricFogInfo) * volumetricFogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});
	vkCmdUpdateBuffer(cmd, volumetricFogInfosBuffer.buffer.buffer, 0, sizeof(shaderio::VolumetricFogInfo) * volumetricFogCount, volumetricFogInfos.data());
	//-----------------------------------------------------------高度----------------------------------------------------
	if (volumetricFogHeightCount > 0) {
		volumetricFogHeightInfoBuffer = FzbRenderer::Buffer("volumetricFogHeightInfoBuffer", false);
		volumetricFogHeightInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::HeightFogInfo) * volumetricFogHeightCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	//-----------------------------------------------------------流体----------------------------------------------------
	if (volumetricFogFluidCount > 0) {
		volumetricFogFluidInfoBuffer = FzbRenderer::Buffer("volumetricFogFluidInfoBuffer", false);
		volumetricFogFluidInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::FluidFogInfo) * volumetricFogFluidCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		uint32_t fogIndex = volumetricFogFluidIndexMap[i];

		createVolumetricFogImage(volumetricFogFluidVoxelVelocityImages[i], volumetricFogInfos[fogIndex].fogVoxelGridSize);

		volumetricFogFluidVoxelInfoBuffers[i] = FzbRenderer::Buffer("volumetricFogFluidVoxelInfoBuffer" + std::to_string(i), false);
		volumetricFogFluidVoxelInfoBuffers[i].init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::VolumetricFogFluidVoxelInfo) * volumetricFogInfos[fogIndex].fogVoxelGridSize.x * volumetricFogInfos[fogIndex].fogVoxelGridSize.y * volumetricFogInfos[fogIndex].fogVoxelGridSize.z,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});

		createVolumetricFogImage(volumetricFogFluidExtinctionImages[i], volumetricFogInfos[fogIndex].fogVoxelGridSize);

		//image init的时候会全部置0
		//vkCmdFillBuffer(cmd, volumetricFogVoxelInfoBuffers[i].buffer.buffer, 0, volumetricFogVoxelInfoBuffers[i].allocMemSize, 0);
	}
	//-----------------------------------------------------------噪声----------------------------------------------------
	if (volumetricFogNoiseCount > 0) {
		volumetricFogNoiseInfoBuffer = FzbRenderer::Buffer("volumetricFogNoiseInfoBuffer", false);
		volumetricFogNoiseInfoBuffer.init({
			.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
			.size = sizeof(shaderio::NoiseFogInfo) * volumetricFogNoiseCount,
			.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			});
	}
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void VolumetricFog::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	{
		bindings.addBinding({ .binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
				.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage,
				.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
				.descriptorCount = 1,
				.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVisibleVolumetricFogIndexBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------高度-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogHeightCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------流体-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogFluidCount,
		.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogFluidCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });

		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidExtinctionImages,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
		bindings.addBinding({
			.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidExtinctionImages_sampler,
			.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
			.descriptorCount = volumetricFogFluidCount,
			.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//----------------------------------------------------------------噪声-----------------------------------------------------------------------------
	{
		bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = volumetricFogNoiseCount,
		.stageFlags = VK_SHADER_STAGE_ALL });
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eShadowMap,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	LOGI("Fzb PathGuiding static descriptor layout created\n");
	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void VolumetricFog::createDescriptorSet() {
	nvvk::WriteSetContainer write{};

	{
		if (!Application::sceneResource.textures.empty()) {
			VkWriteDescriptorSet    allTextures =
				staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eTextures, 0, 0, uint32_t(Application::sceneResource.textures.size()));
			nvvk::Image* allImages = Application::sceneResource.textures.data();
			write.append(allTextures, allImages);
		}

		VkWriteDescriptorSet	gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAlbedoImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eAlbedo));

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eNormalImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eNormal));

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eDepthImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_READ_ONLY_OPTIMAL, gBuffers.m_res.gBufferDepth.descriptor.sampler);

		gBuffersWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eEmissiveImage, 0, 0, 1);
		write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eEmissive));
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	{
		VkWriteDescriptorSet globalInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer, 0, 0, 1);
		write.append(globalInfoWrite, GlobalInfoBuffer.buffer);

		VkWriteDescriptorSet	volumetricFogInfoWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer, 0, 0, 1);
		write.append(volumetricFogInfoWrite, volumetricFogInfosBuffer.buffer);

		VkWriteDescriptorSet	visibleVolumetricFogIndexWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVisibleVolumetricFogIndexBuffer, 0, 0, 1);
		write.append(visibleVolumetricFogIndexWrite, visibleVolumetricFogIndexBuffer.buffer);
	}
	//------------------------------------------------------------------高度---------------------------------------------------------------------------
	if (volumetricFogHeightCount > 0) {
		VkWriteDescriptorSet	volumetricFogHeightWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer, 0, 0, 1);
		write.append(volumetricFogHeightWrite, volumetricFogHeightInfoBuffer.buffer);
	}
	//------------------------------------------------------------------流体---------------------------------------------------------------------------
	if (volumetricFogFluidCount > 0) {
		std::vector<nvvk::Buffer> volumetricFogBuffer_nvvk(volumetricFogFluidCount);
		nvvk::Buffer* volumetricFogBufferPtr = nullptr;
		auto getNvvkBufferPtr = [&](std::vector<FzbRenderer::Buffer> buffers) {
			for (int i = 0; i < volumetricFogFluidCount; ++i) volumetricFogBuffer_nvvk[i] = buffers[i].buffer;
			volumetricFogBufferPtr = volumetricFogBuffer_nvvk.data();
		};
		std::vector<nvvk::Image> volumetricFogImage_nvvk(volumetricFogFluidCount);
		nvvk::Image* volumetricFogImagePtr;
		auto getNvvkImagePtr = [&](std::vector<FzbRenderer::Image> images) {
			for (int i = 0; i < volumetricFogFluidCount; ++i) volumetricFogImage_nvvk[i] = images[i].image;
			volumetricFogImagePtr = volumetricFogImage_nvvk.data();
		};

		VkWriteDescriptorSet	volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer, 0, 0, 1);
		write.append(volumetricFogFluidWrite, volumetricFogFluidInfoBuffer.buffer);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer, 0, 0, volumetricFogFluidCount);
		getNvvkBufferPtr(volumetricFogFluidVoxelInfoBuffers);
		write.append(volumetricFogFluidWrite, volumetricFogBufferPtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage, 0, 0, volumetricFogFluidCount);
		getNvvkImagePtr(volumetricFogFluidVoxelVelocityImages);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample, 0, 0, volumetricFogFluidCount);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidExtinctionImages, 0, 0, volumetricFogFluidCount);
		getNvvkImagePtr(volumetricFogFluidExtinctionImages);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);

		volumetricFogFluidWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogFluidExtinctionImages_sampler, 0, 0, volumetricFogFluidCount);
		write.append(volumetricFogFluidWrite, volumetricFogImagePtr);
	}
	//------------------------------------------------------------------噪声---------------------------------------------------------------------------
	if (volumetricFogNoiseCount > 0) {
		VkWriteDescriptorSet	volumetricFogNoiseWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer, 0, 0, 1);
		write.append(volumetricFogNoiseWrite, volumetricFogNoiseInfoBuffer.buffer);
	}
	//---------------------------------------------------------------------------------------------------------------------------------------------
	VkWriteDescriptorSet	shadowMapWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eShadowMap, 0, 0, 1);
	write.append(shadowMapWrite, shadowMap.shadowMaps[0].image);

	VkWriteDescriptorSet	renderedImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(renderedImageWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void VolumetricFog::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant)
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
void VolumetricFog::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "createGBuffers.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::VolumetricFogPushConstant),
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
	//--------------------------------------------------------------------------------------
	shaderSource = shaderPath / "createVolumetricFog.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, computeShader_getVisibleVolumetricFog, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_getVisibleVolumetricFog";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_getVisibleVolumetricFog);
	NVVK_DBG_NAME(computeShader_getVisibleVolumetricFog);

	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createVolumetricFog";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog);
	NVVK_DBG_NAME(computeShader_createVolumetricFog);

	{
		vkDestroyShaderEXT(device, computeShader_createVolumetricFogFluid, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFogFluid";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFogFluid);
		NVVK_DBG_NAME(computeShader_createVolumetricFogFluid);

		vkDestroyShaderEXT(device, computeShader_initVolumetricFogFluid, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_initVolumetricFogFluid";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initVolumetricFogFluid);
		NVVK_DBG_NAME(computeShader_initVolumetricFogFluid);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_A, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_A";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_A);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_A);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_D, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_D";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_D);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_D);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_F, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_F";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_F);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_F);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_P, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_P";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_P);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_P);

		vkDestroyShaderEXT(device, computeShader_createVolumetricFog_Fluid_S, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_createVolumetricFog_Fluid_S";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog_Fluid_S);
		NVVK_DBG_NAME(computeShader_createVolumetricFog_Fluid_S);
	}

	vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createLightAttenuationEstimator";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createLightAttenuationEstimator);
	NVVK_DBG_NAME(computeShader_createLightAttenuationEstimator);
	//--------------------------------------------------------------------------------------
	shaderSource = shaderPath / "deferredRendering.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_deferredRendering";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_deferredRenderring);
	NVVK_DBG_NAME(computeShader_deferredRenderring);

#ifndef NDEBUG
	shaderSource = shaderPath / "renderVolumetricFogVoxelGrid.slang";
	shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(device, vertexShader_renderVoxelGrid, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader_renderVoxelGrid);
	NVVK_DBG_NAME(vertexShader_renderVoxelGrid);

	vkDestroyShaderEXT(device, fragmentShader_renderVoxelGrid, nullptr);
	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader_renderVoxelGrid);
	NVVK_DBG_NAME(fragmentShader_renderVoxelGrid);
#endif
}
void VolumetricFog::updateDataPerFrame(VkCommandBuffer cmd) {
	if (volumetricFogCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if(volumetricFogHeightCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogHeightInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if(volumetricFogFluidCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogFluidInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if(volumetricFogNoiseCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogNoiseInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });

	bool heightModified = false, fluidModified = false, noiseModified = false;
	for (int i = 0; i < volumetricFogCount; ++i) {
		if (!volumetricFogInfoModified[i] && !firstFrame) continue;
		vkCmdUpdateBuffer(cmd, volumetricFogInfosBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * i, sizeof(shaderio::VolumetricFogInfo), &volumetricFogInfos[i]);

		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[i];
		uint32_t typeFogIndex = fogInfo.volumetricFogTypeIndex;
		if (fogInfo.type == shaderio::VolumetricFogType::Height) {
			heightModified = true;
			vkCmdUpdateBuffer(cmd, volumetricFogHeightInfoBuffer.buffer.buffer, sizeof(shaderio::HeightFogInfo) * typeFogIndex, sizeof(shaderio::HeightFogInfo), &volumetricFogHeightInfos[typeFogIndex]);
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Fluid) {
			fluidModified = true;
			vkCmdUpdateBuffer(cmd, volumetricFogFluidInfoBuffer.buffer.buffer, sizeof(shaderio::FluidFogInfo) * typeFogIndex, sizeof(shaderio::FluidFogInfo), &volumetricFogFluidInfos[typeFogIndex]);
		}
		else if (fogInfo.type == shaderio::VolumetricFogType::Noise) {
			noiseModified = true;
			vkCmdUpdateBuffer(cmd, volumetricFogNoiseInfoBuffer.buffer.buffer, sizeof(shaderio::NoiseFogInfo) * typeFogIndex, sizeof(shaderio::NoiseFogInfo), &volumetricFogNoiseInfos[typeFogIndex]);
		}
	}
	if (volumetricFogCount > 0) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if (heightModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogHeightInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if (fluidModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogFluidInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	if (noiseModified) nvvk::cmdBufferMemoryBarrier(cmd, { volumetricFogNoiseInfoBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });

}

void VolumetricFog::initVolumetricFogFluid(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	for (int i = 0; i < volumetricFogFluidCount; ++i) {
		uint32_t fogIndex = volumetricFogFluidIndexMap[i];
		pushConstant.instanceIndex = fogIndex;
		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
		VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ fogInfo.fogVoxelGridSize.x, fogInfo.fogVoxelGridSize.y, fogInfo.fogVoxelGridSize.z }, VkExtent3D{ 4, 4, 4 });

		VkShaderEXT initShader = computeShader_initVolumetricFogFluid;
		if (volumetricFogInfoModified[fogIndex] || firstFrame) initShader = computeShader_createVolumetricFogFluid;
		vkCmdBindShadersEXT(cmd, 1, &stage, &initShader);

		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	}
}
void VolumetricFog::createGBuffers(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	uint32_t numColorAttachments = (uint32_t)GBuffers_VolumetricFog::eEmissive + 1;
	std::vector<VkRenderingAttachmentInfo> colorAttachments(numColorAttachments);
	for (int i = 0; i <= (uint32_t)GBuffers_VolumetricFog::eEmissive; ++i) {
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

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		//pushConstant.volumetricFogFluidIndex = -1;
		pushConstant.instanceVelocity = shaderio::float3(0.0f);
		if (Application::sceneResource.periodInstanceIndexToInstanceSetIndex.count(i)) {
			uint32_t instanceSetIndex = Application::sceneResource.periodInstanceIndexToInstanceSetIndex[i];
			FzbRenderer::InstanceSet* instanceSet = &Application::sceneResource.periodInstanceSets[instanceSetIndex];

			//先不考虑旋转带来的力
			shaderio::float3 pos0 = shaderio::float3(instanceSet->transform * shaderio::float4(0.0f, 0.0f, 0.0f, 1.0f));
			shaderio::float3 pos1 = shaderio::float3(instanceSet->transform_lastTime * shaderio::float4(0.0f, 0.0f, 0.0f, 1.0f));
			pushConstant.instanceVelocity = (pos1 - pos0) / pushConstant.dt;

			//pushConstant.volumetricFogFluidIndex = 1;	//表示会与流体进行交互，因此几何需要与流体进行判断
		}
		//else {		//静态的先用AABB与流体AABB进行判断，如果不相交，则几何无需与流体进行判断; 动态的每帧需要重新计算AABB，还不如直接FS中进行判断呢
		//	uint32_t instanceSetIndex = Application::sceneResource.staticInstanceIndexToInstanceSetIndex[i];
		//	FzbRenderer::InstanceSet* instanceSet = &Application::sceneResource.staticInstanceSets[instanceSetIndex];
		//
		//	shaderio::AABB instanceAABB = instanceSet->aabb;
		//
		//	for (int j = 0; j < volumetricFogFluidCount; ++j) {
		//		uint32_t fluidFogIndex = volumetricFogFluidIndexMap[j];
		//		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fluidFogIndex];
		//		shaderio::AABB fogAABB = { .minimum = fogInfo.fogStartPos, .maximum = fogInfo.fogStartPos + (shaderio::float3)fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
		//
		//		if (instanceAABB.maximum.x >= fogAABB.minimum.x && instanceAABB.minimum.x <= fogAABB.maximum.x &&
		//			instanceAABB.maximum.y >= fogAABB.minimum.y && instanceAABB.minimum.y <= fogAABB.maximum.y &&
		//			instanceAABB.maximum.z >= fogAABB.minimum.z && instanceAABB.minimum.z <= fogAABB.maximum.z
		//			) {
		//			pushConstant.volumetricFogFluidIndex = 1;
		//			break;
		//		}
		//	}
		//}

		pushConstant.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	for (int i = 0; i <= (uint32_t)GBuffers_VolumetricFog::eEmissive; ++i)
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(i), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void VolumetricFog::createVolumetricFog(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_getVisibleVolumetricFog);
	vkCmdPushConstants2(cmd, &pushInfo);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{ volumetricFogCount, 1, 1 }, VkExtent3D{ 1024, 1, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	//---------------------------------------------------------------------------------------------------------------------------
	/*
	for (int i = 0; i < volumetricFogNoFluidCount; ++i) {
		uint32_t fogIndex = volumetricFogNoFluidIndexMap[i];
		pushConstant.instanceIndex = fogIndex;
		shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
		if (fogInfo.type == shaderio::VolumetricFogType::Noise) continue;
		groupSize = nvvk::getGroupCounts(VkExtent3D{ fogInfo.fogVoxelGridSize.x, fogInfo.fogVoxelGridSize.y, fogInfo.fogVoxelGridSize.z }, VkExtent3D{ 4, 4, 4 });

		vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog);
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
	}
	*/
	//------------------------------------------------------------流体---------------------------------------------------------------
	{
		auto barrierImage = [&](VkImage image) {
			VkImageMemoryBarrier2 b = nvvk::makeImageMemoryBarrier({
				.image = image,
				.oldLayout = VK_IMAGE_LAYOUT_GENERAL,
				.newLayout = VK_IMAGE_LAYOUT_GENERAL,
				.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
				.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
			});
			VkDependencyInfo depInfo{ 
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.imageMemoryBarrierCount = 1, 
				.pImageMemoryBarriers = &b };
			vkCmdPipelineBarrier2(cmd, &depInfo);
		};
		auto barrierBuffer = [&](FzbRenderer::Buffer buffer) {
			VkBufferMemoryBarrier2 b = nvvk::makeBufferMemoryBarrier({
				.buffer = buffer.buffer.buffer,
				.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
				.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT,
				.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT,
				});
			VkDependencyInfo depInfo{
				.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
				.bufferMemoryBarrierCount = 1,
				.pBufferMemoryBarriers = &b };
			vkCmdPipelineBarrier2(cmd, &depInfo);
			};

		auto barrierVolumeAll = [&](int fluidIndex) {
			barrierImage(volumetricFogFluidVoxelVelocityImages[fluidIndex].image.image);
			barrierBuffer(volumetricFogFluidVoxelInfoBuffers[fluidIndex]);
			barrierImage(volumetricFogFluidExtinctionImages[fluidIndex].image.image);
		};

		auto barrierVolumePingPong = [&](int idx) {
			barrierBuffer(volumetricFogFluidVoxelInfoBuffers[idx]);
		};

		for (int i = 0; i < volumetricFogFluidCount; ++i) {
			uint32_t fogIndex = volumetricFogFluidIndexMap[i];
			pushConstant.instanceIndex = fogIndex;
			shaderio::VolumetricFogInfo fogInfo = volumetricFogInfos[fogIndex];
			groupSize = nvvk::getGroupCounts(VkExtent3D{ fogInfo.fogVoxelGridSize.x, fogInfo.fogVoxelGridSize.y, fogInfo.fogVoxelGridSize.z }, VkExtent3D{ 4, 4, 4 });

			// --- Stage A: Advection ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_A);
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

			// --- Stage D: Diffusion (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_D);
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter;
				vkCmdPushConstants2(cmd, &pushInfo);
				vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
				barrierVolumePingPong(i);
			}

			// --- Stage F: External forces ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_F);
			pushConstant.iteration = Jacobi_Iteration_Count & 1;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);

			// --- Stage P: Pressure solve (Jacobi) ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_P);
			for (uint32_t iter = 0; iter < Jacobi_Iteration_Count; ++iter) {
				pushConstant.iteration = iter + (Jacobi_Iteration_Count & 1);
				vkCmdPushConstants2(cmd, &pushInfo);
				vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
				barrierVolumePingPong(i);
			}

			// --- Stage S: Subtract pressure gradient ---
			vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog_Fluid_S);
			pushConstant.iteration = Jacobi_Iteration_Count;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);
			barrierVolumeAll(i);
		}
	}
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createLightAttenuationEstimator);
	for (int i = 0; i < volumetricFogCount; ++i) {
		if (volumetricFogInfos[i].type == shaderio::VolumetricFogType::Height || volumetricFogInfos[i].type == shaderio::VolumetricFogType::Noise) continue;
		pushConstant.instanceIndex = i;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDispatch(cmd, 1, 1, 1);
	}
}
void VolumetricFog::deferredRenderring(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	//nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_deferredRenderring);

	pushConstant.lightVP = shadowMap.pushConstant.lightVP;
	vkCmdPushConstants2(cmd, &pushInfo);

	VkExtent2D groupSize = nvvk::getGroupCounts(gBuffers.getSize(), VkExtent2D{16, 16});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);

	//nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}

#ifndef NDEBUG
void VolumetricFog::renderVolumetricFogVoxelGrid(VkCommandBuffer cmd) {
	bool show = showVolumetricFogVoxelGrid;
	for (int i = 0; i < volumetricFogCount; ++i) show |= showVolumetricFogVoxelGrids[i] == 1;
	if (!show) return;

	NVVK_DBG_SCOPE(cmd);

	uint32_t renderedImage = uint32_t(GBuffers_VolumetricFog::eRendered);
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
	colorAttachment.imageView = gBuffers.getColorImageView(renderedImage);

	VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
	depthAttachment.imageView = gBuffers.getDepthImageView();
	depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = &depthAttachment;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//�����ʹ�����߿�������rasterizationLineState
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.rasterizationState.lineWidth = 2.0f;
	graphicsDynamicPipeline.rasterizationState.polygonMode = VK_POLYGON_MODE_LINE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_renderVoxelGrid, .fragment = fragmentShader_renderVoxelGrid });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	uint32_t wireframeMeshIndex = 0;
	const shaderio::Mesh& mesh = scene.meshes[wireframeMeshIndex];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t bufferIndex = scene.getMeshBufferIndex(wireframeMeshIndex);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];

	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));
	for (int i = 0; i < volumetricFogCount; ++i) {
		if (showVolumetricFogVoxelGrids[i] == 1) {
			pushConstant.instanceIndex = i;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDrawIndexed(cmd, triMesh.indices.count, volumetricFogInfos[i].fogVoxelGridSize.x * volumetricFogInfos[i].fogVoxelGridSize.y * volumetricFogInfos[i].fogVoxelGridSize.z, 0, 0, 0);
		}

	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
#endif