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

	if (pugi::xml_node volumetricFogCountNode = rendererNode.child("volumetricFogCount_local"))
		localVolumetricFogCount = std::stoi(volumetricFogCountNode.attribute("value").value());
	localVolumetricFogCount = 2;

	localVolumetricFogImages.resize(localVolumetricFogCount);
	localVolumetricFogInfos.resize(localVolumetricFogCount);
	localVolumetricFogInfoModified.resize(localVolumetricFogCount);
#ifndef NDEBUG
	showLocalVolumetricFogVoxelGrids.resize(localVolumetricFogCount);
#endif
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
}
void VolumetricFog::clean() {
	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_createGBuffer, nullptr);
	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);
	vkDestroyShaderEXT(device, computeShader_createLightAttenuationEstimator, nullptr);
	vkDestroyShaderEXT(device, computeShader_deferredRenderring, nullptr);

	GlobalInfoBuffer.clean();

	volumetricFogImage.clean();
	lightAttenuationEstimatorBuffer.clean();
	for (int i = 0; i < localVolumetricFogCount; ++i) localVolumetricFogImages[i].clean();
	localVolumetricFogInfosBuffer.clean();

	adjacentLocalVolumetricFogIndexBuffer.clean();

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

	if (ImGui::Begin("Volumetric Fog Setting")) {
		if (ImGui::CollapsingHeader("Global Volumetric Fog Properties", ImGuiTreeNodeFlags_DefaultOpen)) {
			UIModified |= ImGui::DragFloat3("Volumetric Fog Start Pos", (float*)&pushConstant.fogStartPos);

			ImGui::BeginDisabled(true);
			bool change = ImGui::DragInt3("Volumetric Fog Voxel Grid Size", (int*)&pushConstant.fogVoxelGridSize);
			ImGui::EndDisabled();
			if (change) {
				//createVolumetricFogData();
				UIModified = true;
			}

			UIModified |= ImGui::DragFloat3("Volumetric Fog Voxel Size", (float*)&pushConstant.fogVoxelSize);

			UIModified |= ImGui::DragFloat2("Extinction Coefficient", (float*)&pushConstant.absorption, 0.1f, 0.0f);
			UIModified |= ImGui::DragFloat("Scatter Coefficient", (float*)&pushConstant.scattering, 0.1f, 0.0f, 1.0f);
			UIModified |= ImGui::DragFloat("Asymmetric Parameters", (float*)&pushConstant.phase, 0.1f, -1.0f, 1.0f);
			UIModified |= ImGui::DragFloat("Light Attenuation Strength", (float*)&pushConstant.lightAttenuationStrength, 0.1f, 0.0f, 1.0f);

			UIModified |= ImGui::Checkbox("show voxel grid", (bool*)&showVolumetricFogVoxelGrid);
		}
		for (int i = 0; i < localVolumetricFogCount; ++i) {
			localVolumetricFogInfoModified[i] = false;
			if (ImGui::CollapsingHeader(std::string("local Volumetric Fog Properties " + std::to_string(i)).c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
				localVolumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Start Pos " + std::to_string(i)).c_str(), (float*)&localVolumetricFogInfos[i].fogStartPos);

				ImGui::BeginDisabled(true);
				bool change = ImGui::DragInt3(std::string("Volumetric Fog Voxel Grid Size " + std::to_string(i)).c_str(), (int*)&localVolumetricFogInfos[i].fogVoxelGridSize);
				ImGui::EndDisabled();
				if (change) {
					//createVolumetricFogData();
					localVolumetricFogInfoModified[i] = true;
				}

				localVolumetricFogInfoModified[i] |= ImGui::DragFloat3(std::string("Volumetric Fog Voxel Size " + std::to_string(i)).c_str(), (float*)&localVolumetricFogInfos[i].fogVoxelSize);

				localVolumetricFogInfoModified[i] |= ImGui::DragFloat2(std::string("Extinction Coefficient " + std::to_string(i)).c_str(), (float*)&localVolumetricFogInfos[i].absorption, 0.1f, 0.0f);
				localVolumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Scatter Coefficient " + std::to_string(i)).c_str(), (float*)&localVolumetricFogInfos[i].scattering, 0.1f, 0.0f, 1.0f);
				localVolumetricFogInfoModified[i] |= ImGui::DragFloat(std::string("Asymmetric Parameters " + std::to_string(i)).c_str(), (float*)&localVolumetricFogInfos[i].phase, 0.1f, -1.0f, 1.0f);

				localVolumetricFogInfoModified[i] |= ImGui::Checkbox(std::string("show voxel grid " + std::to_string(i)).c_str(), (bool*)&showLocalVolumetricFogVoxelGrids[i]);
			}
			UIModified |= localVolumetricFogInfoModified[i];
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

		VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

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

		Application::app->submitAndWaitTempCmdBuffer(cmd);
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

	//{
	//	VkViewportSwizzleNV identitySwizzle = {
	//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_X_NV,
	//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Y_NV,
	//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_Z_NV,
	//		VK_VIEWPORT_COORDINATE_SWIZZLE_POSITIVE_W_NV
	//	};
	//	vkCmdSetViewportSwizzleNV(cmd, 0, 1, &identitySwizzle);
	//}

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

	image.init(colorImageCreateInfo);
}
void VolumetricFog::createVolumetricFogData() {
	GlobalInfoBuffer = FzbRenderer::Buffer("GlobalInfoBuffer", false);
	GlobalInfoBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::GlobalInfo_VolumetricFog),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	createVolumetricFogImage(volumetricFogImage, pushConstant.fogVoxelGridSize);
	lightAttenuationEstimatorBuffer = FzbRenderer::Buffer("lightAttenuationEstimatorBuffer", false);
	lightAttenuationEstimatorBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::float3),
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});

	for (int i = 0; i < localVolumetricFogCount; ++i) {
		shaderio::VolumetricFogInfo info;
		info.fogVoxelGridSize = { 32, 32, 32 };
		info.fogStartPos = { 5084.0f, -77.0f, -4486.0f };
		info.fogVoxelSize = { 0.2, 0.2, 0.2 };
		info.absorption = { 0.0, 0.3 };
		info.scattering = 0.3;
		info.phase = -0.5f;
		if (i == 0) {
			info.fogStartPos = { 5111.0f, -71.0f, -4453.0f };
		}

		createVolumetricFogImage(localVolumetricFogImages[i], info.fogVoxelGridSize);
		
		localVolumetricFogInfos[i] = info;
	}
	localVolumetricFogInfosBuffer = FzbRenderer::Buffer("localVolumetricFogInfosBuffer", false);
	localVolumetricFogInfosBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(shaderio::VolumetricFogInfo) * localVolumetricFogCount,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
		});		//需要在shader中修改lightAttenuationEstimator，所以是SSBO;	追求性能可以拆开

	adjacentLocalVolumetricFogIndexBuffer = FzbRenderer::Buffer("adjacentLocalVolumetricFogIndexBuffer", false);
	adjacentLocalVolumetricFogIndexBuffer.init({
		.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
		.size = sizeof(uint32_t) * 10,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
	});
}
void VolumetricFog::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
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

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLightAttenuationEstimatorBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogInfosBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = localVolumetricFogCount,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAdjacentLocalVolumetricFogIndexBuffer,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogImage_sampler,
		.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
		.descriptorCount = localVolumetricFogCount,
		.stageFlags = VK_SHADER_STAGE_ALL });


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

	VkWriteDescriptorSet globalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer, 0, 0, 1);
	write.append(globalInfoWrite, GlobalInfoBuffer.buffer);

	VkWriteDescriptorSet	volumetricFogWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage, 0, 0, 1);
	write.append(volumetricFogWrite, volumetricFogImage.image);

	volumetricFogWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLightAttenuationEstimatorBuffer, 0, 0, 1);
	write.append(volumetricFogWrite, lightAttenuationEstimatorBuffer.buffer);

	volumetricFogWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogInfosBuffer, 0, 0, 1);
	write.append(volumetricFogWrite, localVolumetricFogInfosBuffer.buffer);

	volumetricFogWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogImage, 0, 0, localVolumetricFogCount);
	std::vector<nvvk::Image> localVolumetricFogImage_nvvk(localVolumetricFogCount);
	for (int i = 0; i < localVolumetricFogCount; ++i) localVolumetricFogImage_nvvk[i] = localVolumetricFogImages[i].image;
	nvvk::Image* localVolumetricFogImagePtr = localVolumetricFogImage_nvvk.data();
	write.append(volumetricFogWrite, localVolumetricFogImagePtr);

	VkWriteDescriptorSet volumetricFogSamplerWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler, 0, 0, 1);
	write.append(volumetricFogSamplerWrite, volumetricFogImage.image);

	volumetricFogSamplerWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eAdjacentLocalVolumetricFogIndexBuffer, 0, 0, 1);
	write.append(volumetricFogSamplerWrite, adjacentLocalVolumetricFogIndexBuffer.buffer);

	volumetricFogSamplerWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eLocalVolumetricFogImage, 0, 0, localVolumetricFogCount);
	write.append(volumetricFogSamplerWrite, localVolumetricFogImagePtr);

	VkWriteDescriptorSet	shadowMapWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eShadowMap, 0, 0, 1);
	write.append(shadowMapWrite, shadowMap.shadowMaps[0].image);

	VkWriteDescriptorSet	renderedImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VolumetricFog::eRenderedImage, 0, 0, 1);
	write.append(gBuffersWrite, gBuffers.getDescriptorImageInfo((uint32_t)GBuffers_VolumetricFog::eRendered));

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

	vkDestroyShaderEXT(device, computeShader_createVolumetricFog, nullptr);

	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "computeMain_createVolumetricFog";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_createVolumetricFog);
	NVVK_DBG_NAME(computeShader_createVolumetricFog);

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
	nvvk::cmdBufferMemoryBarrier(cmd, { localVolumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT });
	for (int i = 0; i < localVolumetricFogCount; ++i) {
		if(localVolumetricFogInfoModified[i])
			vkCmdUpdateBuffer(cmd, localVolumetricFogInfosBuffer.buffer.buffer, sizeof(shaderio::VolumetricFogInfo) * i, sizeof(shaderio::VolumetricFogInfo), &localVolumetricFogInfos[i]);
	}
	nvvk::cmdBufferMemoryBarrier(cmd, { localVolumetricFogInfosBuffer.buffer.buffer, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT });
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
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createVolumetricFog);
	VkExtent3D groupSize = nvvk::getGroupCounts(VkExtent3D{pushConstant.fogVoxelGridSize.x, pushConstant.fogVoxelGridSize.y, pushConstant.fogVoxelGridSize.z}, VkExtent3D{4, 4, 4});
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, groupSize.depth);

	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_createLightAttenuationEstimator);
	vkCmdDispatch(cmd, 1, 1, 1);
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
	for (int i = 0; i < localVolumetricFogCount; ++i) show |= showLocalVolumetricFogVoxelGrids[i] == 1;
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
	graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;		//如果想使用虚线可以设置rasterizationLineState
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
	if (showVolumetricFogVoxelGrid) {
		pushConstant.showVoxelGridIndex = -1;
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDrawIndexed(cmd, triMesh.indices.count, pushConstant.fogVoxelGridSize.x * pushConstant.fogVoxelGridSize.y * pushConstant.fogVoxelGridSize.z, 0, 0, 0);
	}
		
	for (int i = 0; i < localVolumetricFogCount; ++i) {
		if (showLocalVolumetricFogVoxelGrids[i] == 1) {
			pushConstant.showVoxelGridIndex = i;
			vkCmdPushConstants2(cmd, &pushInfo);
			vkCmdDrawIndexed(cmd, triMesh.indices.count, localVolumetricFogInfos[i].fogVoxelGridSize.x * localVolumetricFogInfos[i].fogVoxelGridSize.y * localVolumetricFogInfos[i].fogVoxelGridSize.z, 0, 0, 0);
		}
			
	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(renderedImage), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
#endif