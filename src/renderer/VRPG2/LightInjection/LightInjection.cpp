#include <common/Application/Application.h>
#include "../VRPG2.h"
#include "./LightInjection.h"
#include "common/utils.hpp"
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <bit>
#include <cstdint>
#include <cstring>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

LightInjection_VRPG::LightInjection_VRPG(pugi::xml_node& featureNode) {
	if (pugi::xml_node spvNode = featureNode.child("sampleCountPerVoxel"))
		VRPG2Renderer::setting.spv_lightInjection = std::stoi(spvNode.attribute("value").value());
	else VRPG2Renderer::setting.spv_lightInjection = 32;

	if (pugi::xml_node bounceNode = featureNode.child("bounces"))
		VRPG2Renderer::setting.bounces_lightInjection = std::stoi(bounceNode.attribute("value").value());
	else VRPG2Renderer::setting.bounces_lightInjection = 1;
}

void LightInjection_VRPG::init() {
#ifndef NDEBUG
	VkExtent2D resolution = { VRPG2Renderer::setting.rasterVoxelizatioResolution.x, VRPG2Renderer::setting.rasterVoxelizatioResolution.y };
	Feature::createGBuffer(true, false, 1, resolution);
#endif
	//---------------------------------------------------------------------------------------------
	createSource();
	createDescriptorSetLayout();
	createDescriptorSet();
	// set 1 is the push descriptor set holding the TLAS (the injection shader traces rays)
	{
		const VkPushConstantRange pushConstantRange{ .stageFlags = VK_SHADER_STAGE_ALL, .offset = 0, .size = sizeof(shaderio::LightInjectPushConstant_VRPG) };
		std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
		const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = (uint32_t)layouts.size(),
			.pSetLayouts = layouts.data(),
			.pushConstantRangeCount = 1,
			.pPushConstantRanges = &pushConstantRange,
		};
		NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
		NVVK_DBG_NAME(pipelineLayout);
	}
	compileAndCreateShaders();

#ifndef NDEBUG
	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createCube(false, false);
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Cube", primitive);
		scene.addMeshSet(mesh);
		scene.createSceneInfoBuffer();
	}
#endif
}
void LightInjection_VRPG::clean() {
	// Debug builds create a GBuffer here; Feature::clean also destroys the descriptor packs and
	// the pipeline layout created in init().
	Feature::clean();

	Application::allocator.destroyBuffer(hasGeometryNodeInfoBuffer);
	Application::allocator.destroyBuffer(globalInfoBuffer);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_init, nullptr);
	vkDestroyShaderEXT(device, computeShader_dispatchLightInjection, nullptr);
	vkDestroyShaderEXT(device, computeShader_lightInjection, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);
#endif
}
void LightInjection_VRPG::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::CollapsingHeader(std::string("Light Injection ").c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
		PE::begin();
		bool spvChange = PE::SliderInt("SPV", &VRPG2Renderer::setting.spv_lightInjection, 1, LIGHTINJECT_SAMPLE_COUNT_STATIC_SCENE, "%d", ImGuiSliderFlags_AlwaysClamp, "Sample Count Per Voxel");
		bool bounceChange = PE::SliderInt("Bounces", &VRPG2Renderer::setting.bounces_lightInjection, 1, 10, "%d", ImGuiSliderFlags_AlwaysClamp, "Light Injection Bounces");
		PE::end();
		UIModified |= spvChange || bounceChange;
		VRPG2Renderer::setting.lightChange |= spvChange || bounceChange;
#ifndef NDEBUG
		ImGui::Checkbox("Show Light Injection", &showInjectionResult);
#endif
	}
#ifndef NDEBUG
	if (showInjectionResult) Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)shaderio::LightInjectGBuffer_VRPG::eInjectionResult);
#endif
}
void LightInjection_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
#ifndef NDEBUG
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
#endif
};
void LightInjection_VRPG::preRender() {}
void LightInjection_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	VRPG2Renderer::setting.spv_lightInjection = (Application::sceneResource.isStaticScene && !VRPG2Renderer::setting.rebuildPerFrame)
		? LIGHTINJECT_SAMPLE_COUNT_STATIC_SCENE : LIGHTINJECT_SAMPLE_COUNT;

	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.randomFrameIndex = VRPG2Renderer::setting.randomFrameIndex;

	pushConstant.spv = VRPG2Renderer::setting.spv_lightInjection;
	pushConstant.bounces = VRPG2Renderer::setting.bounces_lightInjection;

	int layerDiff = VRPG2Renderer::setting.geometryOctreeDepth - VRPG2Renderer::setting.lightOctreeDepth;
	pushConstant.nodeSize = shaderio::float4(VRPG2Renderer::setting.voxelSize * float(1 << (3 * layerDiff)), 1.0f);
	pushConstant.nodeGridStartPos = shaderio::float4(VRPG2Renderer::setting.voxelGridStartPos, 1.0f);
	pushConstant.nodeCount = (int)std::pow(VRPG2Renderer::setting.voxelGridResolution, 3) / (1 << (3 * layerDiff));

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	// the light tree and the candidate tables have to follow the geometry VGB
	VRPG2Renderer::setting.lightChange |= VRPG2Renderer::setting.geometryChange;

	float angle = FzbRenderer::rand(pushConstant.randomFrameIndex) * glm::two_pi<float>();
	pushConstant.randomRotateMatrix = glm::mat3(glm::rotate(glm::mat4(1.0f), angle, glm::vec3(0, 0, 1)));
}
void LightInjection_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_VRPG),
		.pValues = &pushConstant,
	};

	injection(cmd);
}

void LightInjection_VRPG::createSource() {
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	int layerDiff = VRPG2Renderer::setting.geometryOctreeDepth - VRPG2Renderer::setting.lightOctreeDepth;
	uint32_t voxelTotalCount = (uint32_t)pow(VRPG2Renderer::setting.voxelGridResolution, 3) * 6 / (1 << (3 * layerDiff));
	uint32_t bufferSize = voxelTotalCount * sizeof(shaderio::HasGeometryNodeInfo_VRPG);
	allocator->createBuffer(hasGeometryNodeInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(hasGeometryNodeInfoBuffer.buffer);

	bufferSize = sizeof(shaderio::LightInjectGlobalInfo_VRPG);
	allocator->createBuffer(globalInfoBuffer, bufferSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_2_INDIRECT_BUFFER_BIT);
	NVVK_DBG_NAME(globalInfoBuffer.buffer);
}
void LightInjection_VRPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eGeoClusterData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eHasGeometryNodeInfo_VRPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eLightInjectionGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

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
}
void LightInjection_VRPG::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet    GeoOctreeWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eGeoClusterData, 0, 0, 1);
	nvvk::Buffer geoClusterDataBuffer = VRPG2Renderer::source.clusterDataBuffer_Octree[VRPG2Renderer::setting.lightOctreeDepth];
	write.append(GeoOctreeWrite, geoClusterDataBuffer, 0, geoClusterDataBuffer.bufferSize);

	VkWriteDescriptorSet    hasGeometryVoxelInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eHasGeometryNodeInfo_VRPG, 0, 0, 1);
	write.append(hasGeometryVoxelInfoWrite, hasGeometryNodeInfoBuffer, 0, hasGeometryNodeInfoBuffer.bufferSize);

	VkWriteDescriptorSet    globalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eGlobalInfo, 0, 0, 1);
	write.append(globalInfoWrite, VRPG2Renderer::source.globalInfoBuffer, 0, VRPG2Renderer::source.globalInfoBuffer.bufferSize);

	VkWriteDescriptorSet    lightInjectionGlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_LightInject_VRPG::eLightInjectionGlobalInfo, 0, 0, 1);
	write.append(lightInjectionGlobalInfoWrite, globalInfoBuffer, 0, globalInfoBuffer.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void LightInjection_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	//缂傚倹鐗為惁褔宕ユ惔锝嗙暠闁轰胶澧楀畵渚€寮ㄩ幆褎韬ù婊冩晞langCompiler�?
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_VRPG),
	};

	std::array<VkDescriptorSetLayout, 2> shaderLayouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = (uint32_t)shaderLayouts.size(),
		.pSetLayouts = shaderLayouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	VkDevice device = Application::app->getDevice();
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "injection.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_init, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_init";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_init);
		NVVK_DBG_NAME(computeShader_init);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_dispatchLightInjection, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_dispatchLightInjection";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_dispatchLightInjection);
		NVVK_DBG_NAME(computeShader_dispatchLightInjection);
		//--------------------------------------------------------------------------------------
		vkDestroyShaderEXT(device, computeShader_lightInjection, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_LightInject";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_lightInjection);
		NVVK_DBG_NAME(computeShader_lightInjection);
	}
	//----------------------------------------debug----------------------------------------
#ifndef NDEBUG
	{
		shaderSource = shaderPath / "debug.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader_Cube, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader_Cube);
		NVVK_DBG_NAME(vertexShader_Cube);

		vkDestroyShaderEXT(device, fragmentShader_Cube, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "fragmentMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &fragmentShader_Cube);
		NVVK_DBG_NAME(fragmentShader_Cube);
	}
#endif
}

#ifndef NDEBUG
// Draws one solid cube per injected voxel, colored by the average irradiance of the voxel.
// Ported from VRPG's LightInject::debug_Cube (shaders/lightInject2.slang vertexMain_Cube).
void LightInjection_VRPG::injectionCube(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::LightInjectGBuffer_VRPG::eInjectionResult), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView((uint32_t)shaderio::LightInjectGBuffer_VRPG::eInjectionResult);
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

	debugPushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LightInjectPushConstant_VRPG),
		.pValues = &pushConstant,
	};

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader_Cube, .fragment = fragmentShader_Cube });

	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	vkCmdPushConstants2(cmd, &debugPushInfo);

	const shaderio::Mesh& mesh = scene.meshes[0];
	const shaderio::TriangleMesh& triMesh = mesh.triMesh;

	uint32_t bufferIndex = scene.getMeshBufferIndex(0);
	const nvvk::Buffer& v = scene.bDatas[bufferIndex];
	vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

	// one instance per voxel of every normal partition
	uint32_t instanceCount = uint32_t(pushConstant.nodeCount) * 6;
	vkCmdDrawIndexed(cmd, triMesh.indices.count, instanceCount, 0, 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage((uint32_t)shaderio::LightInjectGBuffer_VRPG::eInjectionResult), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
#endif

void LightInjection_VRPG::postProcess(VkCommandBuffer cmd) {
#ifndef NDEBUG
	if (!showInjectionResult) return;

	injectionCube(cmd);
#endif
}

void LightInjection_VRPG::injection(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);
	{
		nvvk::WriteSetContainer write{};
		write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), VRPG2Renderer::source.asManager->asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());
	}
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdFillBuffer(cmd, globalInfoBuffer.buffer, 0, sizeof(shaderio::LightInjectGlobalInfo_VRPG), 0);
	nvvk::Buffer& lightNodeLeafDataBuffer = VRPG2Renderer::source.lightNodeDataBuffer_Octree[VRPG2Renderer::setting.lightOctreeDepth];
	vkCmdFillBuffer(cmd, lightNodeLeafDataBuffer.buffer, 0, lightNodeLeafDataBuffer.bufferSize, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	int layerDiff = VRPG2Renderer::setting.geometryOctreeDepth - VRPG2Renderer::setting.lightOctreeDepth;
	uint32_t nodeCount = (uint32_t)pow(VRPG2Renderer::setting.voxelGridResolution, 3) * 6 / (1 << (3 * layerDiff));

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_init);
	VkExtent2D groupSize = nvvk::getGroupCounts({ nodeCount , 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_dispatchLightInjection);
	vkCmdDispatch(cmd, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT);

	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_lightInjection);
	vkCmdDispatchIndirect(cmd, VRPG2Renderer::source.globalInfoBuffer.buffer, 0);
}
