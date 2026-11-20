#include <common/Application/Application.h>
#include "../VRPG2.h"
#include "./GeometryInjection.h"
#include "common/utils.hpp"
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <bit>
#include <cstdint>
#include <cstring>
#include <nvgui/property_editor.hpp>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

GeometryInjection_VRPG::GeometryInjection_VRPG(pugi::xml_node& featureNode) {
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
		VRPG2Renderer::setting.rasterVoxelizatioResolution = shaderio::int2(resolution.x, resolution.y);
	}
	else VRPG2Renderer::setting.rasterVoxelizatioResolution = { 4096, 4096 };

	if (pugi::xml_node voxelCountNode = featureNode.child("voxelCount"))
		VRPG2Renderer::setting.voxelGridResolution = std::stoi(voxelCountNode.attribute("value").value());
	else VRPG2Renderer::setting.voxelGridResolution = 16;

#ifndef NDEBUG
	//-------------------------------------DebugSetting-------------------------------------------------
	Application::vkContext->getPhysicalDeviceFeatures11_notConst().multiview = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().multiViewport = VK_TRUE;

	Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
#endif
}

void GeometryInjection_VRPG::init() {
	// keep the resolution configured in rendererInfo.xml (VRPG does the same for static scenes)
#ifndef NDEBUG
	VkExtent2D resolution = { VRPG2Renderer::setting.rasterVoxelizatioResolution.x, VRPG2Renderer::setting.rasterVoxelizatioResolution.y };
	Feature::createGBuffer(false, false, 1, resolution);
#endif
	//---------------------------------------------------------------------------------------------
	createVGBs();
	createDescriptorSetLayout();
	createDescriptorSet();
	Feature::createPipelineLayout(sizeof(shaderio::GeometryInjectionPushConstant_VRPG));
	compileAndCreateShaders();
}
void GeometryInjection_VRPG::clean() {
	Feature::clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_initVGB, nullptr);
	vkDestroyShaderEXT(device, vertexShader, nullptr);
	vkDestroyShaderEXT(device, geometryShader, nullptr);
	vkDestroyShaderEXT(device, fragmentShader, nullptr);
#ifndef NDEBUG
	vkDestroyShaderEXT(device, geometryShader_ThreeView, nullptr);
	vkDestroyShaderEXT(device, fragmentShader_ThreeView, nullptr);
#endif
}
void GeometryInjection_VRPG::uiRender() {
#ifndef NDEBUG
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	if (ImGui::CollapsingHeader(std::string("Geometry Injection ").c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::TextDisabled("Geometry VGB: %d^3", int(VRPG2Renderer::setting.voxelGridResolution));

		ImGui::Checkbox("Show Three View Rsult", &showThreeView);
		if(showThreeView) Application::viewportImage = gBuffers.getDescriptorSet(0);
	}
#endif
}
void GeometryInjection_VRPG::resize(VkCommandBuffer cmd, const VkExtent2D& size) {};
void GeometryInjection_VRPG::preRender() {}
void GeometryInjection_VRPG::updateDataPerFrame(VkCommandBuffer cmd) {
	pushConstant.frameIndex = VRPG2Renderer::setting.frameIndex;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;

	// frameIndex is reset by resetFrame(), so a camera / UI reset also rebuilds the VGB
	VRPG2Renderer::setting.geometryChange = !Application::sceneResource.isStaticScene || VRPG2Renderer::setting.rebuildPerFrame || VRPG2Renderer::setting.frameIndex == 0 || showThreeView;
}
void GeometryInjection_VRPG::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd, "GeometryInjection_render");

	updateDataPerFrame(cmd);

	pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::GeometryInjectionPushConstant_VRPG),
		.pValues = &pushConstant,
	};
	
	initVGB(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT);

	if (showThreeView) injection_ThreeView(cmd);
	else injection(cmd);
}

void GeometryInjection_VRPG::createVGBs() {
	{
		shaderio::AABB aabb;
		aabb.minimum = { FLT_MAX, FLT_MAX, FLT_MAX };
		aabb.maximum = -aabb.minimum;
		FzbRenderer::Scene& sceneResource = Application::sceneResource;
		for (int i = 0; i < sceneResource.instances.size(); ++i) {
			uint32_t meshIndex = sceneResource.instances[i].meshIndex;
			MeshInfo meshInfo = sceneResource.getMeshInfo(meshIndex);
			shaderio::AABB meshAABB = meshInfo.getAABB(sceneResource.instances[i].transform);

			aabb.minimum.x = std::min(meshAABB.minimum.x, aabb.minimum.x);
			aabb.minimum.y = std::min(meshAABB.minimum.y, aabb.minimum.y);
			aabb.minimum.z = std::min(meshAABB.minimum.z, aabb.minimum.z);
			aabb.maximum.x = std::max(meshAABB.maximum.x, aabb.maximum.x);
			aabb.maximum.y = std::max(meshAABB.maximum.y, aabb.maximum.y);
			aabb.maximum.z = std::max(meshAABB.maximum.z, aabb.maximum.z);
		}

		glm::vec3 distance = (aabb.maximum - aabb.minimum) * 1.1f;
		//float maxDistance = std::max(distance.x, std::max(distance.y, distance.z));
		glm::vec3 center = (aabb.maximum + aabb.minimum) * 0.5f;
		glm::vec3 minimum = center - distance * 0.5f;
		glm::vec3 maximum = center + distance * 0.5f;

		glm::mat4 VP[3];
		glm::vec3 viewPoint = glm::vec3(center.x, center.y, maximum.z + 0.1f);
		glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.z + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[0] = orthoMatrix * viewMatrix;

		viewPoint = glm::vec3(minimum.x - 0.1f, center.y, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.z, 0.5f * distance.z, -0.5f * distance.y, 0.5f * distance.y, 0.1f, distance.x + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[1] = orthoMatrix * viewMatrix;

		viewPoint = glm::vec3(center.x, minimum.y - 0.1f, center.z);
		viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		orthoMatrix = glm::orthoRH_ZO(-0.5f * distance.x, 0.5f * distance.x, -0.5f * distance.z, 0.5f * distance.z, 0.1f, distance.y + 0.1f);
		orthoMatrix[1][1] *= -1;
		VP[2] = orthoMatrix * viewMatrix;

		for (int i = 0; i < 3; ++i) pushConstant.VP[i] = VP[i];

		VRPG2Renderer::setting.voxelGridStartPos = minimum;
		VRPG2Renderer::setting.voxelSize = distance / float(VRPG2Renderer::setting.voxelGridResolution);

		pushConstant.voxelGroupStartPos = VRPG2Renderer::setting.voxelGridStartPos;
		pushConstant.voxelSize_Count = glm::vec4(VRPG2Renderer::setting.voxelSize, VRPG2Renderer::setting.voxelGridResolution);
	}

	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t voxelTotalCount = std::pow(VRPG2Renderer::setting.voxelGridResolution, 3);
	uint32_t VGBByteSize = voxelTotalCount * sizeof(shaderio::VoxelData_VRPG);

	VRPG2Renderer::source.VGBs.resize(6);
	for (int i = 0; i < 6; ++i) {
		allocator->createBuffer(VRPG2Renderer::source.VGBs[i], VGBByteSize,
			VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
		NVVK_DBG_NAME(VRPG2Renderer::source.VGBs[i].buffer);
	}
}
void GeometryInjection_VRPG::createDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = (uint32_t)shaderio::GeometryInjectionBinding_VRPG::eTextures,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({ .binding = (uint32_t)shaderio::GeometryInjectionBinding_VRPG::eVGB,
						 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						 .descriptorCount = (uint32_t)VRPG2Renderer::source.VGBs.size(),
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(staticDescPack.getLayout());
	NVVK_DBG_NAME(staticDescPack.getPool());
	NVVK_DBG_NAME(staticDescPack.getSet(0));
}
void GeometryInjection_VRPG::createDescriptorSet() {
	Feature::addTextureArrayDescriptor((uint32_t)shaderio::GeometryInjectionBinding_VRPG::eTextures);

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::GeometryInjectionBinding_VRPG::eVGB, 0, 0, VRPG2Renderer::source.VGBs.size());
	nvvk::Buffer* VGBsPtr = VRPG2Renderer::source.VGBs.data();
	write.append(VGBWrite, VGBsPtr);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void GeometryInjection_VRPG::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	//缂傚倹鐗為惁褔宕ユ惔锝嗙暠闁轰胶澧楀畵渚€寮ㄩ幆褎韬ù婊冩晞langCompiler�?
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource;
	VkShaderModuleCreateInfo shaderCode;

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::GeometryInjectionPushConstant_VRPG),
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
	{
		shaderSource = shaderPath / "init.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, computeShader_initVGB, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_clearVGB";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_initVGB);
		NVVK_DBG_NAME(computeShader_initVGB);
	}
	//--------------------------------------------------------------------------------------
	{
		shaderSource = shaderPath / "injection.slang";
		shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

		vkDestroyShaderEXT(device, vertexShader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		shaderInfo.nextStage = VK_SHADER_STAGE_GEOMETRY_BIT;
		shaderInfo.pName = "vertexMain";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &vertexShader);
		NVVK_DBG_NAME(vertexShader);

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
#endif
	}

}

void GeometryInjection_VRPG::initVGB(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_initVGB);

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdPushConstants2(cmd, &pushInfo);

	uint32_t totalVoxelCount = pow(VRPG2Renderer::setting.voxelGridResolution, 3) * 6;
	VkExtent2D groupSize = nvvk::getGroupCounts({ totalVoxelCount, 1 }, VkExtent2D{ 1024, 1 });
	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}
void GeometryInjection_VRPG::injection(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;

	VkExtent2D resolution = { VRPG2Renderer::setting.rasterVoxelizatioResolution.x, VRPG2Renderer::setting.rasterVoxelizatioResolution.y };
	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
	graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, resolution);
	graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader, .geometry = geometryShader });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);
}

void GeometryInjection_VRPG::injection_ThreeView(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });

	VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
	colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
	colorAttachment.imageView = gBuffers.getColorImageView(0);
	colorAttachment.clearValue = { .color = {Application::sceneResource.sceneInfo.backgroundColor.x,
											Application::sceneResource.sceneInfo.backgroundColor.y,
											Application::sceneResource.sceneInfo.backgroundColor.z, 1.0f} };

	VkExtent2D resolution = { VRPG2Renderer::setting.rasterVoxelizatioResolution.x, VRPG2Renderer::setting.rasterVoxelizatioResolution.y };
	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = &colorAttachment;
	renderingInfo.pDepthAttachment = nullptr;

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
	graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	graphicsDynamicPipeline.cmdApplyAllStates(cmd);
	//vkCmdSetConservativeRasterizationModeEXT(cmd, VK_CONSERVATIVE_RASTERIZATION_MODE_OVERESTIMATE_EXT);
	//vkCmdSetExtraPrimitiveOverestimationSizeEXT(cmd, 0.75f);
	vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);

	float width = float(resolution.width) * 0.5f;
	float height = float(resolution.height) * 0.5f;
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

		pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(0), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
}