#include <common/Application/Application.h>
#include "./RasterVoxelization.h"
#include "common/utils.hpp"
#include <common/Shader/Shader.h>
#include <nvvk/default_structs.hpp>
#include <bit>
#include <cstdint>

FzbRenderer::RasterVoxelization::RasterVoxelization(pugi::xml_node& featureNode) {
	Application::vkContext->getPhysicalDeviceFeatures_notConst().geometryShader = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().vertexPipelineStoresAndAtomics = VK_TRUE;
	Application::vkContext->getPhysicalDeviceFeatures_notConst().shaderSampledImageArrayDynamicIndexing = VK_TRUE;

	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
	Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures });

	if (pugi::xml_node resolutionNode = featureNode.child("resolution")) {
		glm::vec2 resolution = FzbRenderer::getfloat2FromString(resolutionNode.attribute("value").value());
		setting.resolution = VkExtent2D(resolution.x, resolution.y);
	}
	else setting.resolution = { 4096, 4096 };
		
	if (pugi::xml_node voxelCountNode = featureNode.child("voxelCount"))
		setting.pushConstant.voxelSize_Count.w = std::stoi(voxelCountNode.attribute("value").value());
	else setting.pushConstant.voxelSize_Count.w = 64;

	if (pugi::xml_node debugModeNode = featureNode.child("debugMode")) {
		std::string debugModeStr = debugModeNode.attribute("type").value();
		if (debugModeStr == "Cube") setting.debugMode = DebugMode::Cube;
		else setting.debugMode = DebugMode::Wireframe;
	}
	else setting.debugMode = DebugMode::None;

	if (setting.debugMode == DebugMode::Wireframe) {
		Application::vkContext->getPhysicalDeviceFeatures_notConst().fillModeNonSolid = VK_TRUE;
		Application::vkContext->getPhysicalDeviceFeatures_notConst().wideLines = VK_TRUE;
		setting.lineWidth = std::stof(featureNode.child("debugMode").attribute("value").value());
	}
}

void FzbRenderer::RasterVoxelization::init() {
	std::string shaderioPath = (std::filesystem::path(__FILE__).parent_path() / "shaderio.h").string();
	Application::slangCompiler.addOption({ .name = slang::CompilerOptionName::Include,
		.value = {.kind = slang::CompilerOptionValueKind::String, .stringValue0 = shaderioPath.c_str()}
		});

	createGraphicsDescriptorSetLayout();
	Feature::createGraphicsPipelineLayout(sizeof(shaderio::RasterVoxelizationPushConstant));

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

	glm::vec3 distance = aabb.maximum - aabb.minimum;
	float maxDistance = std::max(distance.x, std::max(distance.y, distance.z));
	glm::vec3 center = (aabb.maximum + aabb.minimum) * 0.5f;
	glm::vec3 minimum = center - maxDistance * 0.5f;
	glm::vec3 maximum = center + maxDistance * 0.5f;

	glm::mat4 VP[3];
	//前面
	glm::vec3 viewPoint = glm::vec3(center.x, center.y, maximum.z + 0.1f);	//世界坐标右手螺旋，即+z朝后
	glm::mat4 viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 0.0f, -1.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	glm::mat4 orthoMatrix = glm::orthoRH_ZO(-0.5f * maxDistance, 0.5f * maxDistance, -0.5f * maxDistance, 0.5f * maxDistance, 0.1f, maxDistance + 0.1f);
	orthoMatrix[1][1] *= -1;
	VP[0] = orthoMatrix * viewMatrix;
	//左边
	viewPoint = glm::vec3(minimum.x - 0.1f, center.y, center.z);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * maxDistance, 0.5f * maxDistance, -0.5f * maxDistance, 0.5f * maxDistance, 0.1f, maxDistance + 0.1f);
	orthoMatrix[1][1] *= -1;
	VP[1] = orthoMatrix * viewMatrix;
	//下面
	viewPoint = glm::vec3(center.x, minimum.y - 0.1f, center.z);
	viewMatrix = glm::lookAt(viewPoint, viewPoint + glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
	orthoMatrix = glm::orthoRH_ZO(-0.5f * maxDistance, 0.5f * maxDistance, -0.5f * maxDistance, 0.5f * maxDistance, 0.1f, maxDistance + 0.1f);
	orthoMatrix[1][1] *= -1;
	VP[2] = orthoMatrix * viewMatrix;

	for(int i = 0; i < 3; ++i) setting.pushConstant.VP[i] = VP[i];
	setting.pushConstant.voxelGroupStartPos = glm::vec4(aabb.minimum, 1.0f);
	setting.pushConstant.voxelSize_Count = glm::vec4(distance / glm::vec3(setting.pushConstant.voxelSize_Count.w), setting.pushConstant.voxelSize_Count.w);

	createGraphicsDescriptorSetLayout();
	Feature::createGraphicsPipelineLayout();
	compileAndCreateShaders();
	//----------------------------------------------创造VGB-----------------------------------------
	nvvk::StagingUploader& stagingUploader = Application::stagingUploader;
	nvvk::ResourceAllocator* allocator = stagingUploader.getResourceAllocator();

	uint32_t voxelTotalCount = std::pow(setting.pushConstant.voxelSize_Count.w, 3);
	uint32_t VGBByteSize = voxelTotalCount * sizeof(shaderio::VGBVoxelData);
	allocator->createBuffer(VGB, VGBByteSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT);
	NVVK_DBG_NAME(VGB.buffer);

	std::vector<shaderio::VGBVoxelData> initVoxelData(voxelTotalCount);
	for (int i = 0; i < voxelTotalCount; ++i) {
		initVoxelData[i].aabbU.minimum = glm::uvec3(std::bit_cast<uint32_t>(FLT_MAX));
		initVoxelData[i].aabbU.maximum = glm::uvec3(std::bit_cast<uint32_t>(-FLT_MAX));
	}
	NVVK_CHECK(stagingUploader.appendBuffer(VGB, 0, std::span<const shaderio::VGBVoxelData>(initVoxelData)));
	//----------------------------------------------更新描述符集合-----------------------------------
	Feature::addTextureArrayDescriptor(shaderio::RasterVoxelizationBindingPoints::eTextures_RV);
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    VGBWrite =
		descPack.makeWrite(shaderio::RasterVoxelizationBindingPoints::eVGB_RV, 0, 0, 1);
	write.append(VGBWrite, VGB, 0, VGBByteSize);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbRenderer::RasterVoxelization::clean() {
	Feature::clean();
	Application::allocator.destroyBuffer(VGB);

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, vertexShader, nullptr);
	vkDestroyShaderEXT(device, geometryShader, nullptr);
	vkDestroyShaderEXT(device, fragmentShader, nullptr);
}
void FzbRenderer::RasterVoxelization::uiRender() {

}
void FzbRenderer::RasterVoxelization::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	setting.pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = graphicPipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
		.offset = 0,
		.size = sizeof(shaderio::RasterVoxelizationPushConstant),
		.pValues = &setting.pushConstant,
	};

	VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
	renderingInfo.renderArea = { {0, 0}, setting.resolution };
	renderingInfo.colorAttachmentCount = 1;
	renderingInfo.pColorAttachments = nullptr;
	renderingInfo.pDepthAttachment = nullptr;

	const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
		.layout = graphicPipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = descPack.getSetPtr(),
	};
	//vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicPipelineLayout, 0, 1,
		descPack.getSetPtr(), 0, nullptr);

	vkCmdBeginRendering(cmd, &renderingInfo);

	//使用VK_EXT_SHADER_OBJECT_EXTENSION_NAME后可以不需要pipeline，直接通过命令设置渲染设置和着色器
	dynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
	dynamicPipeline.depthStencilState.depthTestEnable = VK_FALSE;
	dynamicPipeline.cmdApplyAllStates(cmd);
	dynamicPipeline.cmdSetViewportAndScissor(cmd, setting.resolution);
	dynamicPipeline.cmdBindShaders(cmd, { .vertex = vertexShader, .fragment = fragmentShader, .geometry = geometryShader });

	VkVertexInputBindingDescription2EXT bindingDescription{};
	VkVertexInputAttributeDescription2EXT attributeDescription = {};
	vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	for (size_t i = 0; i < Application::sceneResource.instances.size(); ++i)
	{
		uint32_t meshIndex = Application::sceneResource.instances[i].meshIndex;
		const shaderio::Mesh& mesh = Application::sceneResource.meshes[meshIndex];
		const shaderio::TriangleMesh& triMesh = mesh.triMesh;

		//setting.pushConstant.normalMatrix = glm::transpose(glm::inverse(glm::mat3(Application::sceneResource.instances[i].transform)));
		setting.pushConstant.instanceIndex = int(i);
		vkCmdPushConstants2(cmd, &pushInfo);

		uint32_t bufferIndex = Application::sceneResource.getMeshBufferIndex(meshIndex);
		const nvvk::Buffer& v = Application::sceneResource.bDatas[bufferIndex];

		vkCmdBindIndexBuffer(cmd, v.buffer, triMesh.indices.offset, VkIndexType(mesh.indexType));

		vkCmdDrawIndexed(cmd, triMesh.indices.count, 1, 0, 0, 0);
	}

	vkCmdEndRendering(cmd);
}

void FzbRenderer::RasterVoxelization::createGraphicsDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints::eTextures_RV,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	bindings.addBinding({ .binding = shaderio::RasterVoxelizationBindingPoints::eVGB_RV,
						 .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
						 .descriptorCount = 1,
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	descPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(descPack.getLayout());
	NVVK_DBG_NAME(descPack.getPool());
	NVVK_DBG_NAME(descPack.getSet(0));
}
void FzbRenderer::RasterVoxelization::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	//编译后的数据放在了slangCompiler中
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "rasterVoxelization.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	vkDestroyShaderEXT(Application::app->getDevice(), vertexShader, nullptr);
	vkDestroyShaderEXT(Application::app->getDevice(), geometryShader, nullptr);
	vkDestroyShaderEXT(Application::app->getDevice(), fragmentShader, nullptr);

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
		.offset = 0,
		.size = sizeof(shaderio::RasterVoxelizationPushConstant),
	};

	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = 1,
		.pSetLayouts = descPack.getLayoutPtr(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};

	shaderInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_GEOMETRY_BIT;
	shaderInfo.pName = "vertexMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &vertexShader);
	NVVK_DBG_NAME(vertexShader);

	shaderInfo.stage = VK_SHADER_STAGE_GEOMETRY_BIT;
	shaderInfo.nextStage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.pName = "geometryMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &geometryShader);
	NVVK_DBG_NAME(geometryShader);

	shaderInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.pName = "fragmentMain";
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;
	vkCreateShadersEXT(Application::app->getDevice(), 1U, &shaderInfo, nullptr, &fragmentShader);
	NVVK_DBG_NAME(fragmentShader);
}
void FzbRenderer::RasterVoxelization::updateDataPerFrame(VkCommandBuffer cmd) {

}