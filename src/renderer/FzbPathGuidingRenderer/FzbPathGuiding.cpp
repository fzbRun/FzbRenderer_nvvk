#include "./FzbPathGuiding.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>

using namespace FzbRenderer;

FzbPathGuidingRenderer::FzbPathGuidingRenderer(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		pushConstant.maxDepth = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node sppNode = rendererNode.child("spp"))
		pushConstant.spp = std::stoi(sppNode.attribute("value").value());
	if (pugi::xml_node rasterVoxelizationNode = rendererNode.child("RasterVoxelization"))
		rasterVoxelization = std::make_shared<RasterVoxelization_FzbPG>(rasterVoxelizationNode);
	if (pugi::xml_node lightInjectNode = rendererNode.child("LightInject"))
		lightInject = std::make_shared<LightInject_FzbPG>(lightInjectNode);
	if (pugi::xml_node octreeNode = rendererNode.child("Octree"))
		octree = std::make_shared<Octree_FzbPG>(octreeNode);

	shadowMap = std::make_shared<ShadowMap>();
}
void FzbPathGuidingRenderer::init() {
	renderStaticScene = Application::sceneResource.isStaticScene;
	if (renderStaticScene) shadowMap->init({ 1024, 1024 });

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	rasterVoxelization->init();
	
	LightInjectCreateInfo_FzbPG lightInjectCreateInfo{
		.VGBs = rasterVoxelization->VGBs,
		.VGBStartPos = rasterVoxelization->setting.pushConstant.voxelGroupStartPos,
		.VGBVoxelSize = glm::vec3(rasterVoxelization->setting.pushConstant.voxelSize_Count),
		.VGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w,
		.ptContext = &ptContext,
		.asManager = &asManager
	};
	lightInject->init(lightInjectCreateInfo);
	
	OctreeCreateInfo_FzbPG octreeCreateInfo{
		.VGBs = rasterVoxelization->VGBs,
		.VGBStartPos = rasterVoxelization->setting.pushConstant.voxelGroupStartPos,
		.VGBVoxelSize = glm::vec3(rasterVoxelization->setting.pushConstant.voxelSize_Count),
		.VGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w,
		.asManager = &asManager,
	};
	octree->init(octreeCreateInfo);

	IF_DEBUG(Feature::createGBuffer(true, true, 1), Feature::createGBuffer(false, true, 1));
	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void FzbPathGuidingRenderer::clean() {
	if (renderStaticScene) shadowMap->clean();

	rasterVoxelization->clean();
	lightInject->clean();
	octree->clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_FzbPathGuiding, nullptr);

	PathTracingRenderer::clean();
};
void FzbPathGuidingRenderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);

	if (ImGui::Begin("FzbPathGuidingSettings"))
	{
		ImGui::SeparatorText("Jitter");
		//UIModified |= ImGui::SliderInt("Max Frames", &maxFrames, 1, MAX_FRAME);
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		PE::end();
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		ImGui::SeparatorText("Bounces");
		{
			PE::begin();
			PE::SliderInt("Bounces Depth", &pushConstant.maxDepth, 1, std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}

		if (ptContext.rtPosFetchFeature.rayTracingPositionFetch == VK_FALSE)
		{
			ImGui::TextColored({ 1, 0, 0, 1 }, "ERROR: Position Fetch not supported!");
			ImGui::Text("This hardware does not support");
			ImGui::Text("VK_KHR_ray_tracing_position_fetch");
			ImGui::Text("Please use RTX 20 series or newer GPU.");
		}
		else
		{
			ImGui::TextColored({ 0, 1, 0, 1 }, "Position Fetch: SUPPORTED");
			ImGui::Separator();
		}
	}
	ImGui::End();

	if (renderStaticScene) shadowMap->uiRender();
	rasterVoxelization->uiRender();
	lightInject->uiRender();
	octree->uiRender();

	if (UIModified) resetFrame();
};
void FzbPathGuidingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

#ifndef NDEBUG
	VkWriteDescriptorSet    depthImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eDepthImage, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	if (renderStaticScene) shadowMap->resize(cmd, size);

	IF_DEBUG(rasterVoxelization->resize(cmd, size, gBuffers, eImgTonemapped), rasterVoxelization->resize(cmd, size));
	lightInject->resize(cmd, size);
	IF_DEBUG(octree->resize(cmd, size, gBuffers, eImgTonemapped), octree->resize(cmd, size));
};
void FzbPathGuidingRenderer::preRender() {
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	Scene& scene = Application::sceneResource;
	if (scene.cameraChange) resetFrame();	//如果相机参数变化，则从新累计帧
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;
	pushConstant.VGBStartPos_Size = shaderio::float4(rasterVoxelization->setting.pushConstant.voxelGroupStartPos, rasterVoxelization->setting.pushConstant.voxelSize_Count.w);
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.maxFrameCount = maxFrames;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.maxOctreeLayer = octree->octreeMaxLayer;
	pushConstant.VGBVoxelSize = shaderio::float3(rasterVoxelization->setting.pushConstant.voxelSize_Count);
	asManager.updateToplevelAS(cmd);

	shadowMap->preRender(cmd);
	rasterVoxelization->preRender(cmd);
	lightInject->preRender();
	octree->preRender();
	octree->pushConstant.maxFrameCount = maxFrames;

	pushConstant.randomRotateMatrix = octree->pushConstant.randomRotateMatrix;

	Application::app->submitAndWaitTempCmdBuffer(cmd);
}
void FzbPathGuidingRenderer::render(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	if (pushConstant.frameIndex >= maxFrames && maxFrames > 1) return;

	if (!renderStaticScene || frameIndex == 0) {
		rasterVoxelization->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		lightInject->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	++frameIndex;

	octree->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	pathGuiding(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Renderer::postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	if(renderStaticScene) shadowMap->postProcess(cmd);
	rasterVoxelization->postProcess(cmd);
	lightInject->postProcess(cmd);
	octree->postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
};

void FzbPathGuidingRenderer::createDescriptorSetLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
					 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					 .descriptorCount = 10,
					 .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
			.binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octree->octreeDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eClusterLayerData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeNodePairData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeNodePairWeight,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#ifdef NEARBYNODE_JITTER_FZBPG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeClusterData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octree->octreeClusterDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eNearbyNodeInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif
#ifndef NDEBUG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_FzbPG::eDepthImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif

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
void FzbPathGuidingRenderer::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeData_G, 0, 0, octree->octreeDataBuffer_G.size());
	nvvk::Buffer* octreeArraysPtr = octree->octreeDataBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eClusterLayerData_E, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->clusterLayerDataBuffer_E, 0, octree->clusterLayerDataBuffer_E.bufferSize);

	VkWriteDescriptorSet NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeNodePairWeight, 0, 0, 1);
	write.append(NodePairInfoWrite, octree->octreeNodePairWeightBuffer, 0, octree->octreeNodePairWeightBuffer.bufferSize);
#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
	NodePairInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeNodePairData, 0, 0, 1);
	write.append(NodePairInfoWrite, octree->octreeNodePairDataBuffer, 0, octree->octreeNodePairDataBuffer.bufferSize);
#endif

	VkWriteDescriptorSet    GlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eGlobalInfo, 0, 0, 1);
	write.append(GlobalInfoWrite, octree->globalInfoBuffer, 0, octree->globalInfoBuffer.bufferSize);

#ifdef NEARBYNODE_JITTER_FZBPG
	VkWriteDescriptorSet    OctreeClusterDataWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eOctreeClusterData_G, 0, 0, octree->octreeClusterDataBuffer_G.size());
	nvvk::Buffer* octreeClusterDataPtr = octree->octreeClusterDataBuffer_G.data();
	write.append(OctreeClusterDataWrite, octreeClusterDataPtr);

	VkWriteDescriptorSet    NearbyDataWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_FzbPG::eNearbyNodeInfos, 0, 0, 1);
	write.append(NearbyDataWrite, octree->nearbyNodeInfoBuffer, 0, octree->nearbyNodeInfoBuffer.bufferSize);
#endif

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void FzbPathGuidingRenderer::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::FzbPathGuidingPushConstant)
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
void FzbPathGuidingRenderer::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "FzbPathGuiding.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::FzbPathGuidingPushConstant),
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
		vkDestroyShaderEXT(device, computeShader_FzbPathGuiding, nullptr);

		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_FzbPathGuiding";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_FzbPathGuiding);
		NVVK_DBG_NAME(computeShader_FzbPathGuiding);
	}
};
void FzbPathGuidingRenderer::updateDataPerFrame(VkCommandBuffer cmd) {}

void FzbPathGuidingRenderer::pathGuiding(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	#ifndef NDEBUG
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
	#endif

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_FzbPathGuiding);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::FzbPathGuidingPushConstant),
		.pValues = &pushConstant,
	};

	VkExtent2D sceneSize = Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y });

	pushConstant.sceneSize = shaderio::uint2(sceneSize.width, sceneSize.height);
	pushConstant.threadGroupCount = shaderio::uint2(groupSize.width, groupSize.height);
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
}