#include "./VRPG.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <cstdlib>

using namespace FzbRenderer;

VRPGRenderer::VRPGRenderer(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		pushConstant.maxDepth = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node sppNode = rendererNode.child("spp"))
		pushConstant.spp = std::stoi(sppNode.attribute("value").value());
	if (pugi::xml_node rasterVoxelizationNode = rendererNode.child("RasterVoxelization"))
		rasterVoxelization = std::make_shared<RasterVoxelization_VRPG>(rasterVoxelizationNode);
	if (pugi::xml_node lightVGBNode = rendererNode.child("LightVGB")) {
		lightRasterVoxelization = std::make_shared<RasterVoxelization_VRPG>(lightVGBNode);
	}
	else {
		pugi::xml_node fallbackNode = rendererNode.child("RasterVoxelization");
		lightRasterVoxelization = std::make_shared<RasterVoxelization_VRPG>(fallbackNode);
	}
	if (pugi::xml_node lightInjectNode = rendererNode.child("LightInject"))
		lightInject = std::make_shared<LightInject_VRPG>(lightInjectNode);
	if (pugi::xml_node octreeNode = rendererNode.child("Octree"))
		octree = std::make_shared<Octree_VRPG>(octreeNode);

	shadowMap = std::make_shared<ShadowMap>();
#ifndef NDEBUG
	Flip = std::make_shared<FLIP>();
#endif
}
void VRPGRenderer::init() {
	renderStaticScene = Application::sceneResource.isStaticScene;
	if (renderStaticScene) shadowMap->init({ 1024, 1024 });

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	// Keep static-scene rebuilds cheap: use the configured raster resolution instead
	// of forcing 4096x4096 for both geometry and light VGBs.
	rasterVoxelization->forceStaticHighResolution = false;
	lightRasterVoxelization->forceStaticHighResolution = false;
	rasterVoxelization->init();
	lightRasterVoxelization->init();

	LightInjectCreateInfo_VRPG lightInjectCreateInfo{};
	lightInjectCreateInfo.VGBs = lightRasterVoxelization->VGBs;
	lightInjectCreateInfo.VGBStartPos = lightRasterVoxelization->setting.pushConstant.voxelGroupStartPos;
	lightInjectCreateInfo.VGBVoxelSize = glm::vec3(lightRasterVoxelization->setting.pushConstant.voxelSize_Count);
	lightInjectCreateInfo.VGBSize = lightRasterVoxelization->setting.pushConstant.voxelSize_Count.w;
	lightInjectCreateInfo.geometryVGBs = rasterVoxelization->VGBs;
	lightInjectCreateInfo.geometryVGBStartPos = rasterVoxelization->setting.pushConstant.voxelGroupStartPos;
	lightInjectCreateInfo.geometryVGBVoxelSize = glm::vec3(rasterVoxelization->setting.pushConstant.voxelSize_Count);
	lightInjectCreateInfo.geometryVGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w;
	lightInjectCreateInfo.ptContext = &ptContext;
	lightInjectCreateInfo.asManager = &asManager;
	lightInject->init(lightInjectCreateInfo);

	OctreeCreateInfo_VRPG octreeCreateInfo{};
	octreeCreateInfo.VGBs = rasterVoxelization->VGBs;
	octreeCreateInfo.VGBStartPos = rasterVoxelization->setting.pushConstant.voxelGroupStartPos;
	octreeCreateInfo.VGBVoxelSize = glm::vec3(rasterVoxelization->setting.pushConstant.voxelSize_Count);
	octreeCreateInfo.VGBSize = rasterVoxelization->setting.pushConstant.voxelSize_Count.w;
	octreeCreateInfo.lightVGBs = lightRasterVoxelization->VGBs;
	octreeCreateInfo.lightVGBStartPos = lightRasterVoxelization->setting.pushConstant.voxelGroupStartPos;
	octreeCreateInfo.lightVGBVoxelSize = glm::vec3(lightRasterVoxelization->setting.pushConstant.voxelSize_Count);
	octreeCreateInfo.lightVGBSize = lightRasterVoxelization->setting.pushConstant.voxelSize_Count.w;
	octreeCreateInfo.asManager = &asManager;
	octree->init(octreeCreateInfo);

	IF_DEBUG(Feature::createGBuffer(true, true,(uint32_t)ImageType_VRPG::eImgTonemapped,Application::renderResolution),
		Feature::createGBuffer(false, true, 1, Application::renderResolution));

	// Keep a tiny placeholder alive so the static descriptor write is always valid.
	// The full W*H path-state buffer is only allocated when wavefront mode is enabled.
	ensureWavefrontPathStateBuffer({ 1u, 1u });

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

#ifndef NDEBUG
	showDebugImages.clear();
	showDebugImages.insert({ (uint32_t)ImageType_VRPG::eImgPGVariance, false });

	{
		std::filesystem::path refImagePath = "C:/Users/fangzanbo/Desktop/FzbRenderer_nvvk/result/bearBox.png";
		refImage.init(refImagePath);

		FLIPCreateInfo flipCreateInfo{
			.referenceImage = refImage.image,
			.testImage = gBuffers.m_res.gBufferColor[((uint32_t)ImageType_VRPG::eImgTonemapped)],
		};
		Flip->init(flipCreateInfo);
		createFlipReadbackBuffer();
	}
#endif

	Renderer::init();
}
void VRPGRenderer::clean() {
	if (renderStaticScene) shadowMap->clean();

	rasterVoxelization->clean();
	lightRasterVoxelization->clean();
	lightInject->clean();
	octree->clean();

	VkDevice device = Application::app->getDevice();
	vkDestroyShaderEXT(device, computeShader_VRPG, nullptr);

	// Experimental wavefront prototype resources.
	vkDestroyShaderEXT(device, computeShader_WavefrontInit, nullptr);
	vkDestroyShaderEXT(device, computeShader_WavefrontBounce, nullptr);
	vkDestroyShaderEXT(device, computeShader_WavefrontResolve, nullptr);
	if (wavefrontPathStateBuffer.buffer != VK_NULL_HANDLE)
		Application::allocator.destroyBuffer(wavefrontPathStateBuffer);

#ifndef NDEBUG
	Flip->clean();
	refImage.clean();
	Application::allocator.destroyBuffer(flipReadbackBuffer);
#endif

	PathTracingRenderer::clean();
};
void VRPGRenderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_VRPG::eImgTonemapped);

	//static int maxFrames_temp = (MAX_FRAME) / 2;
	if (ImGui::Begin("VRPGSettings"))
	{
		ImGui::SeparatorText("VRPG");
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		//maxFrames = maxFrames_temp + 1;
		PE::end();
		ImGui::TextDisabled("Frame: %d", pushConstant.frameIndex);

		// Compile-time VRPG components.
		ImGui::TextDisabled("Spatial Reuse: %d  Temporal Reuse: %d", VRPG_SPATIAL_REUSE, VRPG_TEMPORAL_REUSE);
		ImGui::TextDisabled("Jitter: %d  Nearby: %d",
			VRPG_JITTER_SAMPLING, VRPG_NEARBY_NODE_COUNT);
		UIModified |= ImGui::SliderFloat("Jitter Radius", &jitterRadiusScale, 0.0f, 10.0f, "%.2f");
		ImGui::TextDisabled("Path Sampling Mode: %d", VRPG_PATH_SAMPLE_MODE);
		UIModified |= ImGui::Checkbox("Rotate Outgoing", &octree->rotateOutgoing);
		UIModified |= ImGui::Checkbox("Rebuild Per Frame", &rebuildPerFrame);
		ImGui::TextDisabled("M=%d  K=%d  S=%d  (compile-time)",
			Candidate_Samples_Count, OUTGOING_COUNT_VRPG, HITTEST_COUNT_VRPG);
		ImGui::TextDisabled("Geometry VGB: %d^3   Light VGB: %d^3",
			int(rasterVoxelization->setting.pushConstant.voxelSize_Count.w),
			int(lightRasterVoxelization->setting.pushConstant.voxelSize_Count.w));

		ImGui::SeparatorText("Bounces");{
			PE::begin();
			PE::SliderInt("Bounces Depth", &pushConstant.maxDepth, 1, std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}
		ImGui::SeparatorText("SPP");{
			PE::begin();
			UIModified |= PE::SliderInt("SPP", &pushConstant.spp, 1, 64, "%d", ImGuiSliderFlags_AlwaysClamp,
				"Sample Per Pixel");
			PE::end();
		}

		/*
		ImGui::SeparatorText("Wavefront (Experimental)");
		ImGui::TextDisabled("Compile-time: VRPG_USE_WAVEFRONT = %d", VRPG_USE_WAVEFRONT);
		ImGui::TextDisabled("Bounced guide + BSDF double MIS path state prototype");

		if (ptContext.rtPosFetchFeature.rayTracingPositionFetch == VK_FALSE){
			ImGui::TextColored({ 1, 0, 0, 1 }, "ERROR: Position Fetch not supported!");
			ImGui::Text("This hardware does not support");
			ImGui::Text("VK_KHR_ray_tracing_position_fetch");
			ImGui::Text("Please use RTX 20 series or newer GPU.");
		}else{
			ImGui::TextColored({ 0, 1, 0, 1 }, "Position Fetch: SUPPORTED");
			ImGui::Separator();
		}
		*/

#ifndef NDEBUG
		if (PE::begin()) {
			uint32_t debugImageIndex = (uint32_t)ImageType_VRPG::eImgPGVariance;
			if (PE::entry("PathGuiding Variance", [&] {
				static const ImVec4 highlightColor = ImVec4(118.f / 255.f, 185.f / 255.f, 0.f, 1.f);
				ImVec4 selectedColor = showDebugImages[debugImageIndex] ? highlightColor : ImGui::GetStyleColorVec4(ImGuiCol_Button);
				ImVec4 hoveredColor = ImVec4(selectedColor.x * 1.2f, selectedColor.y * 1.2f, selectedColor.z * 1.2f, 1.f);
				ImGui::PushStyleColor(ImGuiCol_Button, selectedColor);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, hoveredColor);
				ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(5, 5));

				bool result = ImGui::ImageButton("##but", (ImTextureID)gBuffers.getDescriptorSet((uint32_t)ImageType_VRPG::eImgPGVariance),
					ImVec2(100 * gBuffers.getAspectRatio(), 100));

				ImGui::PopStyleColor(2);
				ImGui::PopStyleVar();
				return result;
				}))
			{
				showDebugImages[debugImageIndex] = !showDebugImages[debugImageIndex];
				for (auto& pair : showDebugImages) {
					if (pair.first == debugImageIndex) continue;
					showDebugImages[pair.first] = false;
				}
			}
		}
		PE::end();
#endif
	}
	ImGui::End();

	if (renderStaticScene) shadowMap->uiRender();
	rasterVoxelization->uiRender();
	lightInject->uiRender();
	octree->uiRender();

#ifndef NDEBUG
	Flip->uiRender();
	if (UIModified) {
		lastFLIP = 0.0f;
		flipSum = 0.0;
		flipFrameCount = 0;
		flipReadbackPending = false;
	}
#endif

	if (UIModified) resetFrame();

#ifndef NDEBUG
	for (auto& pair : showDebugImages) {
		if (showDebugImages[pair.first]) 
			Application::viewportImage = gBuffers.getDescriptorSet(pair.first);
	}
#endif
};
void VRPGRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	NVVK_CHECK(gBuffers.update(cmd, size));

	// Resize the wavefront path-state buffer with the render target. When the prototype
	// is disabled only the 1-element placeholder is kept.
	ensureWavefrontPathStateBuffer(wavefront ? size : VkExtent2D{ 1u, 1u });

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    OutImageWrite =
		staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(OutImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VRPG::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

#ifndef NDEBUG
	//濠电姷鏁搁崑鐐哄箰閹间礁绠犻柟鐗堟緲缁犳牕鈹戦悩韫礈閹兼番鍔岀涵鈧梺缁樺姇濡﹪宕搹鍦＝濞达絿鎳撻崝銈嗙節閳ь剟鏌嗗鍛暫?
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

	VkWriteDescriptorSet    depthImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eDepthImage, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    pgVarianceImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::ePGValueImage, 0, 0, 1);
	write.append(pgVarianceImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VRPG::eImgPGValue), VK_IMAGE_LAYOUT_GENERAL);

	pgVarianceImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::ePGValue2Image, 0, 0, 1);
	write.append(pgVarianceImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VRPG::eImgPGValue2), VK_IMAGE_LAYOUT_GENERAL);

	pgVarianceImageWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::ePGVarianceImage, 0, 0, 1);
	write.append(pgVarianceImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VRPG::eImgPGVariance), VK_IMAGE_LAYOUT_GENERAL);
#endif

	// The wavefront prototype keeps its per-pixel path state in the same static set.
	{
		VkWriteDescriptorSet wavefrontPathStateWrite =
			staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPGWavefront::ePathState, 0, 0, 1);
		write.append(wavefrontPathStateWrite, wavefrontPathStateBuffer, 0, wavefrontPathStateBuffer.bufferSize);
	}

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);

	if (renderStaticScene) shadowMap->resize(cmd, size);

	IF_DEBUG(rasterVoxelization->resize(cmd, size, gBuffers, (uint32_t)ImageType_VRPG::eImgTonemapped), rasterVoxelization->resize(cmd, size));
	IF_DEBUG(lightRasterVoxelization->resize(cmd, size, gBuffers, (uint32_t)ImageType_VRPG::eImgTonemapped), lightRasterVoxelization->resize(cmd, size));
	lightInject->resize(cmd, size);
	octree->resize(cmd, size);
	
#ifndef NDEBUG
	if (Flip) {
		Flip->resize(cmd, size, Flip->setting.referenceImage, gBuffers.m_res.gBufferColor[((uint32_t)ImageType_VRPG::eImgTonemapped)]);
	}
	createFlipReadbackBuffer();
#endif
};
void VRPGRenderer::preRender() {
	readFLIP();

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	Scene& scene = Application::sceneResource;
	bool cameraChange = scene.cameraChange;
	if (Application::UIModified || cameraChange || rebuildPerFrame ||
		!renderStaticScene || frameIndex == 0 ||
		octree->spatialReuse || octree->temporalReuse) {
		octreeDirty = true;
	}
	if (cameraChange) {
		resetFrame();
		octree->temporalReset = true;
	}	//婵犵數濮烽。浠嬪焵椤掆偓閸熷潡鍩€椤掆偓缂嶅﹪骞冨Ο璇茬窞闁归偊鍘奸崜顕€姊洪崨濠勨槈闁挎洏鍎抽懞鍗烆潩閼哥數鍘卞┑鐐叉缁绘劙鍩€椤掆偓缂嶅﹤顕ｉ幎鑺ュ亜闁绘挸娴烽悡鎴濃攽閳藉棗鐏犻柟纰卞亰椤㈡俺顦归柡灞诲妼閳藉螣閸噮浼冮梻浣稿閻撳牓宕规潏鈺冪焿闁圭儤鎸婚崕鐔兼煏韫囥儳纾块柛妯诲姉缁辨挻绗熼埀顒傛兜閹间礁鍨傞柣銏犲閺佸洭鏌涘┑鍕姕闁?
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight) maxFrames = 1;
	pushConstant.VGBStartPos_Size = shaderio::float4(rasterVoxelization->setting.pushConstant.voxelGroupStartPos, rasterVoxelization->setting.pushConstant.voxelSize_Count.w);
	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.randomFrameIndex = (VRPG_RANDOM_FRAME_SOURCE != 0)
		? int(Application::randomFrameIndex)
		: Application::frameIndex;
	pushConstant.staticScene = renderStaticScene ? 1 : 0;
	pushConstant.maxFrameCount = maxFrames;
	pushConstant.time = Application::sceneResource.time;
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)Application::sceneResource.bSceneInfo.address;
	pushConstant.maxOctreeLayer = octree->octreeMaxLayer;
	pushConstant.VGBVoxelSize = shaderio::float3(rasterVoxelization->setting.pushConstant.voxelSize_Count);
	pushConstant.lightVGBStartPos_Size = shaderio::float4(
		lightRasterVoxelization->setting.pushConstant.voxelGroupStartPos,
		lightRasterVoxelization->setting.pushConstant.voxelSize_Count.w);
	pushConstant.lightVGBVoxelSize = shaderio::float4(
		glm::vec3(lightRasterVoxelization->setting.pushConstant.voxelSize_Count), 1.0f);
	pushConstant.lightMaxOctreeLayer = octree->lightOctreeMaxLayer;
	asManager.updateToplevelAS(cmd);

	shadowMap->preRender();
	rasterVoxelization->preRender(cmd);
	lightRasterVoxelization->preRender(cmd);
	lightInject->perFrameRebuild = rebuildPerFrame || !renderStaticScene;
	lightInject->preRender();
	octree->preRender();
	pushConstant.spatialRuse = (octree->spatialReuse || octree->temporalReuse) ? 1 : 0;

	pushConstant.ditherSample = VRPG_JITTER_SAMPLING;
	pushConstant.jitterRadiusScale = jitterRadiusScale;
	pushConstant.nearbyNodeCount = VRPG_NEARBY_NODE_COUNT;
	pushConstant.pathSampleMode = VRPG_PATH_SAMPLE_MODE;

	pushConstant.randomRotateMatrix = octree->pushConstant.randomRotateMatrix;

	Application::app->submitAndWaitTempCmdBuffer(cmd);

#ifndef NDEBUG
	Flip->preRender();
	if (cameraChange) {
		lastFLIP = 0.0f;
		flipSum = 0.0;
		flipFrameCount = 0;
		flipReadbackPending = false;
	}
#endif
}
void VRPGRenderer::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	if (pushConstant.frameIndex >= maxFrames && maxFrames > 2) return;

	const bool rebuildGeometryVGB = rebuildPerFrame || !renderStaticScene || frameIndex == 0;
	const bool rebuildLightVGB = rebuildGeometryVGB;

	if (rebuildGeometryVGB) {
		rasterVoxelization->render(cmd);
		// Geometry VGB is written by the fragment shader; make it visible to the
		// cluster/light-injection compute shaders.
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	if (rebuildLightVGB) {
		// Rebuild the light VGB by clustering the existing geometry VGB.
		// No second scene rasterization is needed.
		lightRasterVoxelization->clearVGB(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		lightInject->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		octreeDirty = true;
	}
	++frameIndex;

	if (octreeDirty) {
		octree->renderStatic(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		octreeDirty = false;
	}

	// Leaf selection, hit tests and weights contain per-frame random sampling,
	// so they must run even when the structural tree is cached.
	octree->renderCandidates(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	
	if (wavefront) wavefrontPathTracing(cmd);
	else pathGuiding(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo((uint32_t)ImageType_VRPG::eImgRendered), gBuffers.getDescriptorImageInfo((uint32_t)ImageType_VRPG::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
		VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	//getFLIP(cmd);
	
	//if(renderStaticScene) shadowMap->postProcess(cmd);
	//rasterVoxelization->postProcess(cmd);
	lightInject->postProcess(cmd);
	octree->postProcess(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);

	//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
};

void VRPGRenderer::createDescriptorSetLayout() {
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
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeInfo_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octree->octreeNodeInfoBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeData_E,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octree->octreeNodeDataBuffer_E.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSubTreeData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateLeafNodeData_VRPG,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleData,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSamplePrefixWSum,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeData_G,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = (uint32_t)octree->octreeNodeDataBuffer_G.size(),
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eNearbyNodeInfos,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eReservoir,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

#ifdef SpatialReuse_VRPG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleData2,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleAABB,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
#endif


	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eGlobalInfo,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	// Experimental wavefront path tracer: one path-state entry per pixel.
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPGWavefront::ePathState,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

#ifndef NDEBUG
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::eDepthImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });

	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::ePGValueImage,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::ePGValue2Image,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
		.descriptorCount = 1,
		.stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({
		.binding = (uint32_t)shaderio::StaticBindingPoints_VRPG::ePGVarianceImage,
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
void VRPGRenderer::createDescriptorSet() {
	nvvk::WriteSetContainer write{};
	if (!Application::sceneResource.textures.empty()) {
		VkWriteDescriptorSet    allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0, uint32_t(Application::sceneResource.textures.size()));
		nvvk::Image* allImages = Application::sceneResource.textures.data();
		write.append(allTextures, allImages);
	}

	VkWriteDescriptorSet	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeInfo_G, 0, 0, octree->octreeNodeInfoBuffer_G.size());
	nvvk::Buffer* octreeArraysPtr = octree->octreeNodeInfoBuffer_G.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeData_E, 0, 0, octree->octreeNodeDataBuffer_E.size());
	octreeArraysPtr = octree->octreeNodeDataBuffer_E.data();
	write.append(OctreeArrayWrite, octreeArraysPtr);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSubTreeData, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateSubTreeInfoBuffer, 0, octree->candidateSubTreeInfoBuffer.bufferSize);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateLeafNodeData_VRPG, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateLeafNodeDataBuffer, 0, octree->candidateLeafNodeDataBuffer.bufferSize);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleData, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateSampleDataBuffer, 0, octree->candidateSampleDataBuffer.bufferSize);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSamplePrefixWSum, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateSamplePrefixWSumBuffer, 0, octree->candidateSamplePrefixWSumBuffer.bufferSize);

#ifdef SpatialReuse_VRPG
	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleData2, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateSampleData2Buffer, 0, octree->candidateSampleData2Buffer.bufferSize);

	OctreeArrayWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eCandidateSampleAABB, 0, 0, 1);
	write.append(OctreeArrayWrite, octree->candidateSampleAABBBuffer, 0, octree->candidateSampleAABBBuffer.bufferSize);
#endif

	VkWriteDescriptorSet    OctreeNodeDataGWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eOctreeNodeData_G, 0, 0, (uint32_t)octree->octreeNodeDataBuffer_G.size());
	octreeArraysPtr = octree->octreeNodeDataBuffer_G.data();
	write.append(OctreeNodeDataGWrite, octreeArraysPtr);

	VkWriteDescriptorSet    NearbyDataWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eNearbyNodeInfos, 0, 0, 1);
	write.append(NearbyDataWrite, octree->nearbyNodeInfoBuffer, 0, octree->nearbyNodeInfoBuffer.bufferSize);

	VkWriteDescriptorSet reservoirWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eReservoir, 0, 0, 1);
	write.append(reservoirWrite, octree->reservoirCurrentBuffer, 0, octree->reservoirCurrentBuffer.bufferSize);

	

	VkWriteDescriptorSet    GlobalInfoWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPG::eGlobalInfo, 0, 0, 1);
	write.append(GlobalInfoWrite, octree->globalInfoBuffer, 0, octree->globalInfoBuffer.bufferSize);

	// Placeholder (1x1) until the experimental wavefront mode is enabled.
	VkWriteDescriptorSet wavefrontPathStateWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPGWavefront::ePathState, 0, 0, 1);
	write.append(wavefrontPathStateWrite, wavefrontPathStateBuffer, 0, wavefrontPathStateBuffer.bufferSize);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void VRPGRenderer::createPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VRPGPushConstant)
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
void VRPGRenderer::compileAndCreateShaders() {
	SCOPED_TIMER(__FUNCTION__);

	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "VRPG.slang";
	
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL ,
		.offset = 0,
		.size = sizeof(shaderio::VRPGPushConstant),
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
		vkDestroyShaderEXT(device, computeShader_VRPG, nullptr);
	
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = "computeMain_VRPG";
		shaderInfo.codeSize = shaderCode.codeSize;
		shaderInfo.pCode = shaderCode.pCode;
		vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_VRPG);
		NVVK_DBG_NAME(computeShader_VRPG);
	}
	//--------------------------------------------------------------------------------------
	// Experimental wavefront path tracer: one module, three compute entry points.
	// Failure here must not affect the stable path, so it is reported and skipped.
	{
		std::filesystem::path wavefrontSource = shaderPath / "VRPGWavefront.slang";
		VkShaderModuleCreateInfo wavefrontCode = FzbRenderer::compileSlangShader(wavefrontSource, {});
		if (wavefrontCode.codeSize > 0 && wavefrontCode.pCode != nullptr)
			createWavefrontShaders(wavefrontCode);
		else
			LOGE("Wavefront prototype shader failed to compile; the toggle will fall back to the stable VRPG path.\n");
	}
};
void VRPGRenderer::updateDataPerFrame(VkCommandBuffer cmd) {}

void VRPGRenderer::pathGuiding(VkCommandBuffer cmd) {
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
	vkCmdBindShadersEXT(cmd, 1, &stage, &computeShader_VRPG);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VRPGPushConstant),
		.pValues = &pushConstant,
	};

	VkExtent2D sceneSize = Application::renderResolution;// Application::app->getViewportSize();
	VkExtent2D groupSize = nvvk::getGroupCounts(sceneSize, VkExtent2D{ FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y });

	pushConstant.sceneSize = shaderio::uint2(sceneSize.width, sceneSize.height);
	pushConstant.threadGroupCount = shaderio::uint2(groupSize.width, groupSize.height);

	// Long paths can exceed the Windows GPU watchdog when a single dispatch
	// covers the whole image. Split the image into horizontal bands so each
	// dispatch is shorter; the pixels are disjoint, so no barrier is needed.
	const uint32_t bandCount = pushConstant.maxDepth >= 8 ? 4u : 1u;
	const uint32_t bandGroupHeight = (groupSize.height + bandCount - 1u) / bandCount;
	for (uint32_t band = 0; band < bandCount; ++band) {
		const uint32_t groupY = band * bandGroupHeight;
		if (groupY >= groupSize.height) break;
		const uint32_t dispatchGroupHeight = std::min(bandGroupHeight, groupSize.height - groupY);

		pushConstant.dispatchOffset = shaderio::uint2(0, groupY * FZB_PATHGUIDING_THREADGROUP_SIZE_Y);
		vkCmdPushConstants2(cmd, &pushInfo);
		vkCmdDispatch(cmd, groupSize.width, dispatchGroupHeight, 1);
	}
}

//------------------------------------------------------------------------------------------
// Experimental wavefront path tracer
//------------------------------------------------------------------------------------------
bool VRPGRenderer::wavefrontShadersValid() const {
	return computeShader_WavefrontInit != VK_NULL_HANDLE &&
		computeShader_WavefrontBounce != VK_NULL_HANDLE &&
		computeShader_WavefrontResolve != VK_NULL_HANDLE;
}

void VRPGRenderer::ensureWavefrontPathStateBuffer(const VkExtent2D& extent) {
	const uint32_t width = std::max(extent.width, 1u);
	const uint32_t height = std::max(extent.height, 1u);
	const VkDeviceSize requiredSize =
		VkDeviceSize(width) * VkDeviceSize(height) * sizeof(shaderio::WavefrontPathState);

	if (wavefrontPathStateBuffer.buffer != VK_NULL_HANDLE && wavefrontPathStateBufferSize == requiredSize)
		return;

	if (wavefrontPathStateBuffer.buffer != VK_NULL_HANDLE)
		Application::allocator.destroyBuffer(wavefrontPathStateBuffer);

	NVVK_CHECK(Application::allocator.createBuffer(
		wavefrontPathStateBuffer, requiredSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT));
	NVVK_DBG_NAME(wavefrontPathStateBuffer.buffer);

	wavefrontPathStateBufferSize = requiredSize;
	wavefrontPathStateExtent = { width, height };
	LOGI("Wavefront path state buffer: %ux%u (%.2f MB)\n", width, height,
		double(requiredSize) / (1024.0 * 1024.0));
}

void VRPGRenderer::writeWavefrontPathStateDescriptor() {
	if (wavefrontPathStateBuffer.buffer == VK_NULL_HANDLE) return;

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet pathStateWrite =
		staticDescPack.makeWrite((uint32_t)shaderio::StaticBindingPoints_VRPGWavefront::ePathState, 0, 0, 1);
	write.append(pathStateWrite, wavefrontPathStateBuffer, 0, wavefrontPathStateBuffer.bufferSize);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void VRPGRenderer::createWavefrontShaders(const VkShaderModuleCreateInfo& shaderCode) {
	VkDevice device = Application::app->getDevice();

	vkDestroyShaderEXT(device, computeShader_WavefrontInit, nullptr);
	vkDestroyShaderEXT(device, computeShader_WavefrontBounce, nullptr);
	vkDestroyShaderEXT(device, computeShader_WavefrontResolve, nullptr);

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VRPGPushConstant),
	};
	std::array<VkDescriptorSetLayout, 2> layouts = { {staticDescPack.getLayout(), dynamicDescPack.getLayout()} };

	// Same layouts/push constants as the stable VRPG shader object, but the module
	// exposes three compute entry points and we select them by name.
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "wavefrontInitMain",
		.setLayoutCount = layouts.size(),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	shaderInfo.nextStage = 0;
	shaderInfo.codeSize = shaderCode.codeSize;
	shaderInfo.pCode = shaderCode.pCode;

	VkResult result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_WavefrontInit);
	if (result != VK_SUCCESS) {
		LOGE("Failed to create wavefront init shader (VkResult %d).\n", int(result));
		return;
	}

	shaderInfo.pName = "wavefrontBounceMain";
	result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_WavefrontBounce);
	if (result != VK_SUCCESS) {
		LOGE("Failed to create wavefront bounce shader (VkResult %d).\n", int(result));
		return;
	}

	shaderInfo.pName = "wavefrontResolveMain";
	result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_WavefrontResolve);
	if (result != VK_SUCCESS)
		LOGE("Failed to create wavefront resolve shader (VkResult %d).\n", int(result));

	NVVK_DBG_NAME(computeShader_WavefrontInit);
	NVVK_DBG_NAME(computeShader_WavefrontBounce);
	NVVK_DBG_NAME(computeShader_WavefrontResolve);
}

void VRPGRenderer::wavefrontPathTracing(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	const VkExtent2D sceneSize = Application::renderResolution;
	const VkDeviceSize requiredSize =
		VkDeviceSize(sceneSize.width) * VkDeviceSize(sceneSize.height) * sizeof(shaderio::WavefrontPathState);

	// Never dispatch the prototype against an undersized buffer or missing shaders;
	// fall back to the stable path so the renderer stays usable.
	if (sceneSize.width == 0 || sceneSize.height == 0 || !wavefrontShadersValid() ||
		wavefrontPathStateBuffer.buffer == VK_NULL_HANDLE || wavefrontPathStateBufferSize < requiredSize) {
		LOGW("Wavefront prototype not ready (state buffer %llu/%llu bytes, shaders %d); falling back to stable VRPG.\n",
			(unsigned long long)wavefrontPathStateBufferSize, (unsigned long long)requiredSize,
			wavefrontShadersValid() ? 1 : 0);
		pathGuiding(cmd);
		return;
	}

	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
		staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	const VkExtent2D groupSize = nvvk::getGroupCounts(
		sceneSize, VkExtent2D{ FZB_WAVEFRONT_THREADGROUP_SIZE_X, FZB_WAVEFRONT_THREADGROUP_SIZE_Y });

	pushConstant.sceneSize = shaderio::uint2(sceneSize.width, sceneSize.height);
	pushConstant.threadGroupCount = shaderio::uint2(groupSize.width, groupSize.height);
	pushConstant.dispatchOffset = shaderio::uint2(0, 0);

	VkPushConstantsInfo pushInfo = {
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VRPGPushConstant),
		.pValues = &pushConstant,
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	auto dispatchPass = [&](VkShaderEXT shader) {
		vkCmdBindShadersEXT(cmd, 1, &stage, &shader);
		vkCmdDispatch(cmd, groupSize.width, groupSize.height, 1);
		// Path states are read by the next pass, so order the passes with a barrier.
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	};

	dispatchPass(computeShader_WavefrontInit);

	const int bounceCount = std::max(pushConstant.maxDepth, 1);
	for (int bounce = 0; bounce < bounceCount; ++bounce)
		dispatchPass(computeShader_WavefrontBounce);

	dispatchPass(computeShader_WavefrontResolve);
}

void VRPGRenderer::createFlipReadbackBuffer() {
	if (flipReadbackBuffer.buffer != VK_NULL_HANDLE)
		Application::allocator.destroyBuffer(flipReadbackBuffer);

	flipReadbackExtent = Application::renderResolution;

	VkDeviceSize size = VkDeviceSize(flipReadbackExtent.width) *
		flipReadbackExtent.height * sizeof(float) * 4;

	NVVK_CHECK(Application::allocator.createBuffer(
		flipReadbackBuffer, size,
		VK_BUFFER_USAGE_2_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
		VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT));

	NVVK_DBG_NAME(flipReadbackBuffer.buffer);
	flipReadbackPending = false;
}
void VRPGRenderer::getFLIP(VkCommandBuffer cmd) {
#ifndef NDEBUG
	if (!Flip || !Flip->hasValidInputs())
		return;

	// 1. 缂?FLIP error map
	Flip->evaluate(cmd);

	// 2. 缂傚倸鍊搁崐鐑芥倿閿曞倸绠板┑鐘崇閸婅泛顭块懜闈涘闁绘劕锕弻锝夊箛闂堟稑顫╁銈呯箰閻栧ジ寮?host visible buffer
	VkImage result = Flip->getResultImage().image;
	VkExtent2D extent = { Flip->getResultImage().extent.width, Flip->getResultImage().extent.height };

	nvvk::cmdImageMemoryBarrier(cmd, { result,
		VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL });

	VkBufferImageCopy copy{};
	copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
	copy.imageExtent = { extent.width, extent.height, 1 };

	vkCmdCopyImageToBuffer(cmd, result,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
		flipReadbackBuffer.buffer, 1, &copy);

	nvvk::cmdImageMemoryBarrier(cmd, { result,
		VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });

	flipReadbackPending = true;
#endif
}
void VRPGRenderer::readFLIP() {
#ifndef NDEBUG
	if (!flipReadbackPending || flipReadbackBuffer.mapping == nullptr)
		return;

	// 缂傚倸鍊烽懗鑸垫叏椤撱垹纾婚柟鍓х帛閻撴洘淇婇妶鍕妽缂佲偓婢舵劖鐓欓柣褍鎽滈幃濂告煙閹绘帞鎽犵紒妤冨枑缁绘繈宕熼褎啸缂傚倸鍊烽悞锔剧矙閹次诲洭顢欓崜褏鐣堕梺绋挎湰缁嬫捇鍩㈤弮鍫熺厓鐟滄粓宕滃☉銏犳瀬?GPU 闂備浇顕уù鐑藉箠閹捐瀚夋い鎺戝閸ㄥ倹鎱ㄥΟ鍨厫闁哄拋鍓氱换娑㈠箣閻愭彃姣堥梺璇叉唉鐏忔瑧妲愰幒妤佸€锋い鎺嗗亾妞ゆ洘绮庣槐鎺撴媴鐟欏嫬鍞夐梺鍝勮嫰閻胶绮诲☉銏犵閻庡湱濯崬鐟扳攽閻愬樊鍤熷┑顔诲嵆瀹曟洟鎳犻浣规闂佸湱铏庨崰鏍儗?
	vkQueueWaitIdle(Application::app->getQueue(0).queue);

	const float* data = reinterpret_cast<const float*>(flipReadbackBuffer.mapping);
	const size_t count = size_t(flipReadbackExtent.width) * flipReadbackExtent.height;

	double sum = 0.0;
	for (size_t i = 0; i < count; ++i)
		sum += data[i * 4 + 3];   // A channel 濠电姵顔栭崰妤勬懌闂佺儵鍓濆ú婊堝箲閵忕姭鏋庨柟瀛樻煥娴滅偓绻涢幋鐑嗕紗闁瑰瓨绺块埀顒佸浮瀹曟粍鎷呴搹鐧哥床?FLIP 闂傚倷鑳堕…鍫ユ晝閿曞倸违閻庯綆鍓氶～鏇熺節闂堟稒宸濋柍缁樻⒒閳ь剙绠嶉崕閬嶅箠鎼达絿绀婇柍?debugView 闂佽崵鍠愮划宥咁熆濡櫣鏆︽い鎺戝缁?

	lastFLIP = float(sum / count);
	flipSum += lastFLIP;
	++flipFrameCount;
	flipReadbackPending = false;

	LOGI("FLIP frame %d: %.6f\n", flipFrameCount, lastFLIP);

	static int maxSumCount = 300; 
	auto getAverageFLIP = [&]() {
		return flipFrameCount > 0 ? float(flipSum / flipFrameCount) : 0.0f;
	};
	if(flipFrameCount == maxSumCount)
		LOGI("FLIP average over %d frames: %.6f\n", flipFrameCount, getAverageFLIP());
#endif
}