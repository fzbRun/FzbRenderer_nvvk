#include "./VRPG2.h"
#include <common/Application/Application.h>
#include <nvgui/sky.hpp>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvvk/default_structs.hpp>
#include <cstdlib>
#include <cstdio>

using namespace FzbRenderer;

VRPGSetting VRPG2Renderer::setting;
VRPGSource VRPG2Renderer::source;

VRPG2Renderer::VRPG2Renderer(pugi::xml_node& rendererNode) {
	ptContext.setContextInfo();

	if (pugi::xml_node maxDepthNode = rendererNode.child("maxDepth"))
		setting.maxBounces = std::stoi(maxDepthNode.attribute("value").value());
	if (pugi::xml_node sppNode = rendererNode.child("spp"))
		setting.spp = std::stoi(sppNode.attribute("value").value());

	// module settings live in the <VRPG> block, keep direct children working as well
	pugi::xml_node vrpgNode = rendererNode.child("VRPG");
	if (!vrpgNode) vrpgNode = rendererNode;

	if (pugi::xml_node geoInjecNode = vrpgNode.child("GeometryInjection"))
		geometryInjection = std::make_shared<GeometryInjection_VRPG>(geoInjecNode);
	if (pugi::xml_node geoClusteringNode = vrpgNode.child("GeometryClustering"))
		geometryClustering = std::make_shared<GeometryClustering_VRPG>(geoClusteringNode);
	if (pugi::xml_node lightInjectNode = vrpgNode.child("LightInjection"))
		lightInjection = std::make_shared<LightInjection_VRPG>(lightInjectNode);
	if (pugi::xml_node lightClusteringNode = vrpgNode.child("LightClustering"))
		lightClustering = std::make_shared<LightClustering_VRPG>(lightClusteringNode);
	if (pugi::xml_node pathGuidingNode = vrpgNode.child("PathGuiding"))
		pathGuiding = std::make_shared<PathGuiding_VRPG>(pathGuidingNode);
	else
		pathGuiding = std::make_shared<PathGuiding_VRPG>(vrpgNode);
	if (pugi::xml_node risNode = vrpgNode.child("RIS"))
		ris = std::make_shared<RIS_VRPG>(risNode);
	else
		ris = std::make_shared<RIS_VRPG>(vrpgNode);
	// the modules are required, fail early instead of dereferencing null later
	if (!geometryInjection || !geometryClustering || !lightInjection || !lightClustering) {
		std::fprintf(stderr, "VRPG2: GeometryInjection / GeometryClustering / LightInjection / LightClustering blocks are required in rendererInfo.xml\n");
	}
}

void VRPG2Renderer::init() {
	setting.staticScene = Application::sceneResource.isStaticScene;

	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);
	source.asManager = &asManager;

	// the guided path tracing pass writes into the renderer's own gBuffer
	Renderer::createGBuffer(false, true, 1, Application::renderResolution);
	geometryInjection->init();
	geometryClustering->init();
	lightInjection->init();
	lightClustering->init();
	ris->init();

	PathGuidingCreateInfo_VRPG2 pathGuidingInfo{};
	pathGuidingInfo.gBuffers = &gBuffers;
	pathGuidingInfo.asManager = &asManager;
	pathGuiding->init(pathGuidingInfo);

	Renderer::init();
}
void VRPG2Renderer::clean() {
	geometryInjection->clean();
	geometryClustering->clean();
	lightInjection->clean();
	lightClustering->clean();
	ris->clean();
	if (pathGuiding) pathGuiding->clean();

	// shared state owned by the renderer
	for (nvvk::Buffer& buffer : source.VGBs) Application::allocator.destroyBuffer(buffer);
	source.VGBs.clear();
	for (nvvk::Buffer& buffer : source.clusterDataBuffer_Octree) Application::allocator.destroyBuffer(buffer);
	source.clusterDataBuffer_Octree.clear();
	for (nvvk::Buffer& buffer : source.clusterInfoBuffer_Octree) Application::allocator.destroyBuffer(buffer);
	source.clusterInfoBuffer_Octree.clear();
	for (nvvk::Buffer& buffer : source.lightNodeDataBuffer_Octree) Application::allocator.destroyBuffer(buffer);
	source.lightNodeDataBuffer_Octree.clear();
	Application::allocator.destroyBuffer(source.clusterInfoBuffer_SVO);
	Application::allocator.destroyBuffer(source.nearbyClusterInfoBuffer);
	Application::allocator.destroyBuffer(source.globalInfoBuffer);
	Application::allocator.destroyBuffer(source.clusterPairInfoBuffer);
	Application::allocator.destroyBuffer(source.candidatePoolInfoBuffer);
	Application::allocator.destroyBuffer(source.candidatePoolNodeInfoBuffer);
	Application::allocator.destroyBuffer(source.clusterPairHitInfoBuffer);
	Application::allocator.destroyBuffer(source.candidateSubTreeInfoBuffer);
	Application::allocator.destroyBuffer(source.candidateLeafNodeDataBuffer);
	Application::allocator.destroyBuffer(source.candidateSampleDataBuffer);
	Application::allocator.destroyBuffer(source.candidateSamplePrefixWSumBuffer);

	// the base RT pipeline is not built in this renderer, clean up only what init() created
	Renderer::clean();
	asManager.clean();
	sbtGenerator.deinit();
};
void VRPG2Renderer::uiRender() {
	bool& UIModified = Application::UIModified;

	namespace PE = nvgui::PropertyEditor;
	Application::viewportImage = gBuffers.getDescriptorSet(eImgTonemapped);

	if (ImGui::Begin("VRPGSettings"))
	{
		ImGui::SeparatorText("VRPG");
		PE::begin();
		UIModified |= PE::DragInt("Max Frames", &maxFrames);
		//maxFrames = maxFrames_temp + 1;
		PE::end();
		ImGui::TextDisabled("Frame: %d", setting.frameIndex);

		// Compile-time VRPG components.
		//ImGui::TextDisabled("Spatial Reuse: %d  Temporal Reuse: %d", VRPG_SPATIAL_REUSE, VRPG_TEMPORAL_REUSE);
		//ImGui::TextDisabled("Jitter: %d  Nearby: %d",
		//	VRPG_JITTER_SAMPLING, VRPG_NEARBY_NODE_COUNT);
		UIModified |= ImGui::SliderFloat("Jitter Radius", &setting.jitterRadiusScale, 0.0f, 10.0f, "%.2f");
		//ImGui::TextDisabled("Path Sampling Mode: %d", VRPG_PATH_SAMPLE_MODE);
		//UIModified |= ImGui::Checkbox("Rotate Outgoing", &octree->rotateOutgoing);
		UIModified |= ImGui::Checkbox("Rebuild Per Frame", &setting.rebuildPerFrame);
		//ImGui::TextDisabled("M=%d  K=%d  S=%d  (compile-time)",
		//	Candidate_Samples_Count, OUTGOING_COUNT_VRPG, HITTEST_COUNT_VRPG);
		// ImGui::TextDisabled("Light VGB: %d^3", int(lightRasterVoxelization->setting.pushConstant.voxelSize_Count.w));

		ImGui::SeparatorText("Bounces"); {
			PE::begin();
			PE::SliderInt("Bounces Depth", &setting.maxBounces, 1, std::min(MAX_DEPTH, ptContext.rtProperties.maxRayRecursionDepth), "%d", ImGuiSliderFlags_AlwaysClamp,
				"Maximum Bounces depth");
			PE::end();
		}
		ImGui::SeparatorText("SPP"); {
			PE::begin();
			UIModified |= PE::SliderInt("SPP", &setting.spp, 1, 64, "%d", ImGuiSliderFlags_AlwaysClamp,
				"Sample Per Pixel");
			PE::end();
		}
		geometryInjection->uiRender();
		geometryClustering->uiRender();
		lightInjection->uiRender();
		lightClustering->uiRender();
		ris->uiRender();
		if (pathGuiding) pathGuiding->uiRender();
	}
	ImGui::End();

	if (UIModified) resetFrame();
};
void VRPG2Renderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) {
	geometryInjection->resize(cmd, size);
	geometryClustering->resize(cmd, size);
	lightInjection->resize(cmd, size);
	lightClustering->resize(cmd, size);
}
void VRPG2Renderer::preRender() {
	geometryInjection->preRender();
	geometryClustering->preRender();
	lightInjection->preRender();
	lightClustering->preRender();
	ris->preRender();
	pathGuiding->preRender();
}
void VRPG2Renderer::updateDataPerFrame(VkCommandBuffer cmd) {
	Scene& scene = Application::sceneResource;
	bool cameraChange = scene.cameraChange;
	if (cameraChange || Application::UIModified) resetFrame();

	setting.frameIndex = Application::frameIndex;
	setting.maxFrameCount = maxFrames;

	geometryInjection->updateDataPerFrame(cmd);
	geometryClustering->updateDataPerFrame(cmd);
	lightInjection->updateDataPerFrame(cmd);
	lightClustering->updateDataPerFrame(cmd);
	ris->updateDataPerFrame(cmd);
}
void VRPG2Renderer::render(VkCommandBuffer* cmdPtr) {
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);

	const bool geometryDirty = setting.geometryChange;
	const bool clusteringDirty = setting.clusteringChange;
	const bool lightDirty = setting.lightChange;

	if (geometryDirty) {
		geometryInjection->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	if (clusteringDirty) {
		geometryClustering->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	if (lightDirty) {
		lightInjection->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		//lightClustering->render(cmd);
		//nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
	return;
	{
		setting.geometryChange = false;
		setting.clusteringChange = false;
		setting.lightChange = false;
		++setting.randomFrameIndex;

		// debug views of the geometry octree / SVO / light tree / light injection: drawn right
		// after the clustering data is valid, before any early-out of this frame. uiRender points
		// the viewport at the corresponding debug GBuffer.
		geometryClustering->postProcess(cmd);
		lightInjection->postProcess(cmd);
		lightClustering->postProcess(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
			gBuffers.getDescriptorImageInfo((uint32_t)eImgRendered),
			gBuffers.getDescriptorImageInfo((uint32_t)eImgTonemapped));
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		return;
	}

	ris->render(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// the dirty flags are one-shot: they are recomputed by the modules on the next frame
	setting.geometryChange = false;
	setting.clusteringChange = false;
	setting.lightChange = false;
	++setting.randomFrameIndex;

	// guided path tracing into gBuffers, then tone map for the viewport
	if (setting.frameIndex < maxFrames || maxFrames <= 2) {
		pathGuiding->render(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
			gBuffers.getDescriptorImageInfo((uint32_t)eImgRendered),
			gBuffers.getDescriptorImageInfo((uint32_t)eImgTonemapped));
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
			VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

