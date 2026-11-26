#include "./LPV.h"

#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvgui/property_editor.hpp>
#include <nvvk/default_structs.hpp>
#include <nvutils/file_operations.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

using namespace FzbRenderer;

namespace {
glm::vec3 lpvUpVector(const glm::vec3& direction){
	return std::abs(direction.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
}
// Whitespace trimming for the "key = value" parameter file.
std::string lpvTrim(const std::string& text)
{
	const size_t first = text.find_first_not_of(" \t\r\n");
	if (first == std::string::npos)
		return {};
	const size_t last = text.find_last_not_of(" \t\r\n");
	return text.substr(first, last - first + 1);
}
} 
//------------------------Construction / settings------------------------
LPVRenderer::LPVRenderer(pugi::xml_node& rendererNode)
{
	// Ray tracing (ray queries) and the float atomics of the injection passes must be registered
	// before the Vulkan device is created.
	ptContext.setContextInfo();

	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
	bool atomicFloatPresent = false;
	for (const auto& extension : Application::vkContextInitInfo.deviceExtensions)
	{
		if (extension.extensionName != nullptr && std::strcmp(extension.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0)
		{
			atomicFloatPresent = true;
			break;
		}
	}
	if (!atomicFloatPresent)
		Application::vkContextInitInfo.deviceExtensions.push_back({ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures });

	// Defensive XML parsing: a missing or malformed value must never throw inside the constructor.
	auto readInt = [](pugi::xml_node parent, const char* name, int fallback, int minimum, int maximum) {
		pugi::xml_node node = parent.child(name);
		if (!node)
			return fallback;
		const char* text = node.attribute("value").value();
		if (text == nullptr || text[0] == '\0')
			return fallback;
		try
		{
			return std::clamp(std::stoi(text), minimum, maximum);
		}
		catch (...)
		{
			LOGW("LPV: invalid value \"%s\" for <%s>, using %d\n", text, name, fallback);
			return fallback;
		}
	};
	auto readFloat = [](pugi::xml_node parent, const char* name, float fallback, float minimum, float maximum) {
		pugi::xml_node node = parent.child(name);
		if (!node)
			return fallback;
		const char* text = node.attribute("value").value();
		if (text == nullptr || text[0] == '\0')
			return fallback;
		try
		{
			return std::clamp(std::stof(text), minimum, maximum);
		}
		catch (...)
		{
			LOGW("LPV: invalid value \"%s\" for <%s>, using %.3f\n", text, name, fallback);
			return fallback;
		}
	};
	auto readBool = [](pugi::xml_node parent, const char* name, bool fallback) {
		pugi::xml_node node = parent.child(name);
		if (!node)
			return fallback;
		const std::string text = node.attribute("value").value();
		if (text.empty())
			return fallback;
		return text != "false" && text != "0";
	};

	// Only the grid / cascade sizes and the iteration count are settings (paper 3.3 and 4); every
	// other quantity of the method is fixed by the paper and lives in the shaders.
	if (pugi::xml_node node = rendererNode.child("LPV"))
	{
		cascadeCount      = readInt(node, "cascades", cascadeCount, 1, LPV_MAX_CASCADES);
		baseResolution    = readInt(node, "resolution", baseResolution, 8, 64);
		baseResolution    = std::max(LPV_VOXEL_THREADGROUP, (baseResolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);
		rsmResolution     = readInt(node, "rsmResolution", rsmResolution, 16, LPV_MAX_RSM_RESOLUTION);
		iterationCount    = readInt(node, "iterations", iterationCount, 0, LPV_MAX_ITERATIONS);
		baseCellSize      = readFloat(node, "cellSize", baseCellSize, 0.0f, 1024.0f);
		useOcclusion      = readBool(node, "useOcclusion", useOcclusion);
		useDampening      = readBool(node, "useDampening", useDampening);
		saveFrame         = readInt(node, "saveFrame", saveFrame, -1, MAX_FRAME);
	}
	// Debug / automation hooks. Empty values are ignored so that exporting a variable as "" from a
	// script cannot silently change a setting.
	auto readEnvInt = [](const char* name, int& value, int minimum, int maximum) {
		const char* text = std::getenv(name);
		if (text == nullptr || text[0] == '\0')
			return;
		value = std::clamp(std::atoi(text), minimum, maximum);
	};
	readEnvInt("LPV_SAVE_FRAME", saveFrame, -1, MAX_FRAME);
	readEnvInt("LPV_ITERATIONS", iterationCount, 0, LPV_MAX_ITERATIONS);
	readEnvInt("LPV_CASCADES", cascadeCount, 1, LPV_MAX_CASCADES);
	if (const char* debugCascade = std::getenv("LPV_DEBUG_CASCADE"))
		debugCascadeId = (debugCascade[0] != '\0') && std::atoi(debugCascade) != 0;
	if (const char* debugTransmit = std::getenv("LPV_DEBUG_TRANSMIT"))
		debugTransmittance = (debugTransmit[0] != '\0') && std::atoi(debugTransmit) != 0;
	if (const char* debugShEnv = std::getenv("LPV_DEBUG_SH"))
		debugSH = (debugShEnv[0] != '\0') && std::atoi(debugShEnv) != 0;
	if (const char* noShadow = std::getenv("LPV_NO_SHADOW"))
		skipDirectShadow = (noShadow != nullptr && noShadow[0] != '\0') && std::atoi(noShadow) != 0;
	// Same as the "Remove direct light" button, for scripted comparisons.
	if (const char* noDirect = std::getenv("LPV_NO_DIRECT"))
		showDirectLight = !((noDirect != nullptr && noDirect[0] != '\0') && std::atoi(noDirect) != 0);

	// The volume passes dispatch 8^3 threads, so the resolution has to be a multiple of 8.
	gridResolution = std::max(LPV_VOXEL_THREADGROUP, (gridResolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);

	LOGW("LPV: %d cascades of %d^3 cells, %d iterations, %d^2 virtual RSM\n", cascadeCount, gridResolution, iterationCount,
	     rsmResolution);
}
//---------------------------------------Life cycle----------------------------------------------
void LPVRenderer::init(){
	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	currentResolution = Application::renderResolution;
	// Nothing in this renderer rasterises, so no depth attachment is needed: attachment 0 is the
	// HDR image the compute passes write, attachment 1 the tonemapped one.
	Feature::createGBuffer(true, true, 1, Application::renderResolution);

	updateSceneBounds();
	updateCascadeLayout();
	createResources();

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}
void LPVRenderer::clean()
{
	VkDevice device = Application::app->getDevice();

	for (VkShaderEXT* shader : { &shaderGBuffer, &shaderRsmInject, &shaderGvMerge, &shaderPropagate, &shaderGather,
		&shaderCreateDepthImage, &shaderShowGBuffer, &shaderShowGrid_vertex, &shaderShowGrid_fragment, &shaderShowRSM,
		&shaderShowSH_vertex, &shaderShowSH_fragment, &shaderInjectGrid, &shaderPrintSH })
	{
		if (*shader != VK_NULL_HANDLE)
		{
			vkDestroyShaderEXT(device, *shader, nullptr);
			*shader = VK_NULL_HANDLE;
		}
	}

	destroyResources();
	PathTracingRenderer::clean();
}
void LPVRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size){
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

	const bool resolutionChanged = (size.width != currentResolution.width) || (size.height != currentResolution.height);
	if (resolutionChanged)
	{
		destroyResources();
		currentResolution = size;
		createResources();
		createDescriptorSet();
	}

	// The colour attachment was re-created, so the out image descriptor always has to be rewritten.
	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    outImageWrite = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(outImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_LPV::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	VkWriteDescriptorSet    depthImageWrite = staticDescPack.makeWrite((uint32_t)shaderio::LPVStaticBindingPoints::eDepthImage, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void LPVRenderer::preRender()
{
	Scene& scene = Application::sceneResource;

	if (scene.cameraChange)
		resetFrame();
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight)
		maxFrames = 1;

	const bool dynamicScene = scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight;
	if (!sceneBoundsValid || dynamicScene || scene.instances.size() != sceneBoundsInstanceCount)
	{
		updateSceneBounds();
		sceneBoundsValid         = true;
		sceneBoundsInstanceCount = scene.instances.size();
	}
	updateCascadeLayout();

	pushConstant.frameIndex       = Application::frameIndex;
	pushConstant.cascadeCount     = cascadeCount;
	pushConstant.gridResolution   = gridResolution;
	pushConstant.rsmResolution    = rsmResolution;
	pushConstant.screenSize       = shaderio::uint2(currentResolution.width, currentResolution.height);
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)scene.bSceneInfo.address;

	// The debug passes (showGBufferMain / showRSMMain) read their own push constant struct: without
	// this they dispatch with screenSize == 0 and every thread returns immediately (black image).
	debugPushConstant.screenSize = pushConstant.screenSize;
	debugPushConstant.sceneInfoAddress = pushConstant.sceneInfoAddress;
	debugPushConstant.cascadeCount     = cascadeCount;
	debugPushConstant.frameIndex       = Application::frameIndex;
	pushConstant.lightIndex       = 0;
	pushConstant.time             = scene.time;
	pushConstant.gatherOcclusion  = gatherOcclusionEnabled ? gatherOcclusion : 0.0f;
	pushConstant.cascadeBlendCells = cascadeBlendCells;
	pushConstant.iterationCount    = iterationCount;

	if (!tlasBuilt || dynamicScene || scene.instances.size() != tlasInstanceCount)
	{
		VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
		asManager.updateToplevelAS(cmd);
		Application::app->submitAndWaitTempCmdBuffer(cmd);
		tlasBuilt         = true;
		tlasInstanceCount = scene.instances.size();
	}

	// Debug / automation hook: write the tonemapped image of the previous frame at a given index.
	if (saveFrame >= 0 && Application::frameIndex == saveFrame)
	{
		vkDeviceWaitIdle(Application::app->getDevice());
		std::filesystem::path path = std::filesystem::path(nvutils::getExecutablePath()).replace_filename("lpv_snapshot.png");
		Application::app->saveImageToFile(gBuffers.getColorImage((uint32_t)ImageType_LPV::eImgTonemapped), gBuffers.getSize(), path.string());
		LOGI("LPV: wrote %s\n", path.string().c_str());
		saveFrame = -1;
	}
}
void LPVRenderer::updateDataPerFrame(VkCommandBuffer cmd) {
	updateFrameInfo();
	vkCmdUpdateBuffer(cmd, frameInfoBuffer.buffer, 0, sizeof(shaderio::LPVFrameInfo), &frameInfo);

	pushConstant.flags = int((useOcclusion ? uint32_t(shaderio::LPVFlags::eLPVFlagOcclusion) : 0u) |
		(useDampening ? uint32_t(shaderio::LPVFlags::eLPVFlagDampening) : 0u) |
		(showDirectLight ? 0u : uint32_t(shaderio::LPVFlags::eLPVFlagNoDirect)) |
		(useTrilinearSample ? 0u : uint32_t(shaderio::LPVFlags::eLPVFlagNearestGather)) |
		(debugCascadeId ? uint32_t(shaderio::LPVFlags::eLPVFlagDebugCascadeId) : 0u) |
		(debugTransmittance ? uint32_t(shaderio::LPVFlags::eLPVFlagDebugTransmittance) : 0u) |
		(debugSH ? uint32_t(shaderio::LPVFlags::eLPVFlagDebugSH) : 0u) |
		(skipDirectShadow ? uint32_t(shaderio::LPVFlags::eLPVFlagNoShadow) : 0u));

	const size_t uploadCount = std::min<size_t>(instanceSizes.size(), 12288);
	if (uploadCount > 0)
		vkCmdUpdateBuffer(cmd, instanceSizeBuffer.buffer, 0, VkDeviceSize(uploadCount) * sizeof(float), instanceSizes.data());
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	{
		debugPushConstant.frameIndex = pushConstant.frameIndex;
		debugPushConstant.rsmResolution = pushConstant.rsmResolution;

		for (int cascade = 0; cascade < cascadeCount; ++cascade) {
			debugPushConstant.gridInfo[cascade].startPos = cascades[cascade].startPos;
			debugPushConstant.gridInfo[cascade].voxelSize = cascades[cascade].cellSize;
			debugPushConstant.gridInfo[cascade].resolution = cascadeResolution[cascade];
		}

		debugPushConstant.gvSource = debugPushConstant.showSHType == 0 ? 0 : 1;
		debugPushConstant.iterationCount = iterationCount;
	}
}
void LPVRenderer::render(VkCommandBuffer* cmdPtr){
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	if (shaderGBuffer == VK_NULL_HANDLE || shaderGather == VK_NULL_HANDLE) return;

	{
		pushInfo = {
			.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
			.layout = pipelineLayout,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = 256,
			.pValues = &pushConstant,
		};
		vkCmdPushConstants2(cmd, &pushInfo);

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		nvvk::WriteSetContainer write{};
		write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
		vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());
	}

	// Every frame is rebuilt from scratch: the LPV is a stateless, fully dynamic representation.
	clearVolumes(cmd);
	updateDataPerFrame(cmd);  // uploads the frame info and barriers transfer -> compute

	runGBufferPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runRsmInjectionPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runGvMergePass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);

	runPropagationPasses(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runGatherPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	const bool debugViews = showGBuffer || showRSM || showSH || showGrid || printSH;
	if (debugViews)
	{
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
		createDeptImage(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

		if (showGBuffer) showGBufferResult(cmd);
		else if (showRSM) {
			showRSMResult(cmd);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
			tonemapping(cmd);
			return;
		}

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
		if (showSH) showSHResult(cmd);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
		if (showGrid) showGridResult(cmd);

		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
		if (printSH) printSHResult(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
	}

	tonemapping(cmd);
}
//----------------------------Scene bounds / cascade placement----------------------------
void LPVRenderer::updateSceneBounds(){
	Scene& scene = Application::sceneResource;

	shaderio::float3 boundsMin(FLT_MAX, FLT_MAX, FLT_MAX);
	shaderio::float3 boundsMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	instanceSizes.assign(scene.instances.size(), 1e30f);
	for (size_t instanceIndex = 0; instanceIndex < scene.instances.size(); ++instanceIndex){
		const uint32_t meshIndex = scene.instances[instanceIndex].meshIndex;
		MeshInfo& meshInfo = scene.getMeshInfo(meshIndex);
		shaderio::AABB meshAABB  = meshInfo.getAABB(scene.instances[instanceIndex].transform);

		boundsMin = glm::min(boundsMin, meshAABB.minimum);
		boundsMax = glm::max(boundsMax, meshAABB.maximum);

		// Paper 4.1: the size of an object is determined using its bounding box; objects smaller than
		// a cell of a grid are not rendered into that grid's RSM. Largest extent of the world AABB.
		const shaderio::float3 extent = meshAABB.maximum - meshAABB.minimum;
		instanceSizes[instanceIndex]  = glm::max(extent.x, glm::max(extent.y, extent.z));
	}

	if (scene.instances.empty() || boundsMin.x > boundsMax.x){
		boundsMin = shaderio::float3(-5.0f, -5.0f, -5.0f);
		boundsMax = shaderio::float3(5.0f, 5.0f, 5.0f);
	}

	shaderio::float3 len3 = boundsMax - boundsMin;
	boundsMin -= 0.1f * len3;
	boundsMax += 0.1f * len3;
	len3 = boundsMax - boundsMin;
	
	sceneRadius = std::max(len3.x, std::max(len3.y, len3.z)) * 0.5f;	// std::max(glm::length(boundsMax - boundsMin) * 0.5f, 0.1f);
	sceneHalfDiagonal = length(len3) * 0.5;

	sceneCenter = (boundsMin + boundsMax) * 0.5f;
	sceneBoundsMin = sceneCenter - sceneRadius;
}
void LPVRenderer::updateCascadeLayout(){
	const float extent = std::max(sceneRadius * 2.0f, 0.2f);

	for (int cascade = 0; cascade < cascadeCount; ++cascade) {
		int resolution = std::clamp(baseResolution << cascade, 8, LPV_MAX_GRID_RESOLUTION);
		resolution     = std::max(LPV_VOXEL_THREADGROUP, (resolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);

		cascadeResolution[cascade]  = resolution;
		cascades[cascade].cellSize  = extent / float(resolution);
		cascades[cascade].startPos  = sceneBoundsMin;
	}
	gridResolution = cascadeResolution[std::max(cascadeCount - 1, 0)];
}
//-------------------------------------GPU resources-------------------------------------
int LPVRenderer::debugGridResolution() const
{
	const int cascade = std::clamp(debugPushConstant.cascadeIndex, 0, std::max(cascadeCount - 1, 0));
	return cascadeResolution[cascade];
}

uint32_t LPVRenderer::debugVoxelCount() const
{
	const uint32_t n = uint32_t(std::max(debugGridResolution(), 0));
	return n * n * n;
}
void LPVRenderer::logDebugViewCounts() const
{
	static int sLastCascade = -1;
	static int sLastCount   = -1;
	const int  cascade      = std::clamp(debugPushConstant.cascadeIndex, 0, std::max(cascadeCount - 1, 0));
	const int  count        = int(debugVoxelCount());
	if (cascade == sLastCascade && count == sLastCount)
		return;
	sLastCascade = cascade;
	sLastCount   = count;
	LOGI("LPV debug: cascade %d, grid %d^3, voxels %d, cascadeCount %d, resolutions %d/%d\n", cascade, debugGridResolution(),
	     count, cascadeCount, cascadeResolution[0], cascadeResolution[std::max(cascadeCount - 1, 0)]);
}
VkDeviceSize LPVRenderer::volumeFloatCount() const
{
	const VkDeviceSize voxels = VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution);
	return voxels * LPV_VOXEL_FLOATS;
}
void LPVRenderer::createResources(){
	nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
	const VkBufferUsageFlags2 usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
	                                  VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;

	auto create = [&](nvvk::Buffer& buffer, VkDeviceSize size, const char* name) {
		NVVK_CHECK(allocator->createBuffer(buffer, size, usage));
		NVVK_DBG_NAME(buffer.buffer);
		LOGI("LPV: %s = %.2f MB\n", name, double(size) / (1024.0 * 1024.0));
	};

	const VkDeviceSize pixelCount  = VkDeviceSize(currentResolution.width) * VkDeviceSize(currentResolution.height);

	create(pixelDataBuffer, pixelCount * sizeof(shaderio::LPVPixelData), "pixelData");
	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade)
	{
		const int          resolution  = std::max(cascadeResolution[std::min(cascade, std::max(cascadeCount - 1, 0))], 8);
		const VkDeviceSize voxelCount  = VkDeviceSize(resolution) * resolution * resolution;
		const VkDeviceSize volumeBytes = voxelCount * LPV_VOXEL_FLOATS * sizeof(float);
		const VkDeviceSize gvVoxels    = VkDeviceSize(resolution + 1) * (resolution + 1) * (resolution + 1);
		const VkDeviceSize gvBytes     = gvVoxels * LPV_GV_FLOATS * sizeof(float);
		create(lpvVolumesBuffer[cascade], 2 * volumeBytes, "lpvVolumes");
		create(lpvAccumBuffer[cascade], volumeBytes, "lpvAccum");
		create(gvMergedBuffer[cascade], gvBytes, "gvMerged");
		create(gvCameraBuffer[cascade], gvBytes, "gvCamera");
		create(gvRsmBuffer[cascade], gvBytes, "gvRsm");
	}
	create(frameInfoBuffer, sizeof(shaderio::LPVFrameInfo), "frameInfo");
	create(instanceSizeBuffer, VkDeviceSize(std::max<size_t>(instanceSizes.size(), 1)) * sizeof(float), "instanceSize");

	const VkDeviceSize gridVoxels = VkDeviceSize(gridResolution) * gridResolution * gridResolution;
	const VkDeviceSize gridDataBytes = gridVoxels * sizeof(shaderio::GridDebugInfo);
	create(gridDataBuffer, gridDataBytes, "gridDebugData");

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	clearVolumes(cmd);
	vkCmdFillBuffer(cmd, frameInfoBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	Application::app->submitAndWaitTempCmdBuffer(cmd);

	{
		nvutils::PrimitiveMesh primitive = FzbRenderer::MeshSet::createWireframe();
		FzbRenderer::MeshSet mesh = FzbRenderer::MeshSet("Wireframe", primitive);
		scene.addMeshSet(mesh);

		primitive = FzbRenderer::MeshSet::createCube(false, false);
		mesh = FzbRenderer::MeshSet("Cube", primitive);
		scene.addMeshSet(mesh);

		primitive = FzbRenderer::MeshSet::createSphere(false, false);
		mesh = FzbRenderer::MeshSet("Sphere", primitive);
		scene.addMeshSet(mesh);

		scene.createSceneInfoBuffer();
	}
}
void LPVRenderer::destroyResources()
{
	for (nvvk::Buffer* buffer : { &pixelDataBuffer, &frameInfoBuffer, &instanceSizeBuffer, &gridDataBuffer })
	{
		if (buffer->buffer != VK_NULL_HANDLE)
			Application::allocator.destroyBuffer(*buffer);
		*buffer = {};
	}
	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade)
	{
		for (nvvk::Buffer* buffer : { &lpvVolumesBuffer[cascade], &lpvAccumBuffer[cascade], &gvMergedBuffer[cascade],
		                              &gvCameraBuffer[cascade], &gvRsmBuffer[cascade] })
		{
			if (buffer->buffer != VK_NULL_HANDLE)
				Application::allocator.destroyBuffer(*buffer);
			*buffer = {};
		}
	}
}
void LPVRenderer::clearVolumes(VkCommandBuffer cmd){
	NVVK_DBG_SCOPE(cmd);
	// lpvAccum is fully overwritten by the copy of the injected volume, so it is not cleared here.
	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade)
	{
		vkCmdFillBuffer(cmd, lpvVolumesBuffer[cascade].buffer, 0, VK_WHOLE_SIZE, 0u);
		vkCmdFillBuffer(cmd, gvCameraBuffer[cascade].buffer, 0, VK_WHOLE_SIZE, 0u);
		vkCmdFillBuffer(cmd, gvRsmBuffer[cascade].buffer, 0, VK_WHOLE_SIZE, 0u);
	}
	vkCmdFillBuffer(cmd, gridDataBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
}
//-----------------------Descriptors / pipeline layout / shaders-----------------------
void LPVRenderer::createDescriptorSetLayout(){
	SCOPED_TIMER(__FUNCTION__);

	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eTextures_PT,
	                      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
	                      .descriptorCount = std::max(uint32_t(Application::sceneResource.textures.size()), 1u),
	                      .stageFlags = VK_SHADER_STAGE_ALL });
	bindings.addBinding({ .binding = shaderio::StaticSetBindingPoints_PT::eOutImage_PT,
	                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
	                      .descriptorCount = 1,
	                      .stageFlags = VK_SHADER_STAGE_ALL });

	const uint32_t bufferBindings[] = {
		(uint32_t)shaderio::LPVStaticBindingPoints::ePixelData,  (uint32_t)shaderio::LPVStaticBindingPoints::eFrameInfo,
		(uint32_t)shaderio::LPVStaticBindingPoints::eInstanceSize, (uint32_t)shaderio::LPVStaticBindingPoints::eGridDebugData,
	};
	for (uint32_t binding : bufferBindings)
	{
		bindings.addBinding({ .binding = binding,
		                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      .descriptorCount = 1,
		                      .stageFlags = VK_SHADER_STAGE_ALL });
	}
	const uint32_t cascadeBufferBindings[] = {
		(uint32_t)shaderio::LPVStaticBindingPoints::eLPVVolumes, (uint32_t)shaderio::LPVStaticBindingPoints::eLPVAccum,
		(uint32_t)shaderio::LPVStaticBindingPoints::eGvMerged,   (uint32_t)shaderio::LPVStaticBindingPoints::eGvCamera,
		(uint32_t)shaderio::LPVStaticBindingPoints::eGvRsm,
	};
	for (uint32_t binding : cascadeBufferBindings)
	{
		bindings.addBinding({ .binding = binding,
		                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      .descriptorCount = LPV_MAX_CASCADES,
		                      .stageFlags = VK_SHADER_STAGE_ALL });
	}

	bindings.addBinding({ .binding = (uint32_t)shaderio::LPVStaticBindingPoints::eDepthImage,
							  .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
							  .descriptorCount = 1,
							  .stageFlags = VK_SHADER_STAGE_ALL });

	staticDescPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
	                    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	NVVK_DBG_NAME(staticDescPack.getLayout());

	bindings.clear();
	bindings.addBinding({ .binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
	                      .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
	                      .descriptorCount = 1,
	                      .stageFlags = VK_SHADER_STAGE_ALL });
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
}
void LPVRenderer::createDescriptorSet(){
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty())
	{
		VkWriteDescriptorSet allTextures = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0,
		                                                            uint32_t(Application::sceneResource.textures.size()));
		write.append(allTextures, Application::sceneResource.textures.data());
	}

	VkWriteDescriptorSet outImageWrite = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(outImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_LPV::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	auto appendBuffer = [&](shaderio::LPVStaticBindingPoints binding, const nvvk::Buffer& buffer) {
		VkWriteDescriptorSet writeSet = staticDescPack.makeWrite((uint32_t)binding, 0, 0, 1);
		write.append(writeSet, buffer, 0, buffer.bufferSize);
	};

	appendBuffer(shaderio::LPVStaticBindingPoints::ePixelData, pixelDataBuffer);

	auto appendBufferArray = [&](shaderio::LPVStaticBindingPoints binding, const std::array<nvvk::Buffer, LPV_MAX_CASCADES>& buffers) {
		VkWriteDescriptorSet writeSet = staticDescPack.makeWrite((uint32_t)binding, 0, 0, LPV_MAX_CASCADES);
		write.append(writeSet, buffers.data());
	};
	appendBufferArray(shaderio::LPVStaticBindingPoints::eLPVVolumes, lpvVolumesBuffer);
	appendBufferArray(shaderio::LPVStaticBindingPoints::eLPVAccum, lpvAccumBuffer);
	appendBufferArray(shaderio::LPVStaticBindingPoints::eGvMerged, gvMergedBuffer);
	appendBufferArray(shaderio::LPVStaticBindingPoints::eGvCamera, gvCameraBuffer);
	appendBufferArray(shaderio::LPVStaticBindingPoints::eGvRsm, gvRsmBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eFrameInfo, frameInfoBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eInstanceSize, instanceSizeBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eGridDebugData, gridDataBuffer);

	VkWriteDescriptorSet    depthImageWrite = staticDescPack.makeWrite((uint32_t)shaderio::LPVStaticBindingPoints::eDepthImage, 0, 0, 1);
	write.append(depthImageWrite, gBuffers.getDepthImageView(), VK_IMAGE_LAYOUT_GENERAL);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}
void LPVRenderer::createPipelineLayout(){
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout() } };
	const VkPipelineLayoutCreateInfo      pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = uint32_t(layouts.size()),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &pipelineLayout));
	NVVK_DBG_NAME(pipelineLayout);
}
void LPVRenderer::compileAndCreateShaders(){
	SCOPED_TIMER(__FUNCTION__);

	VkDevice                device      = Application::app->getDevice();
	std::filesystem::path   shaderPath  = std::filesystem::path(__FILE__).parent_path() / "shaders";

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = 256,
	};
	std::array<VkDescriptorSetLayout, 2> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout() } };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = uint32_t(layouts.size()),
		.pSetLayouts = layouts.data(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};

	auto createShader = [&](VkShaderEXT& shader, const VkShaderModuleCreateInfo& code, const char* entryPoint) {
		if (code.codeSize == 0 || code.pCode == nullptr) return;
		vkDestroyShaderEXT(device, shader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = entryPoint;
		shaderInfo.codeSize = code.codeSize;
		shaderInfo.pCode = code.pCode;
		const VkResult result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader);
		if (result != VK_SUCCESS)
			LOGE("LPV: failed to create shader object '%s' (%d)\n", entryPoint, int(result));
		else NVVK_DBG_NAME(shader);
	};

	VkShaderModuleCreateInfo passesCode = FzbRenderer::compileSlangShader(shaderPath / "GBuffer.slang", {});
	createShader(shaderGBuffer, passesCode, "lpvGBufferMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "RSM.slang", {});
	createShader(shaderRsmInject, passesCode, "lpvRsmInjectMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "GvMerge.slang", {});
	createShader(shaderGvMerge, passesCode, "lpvGvMergeMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "Propagate.slang", {});
	createShader(shaderPropagate, passesCode, "lpvPropagateMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "Rendering.slang", {});
	createShader(shaderGather, passesCode, "lpvGatherMain");
	//----------------------------------------------------------------------------------
	auto createShaderVF = [&](VkShaderEXT& shader, const VkShaderModuleCreateInfo& code, const char* entryPoint, bool vertex = true) {
		if (code.codeSize == 0 || code.pCode == nullptr) return;
		vkDestroyShaderEXT(device, shader, nullptr);
		shaderInfo.stage = vertex ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
		shaderInfo.nextStage = vertex ? VK_SHADER_STAGE_FRAGMENT_BIT : 0;
		shaderInfo.pName = entryPoint;
		shaderInfo.codeSize = code.codeSize;
		shaderInfo.pCode = code.pCode;
		const VkResult result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader);
		if (result != VK_SUCCESS)
			LOGE("LPV: failed to create shader object '%s' (%d)\n", entryPoint, int(result));
		else NVVK_DBG_NAME(shader);
		};

	// The debug entry points (showGBufferMain / showRSMMain / ShowGrid / ShowSH) only exist under
	// LPV_DEBUG. The compiler scopes its macro list per compile: every compileFile() rebuilds the
	// session from m_macros / m_options (vslang/slang.cpp:141 -> createSession), so pushing the macro
	// here and popping it after this batch is a real scope switch. The main passes above are already
	// compiled, and neither F5 ("Reload shaders") nor any other renderer ever observes the macro.
	// The empty value matches ShowSH.slang's own `#define LPV_DEBUG`, so the two do not conflict.
	auto&      slangMacros = Application::slangCompiler.macros();
	const auto savedMacros = slangMacros;
	slangMacros.push_back({ "LPV_DEBUG", "" });

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "CreateDepthImage.slang", {});
	createShader(shaderCreateDepthImage, passesCode, "createDepthImageMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "ShowGrid.slang", {});
	createShaderVF(shaderShowGrid_vertex, passesCode, "vertexMain");
	createShaderVF(shaderShowGrid_fragment, passesCode, "fragmentMain", false);

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "GBuffer.slang", {});
	createShader(shaderShowGBuffer, passesCode, "showGBufferMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "RSM.slang", {});
	createShader(shaderShowRSM, passesCode, "showRSMMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "ShowSH.slang", {});
	createShaderVF(shaderShowSH_vertex, passesCode, "vertexMain");
	createShaderVF(shaderShowSH_fragment, passesCode, "fragmentMain", false);

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "ShowGrid.slang", {});
	createShader(shaderInjectGrid, passesCode, "injectGridMain");

	passesCode = FzbRenderer::compileSlangShader(shaderPath / "PrintSH.slang", {});
	createShader(shaderPrintSH, passesCode, "printfSHMain");

	slangMacros = savedMacros;
}
//-----------------------------Per frame state-----------------------------
void LPVRenderer::updateFrameInfo(){
	frameInfo = shaderio::LPVFrameInfo{};

	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade){
		const CascadeLayout& layout = cascades[cascade];
		frameInfo.cascadeOriginSize[cascade] = shaderio::float4(layout.startPos, layout.cellSize);
	}

	Scene& scene = Application::sceneResource;

	uint32_t lightCount = 0;
	for (uint32_t lightIndex = 0; lightIndex < LIGHT_COUNT; ++lightIndex)
	{
		if (int(scene.sceneInfo.lights[lightIndex].type) == int(shaderio::LightType::Non))
			break;
		++lightCount;
	}
	activeLightCount = lightCount;

	// The LPV only injects punctual lights, so report what was found once: a scene lit by emissive
	// meshes or by the sky alone will look flat with this renderer.
	static bool sReportedLights = false;
	if (!sReportedLights)
	{
		sReportedLights = true;
		LOGI("LPV: %u punctual light(s) (numLights=%d), useSky=%d\n", lightCount, scene.sceneInfo.numLights, scene.sceneInfo.useSky);
		for (uint32_t i = 0; i < lightCount; ++i)
		{
			const shaderio::Light& l = scene.sceneInfo.lights[i];
			LOGI("LPV: light[%u] type=%d intensity=%.3f direction=(%.3f %.3f %.3f)\n", i, int(l.type), l.intensity, l.direction.x,
				l.direction.y, l.direction.z);
		}
	}

	// w = number of entries in the instance size buffer (paper 4.1 small object test)
	frameInfo.info = shaderio::int4(cascadeCount, gridResolution, rsmResolution, int(instanceSizes.size()));

	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade)
	{
		const int index = std::min(cascade, std::max(cascadeCount - 1, 0));
		frameInfo.cascadeResolution[cascade] = cascadeResolution[index];
		frameInfo.cascadeReach[cascade] = float(iterationCount) * cascades[index].cellSize;
	}

	const float rsm = float(rsmResolution);
	for (uint32_t lightIndex = 0; lightIndex < LIGHT_COUNT; ++lightIndex)
	{
		frameInfo.lightViewProjInverse[lightIndex] = glm::mat4(1.0f);
		frameInfo.lightRsmParams[lightIndex] = shaderio::float4(0.0f);

		if (lightIndex >= activeLightCount)
			continue;

		const shaderio::Light& light = scene.sceneInfo.lights[lightIndex];

		if (int(light.type) == int(shaderio::LightType::Direction))
		{
			glm::vec3 lightDirection = light.direction;
			if (glm::length(lightDirection) < 1e-5f)
				lightDirection = glm::vec3(0.0f, -1.0f, 0.0f);
			lightDirection = glm::normalize(lightDirection);

			const float halfExtent = sceneHalfDiagonal * 1.05f;
			const float distance = sceneHalfDiagonal * 2.0f + 1.0f;
			const glm::vec3 lightEye = sceneCenter - lightDirection * distance;

			const glm::mat4 view = glm::lookAt(lightEye, sceneCenter, lpvUpVector(lightDirection));
			const glm::mat4 projection = glm::orthoRH_ZO(-halfExtent, halfExtent, -halfExtent, halfExtent, 0.0f, distance * 2.0f);

			frameInfo.lightViewProjInverse[lightIndex] = glm::inverse(projection * view);

			const float texelSize = 2.0f * halfExtent / rsm;
			frameInfo.lightRsmParams[lightIndex] = shaderio::float4(texelSize * texelSize, 0.0f, 0.0f, 0.0f);
		}
		else
		{
			glm::vec3 toCenter = sceneCenter - light.pos;
			float     distance = glm::length(toCenter);
			if (distance < 1e-4f)
				distance = 1e-4f;
			const glm::vec3 direction = toCenter / distance;

			float fovY = 2.0f * std::atan(sceneHalfDiagonal * 1.15f / std::max(distance, sceneHalfDiagonal * 0.2f));
			fovY = std::clamp(fovY, 0.1f, 2.9f);

			const glm::mat4 view = glm::lookAt(light.pos, sceneCenter, lpvUpVector(direction));
			const glm::mat4 projection = glm::perspectiveRH_ZO(fovY, 1.0f, std::max(distance - sceneHalfDiagonal * 1.2f, 0.02f),
				distance + sceneHalfDiagonal * 2.5f + 1.0f);

			frameInfo.lightViewProjInverse[lightIndex] = glm::inverse(projection * view);

			const float texelAngle = 2.0f * std::tan(fovY * 0.5f) / rsm;
			frameInfo.lightRsmParams[lightIndex] = shaderio::float4(0.0f, texelAngle * texelAngle, 0.0f, 0.0f);
		}
	}
}
//------------------------------------Passes-------------------------------
void dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ) {
	if (shader == VK_NULL_HANDLE || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0) return;
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &shader);
	vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
}
void LPVRenderer::runGBufferPass(VkCommandBuffer cmd){
	NVVK_DBG_SCOPE(cmd);
	const uint32_t groupCountX = (currentResolution.width + LPV_PIXEL_THREADGROUP_X - 1) / LPV_PIXEL_THREADGROUP_X;
	const uint32_t groupCountY = (currentResolution.height + LPV_PIXEL_THREADGROUP_Y - 1) / LPV_PIXEL_THREADGROUP_Y;
	dispatchCompute(cmd, shaderGBuffer, groupCountX, groupCountY, 1);
}
void LPVRenderer::runRsmInjectionPass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	const uint32_t groupCount = (uint32_t(rsmResolution) + LPV_RSM_THREADGROUP_X - 1) / LPV_RSM_THREADGROUP_X;
	for (uint32_t lightIndex = 0; lightIndex < activeLightCount; ++lightIndex)
	{
		pushConstant.lightIndex = int(lightIndex);
		vkCmdPushConstants2(cmd, &pushInfo);
		dispatchCompute(cmd, shaderRsmInject, groupCount, groupCount, 1);
		// Both dispatches accumulate into the same volumes with atomics.
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}
void LPVRenderer::runGvMergePass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	for (int cascade = 0; cascade < cascadeCount; ++cascade)
	{
		const uint32_t resolution = uint32_t(cascadeResolution[cascade]) + 1u;
		const uint32_t groupCount = (resolution + LPV_VOXEL_THREADGROUP - 1) / LPV_VOXEL_THREADGROUP;
		pushConstant.cascadeIndex = cascade;
		vkCmdPushConstants2(cmd, &pushInfo);
		dispatchCompute(cmd, shaderGvMerge, groupCount, groupCount, groupCount);
	}
}
void LPVRenderer::runPropagationPasses(VkCommandBuffer cmd){
	NVVK_DBG_SCOPE(cmd);

	{
		for (int cascade = 0; cascade < cascadeCount; ++cascade){
			const VkDeviceSize cascadeVoxels =
			    VkDeviceSize(cascadeResolution[cascade]) * cascadeResolution[cascade] * cascadeResolution[cascade];
			VkBufferCopy2 copyRegion{
				.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
				.srcOffset = 0,
				.dstOffset = 0,
				.size = cascadeVoxels * LPV_VOXEL_FLOATS * sizeof(float),
			};
			VkCopyBufferInfo2 copyInfo{
				.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
				.srcBuffer = lpvVolumesBuffer[cascade].buffer,
				.dstBuffer = lpvAccumBuffer[cascade].buffer,
				.regionCount = 1,
				.pRegions = &copyRegion,
			};
			vkCmdCopyBuffer2(cmd, &copyInfo);
		}
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	if (iterationCount <= 0) return;
	for (int cascade = 0; cascade < cascadeCount; ++cascade){
		const uint32_t groupCount = uint32_t(cascadeResolution[cascade]) / LPV_VOXEL_THREADGROUP;

		pushConstant.cascadeIndex      = cascade;
		pushConstant.cascadeIterations = iterationCount;
		for (int iteration = 0; iteration < iterationCount; ++iteration){
			pushConstant.iteration = iteration;
			vkCmdPushConstants2(cmd, &pushInfo);
			dispatchCompute(cmd, shaderPropagate, groupCount, groupCount, groupCount);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
	}
}
void LPVRenderer::runGatherPass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	// The gather shader ignores the iteration; the injection passes are done by now.
	pushConstant.iteration = 0;
	vkCmdPushConstants2(cmd, &pushInfo);
	const uint32_t groupCountX = (currentResolution.width + LPV_PIXEL_THREADGROUP_X - 1) / LPV_PIXEL_THREADGROUP_X;
	const uint32_t groupCountY = (currentResolution.height + LPV_PIXEL_THREADGROUP_Y - 1) / LPV_PIXEL_THREADGROUP_Y;
	dispatchCompute(cmd, shaderGather, groupCountX, groupCountY, 1);
}
void LPVRenderer::tonemapping(VkCommandBuffer cmd) {
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
		gBuffers.getDescriptorImageInfo((uint32_t)ImageType_LPV::eImgRendered),
		gBuffers.getDescriptorImageInfo((uint32_t)ImageType_LPV::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
}
//----------------------------------Parameter presets----------------------------------
void LPVRenderer::rebuildVolumes()
{
	vkDeviceWaitIdle(Application::app->getDevice());
	updateCascadeLayout();
	destroyResources();
	createResources();
	createDescriptorSet();
	pushConstant.cascadeCount      = cascadeCount;
	pushConstant.gridResolution    = gridResolution;
	pushConstant.rsmResolution     = rsmResolution;
	debugPushConstant.cascadeCount = cascadeCount;
	resetFrame();
}
void LPVRenderer::saveParams()
{
	const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() / "LPVParams.md";
	std::ofstream               file(path, std::ios::trunc);
	if (!file)
	{
		LOGE("LPV: cannot write %s\n", path.string().c_str());
		return;
	}

	file << "# LPV renderer parameters\n";
	file << "# key = value, one entry per line. The 'Load params' button reads this file back.\n\n";
	file << "cascadeCount = " << cascadeCount << "\n";
	file << "baseResolution = " << baseResolution << "\n";
	file << "gridResolution = " << baseResolution << "\n";
	file << "rsmResolution = " << rsmResolution << "\n";
	file << "iterationCount = " << iterationCount << "\n";
	file << "cascadeBlendCells = " << cascadeBlendCells << "\n";
	file << "baseCellSize = " << baseCellSize << "\n\n";
	file << "useOcclusion = " << int(useOcclusion) << "\n";
	file << "gatherOcclusion = " << gatherOcclusion << "\n";
	file << "gatherOcclusionEnabled = " << int(gatherOcclusionEnabled) << "\n";
	file << "useDampening = " << int(useDampening) << "\n";
	file << "showDirectLight = " << int(showDirectLight) << "\n";
	file << "useTrilinearSample = " << int(useTrilinearSample) << "\n\n";
	file << "showGrid = " << int(showGrid) << "\n";
	file << "showSH = " << int(showSH) << "\n";
	file << "showGBuffer = " << int(showGBuffer) << "\n";
	file << "showRSM = " << int(showRSM) << "\n\n";
	file << "gridDataType = " << debugPushConstant.gridDataType << "\n";
	file << "cascadeIndex = " << debugPushConstant.cascadeIndex << "\n";
	file << "showGridVoxelRangeMin = " << debugPushConstant.showGridVoxelRangeMin.x << " " << debugPushConstant.showGridVoxelRangeMin.y
	     << " " << debugPushConstant.showGridVoxelRangeMin.z << "\n";
	file << "showGridVoxelRangeMax = " << debugPushConstant.showGridVoxelRangeMax.x << " " << debugPushConstant.showGridVoxelRangeMax.y
	     << " " << debugPushConstant.showGridVoxelRangeMax.z << "\n";
	file << "GBufferIndex = " << debugPushConstant.GBufferIndex << "\n";
	file << "lightIndex = " << debugPushConstant.lightIndex << "\n";
	file << "gvSource = " << debugPushConstant.gvSource << "\n";
	file << "showSHType = " << debugPushConstant.showSHType << "\n";

	LOGI("LPV: parameters saved to %s\n", path.string().c_str());
}
void LPVRenderer::loadParams()
{
	const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() / "LPVParams.md";
	std::ifstream               file(path);
	if (!file)
	{
		LOGE("LPV: cannot read %s\n", path.string().c_str());
		return;
	}

	const int oldCascadeCount   = cascadeCount;
	const int oldBaseResolution = baseResolution;
	const int oldRsmResolution  = rsmResolution;

	auto toBool = [](const std::string& text) { return !(text == "0" || text == "false" || text == "False" || text == "off"); };

	std::string line;
	while (std::getline(file, line))
	{
		const size_t comment = line.find('#');
		if (comment != std::string::npos)
			line = line.substr(0, comment);
		const size_t separator = line.find('=');
		if (separator == std::string::npos)
			continue;

		const std::string key   = lpvTrim(line.substr(0, separator));
		const std::string value = lpvTrim(line.substr(separator + 1));
		if (key.empty() || value.empty())
			continue;

		try
		{
			if (key == "cascadeCount")
				cascadeCount = std::clamp(std::stoi(value), 1, LPV_MAX_CASCADES);
			else if (key == "baseResolution")
				baseResolution = std::clamp(std::stoi(value), 8, 64);
			else if (key == "gridResolution")
				baseResolution = std::clamp(std::stoi(value), 8, 64);
			else if (key == "rsmResolution")
				rsmResolution = std::clamp(std::stoi(value), 16, LPV_MAX_RSM_RESOLUTION);
			else if (key == "iterationCount")
				iterationCount = std::clamp(std::stoi(value), 0, LPV_MAX_ITERATIONS);
			else if (key == "cascadeBlendCells")
				cascadeBlendCells = std::clamp(std::stoi(value), 1, 8);
			else if (key == "baseCellSize")
				baseCellSize = std::clamp(std::stof(value), 0.0f, 1024.0f);
			else if (key == "useOcclusion")
				useOcclusion = toBool(value);
			else if (key == "gatherOcclusion")
				gatherOcclusion = std::clamp(std::stof(value), 0.0f, 1.0f);
			else if (key == "gatherOcclusionEnabled")
				gatherOcclusionEnabled = toBool(value);
			else if (key == "useDampening")
				useDampening = toBool(value);
			else if (key == "showDirectLight")
				showDirectLight = toBool(value);
			else if (key == "useTrilinearSample")
				useTrilinearSample = toBool(value);
			else if (key == "showGrid")
				showGrid = toBool(value);
			else if (key == "showSH")
				showSH = toBool(value);
			else if (key == "showGBuffer")
				showGBuffer = toBool(value);
			else if (key == "showRSM")
				showRSM = toBool(value);
			else if (key == "gridDataType")
				debugPushConstant.gridDataType = std::clamp(std::stoi(value), 0, 2);
			else if (key == "cascadeIndex")
				debugPushConstant.cascadeIndex = std::stoi(value);
			else if (key == "GBufferIndex")
				debugPushConstant.GBufferIndex = std::clamp(std::stoi(value), 0, 3);
			else if (key == "lightIndex")
				debugPushConstant.lightIndex = std::max(std::stoi(value), 0);
			else if (key == "gvSource")
				debugPushConstant.gvSource = std::clamp(std::stoi(value), 0, 2);
			else if (key == "showSHType")
				debugPushConstant.showSHType = std::clamp(std::stoi(value), 0, 4);
			else if (key == "showGridVoxelRangeMin" || key == "showGridVoxelRangeMax")
			{
				std::istringstream stream(value);
				shaderio::int3     range{};
				stream >> range.x >> range.y >> range.z;
				if (!stream.fail())
				{
					shaderio::int3& target = (key == "showGridVoxelRangeMin") ? debugPushConstant.showGridVoxelRangeMin :
					                                                            debugPushConstant.showGridVoxelRangeMax;
					target = shaderio::clamp(range, shaderio::int3(0), shaderio::int3(std::max(debugGridResolution() - 1, 0)));
				}
			}
			else
				LOGW("LPV: unknown parameter \"%s\" in %s\n", key.c_str(), path.string().c_str());
		}
		catch (const std::exception&)
		{
			LOGW("LPV: invalid value \"%s\" for \"%s\", keeping the current one\n", value.c_str(), key.c_str());
		}
	}

	baseResolution = std::max(LPV_VOXEL_THREADGROUP, (baseResolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);
	rsmResolution  = std::max(16, (rsmResolution / 16) * 16);
	updateCascadeLayout();
	debugPushConstant.cascadeIndex = std::clamp(debugPushConstant.cascadeIndex, 0, std::max(cascadeCount - 1, 0));
	debugPushConstant.showGridVoxelRangeMin =
	    shaderio::clamp(debugPushConstant.showGridVoxelRangeMin, shaderio::int3(0), shaderio::int3(gridResolution - 1));
	debugPushConstant.showGridVoxelRangeMax =
	    shaderio::clamp(debugPushConstant.showGridVoxelRangeMax, shaderio::int3(0), shaderio::int3(gridResolution - 1));

	if (cascadeCount != oldCascadeCount || baseResolution != oldBaseResolution || rsmResolution != oldRsmResolution)
	{
		LOGI("LPV: the grid layout changed, rebuilding the volumes\n");
		rebuildVolumes();
	}
	else
	{
		resetFrame();
	}

	LOGI("LPV: parameters loaded from %s\n", path.string().c_str());
}
//--------------------------------------UI--------------------------------------
void LPVRenderer::uiRender(){
	bool uiModified = Application::UIModified;

	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_LPV::eImgTonemapped);

	if (ImGui::Begin("LPVSettings")){
		ImGui::SeparatorText("Light Propagation Volumes");
		ImGui::TextDisabled("Frame: %d", Application::frameIndex);

		ImGui::Dummy(ImVec2(0.0f, ImGui::GetTextLineHeightWithSpacing()));
		{
			ImGui::TextDisabled("Cascades: %d x %d^3 = %d voxels", cascadeCount, gridResolution, cascadeCount * gridResolution * gridResolution * gridResolution);
			ImGui::TextDisabled("Iterations: %d (0 = injection only, no propagation), virtual RSM: %d^2 per light", iterationCount, rsmResolution);

			uiModified |= ImGui::Checkbox("Fuzzy occlusion", &useOcclusion);
			uiModified |= ImGui::Checkbox("Gather occlusion", &gatherOcclusionEnabled);
			if (gatherOcclusionEnabled)
				uiModified |= ImGui::SliderFloat("Gather strength", &gatherOcclusion, 0.0f, 1.0f, "%.2f");
			uiModified |= ImGui::Checkbox("Fuzzy occlusion Two Face", (bool*)&pushConstant.occTwoFace);
			uiModified |= ImGui::Checkbox("Derivative dampening", &useDampening);
			uiModified |= ImGui::Checkbox("Show Directional Light", &showDirectLight);
			uiModified |= ImGui::Checkbox("Trilinear gather", &useTrilinearSample);

			ImGui::TextDisabled("Everything else (half cell offsets, 5 exit faces, cascade blend) follows the paper.");
		}
		{
			const int oldCascadeCount   = cascadeCount;
			const int oldBaseResolution = baseResolution;

			uiModified |= ImGui::SliderInt("Cascade count", &cascadeCount, 1, LPV_MAX_CASCADES);
			uiModified |= ImGui::SliderInt("Base resolution", &baseResolution, 8, 64);
			baseResolution = std::max(LPV_VOXEL_THREADGROUP, (baseResolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);

			if (cascadeCount != oldCascadeCount || baseResolution != oldBaseResolution)
			{
				rebuildVolumes();
				uiModified = true;
			}
			for (int cascade = 0; cascade < cascadeCount; ++cascade)
				ImGui::TextDisabled("cascade %d: %d^3, cell %.3f m, reach %.2f m", cascade, cascadeResolution[cascade],
				                    double(cascades[cascade].cellSize), double(iterationCount * cascades[cascade].cellSize));
		}
		{
			ImGui::SliderInt("RSM resolution", &rsmResolution, 32, 1024);
			rsmResolution = std::max(16, (rsmResolution / 16) * 16);
		}
		{
			uiModified |= ImGui::SliderInt("Propagation iterations", &iterationCount, 0, 16);
			if (cascadeCount > 1)
				uiModified |= ImGui::SliderInt("Cascade blend", &cascadeBlendCells, 1, 8);
		}

		if (ImGui::Button("Rebuild volumes")) rebuildVolumes();

		if (ImGui::Button("Save snapshot")) saveFrame = Application::frameIndex + 2;
		if (ImGui::Button("Save params")) saveParams();
		if (ImGui::Button("Load params")) loadParams();

		ImGui::SeparatorText("Diagnostics");
		ImGui::TextDisabled("Buffers: volumes %.1f MB, geometry %.1f MB",
		                    double(2 * volumeFloatCount() * sizeof(float)) / (1024.0 * 1024.0),
		                    double(3 * VkDeviceSize(cascadeCount) * VkDeviceSize(gridResolution) * gridResolution * gridResolution * LPV_GV_FLOATS * sizeof(float)) / (1024.0 * 1024.0));
	}
	ImGui::End();

	if (ImGui::Begin("Debug")) {
		if (ImGui::CollapsingHeader("Show Grid")) {
			static const char* const hasDataTypeNames[] = { "gBuffer", "RSM radiance", "Propagate SH radiance" };
			ImGui::Combo("Grid Type", &debugPushConstant.gridDataType, hasDataTypeNames, IM_ARRAYSIZE(hasDataTypeNames));
			uiModified |= ImGui::IsItemEdited();

			if (cascadeLabelPtrs.size() != size_t(cascadeCount)) {
				cascadeLabels.clear();
				cascadeLabelPtrs.clear();
				for (int i = 0; i < cascadeCount; ++i) cascadeLabels.push_back("cascade " + std::to_string(i));
				for (const auto& label : cascadeLabels) cascadeLabelPtrs.push_back(label.c_str());
			}
			ImGui::Combo("Cascade Index", &debugPushConstant.cascadeIndex, cascadeLabelPtrs.data(), static_cast<int>(cascadeLabelPtrs.size()));
			uiModified |= ImGui::IsItemEdited();

			ImGui::DragInt3("Show Voxel Min ", (int*)&debugPushConstant.showGridVoxelRangeMin, 1, 0, debugGridResolution() - 1);
			ImGui::DragInt3("Show Voxel max ", (int*)&debugPushConstant.showGridVoxelRangeMax, 1, 0, debugGridResolution() - 1);
			uiModified |= ImGui::Checkbox("show Grid", &showGrid);
		}
		if (ImGui::CollapsingHeader("SH")) {
			static const char* const hasDataTypeNames[] = { "gBuffer Gv", "RSM Gv", "Gv Merge", "Propagate Radiance", "Propagate Acc Radiance" };
			ImGui::Combo("SH Type", &debugPushConstant.showSHType, hasDataTypeNames, IM_ARRAYSIZE(hasDataTypeNames));
			uiModified |= ImGui::IsItemEdited();

			uiModified |= ImGui::Checkbox("show SH Sphere", &showSH);
			printSH = false;
			printSH = ImGui::Button("print SH");
		}
		if (ImGui::CollapsingHeader("GBuffers")) {
			static const char* const gBufferIndexNames[] = { "normal", "albedo", "specular", "directColor" };
			ImGui::Combo("normal", &debugPushConstant.GBufferIndex, gBufferIndexNames, IM_ARRAYSIZE(gBufferIndexNames));

			uiModified |= ImGui::Checkbox("show GBuffers", &showGBuffer);
		}
		if (ImGui::CollapsingHeader("RSM")) {
			ImGui::SliderInt("Light Index", &debugPushConstant.lightIndex, 0,
				std::max(int(Application::sceneResource.sceneInfo.numLights) - 1, 0));
			uiModified |= ImGui::Checkbox("show RSM", &showRSM);
		}
	}
	ImGui::End();

	if (uiModified) resetFrame();
}
//--------------------------------------Debug--------------------------------------
void LPVRenderer::createDeptImage(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	VkClearDepthStencilValue clearValue = DEFAULT_VkClearDepthStencilValue;
	VkImageSubresourceRange range = {};
	range.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
	range.baseMipLevel = 0;
	range.levelCount = 1;
	range.baseArrayLayer = 0;
	range.layerCount = 1;
	vkCmdClearDepthStencilImage(cmd, gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &range);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	const uint32_t groupCountX = (currentResolution.width + 15) / 16;
	const uint32_t groupCountY = (currentResolution.height + 15) / 16;
	dispatchCompute(cmd, shaderCreateDepthImage, groupCountX, groupCountY, 1);
}
void LPVRenderer::showGBufferResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	if (shaderShowGBuffer == VK_NULL_HANDLE)
		LOGE("LPV: showGBufferMain was not created, the G-buffer debug view stays empty\n");

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	const uint32_t groupCountX = (currentResolution.width + 15) / 16;
	const uint32_t groupCountY = (currentResolution.height + 15) / 16;
	dispatchCompute(cmd, shaderShowGBuffer, groupCountX, groupCountY, 1);
}
void LPVRenderer::showRSMResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	const uint32_t groupCountX = (currentResolution.width + 15) / 16;
	const uint32_t groupCountY = (currentResolution.height + 15) / 16;
	dispatchCompute(cmd, shaderShowRSM, groupCountX, groupCountY, 1);
}
void LPVRenderer::showSHResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	// Bail out before anything is recorded, so that no dynamic rendering instance is left open.
	if (shaderShowSH_vertex == VK_NULL_HANDLE || shaderShowSH_fragment == VK_NULL_HANDLE){
		LOGE("LPV: the SH probe shaders were not created, skipping the probe\n");
		return;
	}

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	{
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(ImageType_LPV::eImgRendered)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

		VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(ImageType_LPV::eImgRendered));

		VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depthAttachment.clearValue = { .depthStencil = DEFAULT_VkClearDepthStencilValue };
		//colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
		depthAttachment.imageView = gBuffers.getDepthImageView();

		VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
		renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;
		renderingInfo.pDepthAttachment = &depthAttachment;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		vkCmdBeginRendering(cmd, &renderingInfo);

		graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
		// A debug overlay is drawn regardless of the winding convention of the project's projection
		// matrix; culling back faces here silently removed the whole probe.
		graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
		graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
		graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;
		graphicsDynamicPipeline.depthStencilState.depthBoundsTestEnable = VK_FALSE;
		graphicsDynamicPipeline.cmdApplyAllStates(cmd);
		vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
		graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
		graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = shaderShowSH_vertex, .fragment = shaderShowSH_fragment });

		VkVertexInputBindingDescription2EXT bindingDescription{};
		VkVertexInputAttributeDescription2EXT attributeDescription = {};
		vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

	}

	constexpr uint32_t kProbeVertexCount = 8u * 4u * 2u * 3u;
	vkCmdDraw(cmd, kProbeVertexCount, debugVoxelCount(), 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(ImageType_LPV::eImgRendered)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}
void LPVRenderer::printSHResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	const uint32_t groupCountX = (debugVoxelCount() + 1023) / 1024;
	dispatchCompute(cmd, shaderPrintSH, groupCountX, 1, 1);
}
void LPVRenderer::showGridResult(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	pushInfo.pValues = &debugPushConstant;
	vkCmdPushConstants2(cmd, &pushInfo);

	vkCmdFillBuffer(cmd, gridDataBuffer.buffer, 0, gridDataBuffer.bufferSize, 0);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	logDebugViewCounts();
	int threadGroupCount = 0;
	if (debugPushConstant.gridDataType == 0) threadGroupCount = currentResolution.width * currentResolution.height;
	else if (debugPushConstant.gridDataType == 1) threadGroupCount = rsmResolution * rsmResolution;
	else if (debugPushConstant.gridDataType == 2) threadGroupCount = debugVoxelCount();
	threadGroupCount = (threadGroupCount + 1023) / 1024;
	dispatchCompute(cmd, shaderInjectGrid, threadGroupCount, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

	{
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(ImageType_LPV::eImgRendered)), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL });
		nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });

		VkRenderingAttachmentInfo colorAttachment = DEFAULT_VkRenderingAttachmentInfo;
		colorAttachment.imageView = gBuffers.getColorImageView(uint32_t(ImageType_LPV::eImgRendered));
		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

		VkRenderingAttachmentInfo depthAttachment = DEFAULT_VkRenderingAttachmentInfo;
		depthAttachment.imageView = gBuffers.getDepthImageView();
		depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;

		VkRenderingInfo renderingInfo = DEFAULT_VkRenderingInfo;
		renderingInfo.renderArea = { {0, 0}, gBuffers.getSize() };
		renderingInfo.colorAttachmentCount = 1;
		renderingInfo.pColorAttachments = &colorAttachment;
		renderingInfo.pDepthAttachment = &depthAttachment;

		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

		vkCmdBeginRendering(cmd, &renderingInfo);

		graphicsDynamicPipeline = nvvk::GraphicsPipelineState();
		graphicsDynamicPipeline.inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
		graphicsDynamicPipeline.rasterizationState.cullMode = VK_CULL_MODE_NONE;
		// No polygonMode = LINE / lineWidth = 2: those need fillModeNonSolid / wideLines, which this
		// renderer does not request. The topology already draws the box edges as lines.
		graphicsDynamicPipeline.depthStencilState.depthTestEnable = VK_TRUE;
		graphicsDynamicPipeline.depthStencilState.depthWriteEnable = VK_TRUE;

		graphicsDynamicPipeline.cmdApplyAllStates(cmd);
		// depth bounds is a dynamic state: it has to be set explicitly before drawing with shader objects
		vkCmdSetDepthBoundsTestEnable(cmd, VK_FALSE);
		graphicsDynamicPipeline.cmdSetViewportAndScissor(cmd, gBuffers.getSize());
		graphicsDynamicPipeline.cmdBindShaders(cmd, { .vertex = shaderShowGrid_vertex, .fragment = shaderShowGrid_fragment });

		VkVertexInputBindingDescription2EXT bindingDescription{};
		VkVertexInputAttributeDescription2EXT attributeDescription = {};
		vkCmdSetVertexInputEXT(cmd, 0, nullptr, 0, nullptr);

		// The box edges are generated in the vertex shader from SV_VertexID (12 edges, 24 vertices):
		// no mesh, no index buffer, no scene lookup.
	}

	// 12 edges x 2 vertices, see ShowGrid.slang.
	vkCmdDraw(cmd, 24, debugVoxelCount(), 0, 0);

	vkCmdEndRendering(cmd);

	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getColorImage(uint32_t(ImageType_LPV::eImgRendered)), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL });
	nvvk::cmdImageMemoryBarrier(cmd, { gBuffers.getDepthImage(), VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, {VK_IMAGE_ASPECT_DEPTH_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS} });
}