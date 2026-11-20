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
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

using namespace FzbRenderer;

namespace {

// Small local helper: uploads the whole push constant block (the renderer mutates a few fields
// between the dispatches of a single pass, exactly like VXPG does).
void lpvPushConstants(VkCommandBuffer cmd, VkPipelineLayout layout, const shaderio::LPVPushConstant& values)
{
	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = layout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LPVPushConstant),
		.pValues = &values,
	};
	vkCmdPushConstants2(cmd, &pushInfo);
}

glm::vec3 lpvUpVector(const glm::vec3& direction)
{
	return std::abs(direction.y) > 0.99f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(0.0f, 1.0f, 0.0f);
}

}  // namespace

//------------------------------------------------------------------------------------------------
// Construction / settings
//------------------------------------------------------------------------------------------------
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
		gridResolution    = readInt(node, "resolution", gridResolution, 8, LPV_MAX_GRID_RESOLUTION);
		rsmResolution     = readInt(node, "rsmResolution", rsmResolution, 16, LPV_MAX_RSM_RESOLUTION);
		iterationCount    = readInt(node, "iterations", iterationCount, 0, LPV_MAX_ITERATIONS);
		baseCellSize      = readFloat(node, "cellSize", baseCellSize, 0.0f, 1024.0f);
		useOcclusion      = readBool(node, "useOcclusion", useOcclusion);
		useDampening      = readBool(node, "useDampening", useDampening);
		saveFrame         = readInt(node, "saveFrame", saveFrame, -1, MAX_FRAME);
		debugMode         = readInt(node, "debugView", debugMode, 0, 5);
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
	readEnvInt("LPV_DEBUG", debugMode, 0, 5);
	readEnvInt("LPV_ITERATIONS", iterationCount, 0, LPV_MAX_ITERATIONS);
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

//------------------------------------------------------------------------------------------------
// Life cycle
//------------------------------------------------------------------------------------------------
void LPVRenderer::init()
{
	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	currentResolution = Application::renderResolution;
	// Nothing in this renderer rasterises, so no depth attachment is needed: attachment 0 is the
	// HDR image the compute passes write, attachment 1 the tonemapped one.
	Feature::createGBuffer(false, true, 1, Application::renderResolution);

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

	for (VkShaderEXT* shader : { &shaderGBuffer, &shaderRsmInject, &shaderGvMerge, &shaderPropagate, &shaderGather })
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

void LPVRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size)
{
	NVVK_CHECK(gBuffers.update(cmd, size));

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
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void LPVRenderer::preRender()
{
	Scene& scene = Application::sceneResource;

	if (scene.cameraChange)
		resetFrame();
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight)
		maxFrames = 1;

	updateSceneBounds();
	updateCascadeLayout();

	pushConstant.frameIndex       = Application::frameIndex;
	pushConstant.cascadeCount     = cascadeCount;
	pushConstant.gridResolution   = gridResolution;
	pushConstant.rsmResolution    = rsmResolution;
	pushConstant.screenSize       = shaderio::uint2(currentResolution.width, currentResolution.height);
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)scene.bSceneInfo.address;
	pushConstant.lightIndex       = 0;
	pushConstant.time             = scene.time;

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	asManager.updateToplevelAS(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);

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

void LPVRenderer::render(VkCommandBuffer* cmdPtr)
{
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	if (shaderGBuffer == VK_NULL_HANDLE || shaderGather == VK_NULL_HANDLE)
		return;

	bindStaticState(cmd);

	// Every frame is rebuilt from scratch: the LPV is a stateless, fully dynamic representation.
	clearVolumes(cmd);
	updateDataPerFrame(cmd);  // uploads the frame info and barriers transfer -> compute

	runGBufferPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runRsmInjectionPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runGvMergePass(cmd);

	// Seed the accumulation grid with the injected volume (the paper accumulates the injected
	// light plus every iteration, so the sum is the final light distribution).
	{
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT);
		VkBufferCopy2 copyRegion{
			.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
			.srcOffset = 0,
			.dstOffset = 0,
			.size = volumeFloatCount() * sizeof(float),
		};
		VkCopyBufferInfo2 copyInfo{
			.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
			.srcBuffer = lpvVolumesBuffer.buffer,
			.dstBuffer = lpvAccumBuffer.buffer,
			.regionCount = 1,
			.pRegions = &copyRegion,
		};
		vkCmdCopyBuffer2(cmd, &copyInfo);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}

	runPropagationPasses(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	runGatherPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
	                                   gBuffers.getDescriptorImageInfo((uint32_t)ImageType_LPV::eImgRendered),
	                                   gBuffers.getDescriptorImageInfo((uint32_t)ImageType_LPV::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                       VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
	                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	nvvk::cmdMemoryBarrier(cmd,
	                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	                       VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}

//------------------------------------------------------------------------------------------------
// Scene bounds / cascade placement
//------------------------------------------------------------------------------------------------
void LPVRenderer::updateSceneBounds()
{
	Scene& scene = Application::sceneResource;

	shaderio::float3 boundsMin(FLT_MAX, FLT_MAX, FLT_MAX);
	shaderio::float3 boundsMax(-FLT_MAX, -FLT_MAX, -FLT_MAX);
	instanceSizes.assign(scene.instances.size(), 1e30f);
	for (size_t instanceIndex = 0; instanceIndex < scene.instances.size(); ++instanceIndex)
	{
		const uint32_t meshIndex = scene.instances[instanceIndex].meshIndex;
		MeshInfo       meshInfo  = scene.getMeshInfo(meshIndex);
		shaderio::AABB meshAABB  = meshInfo.getAABB(scene.instances[instanceIndex].transform);

		boundsMin = glm::min(boundsMin, meshAABB.minimum);
		boundsMax = glm::max(boundsMax, meshAABB.maximum);

		// Paper 4.1: the size of an object is determined using its bounding box; objects smaller than
		// a cell of a grid are not rendered into that grid's RSM. Largest extent of the world AABB.
		const shaderio::float3 extent = meshAABB.maximum - meshAABB.minimum;
		instanceSizes[instanceIndex]  = glm::max(extent.x, glm::max(extent.y, extent.z));
	}

	if (scene.instances.empty() || boundsMin.x > boundsMax.x)
	{
		boundsMin = shaderio::float3(-5.0f, -5.0f, -5.0f);
		boundsMax = shaderio::float3(5.0f, 5.0f, 5.0f);
	}

	sceneBoundsMin = boundsMin;
	sceneBoundsMax = boundsMax;
	sceneCenter    = (boundsMin + boundsMax) * 0.5f;
	sceneRadius    = std::max(glm::length(boundsMax - boundsMin) * 0.5f, 0.1f);
}

void LPVRenderer::updateCascadeLayout()
{
	const auto      cameraManip = Application::sceneResource.cameraManip;
	const glm::vec3 eye         = cameraManip->getEye();
	glm::vec3       forward     = cameraManip->getCenter() - eye;
	if (glm::length(forward) < 1e-5f) forward = glm::vec3(0.0f, 0.0f, -1.0f);
	cameraForward = glm::normalize(forward);

	// A cell size of zero selects one derived from the scene size: the finest cascade then covers
	// roughly the bounding sphere of the scene, and the coarser ones extend the reach.
	float cellSize = baseCellSize;
	if (cellSize <= 0.0f) cellSize = std::max(sceneRadius * 2.0f, 0.2f) / float(std::max(gridResolution, 8));

	for (int cascade = cascadeCount - 1; cascade >= 0; --cascade) {
		CascadeLayout& layout = cascades[cascade];
		layout.cellSize = cellSize * float(1 >> cascade);
		layout.extent = layout.cellSize * float(gridResolution);

		shaderio::int3 cameraVoxelIndex = (eye - sceneBoundsMin) / cellSize;
		layout.origin = glm::floor(eye / layout.cellSize) * layout.cellSize;
	}
	for (int cascade = cascadeCount; cascade < LPV_MAX_CASCADES; ++cascade)
		cascades[cascade] = CascadeLayout{};
}

void LPVRenderer::updateFrameInfo()
{
	frameInfo = shaderio::LPVFrameInfo{};

	for (int cascade = 0; cascade < LPV_MAX_CASCADES; ++cascade)
	{
		const CascadeLayout& layout = cascades[cascade];
		frameInfo.cascadeOriginSize[cascade] = shaderio::float4(layout.origin, layout.cellSize);
		frameInfo.cascadeMisc[cascade] =
			shaderio::float4(layout.extent, layout.startDistance, 1.0f / std::max(layout.extent, 1e-3f), 0.0f);
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

	const float rsm = float(rsmResolution);
	for (uint32_t lightIndex = 0; lightIndex < LIGHT_COUNT; ++lightIndex)
	{
		frameInfo.lightViewProjInverse[lightIndex] = glm::mat4(1.0f);
		frameInfo.lightRsmParams[lightIndex]        = shaderio::float4(0.0f);

		if (lightIndex >= activeLightCount)
			continue;

		const shaderio::Light& light = scene.sceneInfo.lights[lightIndex];

		if (int(light.type) == int(shaderio::LightType::Direction))
		{
			glm::vec3 lightDirection = light.direction;
			if (glm::length(lightDirection) < 1e-5f)
				lightDirection = glm::vec3(0.0f, -1.0f, 0.0f);
			lightDirection = glm::normalize(lightDirection);

			const float halfExtent = sceneRadius * 1.05f;
			const float distance   = sceneRadius * 2.0f + 1.0f;
			const glm::vec3 lightEye = sceneCenter - lightDirection * distance;

			const glm::mat4 view       = glm::lookAt(lightEye, sceneCenter, lpvUpVector(lightDirection));
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

			float fovY = 2.0f * std::atan(sceneRadius * 1.15f / std::max(distance, sceneRadius * 0.2f));
			fovY       = std::clamp(fovY, 0.1f, 2.9f);

			const glm::mat4 view = glm::lookAt(light.pos, sceneCenter, lpvUpVector(direction));
			const glm::mat4 projection = glm::perspectiveRH_ZO(fovY, 1.0f, std::max(distance - sceneRadius * 1.2f, 0.02f),
			                                                   distance + sceneRadius * 2.5f + 1.0f);

			frameInfo.lightViewProjInverse[lightIndex] = glm::inverse(projection * view);

			const float texelAngle = 2.0f * std::tan(fovY * 0.5f) / rsm;
			frameInfo.lightRsmParams[lightIndex] = shaderio::float4(0.0f, texelAngle * texelAngle, 0.0f, 0.0f);
		}
	}
}

//------------------------------------------------------------------------------------------------
// GPU resources
//------------------------------------------------------------------------------------------------
VkDeviceSize LPVRenderer::volumeFloatCount() const
{
	const VkDeviceSize voxels = VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution);
	return VkDeviceSize(cascadeCount) * voxels * LPV_VOXEL_FLOATS;
}

void LPVRenderer::createResources()
{
	nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
	const VkBufferUsageFlags2 usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
	                                  VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;

	auto create = [&](nvvk::Buffer& buffer, VkDeviceSize size, const char* name) {
		NVVK_CHECK(allocator->createBuffer(buffer, size, usage));
		NVVK_DBG_NAME(buffer.buffer);
		LOGI("LPV: %s = %.2f MB\n", name, double(size) / (1024.0 * 1024.0));
	};

	const VkDeviceSize pixelCount  = VkDeviceSize(currentResolution.width) * VkDeviceSize(currentResolution.height);
	const VkDeviceSize voxelCount  = VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution) * VkDeviceSize(gridResolution);
	const VkDeviceSize volumeBytes = volumeFloatCount() * sizeof(float);
	const VkDeviceSize gvBytes     = VkDeviceSize(cascadeCount) * voxelCount * LPV_GV_FLOATS * sizeof(float);

	create(pixelDataBuffer, pixelCount * sizeof(shaderio::LPVPixelData), "pixelData");
	create(lpvVolumesBuffer, 2 * volumeBytes, "lpvVolumes");
	create(lpvAccumBuffer, volumeBytes, "lpvAccum");
	create(gvMergedBuffer, gvBytes, "gvMerged");
	create(gvCameraBuffer, gvBytes, "gvCamera");
	create(gvRsmBuffer, gvBytes, "gvRsm");
	create(frameInfoBuffer, sizeof(shaderio::LPVFrameInfo), "frameInfo");
	create(instanceSizeBuffer, VkDeviceSize(std::max<size_t>(instanceSizes.size(), 1)) * sizeof(float), "instanceSize");

	// One shot clear so that no partially built volume can ever be read.
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	clearVolumes(cmd);
	vkCmdFillBuffer(cmd, frameInfoBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}

void LPVRenderer::destroyResources()
{
	for (nvvk::Buffer* buffer : { &pixelDataBuffer, &lpvVolumesBuffer, &lpvAccumBuffer, &gvMergedBuffer, &gvCameraBuffer, &gvRsmBuffer,
	                              &frameInfoBuffer, &instanceSizeBuffer })
	{
		if (buffer->buffer != VK_NULL_HANDLE)
			Application::allocator.destroyBuffer(*buffer);
		*buffer = {};
	}
}

void LPVRenderer::clearVolumes(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	// lpvAccum is fully overwritten by the copy of the injected volume, so it is not cleared here.
	vkCmdFillBuffer(cmd, lpvVolumesBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, gvCameraBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, gvRsmBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
}

//------------------------------------------------------------------------------------------------
// Descriptors / pipeline layout / shaders
//------------------------------------------------------------------------------------------------
void LPVRenderer::createDescriptorSetLayout()
{
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
		(uint32_t)shaderio::LPVStaticBindingPoints::ePixelData,  (uint32_t)shaderio::LPVStaticBindingPoints::eLPVVolumes,
		(uint32_t)shaderio::LPVStaticBindingPoints::eLPVAccum,   (uint32_t)shaderio::LPVStaticBindingPoints::eGvMerged,
		(uint32_t)shaderio::LPVStaticBindingPoints::eGvCamera,   (uint32_t)shaderio::LPVStaticBindingPoints::eGvRsm,
		(uint32_t)shaderio::LPVStaticBindingPoints::eFrameInfo,  (uint32_t)shaderio::LPVStaticBindingPoints::eInstanceSize,
	};
	for (uint32_t binding : bufferBindings)
	{
		bindings.addBinding({ .binding = binding,
		                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      .descriptorCount = 1,
		                      .stageFlags = VK_SHADER_STAGE_ALL });
	}

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

void LPVRenderer::createDescriptorSet()
{
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
	appendBuffer(shaderio::LPVStaticBindingPoints::eLPVVolumes, lpvVolumesBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eLPVAccum, lpvAccumBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eGvMerged, gvMergedBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eGvCamera, gvCameraBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eGvRsm, gvRsmBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eFrameInfo, frameInfoBuffer);
	appendBuffer(shaderio::LPVStaticBindingPoints::eInstanceSize, instanceSizeBuffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void LPVRenderer::createPipelineLayout()
{
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LPVPushConstant),
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

void LPVRenderer::compileAndCreateShaders()
{
	SCOPED_TIMER(__FUNCTION__);

	VkDevice                device      = Application::app->getDevice();
	std::filesystem::path   shaderPath  = std::filesystem::path(__FILE__).parent_path() / "shaders";

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::LPVPushConstant),
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
		if (code.codeSize == 0 || code.pCode == nullptr)
			return;
		vkDestroyShaderEXT(device, shader, nullptr);
		shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderInfo.nextStage = 0;
		shaderInfo.pName = entryPoint;
		shaderInfo.codeSize = code.codeSize;
		shaderInfo.pCode = code.pCode;
		const VkResult result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader);
		if (result != VK_SUCCESS)
			LOGE("LPV: failed to create shader object '%s' (%d)\n", entryPoint, int(result));
		else
			NVVK_DBG_NAME(shader);
	};

	VkShaderModuleCreateInfo passesCode = FzbRenderer::compileSlangShader(shaderPath / "LPVPasses.slang", {});
	if (passesCode.codeSize > 0)
	{
		createShader(shaderGBuffer, passesCode, "lpvGBufferMain");
		createShader(shaderRsmInject, passesCode, "lpvRsmInjectMain");
		createShader(shaderGvMerge, passesCode, "lpvGvMergeMain");
		createShader(shaderPropagate, passesCode, "lpvPropagateMain");
		createShader(shaderGather, passesCode, "lpvGatherMain");
	}
	else
	{
		LOGE("LPV: LPVPasses.slang failed to compile, see the Slang diagnostics above\n");
	}

	//--------------------------------------------------------------------------------------
	{
		//std::filesystem::path shaderSource = shaderPath / "RSMDebug.slang";
		//VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});
		//
		//vkDestroyShaderEXT(device, computeShader_showRSM, nullptr);
		//
		//shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		//shaderInfo.nextStage = 0;
		//shaderInfo.pName = "computeMain_showRSM";
		//shaderInfo.codeSize = shaderCode.codeSize;
		//shaderInfo.pCode = shaderCode.pCode;
		//vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &computeShader_showRSM);
		//NVVK_DBG_NAME(computeShader_showRSM);
	}
}

//------------------------------------------------------------------------------------------------
// Per frame state
//------------------------------------------------------------------------------------------------
void LPVRenderer::updateDataPerFrame(VkCommandBuffer cmd)
{
	updateFrameInfo();
	vkCmdUpdateBuffer(cmd, frameInfoBuffer.buffer, 0, sizeof(shaderio::LPVFrameInfo), &frameInfo);
	// Paper 4.1 small object test: world space bounding box size of every instance. vkCmdUpdateBuffer
	// is limited to maxUpdateBufferSize (64 KiB), so only the first entries are uploaded.
	const size_t uploadCount = std::min<size_t>(instanceSizes.size(), 12288);
	if (uploadCount > 0)
		vkCmdUpdateBuffer(cmd, instanceSizeBuffer.buffer, 0, VkDeviceSize(uploadCount) * sizeof(float), instanceSizes.data());
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void LPVRenderer::bindStaticState(VkCommandBuffer cmd)
{
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	pushConstant.flags = int((useOcclusion ? uint32_t(shaderio::LPVFlags::eLPVFlagOcclusion) : 0u) |
	                     (useDampening ? uint32_t(shaderio::LPVFlags::eLPVFlagDampening) : 0u) |
	                     (showDirectLight ? 0u : uint32_t(shaderio::LPVFlags::eLPVFlagNoDirect)) |
	                     (uint32_t(debugMode) << 3) |
	                     (skipDirectShadow ? uint32_t(shaderio::LPVFlags::eLPVFlagNoShadow) : 0u));
	lpvPushConstants(cmd, pipelineLayout, pushConstant);
}

void LPVRenderer::dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY,
                                 uint32_t groupCountZ)
{
	if (shader == VK_NULL_HANDLE || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0)
		return;
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &shader);
	vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
}

//------------------------------------------------------------------------------------------------
// Passes
//------------------------------------------------------------------------------------------------
void LPVRenderer::runGBufferPass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	lpvPushConstants(cmd, pipelineLayout, pushConstant);
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
		lpvPushConstants(cmd, pipelineLayout, pushConstant);
		dispatchCompute(cmd, shaderRsmInject, groupCount, groupCount, 1);
		// Both dispatches accumulate into the same volumes with atomics.
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

void LPVRenderer::runGvMergePass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	lpvPushConstants(cmd, pipelineLayout, pushConstant);
	const uint32_t resolution    = uint32_t(gridResolution);
	const uint32_t groupCountXZ  = resolution / LPV_VOXEL_THREADGROUP;
	const uint32_t groupCountY   = (resolution * uint32_t(cascadeCount)) / LPV_VOXEL_THREADGROUP;
	dispatchCompute(cmd, shaderGvMerge, groupCountXZ, groupCountXZ, groupCountY);
}

void LPVRenderer::runPropagationPasses(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	const uint32_t resolution   = uint32_t(gridResolution);
	const uint32_t groupCountXY = resolution / LPV_VOXEL_THREADGROUP;
	const uint32_t groupCountZ  = (resolution * uint32_t(cascadeCount)) / LPV_VOXEL_THREADGROUP;

	for (int iteration = 0; iteration < iterationCount; ++iteration)
	{
		pushConstant.iteration = iteration;
		lpvPushConstants(cmd, pipelineLayout, pushConstant);
		dispatchCompute(cmd, shaderPropagate, groupCountXY, groupCountXY, groupCountZ);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	}
}

void LPVRenderer::runGatherPass(VkCommandBuffer cmd)
{
	NVVK_DBG_SCOPE(cmd);
	// The gather shader ignores the iteration; the injection passes are done by now.
	pushConstant.iteration = 0;
	lpvPushConstants(cmd, pipelineLayout, pushConstant);
	const uint32_t groupCountX = (currentResolution.width + LPV_PIXEL_THREADGROUP_X - 1) / LPV_PIXEL_THREADGROUP_X;
	const uint32_t groupCountY = (currentResolution.height + LPV_PIXEL_THREADGROUP_Y - 1) / LPV_PIXEL_THREADGROUP_Y;
	dispatchCompute(cmd, shaderGather, groupCountX, groupCountY, 1);
}

//------------------------------------------------------------------------------------------------
// UI
//------------------------------------------------------------------------------------------------
void LPVRenderer::uiRender()
{
	bool uiModified = Application::UIModified;

	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_LPV::eImgTonemapped);

	if (ImGui::Begin("LPVSettings"))
	{
		ImGui::SeparatorText("Light Propagation Volumes");
		ImGui::TextDisabled("Frame: %d", Application::frameIndex);
		ImGui::TextDisabled("Cascades: %d x %d^3 = %d voxels", cascadeCount, gridResolution, cascadeCount * gridResolution * gridResolution * gridResolution);
		if (cascadeCount > 0)
		{
			ImGui::TextDisabled("Finest cell: %.3f m, coverage: %.2f m", cascades[0].cellSize,
			                    cascades[cascadeCount - 1].startDistance + cascades[cascadeCount - 1].extent);
		}
		ImGui::TextDisabled("Iterations: %d (0 = injection only, no propagation), virtual RSM: %d^2 per light", iterationCount, rsmResolution);

		ImGui::SeparatorText("Indirect lighting");
		uiModified |= ImGui::Checkbox("Fuzzy occlusion", &useOcclusion);
		uiModified |= ImGui::Checkbox("Derivative dampening", &useDampening);
		uiModified |= ImGui::Checkbox("Show Directional Light", &showDirectLight);
		if (ImGui::Combo("Debug view", &debugMode, "Off\0Cascades\0Direct only\0Indirect only\0Reference GI\0RSM\0"))
			uiModified = true;
		uiModified |= ImGui::SliderInt("Propagation iterations", &iterationCount, 0, 16);
		ImGui::TextDisabled("Everything else (half cell offsets, 5 exit faces, cascade blend) follows the paper.");

		ImGui::SeparatorText("Grids");
		int  newCascadeCount = cascadeCount;
		int  newResolution   = gridResolution;
		int  newRsmResolution = rsmResolution;
		bool rebuild         = false;
		ImGui::SliderInt("Cascade count", &newCascadeCount, 1, LPV_MAX_CASCADES);
		ImGui::SliderInt("Grid resolution", &newResolution, 8, 96);
		ImGui::SliderInt("RSM resolution", &newRsmResolution, 32, 1024);
		if (ImGui::Button("Rebuild volumes"))
		{
			cascadeCount   = newCascadeCount;
			gridResolution = std::max(LPV_VOXEL_THREADGROUP, (newResolution / LPV_VOXEL_THREADGROUP) * LPV_VOXEL_THREADGROUP);
			rsmResolution  = std::max(16, (newRsmResolution / 16) * 16);
			rebuild        = true;
		}
		if (rebuild)
		{
			vkDeviceWaitIdle(Application::app->getDevice());
			destroyResources();
			createResources();
			createDescriptorSet();
			updateCascadeLayout();
			pushConstant.cascadeCount   = cascadeCount;
			pushConstant.gridResolution = gridResolution;
			pushConstant.rsmResolution  = rsmResolution;
			resetFrame();
		}

		if (ImGui::Button("Save snapshot"))
			saveFrame = Application::frameIndex + 2;

		ImGui::SeparatorText("Diagnostics");
		ImGui::TextDisabled("Buffers: volumes %.1f MB, geometry %.1f MB",
		                    double(2 * volumeFloatCount() * sizeof(float)) / (1024.0 * 1024.0),
		                    double(3 * VkDeviceSize(cascadeCount) * VkDeviceSize(gridResolution) * gridResolution * gridResolution *
		                           LPV_GV_FLOATS * sizeof(float)) /
		                        (1024.0 * 1024.0));

		if (ImGui::CollapsingHeader("RSM", ImGuiTreeNodeFlags_DefaultOpen)) {
			// Same switch as the "RSM" entry of the debug view combo: the RSM is drawn by the gather
			// pass (one screen pixel per texel), so it needs no separate render path.
			bool showRsm = debugMode == (int)shaderio::LPVDebugView::eLPVDebugRsm;
			if (ImGui::Checkbox("Show RSM", &showRsm))
			{
				debugMode = showRsm ? (int)shaderio::LPVDebugView::eLPVDebugRsm : 0;
				uiModified = true;
			}
		}
	}
	ImGui::End();

	if (uiModified)
		resetFrame();
}
