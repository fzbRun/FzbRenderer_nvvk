#include "./VXPG.h"

#include <common/Application/Application.h>
#include <common/Shader/Shader.h>
#include <nvvk/compute_pipeline.hpp>
#include <nvgui/property_editor.hpp>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <fstream>
#include <sstream>

using namespace FzbRenderer;

namespace {

uint32_t vxpgNextPowerOfTwo(uint32_t value)
{
	uint32_t result = 1;
	while (result < value)
		result <<= 1;
	return result;
}

uint32_t vxpgStageMask()
{
	const char* value = std::getenv("VXPG_STAGE_MASK");
	return value ? uint32_t(std::strtoul(value, nullptr, 0)) : 0xffffffffu;
}

uint32_t vxpgLog2(uint32_t value)
{
	uint32_t result = 0;
	while ((1u << result) < value)
		++result;
	return result;
}

}  // namespace

VXPGRenderer::VXPGRenderer(pugi::xml_node& rendererNode)
{
	// Ray tracing extensions must be registered before the Vulkan device is created.
	ptContext.setContextInfo();

	// The light tree intensities are accumulated bottom up with float atomics (see
	// vxpgTreeAccumulateMain). The device pNext chain may contain the feature struct only once.
	atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
	atomicFloatFeatures.shaderBufferFloat32AtomicAdd = VK_TRUE;
	bool atomicFloatAdded = false;
	for (const auto& extension : Application::vkContextInitInfo.deviceExtensions)
	{
		if (extension.extensionName != nullptr &&
		    std::strcmp(extension.extensionName, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME) == 0)
		{
			atomicFloatAdded = true;
			break;
		}
	}
	if (!atomicFloatAdded)
		Application::vkContextInitInfo.deviceExtensions.push_back(
			{ VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME, &atomicFloatFeatures });

	if (pugi::xml_node node = rendererNode.child("maxDepth"))
		pushConstant.maxDepth = std::max(1, std::stoi(node.attribute("value").value()));
	if (pugi::xml_node node = rendererNode.child("spp"))
		pushConstant.spp = std::max(1, std::stoi(node.attribute("value").value()));

	pushConstant.mode = int(shaderio::VXPG_SampleMode::eGuidePlusBsdf);
	// 0 = one sample MIS (no per voxel pdf weight), 1 = balance, 2 = power.
	pushConstant.misMode = 0;
	pushConstant.visibilityMode = int(shaderio::VXPG_VisibilityMode::eSpixel);
	pushConstant.bounceCount = 1;

	// Sane defaults: the renderer must work with a rendererInfo.xml block that has no <VXPG>
	// settings at all (a zero would mean a 921.600 entry superpixel grid, i.e. ~400 MB of
	// buffers and a pass grid that never finishes).
#ifdef NDEBUG
	constexpr int defaultVoxelResolution = 64;
#else
	// Debug builds run unoptimised shaders: keep the voxel grid (and therefore the light tree)
	// small enough that a frame stays well below the GPU watchdog.
	constexpr int defaultVoxelResolution = 32;
#endif
	pushConstant.voxelResolution = defaultVoxelResolution;
	pushConstant.spixelSize = 32;
	// Jitter ranges for debugging (exposed in the panel):
	//   jitterRadius   guide sampling SP jitter, in voxel sizes
	//   padding0       superpixel lookup jitter, in pixels
	//   padding1       light injection VPL position jitter, in voxel sizes
	pushConstant.jitterRadius = 0.5f;
	pushConstant.padding0 = 8.0f;
	pushConstant.padding1 = 0.0f;   // SV injection jitter, off by default (it costs light)
	pushConstant.fireflyClamp = 0.0f;
	pushConstant.bounceCount = 1;
	// Build the guiding structure once instead of every frame: it removes the per frame
	// flickering of the voxel sized artifacts (the injected set is re-randomised every frame
	// otherwise) and it is by far the cheapest option for the GPU watchdog. The UI toggle
	// "Rebuild Guiding" switches it back to per frame.
	rebuildPerFrame = true;
	// 0 = one sample MIS (no per voxel pdf weight), 1 = balance, 2 = power.
	pushConstant.misMode = 0;

	// Read the settings defensively: a missing or malformed value must never throw inside the
	// renderer constructor (std::stoi would terminate the process).
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
			LOGW("VXPG: invalid value \"%s\" for <%s>, using %d\n", text, name, fallback);
			return fallback;
		}
	};

	if (pugi::xml_node node = rendererNode.child("VXPG"))
	{
		pushConstant.voxelResolution = readInt(node, "voxelResolution", defaultVoxelResolution, 8, VXPG_MAX_VOXEL_RESOLUTION);
		pushConstant.spixelSize = readInt(node, "spixelSize", 32, 4, 256);
		pushConstant.bounceCount = readInt(node, "bounce", 1, 0, 8);
		pushConstant.misMode = readInt(node, "misMode", 0, 0, 2);
		maxFrames = readInt(node, "maxFrames", maxFrames, 1, MAX_FRAME);
		if (pugi::xml_node value = node.child("rebuildPerFrame"))
			rebuildPerFrame = std::string(value.attribute("value").value()) != "false";
	}

	if (const char* debug = std::getenv("VXPG_DEBUG"))
		debugMode = std::atoi(debug);
	if (const char* v = std::getenv("VXPG_SP_JITTER"))
		pushConstant.jitterRadius = std::stof(v);
	if (const char* v = std::getenv("VXPG_SPIXEL_JITTER"))
		pushConstant.padding0 = std::stof(v);
	if (const char* v = std::getenv("VXPG_SV_JITTER"))
		pushConstant.padding1 = std::stof(v);

}

//-------------------------------------------------------------------------------------------------
// Resolution / scene bounds
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::computeResolution()
{
	uint32_t resolution = std::clamp<uint32_t>(uint32_t(std::max(pushConstant.voxelResolution, 8)), 8u,
	                                           uint32_t(VXPG_MAX_VOXEL_RESOLUTION));
	resolution = vxpgNextPowerOfTwo(resolution);

	pushConstant.voxelResolution = int(resolution);
	pushConstant.maxTreeLevel = int(vxpgLog2(resolution));
	pushConstant.spixelSize = std::max(pushConstant.spixelSize, 1);
	pushConstant.clusterCount = VXPG_CLUSTER_COUNT;
	pushConstant.flags = (enableGuiding ? 1 : 0) | (debugMode << 1);

	voxelCount = resolution * resolution * resolution;
	leafCount = voxelCount;
	nodeCount = 2u * leafCount - 1u;

	pushConstant.spixelSize = std::clamp(pushConstant.spixelSize, 4, 256);

	// Cap the superpixel grid: the per superpixel top level tree (64 floats) and the per
	// superpixel cluster weights grow with it, and a tiny superpixel size (e.g. 1 pixel) would
	// mean a 921.600 entry grid and ~400 MB of buffers.
	auto gridWidthFor = [&](int spixelSize) {
		return std::max(1u, (currentResolution.width + uint32_t(spixelSize) - 1u) / uint32_t(spixelSize));
	};
	auto gridHeightFor = [&](int spixelSize) {
		return std::max(1u, (currentResolution.height + uint32_t(spixelSize) - 1u) / uint32_t(spixelSize));
	};
	while (gridWidthFor(pushConstant.spixelSize) * gridHeightFor(pushConstant.spixelSize) > 8192u &&
	       pushConstant.spixelSize < 256)
	{
		pushConstant.spixelSize *= 2;
		LOGW("VXPG: superpixel size raised to %d to keep the superpixel grid bounded\n", pushConstant.spixelSize);
	}

	spixelGridWidth = gridWidthFor(pushConstant.spixelSize);
	spixelGridHeight = gridHeightFor(pushConstant.spixelSize);
	spixelCount = spixelGridWidth * spixelGridHeight;
	pushConstant.spixelCount = int(spixelCount);
}

void VXPGRenderer::updateSceneBounds()
{
	FzbRenderer::Scene& scene = Application::sceneResource;

	glm::vec3 minimum(FLT_MAX);
	glm::vec3 maximum(-FLT_MAX);
	for (const shaderio::Instance& instance : scene.instances)
	{
		MeshInfo meshInfo = scene.getMeshInfo(instance.meshIndex);
		shaderio::AABB meshAABB = meshInfo.getAABB(instance.transform);
		minimum.x = std::min(minimum.x, meshAABB.minimum.x);
		minimum.y = std::min(minimum.y, meshAABB.minimum.y);
		minimum.z = std::min(minimum.z, meshAABB.minimum.z);
		maximum.x = std::max(maximum.x, meshAABB.maximum.x);
		maximum.y = std::max(maximum.y, meshAABB.maximum.y);
		maximum.z = std::max(maximum.z, meshAABB.maximum.z);
	}
	if (minimum.x > maximum.x)
	{
		minimum = glm::vec3(-1.0f);
		maximum = glm::vec3(1.0f);
	}

	// Same construction as the reference VoxelTexInfo: a cube that covers the scene AABB.
	glm::vec3 extent = maximum - minimum;
	float extentMax = std::max(std::max(extent.x, extent.y), extent.z) * 0.5f + 0.01f;
	glm::vec3 center = (maximum + minimum) * 0.5f;

	// The voxel domain is a camera centred cube (the reference implementation's VXGI clipmap).
	// A whole scene AABB at 64^3 gives ~16 cm voxels and the guided estimate then shows the
	// voxel structure as visible squares; a clipmap keeps the voxels small where the camera is.
	// NOTE: a *smaller* clipmap would give finer voxels (and smaller blocks), but it also
	// shrinks the injected set; the value that reproduces the previous, verified behaviour is the
	// scene AABB half extent. Override it with <clipExtent value="..."/> to explore smaller domains.
	autoClipExtent = extentMax;
	sceneCenter = shaderio::float3(center);
	updateVoxelDomain();
}

// Places the cubic voxel domain around the camera and flags the guiding structure as dirty when
// the domain moved by more than half a voxel.
void VXPGRenderer::updateVoxelDomain()
{
	const float extent = (clipExtentSetting > 0.0f) ? clipExtentSetting : autoClipExtent;

	sceneExtent = extent;
	sceneStartPos = shaderio::float3(sceneCenter - shaderio::float3(extent, extent, extent));
	sceneVoxelSize = shaderio::float3(2.0f * extent / float(pushConstant.voxelResolution));

	pushConstant.VGBStartPos = sceneStartPos;
	pushConstant.VGBExtent = sceneExtent;
	pushConstant.VGBVoxelSize = sceneVoxelSize;

	const float voxel = sceneVoxelSize.x;
	if (domainExtentBuild <= 0.0f || std::abs(extent - domainExtentBuild) > 0.25f * voxel)
	{
		domainCenterBuild = glm::vec3(sceneCenter.x, sceneCenter.y, sceneCenter.z);
		domainExtentBuild = extent;
		guidingDirty = true;
	}
}

//-------------------------------------------------------------------------------------------------
// GPU resources
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::createResources()
{
	nvvk::ResourceAllocator* allocator = Application::stagingUploader.getResourceAllocator();
	const VkBufferUsageFlags2 usage = VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_TRANSFER_DST_BIT |
	                                  VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT;

	auto create = [&](nvvk::Buffer& buffer, VkDeviceSize size, const char* name) {
		allocator->createBuffer(buffer, size, usage);
		NVVK_DBG_NAME(buffer.buffer);
		LOGI("VXPG: %s = %.2f MB\n", name, double(size) / (1024.0 * 1024.0));
	};

	const VkDeviceSize pixelCount = VkDeviceSize(currentResolution.width) * VkDeviceSize(currentResolution.height);

	create(voxelIrradianceBuffer, VkDeviceSize(voxelCount) * sizeof(uint32_t), "voxelIrradiance");
	create(voxelVplCountBuffer, VkDeviceSize(voxelCount) * sizeof(uint32_t), "voxelVplCount");
	create(voxelBoundMinBuffer, VkDeviceSize(voxelCount) * 4 * sizeof(uint32_t), "voxelBoundMin");
	create(voxelBoundMaxBuffer, VkDeviceSize(voxelCount) * 4 * sizeof(uint32_t), "voxelBoundMax");
	create(pixelDataBuffer, pixelCount * sizeof(shaderio::VXPGPixelData), "pixelData");
	create(treeNodesBuffer, VkDeviceSize(nodeCount) * sizeof(shaderio::VXPGTreeNode), "treeNodes");
	create(clusterRootsBuffer, VkDeviceSize(VXPG_CLUSTER_COUNT) * sizeof(int32_t), "clusterRoots");
	create(topLevelTreeBuffer, VkDeviceSize(spixelCount) * VXPG_TOP_LEVEL_STRIDE * sizeof(float), "topLevelTree");
	create(spixelInfoBuffer, VkDeviceSize(spixelCount) * sizeof(shaderio::VXPGSpixelInfo), "spixelInfo");
	create(spixelClusterWeightBuffer, VkDeviceSize(spixelCount) * VXPG_CLUSTER_COUNT * sizeof(int32_t), "spixelClusterWeight");
	create(spixelClusterTotalBuffer, VkDeviceSize(spixelCount) * sizeof(int32_t), "spixelClusterTotal");

	// One shot initialization so that a partially built guiding structure can never be
	// dereferenced with uninitialized indices.
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	clearVoxelGrids(cmd);
	vkCmdFillBuffer(cmd, treeNodesBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, clusterRootsBuffer.buffer, 0, VK_WHOLE_SIZE, 0xffffffffu);
	vkCmdFillBuffer(cmd, topLevelTreeBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	Application::app->submitAndWaitTempCmdBuffer(cmd);
}

void VXPGRenderer::destroyResources()
{
	struct BufferRef
	{
		nvvk::Buffer* buffer;
	};
	const BufferRef buffers[] = {
		{ &voxelIrradianceBuffer },      { &voxelVplCountBuffer },     { &voxelBoundMinBuffer },
		{ &voxelBoundMaxBuffer },        { &pixelDataBuffer },         { &treeNodesBuffer },
		{ &clusterRootsBuffer },         { &topLevelTreeBuffer },      { &spixelInfoBuffer },
		{ &spixelClusterWeightBuffer },  { &spixelClusterTotalBuffer },
	};
	for (const BufferRef& ref : buffers)
	{
		if (ref.buffer->buffer != VK_NULL_HANDLE)
			Application::allocator.destroyBuffer(*ref.buffer);
		*ref.buffer = {};
	}
}

//-------------------------------------------------------------------------------------------------
// Renderer lifecycle
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::init()
{
	ptContext.getRayTracingPropertiesAndFeature();
	asManager.init();
	sbtGenerator.init(Application::app->getDevice(), ptContext.rtProperties);

	IF_DEBUG(Feature::createGBuffer(true, true, (uint32_t)ImageType_VXPG::eImgTonemapped, Application::renderResolution),
	         Feature::createGBuffer(false, true, 1, Application::renderResolution));

	currentResolution = Application::renderResolution;
	computeResolution();
	updateSceneBounds();
	createResources();

	createDescriptorSetLayout();
	createDescriptorSet();
	createPipelineLayout();
	compileAndCreateShaders();

	Renderer::init();
}

void VXPGRenderer::clean()
{
	VkDevice device = Application::app->getDevice();
	for (VkShaderEXT* shader : { &shaderVBuffer, &shaderLightInject, &shaderGeometryInject, &shaderTreeInit,
	                             &shaderTreeInternal, &shaderTreeAccumulate, &shaderClusterRoots, &shaderSpixelCenters,
	                             &shaderSpixelAssign, &shaderSpixelAccum, &shaderTopLevel, &shaderIntegrator })
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

void VXPGRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size)
{
	NVVK_CHECK(gBuffers.update(cmd, size));

	const bool resolutionChanged = (size.width != currentResolution.width) || (size.height != currentResolution.height);

	if (resolutionChanged)
	{
		destroyResources();
		currentResolution = size;
		computeResolution();
		updateSceneBounds();
		createResources();
		createDescriptorSet();
	}

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet outImageWrite = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(outImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VXPG::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void VXPGRenderer::preRender()
{
	Scene& scene = Application::sceneResource;
	if (scene.cameraChange)
		resetFrame();
	if (scene.periodInstanceCount + scene.randomInstanceCount > 0 || scene.hasDynamicLight)
		maxFrames = 1;

	pushConstant.frameIndex = Application::frameIndex;
	pushConstant.randomFrameIndex = int(Application::randomFrameIndex);
	pushConstant.maxFrameCount = maxFrames;
	pushConstant.time = scene.time;
	pushConstant.mode = int(shaderio::VXPG_SampleMode::eGuidePlusBsdf);
	pushConstant.flags = (enableGuiding ? 1 : 0) | (debugMode << 1);
	pushConstant.bounceCount = std::clamp(pushConstant.maxDepth - 2, 0, 8);
	// Radius of the camera centred injection volume (2 scene half extents by default).
	updateVoxelDomain();
	pushConstant.sceneInfoAddress = (shaderio::SceneInfo*)scene.bSceneInfo.address;

	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();
	asManager.updateToplevelAS(cmd);
	Application::app->submitAndWaitTempCmdBuffer(cmd);

	// Debug helper: VXPG_DUMP_FRAME=<frame> prints a summary of the guiding buffers.
	if (const char* dumpFrame = std::getenv("VXPG_DUMP_FRAME"))
		if (Application::frameIndex == std::atoi(dumpFrame))
			dumpGuidingState();

	// Debug helper: VXPG_SAVE_FRAME=<frame> stores the previous frame's tonemapped image.
	const char* saveFrame = std::getenv("VXPG_SAVE_FRAME");
	if (saveFrame != nullptr && Application::frameIndex == std::atoi(saveFrame))
	{
		const char* saveName = std::getenv("VXPG_SAVE_NAME");
		std::filesystem::path path = std::filesystem::current_path() / (saveName ? saveName : "vxpg_frame.png");
		Application::app->saveImageToFile(gBuffers.getColorImage((uint32_t)ImageType_VXPG::eImgTonemapped),
		                                  gBuffers.getSize(), path.string());
		LOGI("VXPG: wrote %s\n", path.string().c_str());
	}
}

void VXPGRenderer::updateDataPerFrame(VkCommandBuffer cmd) {}

namespace {
// Flat voxel index -> voxel coordinate (diagnostics only).
glm::ivec3 vxpgVoxelFromFlat(uint32_t flat, uint32_t resolution)
{
	return glm::ivec3(int(flat % resolution), int((flat / resolution) % resolution), int(flat / (resolution * resolution)));
}
}  // namespace

// Debug helper: reads the guiding buffers back to the host and writes a summary to a text
// file (VXPG_DUMP_FRAME selects the frame, VXPG_DUMP_FILE the file name).
void VXPGRenderer::dumpGuidingState()
{
	struct Readback
	{
		nvvk::Buffer stage{};
	};

	std::vector<Readback> reads;
	VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

	auto enqueue = [&](nvvk::Buffer& source, VkDeviceSize offset, VkDeviceSize size) {
		Readback read{};
		Application::allocator.createBuffer(read.stage, size,
			VK_BUFFER_USAGE_2_TRANSFER_DST_BIT | VK_BUFFER_USAGE_2_TRANSFER_SRC_BIT,
			VMA_MEMORY_USAGE_GPU_TO_CPU, VMA_ALLOCATION_CREATE_MAPPED_BIT);
		VkBufferCopy2 region{ .sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2, .srcOffset = offset, .size = size };
		VkCopyBufferInfo2 info{ .sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2, .srcBuffer = source.buffer,
			                    .dstBuffer = read.stage.buffer, .regionCount = 1, .pRegions = &region };
		vkCmdCopyBuffer2(cmd, &info);
		reads.push_back(read);
	};

	const VkDeviceSize treeBytes = VkDeviceSize(nodeCount) * sizeof(shaderio::VXPGTreeNode);
	const VkDeviceSize pixelOffset = VkDeviceSize(currentResolution.width) * 360u * sizeof(shaderio::VXPGPixelData);

	enqueue(clusterRootsBuffer, 0, VkDeviceSize(VXPG_CLUSTER_COUNT) * sizeof(int32_t));
	enqueue(treeNodesBuffer, 0, treeBytes);
	enqueue(topLevelTreeBuffer, 0, VkDeviceSize(spixelCount) * VXPG_TOP_LEVEL_STRIDE * sizeof(float));
	enqueue(spixelClusterTotalBuffer, 0, VkDeviceSize(spixelCount) * sizeof(int32_t));
	enqueue(spixelClusterWeightBuffer, 0, VkDeviceSize(spixelCount) * VXPG_CLUSTER_COUNT * sizeof(int32_t));
	enqueue(voxelVplCountBuffer, 0, VkDeviceSize(voxelCount) * sizeof(uint32_t));
	enqueue(pixelDataBuffer, pixelOffset, VkDeviceSize(4) * sizeof(shaderio::VXPGPixelData));

	Application::app->submitAndWaitTempCmdBuffer(cmd);

	auto mapped = [&](size_t index) { return static_cast<const uint8_t*>(reads[index].stage.mapping); };
	const int* clusterRootsHost = reinterpret_cast<const int*>(mapped(0));
	const shaderio::VXPGTreeNode* nodes = reinterpret_cast<const shaderio::VXPGTreeNode*>(mapped(1));
	const float* topLevel = reinterpret_cast<const float*>(mapped(2));
	const int* totals = reinterpret_cast<const int*>(mapped(3));
	const int* weights = reinterpret_cast<const int*>(mapped(4));
	const uint32_t* vpl = reinterpret_cast<const uint32_t*>(mapped(5));
	const shaderio::VXPGPixelData* pixels = reinterpret_cast<const shaderio::VXPGPixelData*>(mapped(6));

	uint64_t litVoxels = 0, totalVpls = 0;
	uint32_t maxVpl = 0;
	for (uint32_t i = 0; i < voxelCount; ++i)
	{
		if (vpl[i] != 0u)
		{
			++litVoxels;
			totalVpls += vpl[i];
			maxVpl = std::max(maxVpl, vpl[i]);
		}
	}

	uint32_t spixelsWithTotal = 0;
	int64_t sumTotals = 0;
	for (uint32_t i = 0; i < spixelCount; ++i)
	{
		if (totals[i] > 0)
			++spixelsWithTotal;
		sumTotals += totals[i];
	}

	std::ostringstream out;
	out << "frame " << pushConstant.frameIndex << " resolution " << currentResolution.width << "x" << currentResolution.height
	    << " voxels " << pushConstant.voxelResolution << "^3 nodes " << nodeCount << " spixels " << spixelCount << "\n";
	const FzbRenderer::Scene& scene = Application::sceneResource;
	out << "lights=" << scene.sceneInfo.numLights << " useSky=" << scene.sceneInfo.useSky
	    << " bg=(" << scene.sceneInfo.backgroundColor.x << " " << scene.sceneInfo.backgroundColor.y << " "
	    << scene.sceneInfo.backgroundColor.z << ")\n";
	for (int i = 0; i < scene.sceneInfo.numLights && i < 2; ++i)
	{
		const shaderio::Light& light = scene.sceneInfo.lights[i];
		out << "  light " << i << " type=" << light.type << " intensity=" << light.intensity
		    << " color=(" << light.color.x << " " << light.color.y << " " << light.color.z << ")"
		    << " dir=(" << light.direction.x << " " << light.direction.y << " " << light.direction.z << ")"
		    << " pos=(" << light.pos.x << " " << light.pos.y << " " << light.pos.z << ")\n";
	}

	// Spatial extent of the injected VPLs (world space), which tells which surfaces the
	// guiding distribution actually covers.
	glm::ivec3 litMin(INT_MAX), litMax(INT_MIN);
	for (uint32_t i = 0; i < voxelCount; ++i)
	{
		if (vpl[i] == 0u)
			continue;
		glm::ivec3 v = vxpgVoxelFromFlat(i, uint32_t(pushConstant.voxelResolution));
		litMin = glm::ivec3(std::min(litMin.x, v.x), std::min(litMin.y, v.y), std::min(litMin.z, v.z));
		litMax = glm::ivec3(std::max(litMax.x, v.x), std::max(litMax.y, v.y), std::max(litMax.z, v.z));
	}
	if (litVoxels > 0)
	{
		const glm::vec3 sceneMin = sceneStartPos;
		out << "lit voxel range=" << litMin.x << "," << litMin.y << "," << litMin.z << " .. " << litMax.x << ","
		    << litMax.y << "," << litMax.z << " world=(" << sceneMin.x + float(litMin.x) * sceneVoxelSize.x << " "
		    << sceneMin.y + float(litMin.y) * sceneVoxelSize.y << " " << sceneMin.z + float(litMin.z) * sceneVoxelSize.z
		    << ") .. (" << sceneMin.x + float(litMax.x + 1) * sceneVoxelSize.x << " "
		    << sceneMin.y + float(litMax.y + 1) * sceneVoxelSize.y << " "
		    << sceneMin.z + float(litMax.z + 1) * sceneVoxelSize.z << ")\n";
	}
	out << "litVoxels=" << litVoxels << " totalVpls=" << totalVpls << " maxVpl=" << maxVpl
	    << " spixelsWithTotal=" << spixelsWithTotal << "/" << spixelCount << " sumTotals=" << sumTotals << "\n";
	out << "rootIntensity=" << nodes[0].intensity << " rootFlag=0x" << std::hex << nodes[0].flag << std::dec
	    << " rootLeft=" << nodes[0].leftIdx << " rootRight=" << nodes[0].rightIdx << "\n";
	for (int c = 0; c < VXPG_CLUSTER_COUNT; ++c)
	{
		const int root = clusterRootsHost[c];
		out << "cluster " << c << " root=" << root
		    << " intensity=" << ((root >= 0 && root < int(nodeCount)) ? nodes[root].intensity : -1.0f) << "\n";
	}
	out << "topLevel[1]=" << topLevel[1] << " leaves:";
	for (int c = 0; c < VXPG_CLUSTER_COUNT; ++c)
		out << " " << topLevel[VXPG_TOP_LEVEL_LEAF_OFFSET + c];
	out << "\nspixel0 weights:";
	for (int c = 0; c < VXPG_CLUSTER_COUNT; ++c)
		out << " " << weights[c];
	out << "\npixel(row 360, col 0) valid=" << pixels[0].info.w << " spixel=" << int(int32_t(pixels[0].info.x))
	    << " primaryVoxel=" << int(int32_t(pixels[0].info.y)) << " vplVoxel=" << int(int32_t(pixels[0].info.z)) << "\n";
	out << "pixel(row 360, col 1) valid=" << pixels[1].info.w << " spixel=" << int(int32_t(pixels[1].info.x))
	    << " primaryVoxel=" << int(int32_t(pixels[1].info.y)) << " vplVoxel=" << int(int32_t(pixels[1].info.z)) << "\n";
	out << "pixel(row 360, col 2) valid=" << pixels[2].info.w << " spixel=" << int(int32_t(pixels[2].info.x))
	    << " primaryVoxel=" << int(int32_t(pixels[2].info.y)) << " vplVoxel=" << int(int32_t(pixels[2].info.z)) << "\n";

	const char* fileName = std::getenv("VXPG_DUMP_FILE");
	std::ofstream file(fileName ? fileName : "vxpg_dump.txt");
	file << out.str();
	file.close();

	for (Readback& read : reads)
		Application::allocator.destroyBuffer(read.stage);
}

//-------------------------------------------------------------------------------------------------
// Descriptors
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::createDescriptorSetLayout()
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
		(uint32_t)shaderio::VXPGStaticBindingPoints::eVoxelIrradiance,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eVoxelVplCount,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eVoxelBoundMin,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eVoxelBoundMax,
		(uint32_t)shaderio::VXPGStaticBindingPoints::ePixelData,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eTreeNodes,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eClusterRoots,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eTopLevelTree,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eSpixelInfo,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eSpixelClusterWeight,
		(uint32_t)shaderio::VXPGStaticBindingPoints::eSpixelClusterTotal,
	};
	for (uint32_t binding : bufferBindings)
	{
		bindings.addBinding({ .binding = binding,
		                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		                      .descriptorCount = 1,
		                      .stageFlags = VK_SHADER_STAGE_ALL });
	}

	staticDescPack.init(bindings, Application::app->getDevice(), 1,
	                    VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
	                    VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);
	NVVK_DBG_NAME(staticDescPack.getLayout());

	bindings.clear();
	bindings.addBinding({ .binding = shaderio::DynamicSetBindingPoints_PT::eTlas_PT,
	                      .descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
	                      .descriptorCount = 1,
	                      .stageFlags = VK_SHADER_STAGE_ALL });
	dynamicDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);
}

void VXPGRenderer::createDescriptorSet()
{
	nvvk::WriteSetContainer write{};

	if (!Application::sceneResource.textures.empty())
	{
		VkWriteDescriptorSet allTextures =
			staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eTextures_PT, 0, 0,
			                         uint32_t(Application::sceneResource.textures.size()));
		write.append(allTextures, Application::sceneResource.textures.data());
	}

	VkWriteDescriptorSet outImageWrite = staticDescPack.makeWrite(shaderio::StaticSetBindingPoints_PT::eOutImage_PT, 0, 0, 1);
	write.append(outImageWrite, gBuffers.getColorImageView((uint32_t)ImageType_VXPG::eImgRendered), VK_IMAGE_LAYOUT_GENERAL);

	auto appendBuffer = [&](shaderio::VXPGStaticBindingPoints binding, const nvvk::Buffer& buffer) {
		VkWriteDescriptorSet writeSet = staticDescPack.makeWrite((uint32_t)binding, 0, 0, 1);
		write.append(writeSet, buffer, 0, buffer.bufferSize);
	};

	appendBuffer(shaderio::VXPGStaticBindingPoints::eVoxelIrradiance, voxelIrradianceBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eVoxelVplCount, voxelVplCountBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eVoxelBoundMin, voxelBoundMinBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eVoxelBoundMax, voxelBoundMaxBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::ePixelData, pixelDataBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eTreeNodes, treeNodesBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eClusterRoots, clusterRootsBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eTopLevelTree, topLevelTreeBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eSpixelInfo, spixelInfoBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eSpixelClusterWeight, spixelClusterWeightBuffer);
	appendBuffer(shaderio::VXPGStaticBindingPoints::eSpixelClusterTotal, spixelClusterTotalBuffer);

	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
}

void VXPGRenderer::createPipelineLayout()
{
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VXPGPushConstant),
	};

	std::array<VkDescriptorSetLayout, 2> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout() } };
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

void VXPGRenderer::compileAndCreateShaders()
{
	SCOPED_TIMER(__FUNCTION__);

	VkDevice device = Application::app->getDevice();
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";

	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VXPGPushConstant),
	};
	std::array<VkDescriptorSetLayout, 2> layouts = { { staticDescPack.getLayout(), dynamicDescPack.getLayout() } };
	VkShaderCreateInfoEXT shaderInfo{
		.sType = VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT,
		.codeType = VK_SHADER_CODE_TYPE_SPIRV_EXT,
		.pName = "main",
		.setLayoutCount = layouts.size(),
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
		VkResult result = vkCreateShadersEXT(device, 1U, &shaderInfo, nullptr, &shader);
		if (result != VK_SUCCESS)
			LOGE("VXPG: failed to create shader object '%s'\n", entryPoint);
		else
			NVVK_DBG_NAME(shader);
	};

	VkShaderModuleCreateInfo passesCode = FzbRenderer::compileSlangShader(shaderPath / "VXPGPasses.slang", {});
	if (passesCode.codeSize > 0)
	{
		createShader(shaderVBuffer, passesCode, "vxpgVBufferMain");
		createShader(shaderLightInject, passesCode, "vxpgLightInjectMain");
		createShader(shaderGeometryInject, passesCode, "vxpgGeometryInjectMain");
		createShader(shaderTreeInit, passesCode, "vxpgTreeInitMain");
		createShader(shaderTreeInternal, passesCode, "vxpgTreeInternalMain");
		createShader(shaderTreeAccumulate, passesCode, "vxpgTreeAccumulateMain");
		createShader(shaderClusterRoots, passesCode, "vxpgClusterRootsMain");
		createShader(shaderSpixelCenters, passesCode, "vxpgSpixelCentersMain");
		createShader(shaderSpixelAssign, passesCode, "vxpgSpixelAssignMain");
		createShader(shaderSpixelAccum, passesCode, "vxpgSpixelAccumMain");
		createShader(shaderTopLevel, passesCode, "vxpgTopLevelMain");
	}
	else
	{
		LOGE("VXPG: VXPGPasses.slang failed to compile, see the Slang diagnostics above\n");
	}

	VkShaderModuleCreateInfo integratorCode = FzbRenderer::compileSlangShader(shaderPath / "VXPG.slang", {});
	if (integratorCode.codeSize > 0)
		createShader(shaderIntegrator, integratorCode, "computeMain_VXPG");
	else
		LOGE("VXPG: VXPG.slang failed to compile, see the Slang diagnostics above\n");
}

//-------------------------------------------------------------------------------------------------
// Passes
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::bindCommonState(VkCommandBuffer cmd)
{
	vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, staticDescPack.getSetPtr(), 0, nullptr);

	nvvk::WriteSetContainer write{};
	write.append(dynamicDescPack.makeWrite(shaderio::DynamicSetBindingPoints_PT::eTlas_PT), asManager.asBuilder.tlas);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 1, write.size(), write.data());

	pushConstant.screenSize = shaderio::uint2(currentResolution.width, currentResolution.height);
	pushConstant.threadGroupCount = shaderio::uint2(
		(currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
		(currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y);

	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VXPGPushConstant),
		.pValues = &pushConstant,
	};
	vkCmdPushConstants2(cmd, &pushInfo);
}

void VXPGRenderer::dispatchCompute(VkCommandBuffer cmd, VkShaderEXT shader, uint32_t groupCountX, uint32_t groupCountY,
                                   uint32_t groupCountZ)
{
	if (shader == VK_NULL_HANDLE || groupCountX == 0 || groupCountY == 0 || groupCountZ == 0)
		return;
	VkShaderStageFlagBits stage = VK_SHADER_STAGE_COMPUTE_BIT;
	vkCmdBindShadersEXT(cmd, 1, &stage, &shader);
	vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
}

void VXPGRenderer::clearVoxelGrids(VkCommandBuffer cmd)
{
	vkCmdFillBuffer(cmd, voxelIrradianceBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, voxelVplCountBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, voxelBoundMinBuffer.buffer, 0, VK_WHOLE_SIZE, 0xffffffffu);
	vkCmdFillBuffer(cmd, voxelBoundMaxBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
}

// The per superpixel weights are rebuilt from the current frame, so they are always cleared.
void VXPGRenderer::clearSpixelWeights(VkCommandBuffer cmd)
{
	vkCmdFillBuffer(cmd, spixelClusterWeightBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
	vkCmdFillBuffer(cmd, spixelClusterTotalBuffer.buffer, 0, VK_WHOLE_SIZE, 0u);
}

void VXPGRenderer::runVBufferPass(VkCommandBuffer cmd)
{
	dispatchCompute(cmd, shaderVBuffer, (currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
	                (currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y, 1);
}

void VXPGRenderer::runLightInjectionPass(VkCommandBuffer cmd)
{
	dispatchCompute(cmd, shaderLightInject, (currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
	                (currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y, 1);
}

void VXPGRenderer::runGeometryInjectionPass(VkCommandBuffer cmd)
{
	FzbRenderer::Scene& scene = Application::sceneResource;
	for (uint32_t instanceIndex = 0; instanceIndex < uint32_t(scene.instances.size()); ++instanceIndex)
	{
		const uint32_t meshIndex = scene.instances[instanceIndex].meshIndex;
		if (meshIndex >= scene.meshes.size())
			continue;
		const uint32_t indexCount = scene.meshes[meshIndex].triMesh.indices.count;
		const uint32_t triangleCount = indexCount / 3u;
		if (triangleCount == 0u)
			continue;

		pushConstant.instanceIndex = int(instanceIndex);
		pushConstant.triangleCount = int(triangleCount);

		const VkPushConstantsInfo pushInfo{
			.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
			.layout = pipelineLayout,
			.stageFlags = VK_SHADER_STAGE_ALL,
			.offset = 0,
			.size = sizeof(shaderio::VXPGPushConstant),
			.pValues = &pushConstant,
		};
		vkCmdPushConstants2(cmd, &pushInfo);

		dispatchCompute(cmd, shaderGeometryInject, (triangleCount + 63u) / 64u, 1, 1);
	}
	pushConstant.instanceIndex = 0;
	pushConstant.triangleCount = 0;
}

void VXPGRenderer::runTreePasses(VkCommandBuffer cmd)
{
	const uint32_t treeGroups = (nodeCount + VXPG_TREE_THREADGROUP_SIZE - 1) / VXPG_TREE_THREADGROUP_SIZE;
	dispatchCompute(cmd, shaderTreeInit, treeGroups, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	const uint32_t internalGroups = (leafCount + VXPG_TREE_THREADGROUP_SIZE - 1) / VXPG_TREE_THREADGROUP_SIZE;
	dispatchCompute(cmd, shaderTreeInternal, internalGroups, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// Leaf intensities are summed into all ancestors (deterministic, no cross thread handshake).
	dispatchCompute(cmd, shaderTreeAccumulate, internalGroups, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	// Deterministic cluster roots (one thread per cluster).
	dispatchCompute(cmd, shaderClusterRoots, 1, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void VXPGRenderer::runSpixelPasses(VkCommandBuffer cmd)
{
	dispatchCompute(cmd, shaderSpixelCenters, (spixelGridWidth + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
	                (spixelGridHeight + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	dispatchCompute(cmd, shaderSpixelAssign, (currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
	                (currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	dispatchCompute(cmd, shaderSpixelAccum, (currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X,
	                (currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	dispatchCompute(cmd, shaderTopLevel, (spixelCount + 63u) / 64u, 1, 1);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}

void VXPGRenderer::runIntegratorPass(VkCommandBuffer cmd)
{
	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = pipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.offset = 0,
		.size = sizeof(shaderio::VXPGPushConstant),
		.pValues = &pushConstant,
	};

	// The integrator is the most expensive dispatch (three rays per pixel plus the guiding
	// traversal), and a single dispatch over the whole image can exceed the Windows GPU
	// watchdog (TDR), especially in Debug builds where the shaders are unoptimised. The image
	// is therefore traced in horizontal bands; the pixels are disjoint, so no barrier is needed.
	const uint32_t groupCountX = (currentResolution.width + VXPG_THREADGROUP_SIZE_X - 1) / VXPG_THREADGROUP_SIZE_X;
	const uint32_t groupCountY = (currentResolution.height + VXPG_THREADGROUP_SIZE_Y - 1) / VXPG_THREADGROUP_SIZE_Y;
	const uint32_t bandCount = std::clamp<uint32_t>(uint32_t(std::max(pushConstant.maxDepth - 1, 1)) * 2u, 2u, 16u);
	const uint32_t bandGroupHeight = (groupCountY + bandCount - 1u) / bandCount;

	for (uint32_t band = 0; band < bandCount; ++band)
	{
		const uint32_t groupY = band * bandGroupHeight;
		if (groupY >= groupCountY)
			break;
		const uint32_t dispatchGroupHeight = std::min(bandGroupHeight, groupCountY - groupY);
		pushConstant.dispatchOffset = shaderio::uint2(0u, groupY * VXPG_THREADGROUP_SIZE_Y);
		vkCmdPushConstants2(cmd, &pushInfo);
		dispatchCompute(cmd, shaderIntegrator, groupCountX, dispatchGroupHeight, 1);
	}
}

void VXPGRenderer::render(VkCommandBuffer* cmdPtr)
{
	VkCommandBuffer cmd = cmdPtr[0];
	NVVK_DBG_SCOPE(cmd);

	updateDataPerFrame(cmd);
	if (maxFrames > 1 && pushConstant.frameIndex >= maxFrames)
		return;

	if (shaderIntegrator == VK_NULL_HANDLE)
		return;

	bindCommonState(cmd);

	const uint32_t stageMask = vxpgStageMask();
	const bool rebuild = rebuildPerFrame || guidingDirty || Application::frameIndex == 0;
	if (rebuild)
	{
		// The voxel field is accumulated across frames (sum / count is already an average), which
		// makes the injected distribution dense and temporally stable; it is only reset when the
		// domain changes or at the very first frame.
		if (guidingDirty || pushConstant.frameIndex == 0)
			clearVoxelGrids(cmd);
		clearSpixelWeights(cmd);
		nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

		if (stageMask & 1u)
		{
			runVBufferPass(cmd);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
		if (stageMask & 2u)
		{
			runLightInjectionPass(cmd);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
		if (stageMask & 4u)
		{
			runGeometryInjectionPass(cmd);
			nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
		}
		if (stageMask & 8u) runTreePasses(cmd);
		if (stageMask & 16u) runSpixelPasses(cmd);
	}

	if (stageMask & 32u) runIntegratorPass(cmd);
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData,
	                                  gBuffers.getDescriptorImageInfo((uint32_t)ImageType_VXPG::eImgRendered),
	                                  gBuffers.getDescriptorImageInfo((uint32_t)ImageType_VXPG::eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
	                      VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
	                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
	nvvk::cmdMemoryBarrier(cmd,
	                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
	                          VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
	                      VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
}

//-------------------------------------------------------------------------------------------------
// UI
//-------------------------------------------------------------------------------------------------
void VXPGRenderer::uiRender()
{
	bool uiModified = Application::UIModified;

	Application::viewportImage = gBuffers.getDescriptorSet((uint32_t)ImageType_VXPG::eImgTonemapped);

	if (ImGui::Begin("VXPGSettings"))
	{
		ImGui::SeparatorText("VXPG (mode 4: Guide + BSDF)");
		ImGui::TextDisabled("Frame: %d / %d", pushConstant.frameIndex, maxFrames);
		ImGui::TextDisabled("Voxel grid: %d^3 = %d voxels, %d tree nodes", pushConstant.voxelResolution, voxelCount, nodeCount);
		ImGui::TextDisabled("Superpixels: %d x %d = %d, %d clusters", spixelGridWidth, spixelGridHeight, spixelCount,
		                    VXPG_CLUSTER_COUNT);

		uiModified |= ImGui::DragInt("Max Frames", &maxFrames, 1.0f, 1, 100000);
		uiModified |= ImGui::Checkbox("Rebuild Guiding", &rebuildPerFrame);
		uiModified |= ImGui::Checkbox("Enable Guiding", &enableGuiding);

		ImGui::SeparatorText("Sampling");
		uiModified |= ImGui::SliderInt("Bounces (path length)", &pushConstant.maxDepth, 1, 16);
		uiModified |= ImGui::SliderInt("SPP", &pushConstant.spp, 1, 16);
		// Path length is the single source of truth: the first indirect vertex is traced by both
		// techniques, every further vertex only by the BSDF technique.
		pushConstant.bounceCount = std::clamp(pushConstant.maxDepth - 2, 0, 8);
		ImGui::TextDisabled("Indirect vertices after the first: %d  (2 traced by both techniques)", pushConstant.bounceCount);
		int misMode = std::clamp(pushConstant.misMode, 0, 2);
		if (ImGui::Combo("MIS Mode", &misMode, "One-sample 50/50\0Balance\0Power\0"))
		{
			pushConstant.misMode = misMode;
			uiModified = true;
		}

		// Jitter ranges (0 disables the corresponding jitter). Useful to see what each one does:
		//   SP     : shading point offset while sampling the guide, in voxel sizes
		//   Spixel : superpixel lookup offset, in pixels
		//   SV     : injected VPL position offset before quantisation, in voxel sizes
		float spJitter = pushConstant.jitterRadius;
		if (ImGui::SliderFloat("SP Jitter (voxels)", &spJitter, 0.0f, 2.0f, "%.2f"))
		{
			pushConstant.jitterRadius = spJitter;
			uiModified = true;
		}
		float spixelJitter = pushConstant.padding0;
		if (ImGui::SliderFloat("Spixel Jitter (pixels)", &spixelJitter, 0.0f, 32.0f, "%.0f"))
		{
			pushConstant.padding0 = spixelJitter;
			uiModified = true;
		}
		float svJitter = pushConstant.padding1;
		if (ImGui::SliderFloat("SV Injection Jitter (voxels)", &svJitter, 0.0f, 2.0f, "%.2f"))
		{
			pushConstant.padding1 = svJitter;
			uiModified = true;
		}
		float fireflyClamp = pushConstant.fireflyClamp;
		if (ImGui::SliderFloat("Indirect Clamp (0 = off)", &fireflyClamp, 0.0f, 500.0f, "%.0f"))
		{
			pushConstant.fireflyClamp = fireflyClamp;
			uiModified = true;
		}

		ImGui::SeparatorText("Voxel grid");
		int voxelResolution = pushConstant.voxelResolution;
		ImGui::DragInt("Voxel Resolution", &voxelResolution, 1.0f, 8, VXPG_MAX_VOXEL_RESOLUTION);
		int spixelSize = pushConstant.spixelSize;
		ImGui::DragInt("Superpixel Size", &spixelSize, 1.0f, 8, 128);
		if (ImGui::Button("Rebuild Voxel Grid"))
		{
			vkDeviceWaitIdle(Application::app->getDevice());
			destroyResources();
			pushConstant.voxelResolution = voxelResolution;
			pushConstant.spixelSize = spixelSize;
			computeResolution();
			updateSceneBounds();
			createResources();
			createDescriptorSet();
			resetFrame();
		}
	}
	ImGui::End();

	if (uiModified)
		resetFrame();
}