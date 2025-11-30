#include "PathTracingRenderer.h"
#include <common/Application/Application.h>
#include <nvvk/formats.hpp>
#include "common/Shader/shaderio.h"
#include <nvgui/sky.hpp>
#include <nvvk/default_structs.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "common/Shader/Shader.h"

FzbRenderer::PathTracingRenderer::PathTracingRenderer(RendererCreateInfo& createInfo) {
	createInfo.vkContextInfo.deviceExtensions.push_back( { VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, &accelFeature });
	createInfo.vkContextInfo.deviceExtensions.push_back({ VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, &rtPipelineFeature });
	createInfo.vkContextInfo.deviceExtensions.push_back({ VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME });
}
//-----------------------------------------创建加速结构----------------------------------------------------------
/*
void FzbRenderer::PathTracingRenderer::primitiveToGeometry(const shaderio::GltfMesh& gltfMesh,
	VkAccelerationStructureGeometryKHR& geometry, VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
	//这个函数就和我之前cuda实现BVH的思路一摸一样啊

	const shaderio::TriangleMesh triMesh = gltfMesh.triMesh;
	const auto triangleCount = static_cast<uint32_t>(triMesh.indices.count / 3U);

	//从一个大的buffer中将顶点和索引找到，构成三角形的信息
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.positions.offset},
		.vertexStride = triMesh.positions.byteStride,
		.maxVertex = triMesh.positions.count - 1,
		.indexType = VkIndexType(gltfMesh.indexType),
		.indexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.indices.offset},
	};

	//然后使用三角形创建BVH
	geometry = VkAccelerationStructureGeometryKHR{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {.triangles = triangles},
		.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
	};

	rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{ .primitiveCount = triangleCount };
}
void FzbRenderer::PathTracingRenderer::createAccelerationStructure(VkAccelerationStructureTypeKHR asType, nvvk::AccelerationStructure& accelStruct,
	VkAccelerationStructureGeometryKHR& asGeometry, VkAccelerationStructureBuildRangeInfoKHR& asBuildRangeInfo,
	VkBuildAccelerationStructureFlagsKHR flags) {
	VkDevice device = Application::app->getDevice();
	//对齐函数，alignment一定会是2的幂次，所以~(alignment - 1)表示低位全为0，高位全为1；那么与之后就剩下向上对齐的上确界了
	auto alignUp = [](auto value, size_t alignment) noexcept { return ((value + alignment - 1) & ~(alignment - 1)); };

	VkAccelerationStructureBuildGeometryInfoKHR asBuildInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
		.type = asType,
		.flags = flags,
		.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
		.geometryCount = 1,
		.pGeometries = &asGeometry
	};

	std::vector<uint32_t> maxPrimCount(1);
	maxPrimCount[0] = asBuildRangeInfo.primitiveCount;

	//根据buildInfo计算出需要的存储空间大小(AS的存储空间以及创建AS所需要的临时空间）
	VkAccelerationStructureBuildSizesInfoKHR asBuildSize{ .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
	vkGetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &asBuildInfo, maxPrimCount.data(), &asBuildSize);

	//buildScratchSize是创建AS所需的临时空间，minAccelerationStructureScratchOffsetAlignment是硬件要求的最小对齐
	VkDeviceSize scratchSize = alignUp(asBuildSize.buildScratchSize, asProperties.minAccelerationStructureScratchOffsetAlignment);

	nvvk::Buffer scratchBuffer;
	NVVK_CHECK(Application::allocator.createBuffer(scratchBuffer, scratchSize,
		VK_BUFFER_USAGE_2_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		VMA_MEMORY_USAGE_AUTO, {}, asProperties.minAccelerationStructureScratchOffsetAlignment));

	VkAccelerationStructureCreateInfoKHR createInfo{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
		.size = asBuildSize.accelerationStructureSize,
		.type = asType,
	};
	NVVK_CHECK(Application::allocator.createAcceleration(accelStruct, createInfo));		//申请空间

	{
		VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

		asBuildInfo.dstAccelerationStructure = accelStruct.accel;
		asBuildInfo.scratchData.deviceAddress = scratchBuffer.address;

		VkAccelerationStructureBuildRangeInfoKHR* pBuildRangeInfo = &asBuildRangeInfo;
		vkCmdBuildAccelerationStructuresKHR(cmd, 1, &asBuildInfo, &pBuildRangeInfo);	//创建AS

		Application::app->submitAndWaitTempCmdBuffer(cmd);
	}

	Application::allocator.destroyBuffer(scratchBuffer);
}
void FzbRenderer::PathTracingRenderer::createBottomLevelAS() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI("Ready to build %zu bottom-level acceleration structures\n", Application::sceneResource.meshes.size());

	blasAccel.resize(Application::sceneResource.meshes.size());
	for (uint32_t blasId = 0; blasId < Application::sceneResource.meshes.size(); ++blasId) {
		VkAccelerationStructureGeometryKHR asGeometry{};
		VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};

		primitiveToGeometry(Application::sceneResource.meshes[blasId], asGeometry, asBuildRangeInfo);

		createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, blasAccel[blasId], asGeometry,
			asBuildRangeInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
		NVVK_DBG_NAME(blasAccel[blasId].accel);
	}

	LOGI("Bottom-level acceleration structures built successfully\n");
}
void FzbRenderer::PathTracingRenderer::createToLevelAS() {
	SCOPED_TIMER(__FUNCTION__);

	auto toTransformMatrixKHR = [](const glm::mat4& m) {
		VkTransformMatrixKHR t;
		memcpy(&t, glm::value_ptr(glm::transpose(m)), sizeof(t));	//glm的矩阵是列矩阵，而VkTransformMatrixKHR是行矩阵，所以需要先转置
		return t;
	};

	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
	tlasInstances.reserve(Application::sceneResource.instances.size());		//每个mesh一个顶层实例
	for (const shaderio::GltfInstance& instance : Application::sceneResource.instances) {
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = toTransformMatrixKHR(instance.transform);
		asInstance.instanceCustomIndex = instance.meshIndex;
		asInstance.accelerationStructureReference = blasAccel[instance.meshIndex].address;
		asInstance.instanceShaderBindingTableRecordOffset = 0;		//所有的实例都用相同的(索引为0)的(rayGen、miss等)shader
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;	//没有背面剔除
		asInstance.mask = 0xFF;
		tlasInstances.emplace_back(asInstance);
	}
	LOGI("Ready to build top-level acceleration structure with %zu instances\n", tlasInstances.size());

	nvvk::Buffer tlasInstancesBuffer;
	{
		VkCommandBuffer cmd = Application::app->createTempCmdBuffer();

		NVVK_CHECK(Application::allocator.createBuffer(
			tlasInstancesBuffer, std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances).size_bytes(),
			VK_BUFFER_USAGE_2_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_2_SHADER_DEVICE_ADDRESS_BIT
		));
		NVVK_CHECK(Application::stagingUploader.appendBuffer(
			tlasInstancesBuffer, 0,
			std::span<VkAccelerationStructureInstanceKHR const>(tlasInstances)
		));

		NVVK_DBG_NAME(tlasInstancesBuffer.buffer);
		Application::stagingUploader.cmdUploadAppended(cmd);
		Application::app->submitAndWaitTempCmdBuffer(cmd);
	}

	{
		VkAccelerationStructureGeometryKHR asGeometry{};
		VkAccelerationStructureBuildRangeInfoKHR asBuildRangeInfo{};

		VkAccelerationStructureGeometryInstancesDataKHR geometryInstances{
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
			.data = { .deviceAddress = tlasInstancesBuffer.address },
		};
		asGeometry = {
			.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
			.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
			.geometry = { .instances = geometryInstances }
		};
		asBuildRangeInfo = { .primitiveCount = static_cast<uint32_t>(Application::sceneResource.instances.size()) };

		createAccelerationStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, tlasAccel, asGeometry,
			asBuildRangeInfo, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
		NVVK_DBG_NAME(tlasAccel.accel);
	}

	LOGI("Top-level accleration structures built successfully\n");
	Application::allocator.destroyBuffer(tlasInstancesBuffer);
}
//-----------------------------------------创造光追管线----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo) {
	//STB，顾名思义，就是shader大的绑定表；
	//其将pipeline各个阶段的shader组进行绑定
	//每个实例只能用每个阶段的shader组中的一个条目
	//每个组和条目都需要对齐
SCOPED_TIMER(__FUNCTION__);
Application::allocator.destroyBuffer(sbtBuffer);

//STB中shader必须对齐
VkDevice device = Application::app->getDevice();
uint32_t handleSize = rtProperties.shaderGroupHandleSize;		//shader句柄的大小
uint32_t handleAlignment = rtProperties.shaderGroupHandleAlignment;		//每个shader的对齐
uint32_t baseAlignment = rtProperties.shaderGroupBaseAlignment;		//每个shader组的对齐
uint32_t groupCount = rtPipelineInfo.groupCount;

size_t dataSize = handleSize * groupCount;	//目前每组只有一个shader（条目），所以乘以组数
shaderHandles.resize(dataSize);
NVVK_CHECK(vkGetRayTracingShaderGroupHandlesKHR(device, rtPipeline, 0, groupCount, dataSize, shaderHandles.data()));

auto alignUp = [](uint32_t size, uint32_t alignment) {return (size + alignment - 1) & ~(alignment - 1); };
uint32_t raygenSize = alignUp(handleSize, handleAlignment);		//目前没有附加数据，所以都一样
uint32_t missSize = alignUp(handleSize, handleAlignment);
uint32_t hitSize = alignUp(handleSize, handleAlignment);
uint32_t callableSize = 0;

uint32_t raygenOffset = 0;
uint32_t missOffset = alignUp(raygenSize, baseAlignment);
uint32_t hitOffset = alignUp(missOffset + missSize, baseAlignment);
uint32_t callableOffset = alignUp(hitOffset + hitSize, baseAlignment);

size_t bufferSize = callableOffset + callableSize;
NVVK_CHECK(Application::allocator.createBuffer(
	sbtBuffer, bufferSize,
	VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
	VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
));
NVVK_DBG_NAME(sbtBuffer.buffer);
uint8_t* pData = static_cast<uint8_t*>(sbtBuffer.mapping);

memcpy(pData + raygenOffset, shaderHandles.data() + 0 * handleSize, handleSize);
raygenRegion.deviceAddress = sbtBuffer.address + raygenOffset;
raygenRegion.stride = raygenSize;	//raygen中每个shader的条目
raygenRegion.size = raygenSize;		//raygen组总大小

memcpy(pData + missOffset, shaderHandles.data() + 1 * handleSize, handleSize);	//shaderHandles是紧密的
missRegion.deviceAddress = sbtBuffer.address + missOffset;
missRegion.stride = missSize;
missRegion.size = missSize;

memcpy(pData + hitOffset, shaderHandles.data() + 2 * handleSize, handleSize);
hitRegion.deviceAddress = sbtBuffer.address + hitOffset;
hitRegion.stride = hitSize;
hitRegion.size = hitSize;

callableRegion.deviceAddress = 0;
callableRegion.stride = 0;
callableRegion.size = 0;

LOGI(" Shader binding table created and populated\n");
}
void FzbRenderer::PathTracingRenderer::createRayTracingPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI(" Creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), rtPipelineLayout, nullptr);

	enum StageIndices {
		eRaygen,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};
	std::vector<VkPipelineShaderStageCreateInfo> stages(eShaderGroupCount);
	for (auto& s : stages)
		s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

	//VkShaderModuleCreateInfo shaderCode = compileSlangShader("pathTracingShaders.slang", {});
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "pathTracingShaders.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	stages[eRaygen].pNext = &shaderCode;
	stages[eRaygen].pName = "raygenMain";
	stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[eMiss].pNext = &shaderCode;
	stages[eMiss].pName = "rayMissMain";
	stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[eClosestHit].pNext = &shaderCode;
	stages[eClosestHit].pName = "rayClosestHitMain";
	stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;	//表示光线追踪pipeline有几个阶段，光纤生成->打中/没打中
	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	//光线生成shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	shader_groups.push_back(group);

	//光线没打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	shader_groups.push_back(group);

	//光线打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	shader_groups.push_back(group);

	const VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::TutoPushConstant) };

	VkPipelineLayoutCreateInfo pipeline_layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipeline_layout_create_info.pushConstantRangeCount = 1;
	pipeline_layout_create_info.pPushConstantRanges = &push_constant;

	std::vector<VkDescriptorSetLayout> layouts = { { descPack.getLayout(), rtDescPack.getLayout()} };	//二合一
	pipeline_layout_create_info.setLayoutCount = uint32_t(layouts.size());
	pipeline_layout_create_info.pSetLayouts = layouts.data();
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &rtPipelineLayout);
	NVVK_DBG_NAME(rtPipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::max(3U, rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = rtPipelineLayout;
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, &rtPipelineInfo, nullptr, &rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	createShaderBindingTable(rtPipelineInfo);
}
*/
nvvk::AccelerationStructureGeometryInfo FzbRenderer::PathTracingRenderer::primitiveToGeometry(const shaderio::GltfMesh& gltfMesh) {
	//这个函数就和我之前cuda实现BVH的思路一摸一样啊

	nvvk::AccelerationStructureGeometryInfo result{};

	const shaderio::TriangleMesh triMesh = gltfMesh.triMesh;
	const auto triangleCount = static_cast<uint32_t>(triMesh.indices.count / 3U);

	//从一个大的buffer中将顶点和索引找到，构成三角形的信息
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.positions.offset},
		.vertexStride = triMesh.positions.byteStride,
		.maxVertex = triMesh.positions.count - 1,
		.indexType = VkIndexType(gltfMesh.indexType),
		.indexData = {.deviceAddress = VkDeviceAddress(gltfMesh.gltfBuffer) + triMesh.indices.offset},
	};

	//然后使用三角形创建BVH
	result.geometry = VkAccelerationStructureGeometryKHR{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {.triangles = triangles},
		.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR | VK_GEOMETRY_OPAQUE_BIT_KHR,
	};

	result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{ .primitiveCount = triangleCount };

	return result;
}
void FzbRenderer::PathTracingRenderer::createBottomLevelAS() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI("Ready to build %zu bottom-level acceleration structures\n", Application::sceneResource.meshes.size());

	std::vector<nvvk::AccelerationStructureGeometryInfo> geoInfos(Application::sceneResource.meshes.size());
	for (uint32_t blasId = 0; blasId < Application::sceneResource.meshes.size(); ++blasId)
		geoInfos[blasId] = primitiveToGeometry(Application::sceneResource.meshes[blasId]);

	asBuilder.blasSubmitBuildAndWait(geoInfos, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	LOGI("Bottom-level acceleration structures built successfully\n");
}
void FzbRenderer::PathTracingRenderer::createToLevelAS() {
	SCOPED_TIMER(__FUNCTION__);

	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
	tlasInstances.reserve(Application::sceneResource.instances.size());		//每个mesh一个顶层实例
	const VkGeometryInstanceFlagsKHR flgas{ VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV };

	for (const shaderio::GltfInstance& instance : Application::sceneResource.instances) {
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = nvvk::toTransformMatrixKHR(instance.transform);
		asInstance.instanceCustomIndex = instance.meshIndex;
		asInstance.accelerationStructureReference = asBuilder.blasSet[instance.meshIndex].address;
		asInstance.instanceShaderBindingTableRecordOffset = 0;		//所有的实例都用相同的(索引为0)的(rayGen、miss等)shader
		asInstance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV;	//没有背面剔除
		asInstance.mask = 0xFF;
		tlasInstances.emplace_back(asInstance);
	}
	LOGI("Ready to build top-level acceleration structure with %zu instances\n", tlasInstances.size());

	asBuilder.tlasSubmitBuildAndWait(tlasInstances, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

	LOGI("Top-level accleration structures built successfully\n");
}
//-----------------------------------------创造光追管线----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::createRayTracingDescriptorLayout() {
	SCOPED_TIMER(__FUNCTION__);
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({
			.binding = shaderio::BindingPoints::eTlas,
			.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});
	bindings.addBinding({
			.binding = shaderio::BindingPoints::eOutImage,
			.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
			.descriptorCount = 1,
			.stageFlags = VK_SHADER_STAGE_ALL
		});

	rtDescPack.init(bindings, Application::app->getDevice(), 0, VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT);

	LOGI("Ray tracing descriptor layout created\n");
}
void FzbRenderer::PathTracingRenderer::createShaderBindingTable(const VkRayTracingPipelineCreateInfoKHR& rtPipelineInfo) {
	/*
	STB，顾名思义，就是shader大的绑定表；
	其将pipeline各个阶段的shader组进行绑定
	每个实例只能用每个阶段的shader组中的一个条目
	每个组和条目都需要对齐
	*/
	SCOPED_TIMER(__FUNCTION__);
	Application::allocator.destroyBuffer(sbtBuffer);

	size_t bufferSize = sbtGenerator.calculateSBTBufferSize(rtPipeline, rtPipelineInfo);
	NVVK_CHECK(Application::allocator.createBuffer(sbtBuffer, bufferSize, VK_BUFFER_USAGE_2_SHADER_BINDING_TABLE_BIT_KHR,
		VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
		sbtGenerator.getBufferAlignment()));
	NVVK_DBG_NAME(sbtBuffer.buffer);
	NVVK_CHECK(sbtGenerator.populateSBTBuffer(sbtBuffer.address, bufferSize, sbtBuffer.mapping));

	LOGI(" Shader binding table created and populated\n");
}
void FzbRenderer::PathTracingRenderer::createRayTracingPipeline() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI(" Creating ray tracing pipeline Structure\n");

	Application::allocator.destroyBuffer(sbtBuffer);
	vkDestroyPipeline(Application::app->getDevice(), rtPipeline, nullptr);
	vkDestroyPipelineLayout(Application::app->getDevice(), rtPipelineLayout, nullptr);

	enum StageIndices {
		eRaygen,
		eMiss,
		eClosestHit,
		eShaderGroupCount
	};
	std::vector<VkPipelineShaderStageCreateInfo> stages(eShaderGroupCount);
	for(auto& s : stages)
		s.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;

	//VkShaderModuleCreateInfo shaderCode = compileSlangShader("pathTracingShaders.slang", {});
	std::filesystem::path shaderPath = std::filesystem::path(__FILE__).parent_path() / "shaders";
	std::filesystem::path shaderSource = shaderPath / "pathTracingShaders.slang";
	VkShaderModuleCreateInfo shaderCode = FzbRenderer::compileSlangShader(shaderSource, {});

	stages[eRaygen].pNext = &shaderCode;
	stages[eRaygen].pName = "raygenMain";
	stages[eRaygen].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
	stages[eMiss].pNext = &shaderCode;
	stages[eMiss].pName = "rayMissMain";
	stages[eMiss].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
	stages[eClosestHit].pNext = &shaderCode;
	stages[eClosestHit].pName = "rayClosestHitMain";
	stages[eClosestHit].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

	std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;	//表示光线追踪pipeline有几个阶段，光纤生成->打中/没打中
	VkRayTracingShaderGroupCreateInfoKHR group{ VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR };
	group.anyHitShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = VK_SHADER_UNUSED_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.intersectionShader = VK_SHADER_UNUSED_KHR;

	//光线生成shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eRaygen;
	shader_groups.push_back(group);

	//光线没打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
	group.generalShader = eMiss;
	shader_groups.push_back(group);

	//光线打中shader组，此时只有一个条目（shader）
	group.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
	group.generalShader = VK_SHADER_UNUSED_KHR;
	group.closestHitShader = eClosestHit;
	shader_groups.push_back(group);

	const VkPushConstantRange push_constant{ VK_SHADER_STAGE_ALL, 0, sizeof(shaderio::TutoPushConstant) };

	VkPipelineLayoutCreateInfo pipeline_layout_create_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
	pipeline_layout_create_info.pushConstantRangeCount = 1;
	pipeline_layout_create_info.pPushConstantRanges = &push_constant;

	std::vector<VkDescriptorSetLayout> layouts = { { descPack.getLayout(), rtDescPack.getLayout()} };	//二合一
	pipeline_layout_create_info.setLayoutCount = uint32_t(layouts.size());
	pipeline_layout_create_info.pSetLayouts = layouts.data();
	vkCreatePipelineLayout(Application::app->getDevice(), &pipeline_layout_create_info, nullptr, &rtPipelineLayout);
	NVVK_DBG_NAME(rtPipelineLayout);

	VkRayTracingPipelineCreateInfoKHR rtPipelineInfo{ VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR };
	rtPipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
	rtPipelineInfo.pStages = stages.data();
	rtPipelineInfo.groupCount = static_cast<uint32_t>(shader_groups.size());
	rtPipelineInfo.pGroups = shader_groups.data();
	rtPipelineInfo.maxPipelineRayRecursionDepth = std::max(3U, rtProperties.maxRayRecursionDepth);		//最大bounce数
	rtPipelineInfo.layout = rtPipelineLayout;
	vkCreateRayTracingPipelinesKHR(Application::app->getDevice(), {}, {}, 1, & rtPipelineInfo, nullptr, & rtPipeline);
	NVVK_DBG_NAME(rtPipeline);

	LOGI("Ray tracing pipeline layout created successfully\n");

	createShaderBindingTable(rtPipelineInfo);
}
//-----------------------------------------光追函数----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::rayTraceScene(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipeline);

	const VkBindDescriptorSetsInfo bindDescriptorSetsInfo{
		.sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.layout = rtPipelineLayout,
		.firstSet = 0,
		.descriptorSetCount = 1,
		.pDescriptorSets = descPack.getSetPtr()
	};
	vkCmdBindDescriptorSets2(cmd, &bindDescriptorSetsInfo);

	nvvk::WriteSetContainer write{};
	write.append(rtDescPack.makeWrite(shaderio::BindingPoints::eTlas), asBuilder.tlas);
	write.append(rtDescPack.makeWrite(shaderio::BindingPoints::eOutImage), gBuffers.getColorImageView(eImgRendered), VK_IMAGE_LAYOUT_GENERAL);
	vkCmdPushDescriptorSetKHR(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rtPipelineLayout, 1, write.size(), write.data());

	shaderio::TutoPushConstant pushValues{
		.sceneInfoAddress = (shaderio::GltfSceneInfo*)Application::sceneResource.bSceneInfo.address
	};
	const VkPushConstantsInfo pushInfo{
		.sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO,
		.layout = rtPipelineLayout,
		.stageFlags = VK_SHADER_STAGE_ALL,
		.size = sizeof(shaderio::TutoPushConstant),
		.pValues = &pushValues
	};
	vkCmdPushConstants2(cmd, &pushInfo);

	const nvvk::SBTGenerator::Regions& regions = sbtGenerator.getSBTRegions();
	const VkExtent2D& size = Application::app->getViewportSize();
	vkCmdTraceRaysKHR(cmd, &regions.raygen, &regions.miss, &regions.hit, &regions.callable, size.width, size.height, 1);

	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
}
void FzbRenderer::PathTracingRenderer::updateSceneBuffer(VkCommandBuffer cmd) {

}
//-----------------------------------------基础输入----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::createImage() {
	VkSampler linearSampler{};
	NVVK_CHECK(Application::samplerPool.acquireSampler(linearSampler));
	NVVK_DBG_NAME(linearSampler);

	nvvk::GBufferInitInfo gBufferInit{
		.allocator = &Application::allocator,
		.colorFormats = { VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM },
		//.depthFormat = nvvk::findDepthFormat(Application::app->getPhysicalDevice()),
		.imageSampler = linearSampler,
		.descriptorPool = Application::app->getTextureDescriptorPool(),
	};
	gBuffers.init(gBufferInit);
};
void FzbRenderer::PathTracingRenderer::createGraphicsDescriptorSetLayout() {
	nvvk::DescriptorBindings bindings;
	bindings.addBinding({ .binding = shaderio::BindingPoints::eTextures,
						 .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
						 .descriptorCount = 10,
						 .stageFlags = VK_SHADER_STAGE_ALL },
		VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT
		| VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT);
	descPack.init(bindings, Application::app->getDevice(), 1, VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
		VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT);

	NVVK_DBG_NAME(descPack.getLayout());
	NVVK_DBG_NAME(descPack.getPool());
	NVVK_DBG_NAME(descPack.getSet(0));
};
void FzbRenderer::PathTracingRenderer::createGraphicsPipelineLayout() {
	const VkPushConstantRange pushConstantRange{
		.stageFlags = VK_SHADER_STAGE_ALL_GRAPHICS,
		.offset = 0,
		.size = sizeof(shaderio::TutoPushConstant)
	};

	const VkPipelineLayoutCreateInfo pipelineLayoutInfo{
		.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
		.setLayoutCount = 1,
		.pSetLayouts = descPack.getLayoutPtr(),
		.pushConstantRangeCount = 1,
		.pPushConstantRanges = &pushConstantRange,
	};
	NVVK_CHECK(vkCreatePipelineLayout(Application::app->getDevice(), &pipelineLayoutInfo, nullptr, &graphicPipelineLayout));
	NVVK_DBG_NAME(graphicPipelineLayout);
};
void FzbRenderer::PathTracingRenderer::updateTextures() {
	if (Application::textures.empty())
		return;

	nvvk::WriteSetContainer write{};
	VkWriteDescriptorSet    allTextures =
		descPack.makeWrite(shaderio::BindingPoints::eTextures, 0, 1, uint32_t(Application::textures.size()));
	nvvk::Image* allImages = Application::textures.data();
	write.append(allTextures, allImages);
	vkUpdateDescriptorSets(Application::app->getDevice(), write.size(), write.data(), 0, nullptr);
};
//-----------------------------------------渲染器行为----------------------------------------------------------
void FzbRenderer::PathTracingRenderer::init() {
	createImage();
	createGraphicsDescriptorSetLayout();
	createGraphicsPipelineLayout();
	updateTextures();

	VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	prop2.pNext = &rtProperties;
	vkGetPhysicalDeviceProperties2(Application::app->getPhysicalDevice(), &prop2);

	asBuilder.init(&Application::allocator, &Application::stagingUploader, Application::app->getQueue(0));
	sbtGenerator.init(Application::app->getDevice(), rtProperties);

	createBottomLevelAS();
	createToLevelAS();

	createRayTracingDescriptorLayout();
	createRayTracingPipeline();
}
void FzbRenderer::PathTracingRenderer::clean() {
	VkDevice device = Application::app->getDevice();

	descPack.deinit();
	vkDestroyPipelineLayout(device, graphicPipelineLayout, nullptr);
	gBuffers.deinit();

	asBuilder.deinitAccelerationStructures();
	asBuilder.deinit();
	sbtGenerator.deinit();

	vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
	vkDestroyPipeline(device, rtPipeline, nullptr);
	rtDescPack.deinit();
	Application::allocator.destroyBuffer(sbtBuffer);
};
void FzbRenderer::PathTracingRenderer::uiRender() {
	namespace PE = nvgui::PropertyEditor;
	if (ImGui::Begin("Viewport"))
		ImGui::Image(ImTextureID(gBuffers.getDescriptorSet(eImgTonemapped)), ImGui::GetContentRegionAvail());
	ImGui::End();
};
void FzbRenderer::PathTracingRenderer::resize(VkCommandBuffer cmd, const VkExtent2D& size) { NVVK_CHECK(gBuffers.update(cmd, size)); };
void FzbRenderer::PathTracingRenderer::render(VkCommandBuffer cmd) {
	rayTraceScene(cmd);
	postProcess(cmd);
};
void FzbRenderer::PathTracingRenderer::postProcess(VkCommandBuffer cmd) {
	NVVK_DBG_SCOPE(cmd);
	Application::tonemapper.runCompute(cmd, gBuffers.getSize(), Application::tonemapperData, gBuffers.getDescriptorImageInfo(eImgRendered),
		gBuffers.getDescriptorImageInfo(eImgTonemapped));
	nvvk::cmdMemoryBarrier(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT);
};
void FzbRenderer::PathTracingRenderer::onLastHeadlessFrame() {
	Application::app->saveImageToFile(gBuffers.getColorImage(eImgTonemapped), gBuffers.getSize(),
		nvutils::getExecutablePath().replace_extension(".jpg").string());
};

void FzbRenderer::PathTracingRenderer::compileAndCreateShaders() {
	createRayTracingPipeline();
};