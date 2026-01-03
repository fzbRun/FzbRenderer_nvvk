#include "AccelerationStructure.h"
#include <common/Application/Application.h>
#include <nvvk/resource_allocator.hpp>
#include <glm/gtc/type_ptr.hpp>

using namespace FzbRenderer;

void AccelerationStructureManager::init() {
	asBuilder.init(&Application::allocator, &Application::stagingUploader, Application::app->getQueue(0));
	createBottomLevelAS_nvvk();
	createTopLevelAS_nvvk();
}
void AccelerationStructureManager::clean() {
	VkDevice device = Application::app->getDevice();
	for (int i = 0; i < blasAccel.size(); ++i) Application::allocator.destroyBuffer(blasAccel[i].buffer);
	Application::allocator.destroyBuffer(tlasAccel.buffer);

	asBuilder.deinitAccelerationStructures();
	asBuilder.deinit();
}

void AccelerationStructureManager::primitiveToGeometry(const shaderio::Mesh& mesh,
	VkAccelerationStructureGeometryKHR& geometry, VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {

	const shaderio::TriangleMesh triMesh = mesh.triMesh;
	const auto triangleCount = static_cast<uint32_t>(triMesh.indices.count / 3U);

	//从一个大的buffer中将顶点和索引找到，构成三角形的信息
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = {.deviceAddress = VkDeviceAddress(mesh.dataBuffer) + triMesh.positions.offset},
		.vertexStride = triMesh.positions.byteStride,
		.maxVertex = triMesh.positions.count - 1,
		.indexType = VkIndexType(mesh.indexType),
		.indexData = {.deviceAddress = VkDeviceAddress(mesh.dataBuffer) + triMesh.indices.offset},
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
nvvk::AccelerationStructureGeometryInfo AccelerationStructureManager::primitiveToGeometry_nvvk(const shaderio::Mesh& mesh) {
	//这个函数就和我之前cuda实现BVH的思路一摸一样啊

	nvvk::AccelerationStructureGeometryInfo result{};

	const shaderio::TriangleMesh triMesh = mesh.triMesh;
	const auto triangleCount = static_cast<uint32_t>(triMesh.indices.count / 3U);

	//从一个大的buffer中将顶点和索引找到，构成三角形的信息
	VkAccelerationStructureGeometryTrianglesDataKHR triangles{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
		.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
		.vertexData = {.deviceAddress = VkDeviceAddress(mesh.dataBuffer) + triMesh.positions.offset},
		.vertexStride = triMesh.positions.byteStride,
		.maxVertex = triMesh.positions.count - 1,
		.indexType = VkIndexType(mesh.indexType),
		.indexData = {.deviceAddress = VkDeviceAddress(mesh.dataBuffer) + triMesh.indices.offset},
	};

	//然后使用三角形创建BVH
	result.geometry = VkAccelerationStructureGeometryKHR{
		.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
		.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
		.geometry = {.triangles = triangles},
		.flags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR,	//这表明一个三角形不会被多次命中
	};

	result.rangeInfo = VkAccelerationStructureBuildRangeInfoKHR{ .primitiveCount = triangleCount };

	return result;
}

void AccelerationStructureManager::createAccelerationStructure(VkAccelerationStructureTypeKHR asType, nvvk::AccelerationStructure& accelStruct,
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

	VkPhysicalDeviceAccelerationStructurePropertiesKHR asProperties{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR };
	VkPhysicalDeviceProperties2 prop2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
	prop2.pNext = &asProperties;
	vkGetPhysicalDeviceProperties2(Application::app->getPhysicalDevice(), &prop2);

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

void AccelerationStructureManager::createBottomLevelAS() {
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
void AccelerationStructureManager::createBottomLevelAS_nvvk() {
	SCOPED_TIMER(__FUNCTION__);
	LOGI("Ready to build %zu bottom-level acceleration structures\n", Application::sceneResource.meshes.size());

	std::vector<nvvk::AccelerationStructureGeometryInfo> geoInfos(Application::sceneResource.meshes.size());
	for (uint32_t blasId = 0; blasId < Application::sceneResource.meshes.size(); ++blasId)
		geoInfos[blasId] = primitiveToGeometry_nvvk(Application::sceneResource.meshes[blasId]);

	asBuilder.blasSubmitBuildAndWait(geoInfos, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
		VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR);

	LOGI("Bottom-level acceleration structures built successfully\n");
}

void AccelerationStructureManager::createTopLevelAS() {
	SCOPED_TIMER(__FUNCTION__);

	auto toTransformMatrixKHR = [](const glm::mat4& m) {
			VkTransformMatrixKHR t;
			//glm的矩阵是列矩阵，而VkTransformMatrixKHR是行矩阵，所以需要先转置
			memcpy(&t, glm::value_ptr(glm::transpose(m)), sizeof(t));	
			return t;
	};

	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances;
	tlasInstances.reserve(Application::sceneResource.instances.size());		//每个mesh一个顶层实例
	for (const shaderio::Instance& instance : Application::sceneResource.instances) {
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
			.data = {.deviceAddress = tlasInstancesBuffer.address },
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
void AccelerationStructureManager::createTopLevelAS_nvvk() {
	SCOPED_TIMER(__FUNCTION__);

	FzbRenderer::Scene& sceneResorce = Application::sceneResource;

	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(0);
	tlasInstances.reserve(sceneResorce.instances.size() + sceneResorce.dynamicInstances.size());

	staticTlasInstances.resize(0);
	staticTlasInstances.reserve(Application::sceneResource.instances.size());		//每个mesh一个顶层实例
	const VkGeometryInstanceFlagsKHR flgas{ VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV };	//没有背面剔除
	for (const shaderio::Instance& instance : Application::sceneResource.instances) {
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = nvvk::toTransformMatrixKHR(instance.transform);
		asInstance.instanceCustomIndex = instance.meshIndex;
		asInstance.accelerationStructureReference = asBuilder.blasSet[instance.meshIndex].address;
		asInstance.instanceShaderBindingTableRecordOffset = 0;		//实例用SBT中hitGroup中第i个条目（shader）
		asInstance.flags = flgas;
		asInstance.mask = 0xFF;
		staticTlasInstances.emplace_back(asInstance);
	}
	tlasInstances.insert(tlasInstances.end(), staticTlasInstances.begin(), staticTlasInstances.end());

	for (const shaderio::Instance& instance : Application::sceneResource.dynamicInstances) {
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = nvvk::toTransformMatrixKHR(instance.transform);
		asInstance.instanceCustomIndex = instance.meshIndex;
		asInstance.accelerationStructureReference = asBuilder.blasSet[instance.meshIndex].address;
		asInstance.instanceShaderBindingTableRecordOffset = 0;		//实例用SBT中hitGroup中第i个条目（shader）
		asInstance.flags = flgas;
		asInstance.mask = 0xFF;
		tlasInstances.emplace_back(asInstance);
	}

	LOGI("Ready to build top-level acceleration structure with %zu instances\n", tlasInstances.size());

	VkBuildAccelerationStructureFlagsKHR flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	if (Application::sceneResource.dynamicInstances.size() > 0) flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
	asBuilder.tlasSubmitBuildAndWait(tlasInstances, flags);

	LOGI("Top-level accleration structures built successfully\n");
}

void AccelerationStructureManager::updateTopLevelAS_nvvk() {
	if (Application::sceneResource.dynamicInstances.size() == 0) return;

	FzbRenderer::Scene& sceneResorce = Application::sceneResource;

	std::vector<VkAccelerationStructureInstanceKHR> tlasInstances(0);
	tlasInstances.reserve(sceneResorce.instances.size() + sceneResorce.dynamicInstances.size());
	tlasInstances.insert(tlasInstances.end(), staticTlasInstances.begin(), staticTlasInstances.end());

	const VkGeometryInstanceFlagsKHR flgas{ VK_GEOMETRY_INSTANCE_TRIANGLE_CULL_DISABLE_BIT_NV };	//没有背面剔除
	for (const shaderio::Instance& instance : Application::sceneResource.dynamicInstances) {
		VkAccelerationStructureInstanceKHR asInstance{};
		asInstance.transform = nvvk::toTransformMatrixKHR(instance.transform);
		asInstance.instanceCustomIndex = instance.meshIndex;
		asInstance.accelerationStructureReference = asBuilder.blasSet[instance.meshIndex].address;
		asInstance.instanceShaderBindingTableRecordOffset = 0;		//实例用SBT中hitGroup中第i个条目（shader）
		asInstance.flags = flgas;
		asInstance.mask = 0xFF;
		tlasInstances.emplace_back(asInstance);
	}

	NVVK_CHECK(vkDeviceWaitIdle(Application::app->getDevice()));	//等待GPU指令处理完成
	asBuilder.tlasSubmitUpdateAndWait(tlasInstances);
}