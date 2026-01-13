#pragma once

#include "common/Shader/shaderStructType.h"
#include <vulkan/vulkan_core.h>
#include <nvvk/resources.hpp>
#include <nvvk/acceleration_structures.hpp>

#ifndef FZBRENDERER_ACCELERATION_STRUCTURE_H
#define FZBRENDERER_ACCELERATION_STRUCTURE_H

namespace FzbRenderer {
class AccelerationStructureManager {
public:
	void init();
	void clean();

	nvvk::AccelerationStructureGeometryInfo primitiveToGeometry_nvvk(const shaderio::Mesh& mesh);
	
	void createBottomLevelAS_nvvk();
	void createBottomLevelMotionAS_nvvk();		//mesh发生形变时有用

	void createTopLevelAS_nvvk();
	void createTopLevelMotionAS_nvvk();

	void updateTopLevelAS_nvvk();
	void updateTopLevelMotionAS_nvvk();

	std::vector<VkAccelerationStructureInstanceKHR> staticTlasInstances;
	std::vector<nvvk::VkAccelerationStructureMotionInstanceNVPad> motionInstances;

	std::vector<nvvk::AccelerationStructure> blasAccel;
	nvvk::AccelerationStructure              tlasAccel;

	nvvk::AccelerationStructureHelper asBuilder{};
private:
	/*
	1. 获取mesh的顶点数据，放入VkAccelerationStructureGeometryTrianglesDataKHR，得到一个几何数据
	2. 将几何的信息放入VkAccelerationStructureGeometryKHR
	3. 最后得到加速结构中顶点数据的数量，得到创建范围信息VkAccelerationStructureBuildRangeInfoKHR
	*/
	void primitiveToGeometry(const shaderio::Mesh& mesh,
		VkAccelerationStructureGeometryKHR& geometry, VkAccelerationStructureBuildRangeInfoKHR& rangeInfo);

	/*
	1. 根据参数给定的信息(as类型、几何数据、创建范围信息和flags)得到一个创建几何信息VkAccelerationStructureBuildGeometryInfoKHR
	2. 根据创建几何信息和创建范围信息的几何数量算出创建as所需要的空间信息(as所需空间和创建as所需的临时空间)VkAccelerationStructureBuildSizesInfoKHR
	3. 获取设备参数，得到临时空间的最小偏移对齐，得到最终的临时空间大小，创造临时缓冲
	4. 调用vkCreateAccelerationStructureKHR函数，创造as缓冲（分配空间）
	5. 调用vkCmdBuildAccelerationStructuresKHR函数，构建与填充as缓冲（构建数据结构并填入数据）
	*/
	void createAccelerationStructure(VkAccelerationStructureTypeKHR asType, nvvk::AccelerationStructure& accelStruct,
		VkAccelerationStructureGeometryKHR& asGeometry, VkAccelerationStructureBuildRangeInfoKHR& asBuildRangeInfo,
		VkBuildAccelerationStructureFlagsKHR flags);

	/*
	1. 调用primitiveToGeometry函数为所有的mesh创建几何信息和创建范围信息
	2. 调用createAccelerationStructure函数创建底层加速结构
	*/
	void createBottomLevelAS();

	/*
	1. 底层加速结构需要一个mesh一个mesh的创造；而顶层加速结构同样需要一次性对所有的instance进行创造
	2. 首先为所有的instance创造VkAccelerationStructureInstanceKHR
	3. 将VkAccelerationStructureInstanceKHR数组创建为一个buffer
	4. 根据buffer的地址创建VkAccelerationStructureGeometryKHR；根据instance的数量创建VkAccelerationStructureBuildRangeInfoKHR
	5. 调用createAccelerationStructure创建加速结构
	*/
	void createTopLevelAS();
};
}

#endif