#pragma once

#include "common/Shader/shaderStructType.h"

#ifndef FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
#define FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

enum RasterVoxelizationBindingPoints
{
	eTextures_RV = 0,
	eVGB_RV = 1,
	eFragmentCountBuffer_RV = 2,
	eWireframeMap_RV = 3,
	eBaseMap = 4,
};

struct RasterVoxelizationPushConstant
{
	//float3x3 normalMatrix;
	float4x4 VP[3];
	float4 voxelSize_Count;
	float3 voxelGroupStartPos;
	int instanceIndex;
	int frameIndex;		//debug
	SceneInfo* sceneInfoAddress;           // Address of the scene information buffer
};

struct AABBU {
	uint3 minimum;
	uint3 maximum;
};
/*
G和E共用光栅体素化的aabb，而不是在lightInject中得到E的AABB
好处是
1. lightInject中不需要对aabb进行原子运算
2. 可以等待lightInject后统一对G和E的八叉树节点进行初始化，而不需要rasterVoxelization后初始化G -> lightInject后初始化E
3. E的AABB用的是G的，好处是避免样本较少使得E的aabb较小
坏处是
1. E的AABB用的是G的，对于差别较大的地方存在误差
2. 不能复用sumNormal，本来可以先得到sumNormal_G初始化G，然后lightInject后得到sumNormal_E来初始化E
*/
struct VGBVoxelData {
	float4 irradiance;
	float4 sumNormal_G;
	float4 sumNormal_E;
	AABBU aabbU;
	uint32_t materialType_Count;	//first 27 bite is materialCount, last five bite is materialType(assume max 16 material)
};

NAMESPACE_SHADERIO_END()
#endif