#pragma once

#include "common/Shader/shaderStructType.h"

#ifndef FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
#define FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

enum RasterVoxelizationBindingPoints
{
	eTextures_RV = 0,
	eVGB_RV = 1,
};

struct RasterVoxelizationPushConstant
{
	float4x4 VP[3];
	float4 voxelSize_Count;
	float4 voxelGroupStartPos;
	//float3x3 normalMatrix;
	int instanceIndex;
	SceneInfo* sceneInfoAddress;           // Address of the scene information buffer
};

struct AABBU {
	uint3 minimum;
	uint3 maximum;
};
struct VGBVoxelData {
	float4 radiance;
	float4 sumNormal_G;
	float4 sunNormal_E;
	AABBU aabbU;
};

NAMESPACE_SHADERIO_END()
#endif