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
struct VGBVoxelData {
	float4 radiance;
	float4 sumNormal_G;
	float4 sumNormal_E;
	AABBU aabbU;
};

NAMESPACE_SHADERIO_END()
#endif