#pragma once

#include "common/Shader/shaderStructType.h"

#ifndef FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
#define FZBRENDERER_RASTER_VOXELIZATION_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct RasterVoxelizationPushConstant
{
	float3x3 normalMatrix;
	uint3 voxelSize;
	float3 voxelCount;
	float3 voxelGroupStartPos;
	SceneInfo* sceneInfoAddress;           // Address of the scene information buffer
};

NAMESPACE_SHADERIO_END()
#endif