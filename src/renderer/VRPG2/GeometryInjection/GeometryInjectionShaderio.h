#pragma once

#include <common/Shader/shaderStructType.h>
#include <feature/SceneDivision/RasterVoxelization/shaderio.h>

#ifndef FZBRENDERER_Geometry_Injection_VRPG_SHADER_IO_H
#define FZBRENDERER_Geometry_Injection_VRPG_SHADER_IO_H

NAMESPACE_SHADERIO_BEGIN()

struct GeometryInjectionPushConstant_VRPG
{
	float4x4 VP[3];
	float4 voxelSize_Count;
	float3 voxelGroupStartPos;
	int instanceIndex;
	SceneInfo* sceneInfoAddress;
	int frameIndex;
};
enum class GeometryInjectionBinding_VRPG
{
	eTextures = 0,
	eVGB,
};

struct VoxelData_VRPG {
	float4 sumNormal_G;
	AABBI aabbI;
};
NAMESPACE_SHADERIO_END()
#endif