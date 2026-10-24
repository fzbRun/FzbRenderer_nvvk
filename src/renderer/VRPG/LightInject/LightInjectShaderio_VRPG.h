#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/PathTracing/shaderio.h"

#ifndef FZBRENDERER_LIGHTINJECT_VRPG_SHADER_IO_H
#define FZBRENDERER_LIGHTINJECT_VRPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define LIGHTINJECT_SAMPLE_COUNT 32
#define LIGHTINJECT_SAMPLE_COUNT_STATIC_SCENE 512

struct LightInjectPushConstant_VRPG {
	float4 VGBVoxelSize;       // light VGB: xyz size, w unused
	float3 VGBStartPos;        // light VGB
	int frameIndex = 0;
	float3 sceneSize;
	int voxelCount;            // light VGB total voxel count (count^3)
	int lightVoxelCount;       // light VGB count per axis

	float4 geometryVGBVoxelSize; // geometry VGB: xyz size, w count
	float3 geometryVGBStartPos;

	SceneInfo* sceneInfoAddress;
	float3x3 randomRotateMatrix;
	float time;
	uint sampleCount;
#ifndef NDEBUG
	uint normalIndex;
#endif
};
enum class StaticBindingPoints_LightInject_VRPG {
	eVGB = 2,
	eGeometryVGB,
	eHasGeometryVoxelInfo_VRPG,
	eGlobalInfo,
};

struct HasGeometryVoxelInfo_VRPG {
	uint voxelIndex_normalIndex;
};
struct LightInjectGlobalInfo_VRPG {
	DispatchIndirectCommand cmd;
	uint hasGeometryVoxelCount;
};

#define LIGHTINJECT_CS_THREADGROUP_SIZE 256

NAMESPACE_SHADERIO_END()
#endif