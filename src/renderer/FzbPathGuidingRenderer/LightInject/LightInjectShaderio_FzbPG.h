#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/PathTracing/shaderio.h"

#ifndef FZBRENDERER_LIGHTINJECT_FZBPG_SHADER_IO_H
#define FZBRENDERER_LIGHTINJECT_FZBPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define LIGHTINJECT_SAMPLE_COUNT 32

struct LightInjectPushConstant_FzbPG {
	float4 VGBVoxelSize;
	float3 VGBStartPos;
	int frameIndex = 0;
	float3 sceneSize;
	int voxelCount;
	SceneInfo* sceneInfoAddress;
	float3x3 randomRotateMatrix;
	float time;
#ifndef NDEBUG
	uint normalIndex;
#endif
};
enum class StaticBindingPoints_LightInject_FzbPG {
	eVGB = 2,
	eHasGeometryVoxelInfo,
	eGlobalInfo,
};

struct HasGeometryVoxelInfo {
	uint voxelIndex_normalIndex;
};
struct LightInjectGlobalInfo {
	DispatchIndirectCommand cmd;
	uint hasGeometryVoxelCount;
};

#define LIGHTINJECT_CS_THREADGROUP_SIZE 256

NAMESPACE_SHADERIO_END()
#endif