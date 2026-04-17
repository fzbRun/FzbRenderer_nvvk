#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/PathTracing/shaderio.h"

#ifndef FZBRENDERER_LIGHTINJECT_SVOPG_SHADER_IO_H
#define FZBRENDERER_LIGHTINJECT_SVOPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SAMPLE_OUTPUT_COUNT 8

struct LightInjectPushConstant_SVOPG {
	float4 VGBVoxelSize;
	float4 VGBStartPos_Size;
	float3 sceneStartPos;
	int frameIndex = 0;
	float3 sceneSize;
	float time;
	SceneInfo* sceneInfoAddress;
};
enum StaticBindingPoints_LightInject_SVOPG {
	eVGB_LightInject_SVOPG = 2,
};

NAMESPACE_SHADERIO_END()
#endif