#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/PathTracing/shaderio.h"

#ifndef FZBRENDERER_LIGHTINJECT_SHADER_IO_H
#define FZBRENDERER_LIGHTINJECT_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct LightInjectPushConstant {
	float4 VGBVoxelSize;
	float4 VGBStartPos_Size;
	int frameIndex = 0;
	float time;
	SceneInfo* sceneInfoAddress;
};
enum StaticBindingPoints_LightInject {
	eVGB_LightInject = 2,
};

NAMESPACE_SHADERIO_END()
#endif