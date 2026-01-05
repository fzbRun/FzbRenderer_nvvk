#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHTRACING_SHADER_IO_H
#define FZBRENDERER_PATHTRACING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct PathTracingPushConstant
{
	int NEEShaderIndex = -1;
	int frameIndex = 0;
	int maxDepth = 3;
	float time = 0.0f;
	SceneInfo* sceneInfoAddress;           // Address of the scene information buffer
};

NAMESPACE_SHADERIO_END()
#endif