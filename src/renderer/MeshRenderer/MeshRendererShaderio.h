#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_MESH_RENDERER_SHADER_IO_H
#define FZBRENDERER_MESH_RENDERER_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct MeshRendererPushConstant {
	int x;
};

NAMESPACE_SHADERIO_END()
#endif