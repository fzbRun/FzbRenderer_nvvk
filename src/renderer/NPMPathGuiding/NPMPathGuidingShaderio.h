#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_NPMPATHGUIDING_SHADER_IO_H
#define FZBRENDERER_NPMPATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct NPMPathGuidingPushConstant{
	int frameIndex;
	int time;
};

enum class StaticBindingPoints_NPMPG {
	eFlowerImage = 0,
};

NAMESPACE_SHADERIO_END()
#endif