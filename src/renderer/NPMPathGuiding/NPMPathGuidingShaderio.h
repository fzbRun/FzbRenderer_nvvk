#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_NPMPATHGUIDING_SHADER_IO_H
#define FZBRENDERER_NPMPATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct NPMPathGuidingPushConstant{
	int frameIndex;
	int time;
	uint2 screenSize;
};

enum class StaticBindingPoints_NPMPG {
	eOutImage = 0,
	eFlowerImage = 1,
	eColorImageWrite = 2,
	eColorImageRead = 3,
};

NAMESPACE_SHADERIO_END()
#endif