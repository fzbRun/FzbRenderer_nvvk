#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
#define FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct KPCNN_DenoisingPTPushConstant {
	int frameIndex;
	int time;
	uint2 screenSize;
};

enum class StaticBindingPoints_KPCNPT {
	eOutImage = 0,
	eFlowerImage = 1,
	eInputTensor = 2,
};

NAMESPACE_SHADERIO_END()
#endif