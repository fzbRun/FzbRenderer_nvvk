#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_ZHIHUCODE_SHADER_IO_H
#define FZBRENDERER_ZHIHUCODE_SHADER_IO_H

//#define Step1_VulkanCudaOp
#define Step2_UseModel

NAMESPACE_SHADERIO_BEGIN()

struct ZhiHuCodePushConstant {
	int frameIndex;
	uint2 screenSize;
};

enum class StaticBindingPoints_ZhiHuCode {
	eOutImage = 0,
	eFlowerImage = 1,
	eInputBuffer = 2,
};

NAMESPACE_SHADERIO_END()
#endif