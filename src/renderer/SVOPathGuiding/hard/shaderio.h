#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct SVOPathGuidingPushConstant
{
	float3x3 normalMatrix;
	int NEEShaderIndex = -1;
	int frameIndex = 0;
	int maxDepth = 3;
	SceneInfo* sceneInfoAddress;
};

enum SVOPGBindingPoints
{
	eTextures_SVOPG = 0,
	eTlas_SVOPG,
	eVGB_SVOPG,
	eOutImage_SVOPG,		//First save the result of lightInject, and finally MIS
	eOctree_G_SVOPG,
	eOctree_E_SVOPG,
	eSVO_G_SVOPG,
	eSVO_E_SVOPG,
	eSVOTlas_SVOPG,
	eWeights_SVOPG,
	eOutImage_MIS_SVOPG
};

NAMESPACE_SHADERIO_END()
#endif