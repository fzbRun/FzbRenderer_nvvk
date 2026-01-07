#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct SVOPathGuidingPushConstant
{
	int NEEShaderIndex = -1;
	int frameIndex = 0;
	int maxDepth = 3;
	float time;
	float3x3 normalMatrix;
	SceneInfo* sceneInfoAddress;
};

enum SVOPGBindingPoints
{
	//eTextures = 0,
	//eTlas_SVOPG = 1,
	//eOutImage = 2,	//First save the result of lightInject, and finally MIS
	eVGB_SVOPG = 3,		
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