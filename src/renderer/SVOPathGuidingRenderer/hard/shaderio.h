#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct SVOPathGuidingPushConstant
{
	float4 VGBStartPos_Size;


	int LightInjectShaderIndex = -1;
	int NEEShaderIndex = -1;
	int frameIndex = 0;
	int maxDepth = 3;
	float time;
	float3x3 normalMatrix;
	SceneInfo* sceneInfoAddress;
};

enum StaticBindingPoints_SVOPG
{
	//eTextures = 0,
	//eOutImage = 1,	//First save the result of lightInject, and finally MIS
	eVGB_SVOPG = 2,
	eSVO_G_SVOPG,
	eSVO_E_SVOPG,
	eWeights_SVOPG,
	eOutImage_MIS_SVOPG
};
enum DynamicBindingPoints_SVOPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG,
};

NAMESPACE_SHADERIO_END()
#endif