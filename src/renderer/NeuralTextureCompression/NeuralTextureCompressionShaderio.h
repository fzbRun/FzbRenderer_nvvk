#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_Neural_Texture_Compression_SHADER_IO_H
#define FZBRENDERER_Neural_Texture_Compression_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define AUTO_DIFF
#define MIPMAP

struct NeuralTextureCompressionPushConstant {
	int2 screenSize;
	int showTextureType = 0;
	SceneInfo* sceneInfoAddress;
};

enum class StaticBindingPoints_NTC {
	eSourceTextures = 0,
	eMipmapSourceTextures,
	eRenderedTexture_Reduced,
	eRenderedTexture_Full,
	eRenderedTexture_Full_2,
	eLossTexture,
	eTonemappedTexture,
};

NAMESPACE_SHADERIO_END()
#endif
