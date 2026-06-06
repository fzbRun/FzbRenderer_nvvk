#pragma once

#include <common/Shader/shaderStructType.h>

#define eps_kpcnn  0.00316 

#ifndef FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
#define FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct KPCNN_DenoisingPTPushConstant {
	int frameIndex;
	int maxFrameCount;
	uint spp;
	float time;
	int maxBounceCount;
	uint2 screenSize;
	SceneInfo* sceneInfoAddress;
};

enum class StaticBindingPoints_KPCNNPT {
	eInputBuffer_irradiance = 2,
	eInputBuffer_irradianceVariance,
	eInputBuffer_gradIrradiance,

	eInputBuffer_normalVariance_diff,
	eInputBuffer_gradNormal_diff,

	eInputBuffer_depthVariance_diff,
	eInputBuffer_gradDepth_diff,

	eInputBuffer_albedoVariance_diff,
	eInputBuffer_gradAlbedo_diff,

	eInputBuffer_spec,
	eInputBuffer_specVariance,
	eInputBuffer_gradSpec,

	eInputBuffer_normalVariance_spec,
	eInputBuffer_gradNormal_spec,

	eInputBuffer_depthVariance_spec,
	eInputBuffer_gradDepth_spec,

	eInputBuffer_albedoVariance_spec,
	eInputBuffer_gradAlbedo_spec,

	eNormalBuffer,
	eDepthBuffer,
	eAlbedoBuffer,
};

#define PATHTRACING_BLOCKSIZE_KPCNN 16

NAMESPACE_SHADERIO_END()
#endif