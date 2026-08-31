#pragma once

#include <common/Shader/shaderStructType.h>

#define eps_kpcnn  0.00316 

//#define SAVE_TRAIN_BUFFERS
#ifdef SAVE_TRAIN_BUFFERS
#define IF_TRAIN_SAMPLE(train, eval) (train)
#else
#define IF_TRAIN_SAMPLE(train, eval) (eval)
#endif

//#define SAVE_SAMPLE_BUFFERS
//#define SAVE_GROUNDTRUTH_BUFFERS

#if defined(SAVE_SAMPLE_BUFFERS) && defined(SAVE_TRAIN_BUFFERS)
#define IF_SAVE_SAMPLE(sample, gt) (sample)
#elif defined(SAVE_GROUNDTRUTH_BUFFERS) && defined(SAVE_TRAIN_BUFFERS)
#define IF_SAVE_SAMPLE(save, gt) (gt)
#else 
#define IF_SAVE_SAMPLE(sample, gt)
#endif

#define KERNEL_SIZE 9

#ifndef FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
#define FZBRENDERER_KPCNN_DENOSING_PATHTRACING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

struct KPCNN_DenoisingPTPushConstant {
	int frameIndex;
	int maxFrameCount;
	int spp = 16;
	float time;
	int maxBounceCount = 10;
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
	eMaxDepthBuffer,
	eAlbedoBuffer,

#ifndef NDEBUG
	eColorDebugImage,
	eDiffuseDebugImage,
	eSpecularDebugImage,
	eIrradianceDebugImage,
	eNormalDebugImage,
	eDepthDebugImage,
	eAlebdoDebugImage,

	eIrradianceVarianceDebugImage,
	eSpecularVariancDebugImage,
	eNormalVarianceDebugImage,
	eDepthVarianceDebugImage,
	eAlbedoVarianceDebugImage,
#endif
};

#define PATHTRACING_BLOCKSIZE_KPCNN 16

NAMESPACE_SHADERIO_END()
#endif