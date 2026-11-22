#pragma once

#include <common/Shader/shaderStructType.h>
#include "feature/PathTracing/shaderio.h"

#ifndef FZBRENDERER_LIGHTINJECT_VRPG_SHADER_IO_H
#define FZBRENDERER_LIGHTINJECT_VRPG_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define LIGHTINJECT_SAMPLE_COUNT 32
#define LIGHTINJECT_SAMPLE_COUNT_STATIC_SCENE 512

struct LightInjectPushConstant_VRPG {
	int frameIndex = 0;
	int randomFrameIndex = 0;

	uint spv;
	int sampleIndex;
	uint bounces;

	int nodeCount;
	float4 nodeSize;
	float4 nodeGridStartPos;

	SceneInfo* sceneInfoAddress;
	float3x3 randomRotateMatrix;

#ifndef NDEBUG
	uint normalIndex;
#endif
};
enum class StaticBindingPoints_LightInject_VRPG {
	eGeoClusterData = 2,
	eHasGeometryNodeInfo_VRPG,
	eGlobalInfo,
	eLightInjectionGlobalInfo,
	eLightNodeData,
};

struct HasGeometryNodeInfo_VRPG {
	uint nodeIndex;
};
struct LightInjectGlobalInfo_VRPG {
	uint hasGeometryNodeCount;
};

#define LIGHTINJECT_CS_THREADGROUP_SIZE 256

//------------------------------------------------------------------------------------------
// Debug GBuffer of the light injection (VRPG: LightInjectGBuffer::LightInjectResult). The
// light injection debug draw renders the per-voxel irradiance into this color target.
enum class LightInjectGBuffer_VRPG : uint32_t {
	eInjectionResult = 0,
	eBufferCount,
};

NAMESPACE_SHADERIO_END()
#endif