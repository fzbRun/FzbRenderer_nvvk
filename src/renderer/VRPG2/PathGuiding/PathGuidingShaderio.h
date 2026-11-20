#pragma once

#include <common/Shader/shaderStructType.h>
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"

#ifndef FZBRENDERER_VRPG2_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_VRPG2_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

//------------------------------------------------------------
// Path guiding compute pass: path tracing plus MIS between a BSDF sample and a guide sample that is
// drawn from the RIS candidate tables of the current pixel's geometric cluster.
//------------------------------------------------------------
#define FZB_PATHGUIDING_THREADGROUP_SIZE_X 16
#define FZB_PATHGUIDING_THREADGROUP_SIZE_Y 16

#ifndef VRPG_PATH_SAMPLE_MODE
// 1 = one sample technique choice with balance heuristic MIS (see pathGuiding.slang)
#define VRPG_PATH_SAMPLE_MODE 1
#endif
#ifndef VRPG_RANDOM_FRAME_SOURCE
#define VRPG_RANDOM_FRAME_SOURCE 1
#endif
// no spatiotemporal reuse and no neighbour jitter in this renderer
#define VRPG_SPATIAL_REUSE 0
#define VRPG_TEMPORAL_REUSE 0
#define VRPG_JITTER_SAMPLING 0
#ifndef NEARBY_NODE_COUNT_VRPG
#define NEARBY_NODE_COUNT_VRPG 4
#endif

struct PathGuidingPushConstant_VRPG2
{
	int maxFrameCount;
	int frameIndex = 0;
	int randomFrameIndex = 0;
	int staticScene = 0;
	float time;

	int maxDepth = 6;
	int spp = 1;

	int spatialRuse = 0;
	int ditherSample = 0;
	int nearbyNodeCount = NEARBY_NODE_COUNT_VRPG;
	int pathSampleMode = VRPG_PATH_SAMPLE_MODE;
	float jitterRadiusScale = 0.0f;

	float3 VGBVoxelSize;
	int maxOctreeLayer;
	float4 VGBStartPos_Size;

	float4 lightVGBStartPos_Size;
	float4 lightVGBVoxelSize;
	int lightMaxOctreeLayer;

	float3x3 randomRotateMatrix;
	uint2 dispatchOffset;
	int padding1;

	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;
};

enum class PathGuidingBindingPoints_VRPG2 : uint32_t
{
	eClusterInfo_Octree = 2,
	eLightNodeData_Octree,

	eCandidateSubTreeInfo,
	eCandidateLeafNodeData,

	eCandidateSampleData,
	eCandidateSamplePrefixWSum,

	eGlobalInfo,

	eClusterData_Octree,

	eOutImage = 0,
	eTextures = 1,
};

NAMESPACE_SHADERIO_END()
#endif
