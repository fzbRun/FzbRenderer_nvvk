#pragma once

#include <common/Shader/shaderStructType.h>
#include <feature/SceneDivision/RasterVoxelization/shaderio.h>
#include <feature/SceneDivision/SparseVoxelOctree/shaderio.h>

#ifndef FZBRENDERER_PATHGUIDING_SHADER_IO_H
#define FZBRENDERER_PATHGUIDING_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define SVO_PATHGUIDING_THREADGROUP_SIZE_X 32
#define SVO_PATHGUIDING_THREADGROUP_SIZE_Y 32
struct SVOPathGuidingPushConstant
{
	float4 VGBStartPos_Size;
	int frameIndex = 0;
	int maxDepth = 3;
	float time;
	SceneInfo* sceneInfoAddress;
	uint2 sceneSize;
	uint2 threadGroupCount;
};

enum StaticBindingPoints_SVOPG
{
	//eTextures = 0,
	//eOutImage = 1,	//First save the result of lightInject, and finally MIS
	eSVO_G_SVOPG = 2,
	eSVO_E_SVOPG,
	eGlobalInfo_SVOPG,
	eWeights_SVOPG,
#ifndef NDEBUG
	eDepthImage_SVOPG,
#endif
};
enum DynamicBindingPoints_SVOPG {
	//eTlas_SVOPG = 0,
	eSVOTlas_SVOPG = 1,
};

struct GlobalInfo_SOVPG {
	uint SVOMaxLayer_G;
	uint SVOMaxLayer_E;
	uint indivisibleNodeCount_G;
	uint totalNodeCount_E;
};
struct SVONodeInfo_G_SVOPG {
	uint label_indivisible;
	AABB aabb;
};
struct SVONodeInfo_E_SVOPG {
	uint label_indivisible;
	AABB aabb;
};
//-------------------------------------------SVORasterVoxelization----------------------------------------
enum RasterVoxelizationBindingPoints_SVOPG
{
	eTextures_SVOPG = 0,
	eVGB_SVOPG,
	eVGBMaterialInfo_SVOPG,
	eFragmentCountBuffer_SVOPG,
	eWireframeMap_SVOPG,
	eBaseMap_SVOPG,
};

struct VGBMaterialInfo_SVOPG {
	uint materialCount[MAX_MATERIAL_COUNT];
};

struct AABBI {
	int4 minimum;
	int4 maximum;
};
struct VGBVoxelData_SVOPG {
	float4 irradiance;
	float4 sumNormal_G;
	float4 sumNormal_E;
	AABBI aabbI;
};
//-------------------------------------------SVO----------------------------------------
#define SVOSize_G 5000	//512 * 6
#define SVOSize_E 800
#if SVOSize_G > SVOSize_E
#define SVOSize SVOSize_G
#else
#define SVOSize SVOSize_E
#endif

enum class BindingPoints_SVOPG {
	eOctreeArray_G_SVOPG = 0,
	eOctreeArray_E_SVOPG,
	eSVO_G_SVOPG,
	eSVO_E_SVOPG,
	eSVOGlobalInfo_SVOPG,
	eSVODivisibleNodeIndices_G_SVOPG,
	eSVODivisibleNodeIndices_E_SVOPG,
	eSVOThreadGroupInfos_SVOPG
};

struct SVOPushConstant_SVOPG {
	uint32_t maxDepth_Octree;
	uint32_t currentDepth_SVO;
#ifndef NDEBUG
	float frameIndex;
	SceneInfo* sceneInfoAddress;
#endif
};

struct SVOGlobalInfo_SVOPG {
	DispatchIndirectCommand cmd;	//CS Dispatch size
#ifndef NDEBUG
	DrawIndexedIndirectCommand drawCmd;
#endif
	uint32_t totalNodeCount_G;
	uint32_t totalNodeCount_E;
	SVOLayerInfo layerInfos_G[MAX_OCTREE_DEPTH];
	SVOLayerInfo layerInfos_E[MAX_OCTREE_DEPTH];
};
//-------------------------------------------SVOWeight----------------------------------------
#define SVOIndivisibleNodeCount_G 2000
#define SVOIndivisibleNodeCount_E 500

#define HITTEST_COUNT 8
#define OUTGOING_COUNT 64
struct SVOWeightPushConstant {
	uint32_t frameIndex;
	uint32_t layerNodeMaxCount_E;
	SceneInfo* sceneInfoAddress;
	uint32_t countdown;
#ifndef NDEBUG
	float3 samplePos;
	float3 outgoing;
	uint instanceIndex;
#endif
};

enum StaticBindingPoints_SVOWeight {
	eSVO_G_SVOWeight = 2,
	eSVO_E_SVOWeight,
	eSVOGlobalInfo_SVOWeight,
	eGlobalInfo_SVOWeight,
	eSVO_IndivisibleNodeInfos_G_SVOWeight,
	eSVO_IndivisibleNodeInfos_E_SVOWeight,
	eSVOWeights_SVOWeight,
	eSVOWeightSums_SVOWeight,
};

struct SVOWeightGlobalInfo {
	DispatchIndirectCommand cmd;
	uint indivisibleNodeCount_G;
	uint indivisibleNodeCount_E;
	uint SVOMaxLayer_G;
	uint SVOMaxLayer_E;
	uint totalNodeCount_E;
#ifndef NDEBUG
	AABB sampelNodeAABB;
	uint sampelNodeLabel;
#endif
	SVOLayerInfo layerInfos_G[MAX_OCTREE_DEPTH];
	SVOLayerInfo layerInfos_E[MAX_OCTREE_DEPTH];
};
struct SVOIndivisibleNodeInfo {
	uint32_t layerIndex;
	uint32_t nodeIndex;
};
struct IndivisibleNodeData_E {
	float3 normal;
	uint singleSide;
	float3x3 TBN;
	uint nodeIndex;
	float irradiance;
	AABB aabb;
};
struct IndivisibleNodeData_G {
	float3 normal;
	uint singleSide;
	float3x3 TBN;
	uint nodeLabel;
	uint materialType;
	AABB aabb;
};

NAMESPACE_SHADERIO_END()
#endif