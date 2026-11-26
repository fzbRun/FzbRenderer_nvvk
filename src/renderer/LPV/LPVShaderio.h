#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_LPV_SHADER_IO_H
#define FZBRENDERER_LPV_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

#define LPV_MAX_CASCADES 4               // upper bound of the cascade count (array sizes)
#define LPV_SH_COEFF_COUNT 4             // 2 SH bands: Y00, Y1-1, Y10, Y11
#define LPV_CHANNEL_COUNT 3              // R, G, B
#define LPV_VOXEL_FLOATS (LPV_SH_COEFF_COUNT * LPV_CHANNEL_COUNT)  // floats per voxel (12)
#define LPV_GV_FLOATS LPV_SH_COEFF_COUNT                          // blocking potential is monochrome
#define LPV_VOXEL_THREADGROUP 8          // 8^3 = 512 threads for the volume passes
#define LPV_PIXEL_THREADGROUP_X 16
#define LPV_PIXEL_THREADGROUP_Y 16
#define LPV_RSM_THREADGROUP_X 16
#define LPV_RSM_THREADGROUP_Y 16
#define LPV_MAX_GRID_RESOLUTION 128      // upper bound of the cubic cascade resolution
#define LPV_MAX_ITERATIONS 32
#define LPV_MAX_RSM_RESOLUTION 2048
#define LPV_INV_PI 0.3183098861837907f

#define LPV_FACE_SOLID_ANGLE 2.0943951023931953f

#define LPV_DIRECT_FACE_SOLID_ANGLE 0.4006696846f
#define LPV_SIDE_FACE_SOLID_ANGLE ((LPV_FACE_SOLID_ANGLE - LPV_DIRECT_FACE_SOLID_ANGLE) * 0.25f)

struct LPVPixelData
{
	float4 positionValid;  // xyz: world space primary hit position, w: 1 when the pixel hit geometry
	float4 normal;         // xyz: shading normal (flipped towards the camera), w: unused
	float4 albedo;         // xyz: diffuse albedo used by the indirect term (0 for purely specular BSDFs)
	float4 directColor;    // xyz: emissive + direct lighting, w: unused
	float4 specular;       // xyz: specular reflectance used by the glossy LPV lookup, w: unused
};

struct LPVFrameInfo
{
	// light clip -> world, one entry per punctual light (direction light: orthographic)
	float4x4 lightViewProjInverse[LIGHT_COUNT];

	// per cascade: xyz = world space minimum corner, w = cell size
	float4 cascadeOriginSize[LPV_MAX_CASCADES];

	// per light: x = world area of one orthographic RSM texel,
	//           y = solid angle of one perspective RSM texel, z/w = unused
	float4 lightRsmParams[LIGHT_COUNT];

	// x: cascade count, y: grid resolution N, z: RSM resolution, w: number of entries in the
	// instance size buffer (paper 4.1 small object test)
	int4 info;

	// per cascade: voxels per axis and the world space distance its propagation reaches
	int   cascadeResolution[LPV_MAX_CASCADES];
	float cascadeReach[LPV_MAX_CASCADES];
};

struct LPVPushConstant
{
	int frameIndex;
	int iteration;   // current propagation iteration, also selects the ping pong volume
	int flags;       // LPVFlags
	int cascadeCount;

	SceneInfo* sceneInfoAddress;
	uint2      screenSize;
	int        gridResolution;
	int        rsmResolution;
	int        lightIndex;  // light currently injected by the virtual RSM pass
	float      time;
	float      gatherOcclusion;
	int        cascadeBlendCells;
	int        iterationCount;
	int        cascadeIndex;
	int        cascadeIterations;

	int occTwoFace;
};

enum LPVFlags
{
	eLPVFlagOcclusion = 1,   // fuzzy occlusion from the geometry volume
	eLPVFlagDampening = 2,   // derivative based dampening (paper section 3.4)
	eLPVFlagNoDirect  = 4,   // comparison: drop the direct light, keep the LPV (and glossy) contribution
	eLPVFlagNoShadow  = 64,  // debug: bit 6, drops the direct shadow ray (LPV_NO_SHADOW=1)
	eLPVFlagNearestGather = 128,  // gather reads the nearest cell instead of interpolating trilinearly
	eLPVFlagDebugCascadeId = 256,
	eLPVFlagDebugTransmittance = 512,
	eLPVFlagDebugSH = 1024,
};

enum LPVGeometrySource
{
	eLPVGeometryFromCamera = 0,
	eLPVGeometryFromRsm    = 1,
	eLPVGeometryMerged     = 2,  // max(camera, rsm), only valid after the GvMerge pass
};

enum class LPVStaticBindingPoints
{
	ePixelData = 2,  // RWStructuredBuffer<LPVPixelData>   W*H
	eLPVVolumes,     // RWStructuredBuffer<float>          2 * C * N^3 * 12   (ping pong pair)
	eLPVAccum,       // RWStructuredBuffer<float>          C * N^3 * 12       (sum of all iterations)
	eGvMerged,       // RWStructuredBuffer<float>          C * N^3 * 4
	eGvCamera,       // RWStructuredBuffer<float>          C * N^3 * 4
	eGvRsm,          // RWStructuredBuffer<float>          C * N^3 * 4
	eFrameInfo,      // RWStructuredBuffer<LPVFrameInfo>   1
	eInstanceSize,   // RWStructuredBuffer<float>          instance count, world size of each instance
	eDepthImage,
	eGridDebugData,  
};

struct GridDebugInfo {
	int hasData;
	float3 color;
};
struct CascadeDebugInfo {
	float3 startPos;
	float voxelSize;
	int resolution;
};
struct LPVDebugPushConstant {
	int frameIndex;
	int rsmResolution;
	uint2 screenSize;
	SceneInfo* sceneInfoAddress;

	int cascadeCount;
	int cascadeIndex;
	CascadeDebugInfo gridInfo[LPV_MAX_CASCADES];

	int3 showGridVoxelRangeMin;
	int3 showGridVoxelRangeMax = { 31, 31, 31 };

	int GBufferIndex;
	int lightIndex;
	int gvSource;
	int gridDataType;
	int showSHType;
	int iterationCount;
};

NAMESPACE_SHADERIO_END()
#endif
