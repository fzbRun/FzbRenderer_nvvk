#pragma once

#include <common/Shader/shaderStructType.h>

#ifndef FZBRENDERER_LPV_SHADER_IO_H
#define FZBRENDERER_LPV_SHADER_IO_H
NAMESPACE_SHADERIO_BEGIN()

//------------------------------------------------------------------------------------------
// Cascaded Light Propagation Volumes, after
//   A. Kaplanyan, C. Dachsbacher, "Cascaded Light Propagation Volumes for Real-Time
//   Indirect Illumination", I3D 2010.
//
// Data flow (everything lives in src/renderer/LPV, see README.md):
//   camera G-buffer (ray query)   -> LPVPixelData + blocking potential of the camera surfels
//   virtual RSM (ray traced)      -> reflective-shadow-map VPLs, injected as 2 band SH
//   geometry volume merge         -> max(camera GV, RSM GV) = fuzzy occlusion
//   N propagation iterations      -> 6 face gather over the cascades, accumulated
//   final gather                  -> trilinear SH lookup, derivative dampening, cascade blend
//
// Stored quantity: SH coefficients of the *radiant intensity* I(w) of a cell, per RGB
// channel, following the paper's normalisation (section 3.1): a VPL of reflected flux Phi
// with normal n contributes  c += Phi * lobe(n)  with lobe() being the SH projection of a
// clamped cosine. The gather converts intensity to irradiance with the paper's half cell
// distance assumption (section 3.4): E = dot(c, lobe(n)) * (2 / cellSize)^2.
//------------------------------------------------------------------------------------------

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
/* Solid angle of one cube face seen from the neighbouring cell centre, 4*pi/6. This is also the
   silhouette of the destination cell seen from the source cell's centre, i.e. the total solid angle
   that one (source -> destination) transfer has to distribute over the destination's exit faces. */
#define LPV_FACE_SOLID_ANGLE 2.0943951023931953f

/* Paper section 3.3 / Fig. 2 (center): one (source -> destination) transfer is split into the 5
   faces of the destination cell *other* than the one the light enters through (the shared face is
   the entry side and gets no VPL). Seen from the source cell's centre those 5 faces subtend

       far face  : 4*atan(1/(3*sqrt(11)))     = 0.4006696846
       side faces: (4*pi/6 - far) / 4        = 0.4234313545   (four of them)

   and they sum to exactly 4*pi/6, the shared face's solid angle. That identity is what keeps the
   total flux of a transfer - and with it the calibrated brightness - independent of how the flux is
   spread over the 5 lobes, so the side angle is derived from it rather than typed in: earlier
   revisions used 0.4234413544 (a typo inherited from the sample code), which made the 5 bins sum to
   4*pi/6 + 4e-5 and slowly brightened the volume (+1.9e-5 per iteration). */
#define LPV_DIRECT_FACE_SOLID_ANGLE 0.4006696846f
#define LPV_SIDE_FACE_SOLID_ANGLE ((LPV_FACE_SOLID_ANGLE - LPV_DIRECT_FACE_SOLID_ANGLE) * 0.25f)

// One record per pixel, written by the G-buffer pass and consumed by the gather pass. The direct
// lighting is evaluated in the G-buffer pass, because that pass already owns the material, the
// texture coordinates and the shading frame of the primary hit.
struct LPVPixelData
{
	float4 positionValid;  // xyz: world space primary hit position, w: 1 when the pixel hit geometry
	float4 normal;         // xyz: shading normal (flipped towards the camera), w: unused
	float4 albedo;         // xyz: diffuse albedo used by the indirect term (0 for purely specular BSDFs)
	float4 directColor;    // xyz: emissive + direct lighting, w: unused
	float4 specular;       // xyz: specular reflectance used by the glossy LPV lookup, w: unused
};

// Per frame constants. Uploaded into one storage buffer (see LPV::updateFrameInfo) so that the
// push constant stays small; the LPV volumes themselves are addressed through it as well.
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
};

// Push constants. Kept 16 byte friendly: the SceneInfo pointer is 8 byte aligned and the struct
// ends on a 16 byte boundary so that the C++ and the Slang (ScalarDataLayout) layout agree.
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
};

// Runtime toggles of the indirect lighting, stored in LPVPushConstant::flags.
// Bits 3..5 hold the debug view (LPVDebugView, 3 bits wide), so the debug-only "no shadow ray"
// switch has to live above them: it used to be bit 5, which collided with debug view 4 and silently
// turned the shadow ray off in the "Reference GI" view.
enum LPVFlags
{
	eLPVFlagOcclusion = 1,   // fuzzy occlusion from the geometry volume
	eLPVFlagDampening = 2,   // derivative based dampening (paper section 3.4)
	eLPVFlagNoDirect  = 4,   // comparison: drop the direct light, keep the LPV (and glossy) contribution
	eLPVFlagNoShadow  = 64,  // debug: bit 6, drops the direct shadow ray (LPV_NO_SHADOW=1)
};

// Where a blocking surfel is injected. The paper accumulates the camera view and every RSM into
// separate geometry volumes and merges them with a maximum, so that a surfel seen by several
// buffers is not counted several times.
enum LPVGeometrySource
{
	eLPVGeometryFromCamera = 0,
	eLPVGeometryFromRsm    = 1,
};

// Debug view of the final gather, stored in bits 3..5 of LPVPushConstant::flags (3 bits wide).
enum LPVDebugView
{
	eLPVDebugOff       = 0,
	eLPVDebugCascade   = 1,  // colour by cascade index
	eLPVDebugDirect    = 2,  // direct lighting only
	eLPVDebugIndirect  = 3,  // indirect (LPV) lighting only
	eLPVDebugReference = 4,  // unbiased single bounce Monte Carlo reference
	eLPVDebugRsm       = 5,  // virtual RSM of the first light, reflected flux per texel
};

// Static descriptor set bindings. 0 and 1 are reserved by pathTracingCommon.slang
// (textures[] and outImage), exactly like the other renderers of this project.
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
};

struct CascadeDebugInfo {
	float3 startPos;
	float voxelSize;
	int resolution;
};
struct LPVDebugPushConstant {
	int frameIndex;
	int padding;
	uint2 screenSize;
	SceneInfo* sceneInfoAddress;

	int GBufferIndex;

	int cascadeCount;
	int gridIndex;
	CascadeDebugInfo gridInfo[LPV_MAX_CASCADES];
	int debugVoxelCount;

	int lightIndex;

	int showGvSH;
	int cascadeIndex_showSH;
	int3 voxelIndex_showSH;
};

NAMESPACE_SHADERIO_END()
#endif
