#pragma once

//------------------------------------------------------------------------------------------
// Shared declarations and helpers of the Light Propagation Volumes renderer.
//
// All math follows Kaplanyan & Dachsbacher, "Cascaded Light Propagation Volumes for Real-Time
// Indirect Illumination" (I3D 2010). The SH convention is (see README.md for the derivation):
//
//   basis(w) = ( Y00, Y1-1, Y10, Y11 )        = ( 0.282095, 0.488603*w.y, 0.488603*w.z, 0.488603*w.x )
//   lobe(n)  = SH projection of max(0, n.w)   = ( 0.886227, 1.023327*n.y, 1.023327*n.z, 1.023327*n.x )
//
// lobe(n) is *both* the SH projection of a clamped cosine and the cosine convolved evaluation
// of an SH vector, because the projection coefficients A0 = pi, A1 = 2*pi/3 are the same for
// both operations. Using it twice would double the convolution, which is the classic LPV
// "my indirect light is 4x too dark / too bright" pitfall.
//------------------------------------------------------------------------------------------

#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/LPV/LPVShaderio.h"

#ifndef LPV_DEBUG
[[vk::push_constant]] ConstantBuffer<LPVPushConstant, ScalarDataLayout> pushConst;
#else
[[vk::push_constant]] ConstantBuffer<LPVDebugPushConstant, ScalarDataLayout> pushConst;
#endif

[[vk::binding(LPVStaticBindingPoints::ePixelData)]] RWStructuredBuffer<LPVPixelData> pixelData;
[[vk::binding(LPVStaticBindingPoints::eLPVVolumes)]] RWStructuredBuffer<float>      lpvVolumes[LPV_MAX_CASCADES];
[[vk::binding(LPVStaticBindingPoints::eLPVAccum)]] RWStructuredBuffer<float>        lpvAccum[LPV_MAX_CASCADES];
[[vk::binding(LPVStaticBindingPoints::eGvMerged)]] RWStructuredBuffer<float>        gvMerged[LPV_MAX_CASCADES];
[[vk::binding(LPVStaticBindingPoints::eGvCamera)]] RWStructuredBuffer<float>        gvCamera[LPV_MAX_CASCADES];
[[vk::binding(LPVStaticBindingPoints::eGvRsm)]] RWStructuredBuffer<float>           gvRsm[LPV_MAX_CASCADES];
[[vk::binding(LPVStaticBindingPoints::eFrameInfo)]] RWStructuredBuffer<LPVFrameInfo> frameInfo;
[[vk::binding(LPVStaticBindingPoints::eInstanceSize)]] RWStructuredBuffer<float> instanceSize;
[[vk::binding(LPVStaticBindingPoints::eDepthImage)]] RWTexture2D<float>     depthImage;
[[vk::binding(LPVStaticBindingPoints::eGridDebugData)]] RWStructuredBuffer<GridDebugInfo, ScalarDataLayout>     gridDebugData;

/* Paper 4.1: "if the size of an object (determined using its bounding box) is smaller than the cell
   size of a grid, we do not create VPLs from this object by not rendering it to the RSM. This is to
   prevent aliasing artifacts." The size is the largest extent of the instance's world space AABB,
   uploaded once per frame by the renderer. */
float LPV_ObjectSize(int instanceIndex)
{
    if (instanceIndex < 0 || instanceIndex >= frameInfo[0].info.w)
        return 1e30;  // unknown instance: never cull it
    return instanceSize[instanceIndex];
}

//------------------------------------------------------------------------------------------
// Spherical harmonics, 2 bands
//------------------------------------------------------------------------------------------
float4 LPV_SH_Basis(float3 w)
{
    return float4(0.2820947918, 0.4886025119 * w.y, 0.4886025119 * w.z, 0.4886025119 * w.x);
}

float4 LPV_SH_Lobe(float3 n)
{
    return float4(0.8862269255, 1.0233267079 * n.y, 1.0233267079 * n.z, 1.0233267079 * n.x);
}

// RGB SH vector of one cell (4 coefficients per channel).
struct LPVSH
{
    float4 r;
    float4 g;
    float4 b;
};

LPVSH LPV_MakeSH(float4 r, float4 g, float4 b)
{
    LPVSH sh;
    sh.r = r;
    sh.g = g;
    sh.b = b;
    return sh;
}

LPVSH LPV_ScaleSH(LPVSH sh, float scale)
{
    return LPV_MakeSH(sh.r * scale, sh.g * scale, sh.b * scale);
}

LPVSH LPV_LerpSH(LPVSH a, LPVSH b, float t)
{
    return LPV_MakeSH(lerp(a.r, b.r, t), lerp(a.g, b.g, t), lerp(a.b, b.b, t));
}

/* The stored coefficient vector `c` is indexed by the light's *travel* direction: a VPL of flux Phi
   with normal n stores Phi * lobe(n), which peaks along n, i.e. along the direction the light leaves
   the surface. Everything downstream must respect that single convention:

       propagation read    I(d)  = dot(c, basis(d))      d = travel direction through the face
       propagation write         c += ... * lobe(d)      the light continues travelling along d
       shading             E(n)  = dot(c, lobe(-n))      light arriving at a surface with normal n
                                                         travels along -w, hence the negative normal
       glossy probe        I(d)  = dot(c, basis(d))      intensity along a marched direction

   Using the cosine lobe where the plain basis belongs (or the other way round) convolves twice and
   rotates band-1 by 90 degrees; using the *positive* normal above inverts band-1, which suppresses
   light from surfaces that face the receiver and boosts light from surfaces that face away. */

// Intensity in direction `d` (W/sr), no cosine convolution.
float3 LPV_SH_EvalIntensity(LPVSH sh, float3 d)
{
    float4 basisValue = LPV_SH_Basis(d);
    return float3(dot(sh.r, basisValue), dot(sh.g, basisValue), dot(sh.b, basisValue));
}

// Irradiance received by a surface with normal n, without the (2 / cellSize)^2 term.
float3 LPV_SH_EvalIrradiance(LPVSH sh, float3 surfaceNormal)
{
    float4 lobe = LPV_SH_Lobe(-surfaceNormal);
    return float3(dot(sh.r, lobe), dot(sh.g, lobe), dot(sh.b, lobe));
}

//------------------------------------------------------------------------------------------
// Cascade / grid addressing
//------------------------------------------------------------------------------------------
int LPV_CascadeCount()
{
    return frameInfo[0].info.x;
}

int LPV_CascadeResolution(int cascade)
{
    return frameInfo[0].cascadeResolution[cascade];
}


uint LPV_VoxelCount(int cascade)
{
    uint n = uint(LPV_CascadeResolution(cascade));
    return n * n * n;
}

float LPV_CascadeReach(int cascade)
{
    return frameInfo[0].cascadeReach[cascade];
}

float3 LPV_CascadeOrigin(int cascade)
{
    return frameInfo[0].cascadeOriginSize[cascade].xyz;
}

float LPV_CascadeCellSize(int cascade)
{
    return frameInfo[0].cascadeOriginSize[cascade].w;
}

float3 LPV_GvOrigin(int cascade)
{
    return LPV_CascadeOrigin(cascade) - 0.5 * LPV_CascadeCellSize(cascade);
}

float3 LPV_CellCenter(int cascade, int3 cell)
{
    return LPV_CascadeOrigin(cascade) + (float3(cell.x, cell.y, cell.z) + 0.5) * LPV_CascadeCellSize(cascade);
}

uint LPV_VolumeFloatIndex(int volume, int cascade, int3 cell, int channel)
{
    int  n       = LPV_CascadeResolution(cascade);
    uint voxels  = LPV_VoxelCount(cascade);
    uint voxel   = uint((cell.z * n + cell.y) * n + cell.x);
    return (uint(volume) * voxels + voxel) * LPV_VOXEL_FLOATS + uint(channel) * LPV_SH_COEFF_COUNT;
}

uint LPV_AccumFloatIndex(int cascade, int3 cell, int channel)
{
    int  n      = LPV_CascadeResolution(cascade);
    uint voxel  = uint((cell.z * n + cell.y) * n + cell.x);
    return voxel * LPV_VOXEL_FLOATS + uint(channel) * LPV_SH_COEFF_COUNT;
}

int LPV_GvResolution(int cascade) { return LPV_CascadeResolution(cascade) + 1; }
uint LPV_GvVoxelCount(int cascade) { int n = LPV_GvResolution(cascade); return uint(n * n * n); }

uint LPV_GvFloatIndex(int cascade, int3 cell, int channel)
{
    int n = LPV_GvResolution(cascade);
    return uint((cell.z * n + cell.y) * n + cell.x) * LPV_GV_FLOATS + uint(channel);
}

float4 LPV_ReadFloat4(RWStructuredBuffer<float> buffer, uint base)
{
    return float4(buffer[base], buffer[base + 1], buffer[base + 2], buffer[base + 3]);
}

void LPV_WriteFloat4(RWStructuredBuffer<float> buffer, uint base, float4 value)
{
    buffer[base + 0] = value.x;
    buffer[base + 1] = value.y;
    buffer[base + 2] = value.z;
    buffer[base + 3] = value.w;
}

void LPV_AtomicAddFloat4(RWStructuredBuffer<float> buffer, uint base, float4 value)
{
    InterlockedAdd(buffer[base + 0], value.x);
    InterlockedAdd(buffer[base + 1], value.y);
    InterlockedAdd(buffer[base + 2], value.z);
    InterlockedAdd(buffer[base + 3], value.w);
}

float4 LPV_ReadVolumes(int volume, int cascade, uint base)
{
    if (cascade == 0) return LPV_ReadFloat4(lpvVolumes[0], base);
    if (cascade == 1) return LPV_ReadFloat4(lpvVolumes[1], base);
    if (cascade == 2) return LPV_ReadFloat4(lpvVolumes[2], base);
    return LPV_ReadFloat4(lpvVolumes[3], base);
}

void LPV_WriteVolumes(int volume, int cascade, uint base, float4 value)
{
    if (cascade == 0) { LPV_WriteFloat4(lpvVolumes[0], base, value); return; }
    if (cascade == 1) { LPV_WriteFloat4(lpvVolumes[1], base, value); return; }
    if (cascade == 2) { LPV_WriteFloat4(lpvVolumes[2], base, value); return; }
    LPV_WriteFloat4(lpvVolumes[3], base, value);
}

void LPV_AtomicAddVolumes(int volume, int cascade, uint base, float4 value)
{
    if (cascade == 0) { LPV_AtomicAddFloat4(lpvVolumes[0], base, value); return; }
    if (cascade == 1) { LPV_AtomicAddFloat4(lpvVolumes[1], base, value); return; }
    if (cascade == 2) { LPV_AtomicAddFloat4(lpvVolumes[2], base, value); return; }
    LPV_AtomicAddFloat4(lpvVolumes[3], base, value);
}

float4 LPV_ReadAccum(int cascade, uint base)
{
    if (cascade == 0) return LPV_ReadFloat4(lpvAccum[0], base);
    if (cascade == 1) return LPV_ReadFloat4(lpvAccum[1], base);
    if (cascade == 2) return LPV_ReadFloat4(lpvAccum[2], base);
    return LPV_ReadFloat4(lpvAccum[3], base);
}

void LPV_AddAccum(int cascade, uint base, float4 value)
{
    if (cascade == 0) { lpvAccum[0][base] += value.x; lpvAccum[0][base + 1] += value.y; lpvAccum[0][base + 2] += value.z; lpvAccum[0][base + 3] += value.w; return; }
    if (cascade == 1) { lpvAccum[1][base] += value.x; lpvAccum[1][base + 1] += value.y; lpvAccum[1][base + 2] += value.z; lpvAccum[1][base + 3] += value.w; return; }
    if (cascade == 2) { lpvAccum[2][base] += value.x; lpvAccum[2][base + 1] += value.y; lpvAccum[2][base + 2] += value.z; lpvAccum[2][base + 3] += value.w; return; }
    lpvAccum[3][base] += value.x; lpvAccum[3][base + 1] += value.y; lpvAccum[3][base + 2] += value.z; lpvAccum[3][base + 3] += value.w;
}

float4 LPV_ReadGv(int gvSource, int cascade, uint base)
{
    if (gvSource == LPVGeometrySource::eLPVGeometryFromCamera)
    {
        if (cascade == 0) return LPV_ReadFloat4(gvCamera[0], base);
        if (cascade == 1) return LPV_ReadFloat4(gvCamera[1], base);
        if (cascade == 2) return LPV_ReadFloat4(gvCamera[2], base);
        return LPV_ReadFloat4(gvCamera[3], base);
    }
    if (gvSource == LPVGeometrySource::eLPVGeometryFromRsm)
    {
        if (cascade == 0) return LPV_ReadFloat4(gvRsm[0], base);
        if (cascade == 1) return LPV_ReadFloat4(gvRsm[1], base);
        if (cascade == 2) return LPV_ReadFloat4(gvRsm[2], base);
        return LPV_ReadFloat4(gvRsm[3], base);
    }
    if (cascade == 0) return LPV_ReadFloat4(gvMerged[0], base);
    if (cascade == 1) return LPV_ReadFloat4(gvMerged[1], base);
    if (cascade == 2) return LPV_ReadFloat4(gvMerged[2], base);
    return LPV_ReadFloat4(gvMerged[3], base);
}

void LPV_AtomicAddGvBase(int gvSource, int cascade, uint base, float4 value)
{
    if (gvSource == LPVGeometrySource::eLPVGeometryFromCamera)
    {
        if (cascade == 0) { LPV_AtomicAddFloat4(gvCamera[0], base, value); return; }
        if (cascade == 1) { LPV_AtomicAddFloat4(gvCamera[1], base, value); return; }
        if (cascade == 2) { LPV_AtomicAddFloat4(gvCamera[2], base, value); return; }
        LPV_AtomicAddFloat4(gvCamera[3], base, value);
        return;
    }
    if (cascade == 0) { LPV_AtomicAddFloat4(gvRsm[0], base, value); return; }
    if (cascade == 1) { LPV_AtomicAddFloat4(gvRsm[1], base, value); return; }
    if (cascade == 2) { LPV_AtomicAddFloat4(gvRsm[2], base, value); return; }
    LPV_AtomicAddFloat4(gvRsm[3], base, value);
}

void LPV_StoreGvMerged(int cascade, uint base, float4 value)
{
    if (cascade == 0) { LPV_WriteFloat4(gvMerged[0], base, value); return; }
    if (cascade == 1) { LPV_WriteFloat4(gvMerged[1], base, value); return; }
    if (cascade == 2) { LPV_WriteFloat4(gvMerged[2], base, value); return; }
    LPV_WriteFloat4(gvMerged[3], base, value);
}

bool LPV_CellInside(int3 cell, int n)
{
    return all(cell >= int3(0, 0, 0)) && all(cell < int3(n, n, n));
}

//------------------------------------------------------------------------------------------
// Ping pong volume access
//------------------------------------------------------------------------------------------
float4 LPV_LoadVolumeSH(int volume, int cascade, int3 cell, int channel)
{
    return LPV_ReadVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, channel));
}

void LPV_StoreVolumeSH(int volume, int cascade, int3 cell, LPVSH sh)
{
    LPV_WriteVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 0), sh.r);
    LPV_WriteVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 1), sh.g);
    LPV_WriteVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 2), sh.b);
}

void LPV_AtomicAddVolumeSH(int volume, int cascade, int3 cell, LPVSH sh)
{
    LPV_AtomicAddVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 0), sh.r);
    LPV_AtomicAddVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 1), sh.g);
    LPV_AtomicAddVolumes(volume, cascade, LPV_VolumeFloatIndex(volume, cascade, cell, 2), sh.b);
}

void LPV_AtomicSplatVolumeSH(int volume, int cascade, float3 position, LPVSH sh)
{
    float  s  = LPV_CascadeCellSize(cascade);
    int    n  = LPV_CascadeResolution(cascade);
    float3 g  = (position - LPV_CascadeOrigin(cascade)) / s - 0.5;
    int3   i0 = int3(floor(g));
    float3 t  = g - float3(i0);

    for (int dz = 0; dz < 2; ++dz)
    {
        float wz = dz == 1 ? t.z : 1.0 - t.z;
        for (int dy = 0; dy < 2; ++dy)
        {
            float wy = dy == 1 ? t.y : 1.0 - t.y;
            for (int dx = 0; dx < 2; ++dx)
            {
                float wx = dx == 1 ? t.x : 1.0 - t.x;
                float w  = wx * wy * wz;
                if (w <= 0.0)
                    continue;

                int3 cell = i0 + int3(dx, dy, dz);
                if (!LPV_CellInside(cell, n))
                    continue;

                LPV_AtomicAddVolumeSH(volume, cascade, cell, LPV_MakeSH(sh.r * w, sh.g * w, sh.b * w));
            }
        }
    }
}

void LPV_AddAccumSH(int cascade, int3 cell, LPVSH sh)
{
    LPV_AddAccum(cascade, LPV_AccumFloatIndex(cascade, cell, 0), sh.r);
    LPV_AddAccum(cascade, LPV_AccumFloatIndex(cascade, cell, 1), sh.g);
    LPV_AddAccum(cascade, LPV_AccumFloatIndex(cascade, cell, 2), sh.b);
}

float4 LPV_LoadAccumSH(int cascade, int3 cell, int channel)
{
    return LPV_ReadAccum(cascade, LPV_AccumFloatIndex(cascade, cell, channel));
}

// Trilinear lookup of one channel of the accumulated volume. Cell centers live at
// origin + (i + 0.5) * cellSize, therefore the texel coordinate is (p - origin) / cellSize - 0.5.
float4 LPV_SampleAccumChannel(int cascade, float3 position, int channel)
{
    int    n = LPV_CascadeResolution(cascade);
    float  s = LPV_CascadeCellSize(cascade);
    float3 o = LPV_CascadeOrigin(cascade);

    float3 c = (position - o) / s - 0.5;
    c = clamp(c, float3(0.0, 0.0, 0.0), float3(n - 1, n - 1, n - 1));

    int3   i0 = int3(floor(c));
    int3   i1 = min(i0 + int3(1, 1, 1), int3(n - 1, n - 1, n - 1));
    float3 t  = c - float3(i0.x, i0.y, i0.z);

    float4 v000 = LPV_LoadAccumSH(cascade, int3(i0.x, i0.y, i0.z), channel);
    float4 v100 = LPV_LoadAccumSH(cascade, int3(i1.x, i0.y, i0.z), channel);
    float4 v010 = LPV_LoadAccumSH(cascade, int3(i0.x, i1.y, i0.z), channel);
    float4 v110 = LPV_LoadAccumSH(cascade, int3(i1.x, i1.y, i0.z), channel);
    float4 v001 = LPV_LoadAccumSH(cascade, int3(i0.x, i0.y, i1.z), channel);
    float4 v101 = LPV_LoadAccumSH(cascade, int3(i1.x, i0.y, i1.z), channel);
    float4 v011 = LPV_LoadAccumSH(cascade, int3(i0.x, i1.y, i1.z), channel);
    float4 v111 = LPV_LoadAccumSH(cascade, int3(i1.x, i1.y, i1.z), channel);

    float4 x00 = lerp(v000, v100, t.x);
    float4 x10 = lerp(v010, v110, t.x);
    float4 x01 = lerp(v001, v101, t.x);
    float4 x11 = lerp(v011, v111, t.x);
    return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z);
}

LPVSH LPV_SampleAccum(int cascade, float3 position)
{
    return LPV_MakeSH(LPV_SampleAccumChannel(cascade, position, 0),
                      LPV_SampleAccumChannel(cascade, position, 1),
                      LPV_SampleAccumChannel(cascade, position, 2));
}

// Nearest lookup, used for the directional derivative of the dampening term (paper section 3.4).
LPVSH LPV_LoadAccumNearest(int cascade, float3 position)
{
    int    n = LPV_CascadeResolution(cascade);
    float  s = LPV_CascadeCellSize(cascade);
    float3 o = LPV_CascadeOrigin(cascade);
    int3   c = int3(clamp(floor((position - o) / s), float3(0.0, 0.0, 0.0), float3(n - 1, n - 1, n - 1)));
    return LPV_MakeSH(LPV_LoadAccumSH(cascade, c, 0), LPV_LoadAccumSH(cascade, c, 1), LPV_LoadAccumSH(cascade, c, 2));
}

//------------------------------------------------------------------------------------------
// Geometry volume (blocking potential). Its cell centers sit on the *corners* of the LPV cells
// (paper section 3.2), i.e. at origin + i * cellSize, which is why the sampling below uses
// (p - origin) / cellSize without the -0.5 of the intensity volume.
//------------------------------------------------------------------------------------------
float4 LPV_LoadGv(int gvSource, int cascade, int3 cell)
{
    return LPV_ReadGv(gvSource, cascade, LPV_GvFloatIndex(cascade, cell, 0));
}

void LPV_AtomicAddGv(int gvSource, int cascade, int3 cell, float4 value)
{
    LPV_AtomicAddGvBase(gvSource, cascade, LPV_GvFloatIndex(cascade, cell, 0), value);
}

float4 LPV_LoadGvMerged(int cascade, int3 cell)
{
    return LPV_ReadGv(LPVGeometrySource::eLPVGeometryMerged, cascade, LPV_GvFloatIndex(cascade, cell, 0));
}

// Load from any of the geometry volumes: 0 = camera, 1 = RSM, 2 = merged (see LPVGeometrySource).
float4 LPV_LoadGvAny(int gvSource, int cascade, int3 cell)
{
    if (gvSource == LPVGeometrySource::eLPVGeometryFromCamera || gvSource == LPVGeometrySource::eLPVGeometryFromRsm)
        return LPV_LoadGv(gvSource, cascade, cell);
    return LPV_LoadGvMerged(cascade, cell);
}

// Trilinear lookup of one of the geometry volumes at a world position.
float4 LPV_SampleGv(int gvSource, int cascade, float3 position)
{
    int    n = LPV_GvResolution(cascade);
    float  s = LPV_CascadeCellSize(cascade);
    float3 o = LPV_GvOrigin(cascade);

    float3 c = clamp((position - o) / s - 0.5, float3(0.0, 0.0, 0.0), float3(n - 1, n - 1, n - 1));
    int3   i0 = int3(floor(c));
    int3   i1 = min(i0 + int3(1, 1, 1), int3(n - 1, n - 1, n - 1));
    float3 t  = c - float3(i0.x, i0.y, i0.z);

    float4 gv000 = LPV_LoadGvAny(gvSource, cascade, int3(i0.x, i0.y, i0.z));
    float4 gv100 = LPV_LoadGvAny(gvSource, cascade, int3(i1.x, i0.y, i0.z));
    float4 gv010 = LPV_LoadGvAny(gvSource, cascade, int3(i0.x, i1.y, i0.z));
    float4 gv110 = LPV_LoadGvAny(gvSource, cascade, int3(i1.x, i1.y, i0.z));
    float4 gv001 = LPV_LoadGvAny(gvSource, cascade, int3(i0.x, i0.y, i1.z));
    float4 gv101 = LPV_LoadGvAny(gvSource, cascade, int3(i1.x, i0.y, i1.z));
    float4 gv011 = LPV_LoadGvAny(gvSource, cascade, int3(i0.x, i1.y, i1.z));
    float4 gv111 = LPV_LoadGvAny(gvSource, cascade, int3(i1.x, i1.y, i1.z));

    float4 x00 = lerp(gv000, gv100, t.x);
    float4 x10 = lerp(gv010, gv110, t.x);
    float4 x01 = lerp(gv001, gv101, t.x);
    float4 x11 = lerp(gv011, gv111, t.x);
    return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z);
}

// Trilinear lookup of the merged geometry volume at a world position.
float4 LPV_SampleGvMerged(int cascade, float3 position)
{
    return LPV_SampleGv(LPVGeometrySource::eLPVGeometryMerged, cascade, position);
}

/* Inject one surfel into the geometry volume of one cascade.

   The paper (3.2) stores the blocking potential of a surfel of *true* area A_s with normal n in a cell
   of grid size s:

       B(w) = (A_s / s^2) * max(0, n.w)

   i.e. the probability that a ray leaving the cell along w hits the surfel: A_s * max(0, n.w) is the
   surfel area projected onto the plane perpendicular to w, and s^2 is the cross section of the cell
   along w (exact for the six face directions, up to sqrt(3) overestimated towards the body diagonal -
   the paper's own approximation).

   Two different cosines are involved and must not be mixed up:
     * the *observation* cosine (light or view direction) has already been divided out by the caller,
       which turns the projected footprint of a texel / pixel into the true area A_s passed here;
     * the *query* cosine n.w is applied at evaluation time, once per propagation direction, by the
       dot product of the stored coefficients with LPV_SH_Basis(w) (which reconstructs 0.25 + 0.5*n.w).

   So only the peak amplitude A_s / s^2 is stored as the SH coefficients of a clamped cosine around
   n; the probability clamp belongs to the evaluation (B(w) is saturated where it is queried), the
   paper's storage step itself has no clamp. The geometry volume is the LPV grid shifted by half a
   cell (LPV_GvOrigin), so its cell centres sit on the corners of the LPV cells (paper 3.2). The
   surfel itself is not moved along its normal. */
void LPV_InjectBlocking(int gvSource, int cascade, float3 position, float3 normal, float surfelArea)
{
    float s = LPV_CascadeCellSize(cascade);
    int   resolution = LPV_GvResolution(cascade);
    int3  cell = int3(floor((position - LPV_GvOrigin(cascade)) / s));
    if (!LPV_CellInside(cell, resolution)) return;

    float scale = surfelArea / (s * s);
    if (scale <= 0.0) return;
    LPV_AtomicAddGv(gvSource, cascade, cell, scale * LPV_SH_Lobe(normal));
}

//------------------------------------------------------------------------------------------
// Ray tracing helpers (ray queries, like the other compute based renderers of this project)
//------------------------------------------------------------------------------------------
struct LPVHit
{
    bool   hit;
    bool   frontFacing;
    bool   hasTexCoords;
    float3 position;
    float3 normal;
    float2 texCoords;
    int    materialIndex;
    int    instanceIndex;
    float  distance;
};

LPVHit LPV_TraceRay(RayDesc ray)
{
    LPVHit result;
    result.hit           = false;
    result.frontFacing   = true;
    result.hasTexCoords  = false;
    result.position      = float3(0.0, 0.0, 0.0);
    result.normal        = float3(0.0, 1.0, 0.0);
    result.texCoords     = float2(0.0, 0.0);
    result.materialIndex = 0;
    result.instanceIndex = -1;
    result.distance      = ray.TMax;

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return result;

    float2 bary2 = q.CommittedTriangleBarycentrics();
    float3 bary  = float3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);

    float4x3 worldToObject = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
    int      instanceIndex = q.CommittedInstanceIndex();
    uint     primitiveIndex = q.CommittedPrimitiveIndex();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance  instance  = sceneInfo.instances[instanceIndex];
    Mesh      mesh      = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    float3 localPos  = triPos[0] * bary.x + triPos[1] * bary.y + triPos[2] * bary.z;
    result.position  = float3(mul(float4(localPos, 1.0), objectToWorld));

    float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    result.normal    = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    float cosineON   = dot(result.normal, -ray.Direction);
    result.frontFacing = cosineON > 0.0;
    if (cosineON <= 0.0)
        result.normal = -result.normal;

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0)
    {
        int3 indices     = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, int(primitiveIndex));
        result.texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, bary);
        result.texCoords.y = 1.0 - result.texCoords.y;
        result.hasTexCoords = true;
    }

    result.materialIndex = int(instance.materialIndex);
    result.instanceIndex = instanceIndex;
    result.hit           = true;
    result.distance      = q.CommittedRayT();
    return result;
}

RayDesc LPV_SpawnRay(float3 origin, float3 direction)
{
    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = normalize(direction);
    ray.TMin      = 0.001;
    ray.TMax      = INFINITE;
    return ray;
}

// Diffuse albedo of a hit, with the material texture applied.
float3 LPV_AlbedoOfHit(LPVHit hit)
{
    SceneInfo    sceneInfo = pushConst.sceneInfoAddress[0];
    BSDFMaterial material  = sceneInfo.materials[hit.materialIndex];
    if (hit.hasTexCoords)
        getBSDFTextureData(material, hit.texCoords);
    return material.albedo;
}

// Camera ray for a pixel (identical to the other renderers of this project).
RayDesc LPV_CameraRay(uint2 pixel, float2 jitter)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float2    launchID  = float2(pixel) + jitter;
    float2    launchSize = float2(pushConst.screenSize);
    float2    clipCoords = launchID / launchSize * 2.0 - 1.0;

    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    RayDesc ray;
    ray.Origin    = sceneInfo.cameraPosition;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin      = 0.001;
    ray.TMax      = INFINITE;
    return ray;
}

float LPV_PixelSolidAngle(uint2 pixel)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float     tanY = 1.0 / max(abs(sceneInfo.projMatrix[1][1]), 1e-6);
    float     tanX = 1.0 / max(abs(sceneInfo.projMatrix[0][0]), 1e-6);

    const float2 screenSize = max(float2(pushConst.screenSize), float2(1.0, 1.0));
    // glm's orthoRH_ZO has P[3][3] = 1, perspectiveRH_ZO has 0; orthographic rays are parallel.
    const bool orthographic = sceneInfo.projMatrix[3][3] > 0.5;

    float2 ndc  = (float2(pixel) + 0.5) / screenSize * 2.0 - 1.0;
    float  tan2 = (ndc.x * tanX) * (ndc.x * tanX) + (ndc.y * tanY) * (ndc.y * tanY);
    float  cosAxis = orthographic ? 1.0 : rsqrt(1.0 + tan2);

    return (2.0 * tanX / screenSize.x) * (2.0 * tanY / screenSize.y) * cosAxis * cosAxis * cosAxis;
}

// Direction towards a punctual light; returns false when the light cannot reach the surface.
bool LPV_LightDirection(const Light light, float3 position, out float3 direction, out float3 radiance, out float distanceToLight)
{
    direction       = float3(0.0, 0.0, 0.0);
    radiance        = float3(0.0, 0.0, 0.0);
    distanceToLight = 0.0;

    if (light.type == uint(LightType::Direction))
    {
        direction       = normalize(-light.direction);
        radiance        = light.color * light.intensity;
        distanceToLight = 1e30;
        return true;
    }

    if (light.type == uint(LightType::Point) || light.type == uint(LightType::Spot) || light.type == uint(LightType::Area))
    {
        float3 toLight  = light.pos - position;
        float  distance = length(toLight);
        if (distance <= 1e-5)
            return false;
        direction       = toLight / distance;
        distanceToLight = distance;
        radiance        = light.color * (light.intensity / (distance * distance));

        if (light.type == uint(LightType::Spot))
        {
            float cosOuter = cos(clamp(light.coneAngle, 1e-3, 3.14159265));
            float cosInner = cos(clamp(light.coneAngle, 1e-3, 3.14159265) * 0.85);
            float cosTheta = dot(-direction, normalize(light.direction));
            radiance *= smoothstep(cosOuter, max(cosInner, cosOuter + 1e-4), cosTheta);
        }
        return true;
    }

    return false;
}

// Occlusion test towards a punctual light.
bool LPV_Occluded(float3 position, float3 direction, float distanceToLight)
{
    RayDesc ray;
    ray.Origin    = position + direction * 1e-4;
    ray.Direction = direction;
    ray.TMin      = max(0.001, 1e-4);
    ray.TMax      = distanceToLight > 1e29 ? INFINITE : max(distanceToLight - 1e-4, 1e-3);

    HitTestPayload payload;
    payload.distance = MISS_DISTANCE;
    hitTestTraceRay(ray, payload);
    return payload.distance != MISS_DISTANCE;
}
