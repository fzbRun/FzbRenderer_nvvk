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
[[vk::binding(LPVStaticBindingPoints::eLPVVolumes)]] RWStructuredBuffer<float>      lpvVolumes;
[[vk::binding(LPVStaticBindingPoints::eLPVAccum)]] RWStructuredBuffer<float>        lpvAccum;
[[vk::binding(LPVStaticBindingPoints::eGvMerged)]] RWStructuredBuffer<float>        gvMerged;
[[vk::binding(LPVStaticBindingPoints::eGvCamera)]] RWStructuredBuffer<float>        gvCamera;
[[vk::binding(LPVStaticBindingPoints::eGvRsm)]] RWStructuredBuffer<float>           gvRsm;
[[vk::binding(LPVStaticBindingPoints::eFrameInfo)]] RWStructuredBuffer<LPVFrameInfo> frameInfo;
[[vk::binding(LPVStaticBindingPoints::eInstanceSize)]] RWStructuredBuffer<float>     instanceSize;

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

int LPV_GridResolution()
{
    return frameInfo[0].info.y;
}

uint LPV_VoxelCount()
{
    uint n = uint(LPV_GridResolution());
    return n * n * n;
}

float3 LPV_CascadeOrigin(int cascade)
{
    return frameInfo[0].cascadeOriginSize[cascade].xyz;
}

float LPV_CascadeCellSize(int cascade)
{
    return frameInfo[0].cascadeOriginSize[cascade].w;
}

float3 LPV_CellCenter(int cascade, int3 cell)
{
    return LPV_CascadeOrigin(cascade) + (float3(cell.x, cell.y, cell.z) + 0.5) * LPV_CascadeCellSize(cascade);
}

// Flat float index of coefficient `channel` of a voxel inside one of the ping pong volumes.
uint LPV_VolumeFloatIndex(int volume, int cascade, int3 cell, int channel)
{
    int  n       = LPV_GridResolution();
    uint voxels  = LPV_VoxelCount();
    uint voxel   = uint(cascade) * voxels + uint((cell.z * n + cell.y) * n + cell.x);
    return (uint(volume) * uint(LPV_CascadeCount()) * voxels + voxel) * LPV_VOXEL_FLOATS + uint(channel) * LPV_SH_COEFF_COUNT;
}

uint LPV_AccumFloatIndex(int cascade, int3 cell, int channel)
{
    int  n      = LPV_GridResolution();
    uint voxels = LPV_VoxelCount();
    return (uint(cascade) * voxels + uint((cell.z * n + cell.y) * n + cell.x)) * LPV_VOXEL_FLOATS + uint(channel) * LPV_SH_COEFF_COUNT;
}

uint LPV_GvFloatIndex(int cascade, int3 cell, int channel)
{
    int  n      = LPV_GridResolution();
    uint voxels = LPV_VoxelCount();
    return (uint(cascade) * voxels + uint((cell.z * n + cell.y) * n + cell.x)) * LPV_GV_FLOATS + uint(channel);
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
    uint base = LPV_VolumeFloatIndex(volume, cascade, cell, channel);
    return float4(lpvVolumes[base], lpvVolumes[base + 1], lpvVolumes[base + 2], lpvVolumes[base + 3]);
}

void LPV_StoreVolumeSH(int volume, int cascade, int3 cell, LPVSH sh)
{
    uint base = LPV_VolumeFloatIndex(volume, cascade, cell, 0);
    lpvVolumes[base + 0] = sh.r.x;
    lpvVolumes[base + 1] = sh.r.y;
    lpvVolumes[base + 2] = sh.r.z;
    lpvVolumes[base + 3] = sh.r.w;
    base = LPV_VolumeFloatIndex(volume, cascade, cell, 1);
    lpvVolumes[base + 0] = sh.g.x;
    lpvVolumes[base + 1] = sh.g.y;
    lpvVolumes[base + 2] = sh.g.z;
    lpvVolumes[base + 3] = sh.g.w;
    base = LPV_VolumeFloatIndex(volume, cascade, cell, 2);
    lpvVolumes[base + 0] = sh.b.x;
    lpvVolumes[base + 1] = sh.b.y;
    lpvVolumes[base + 2] = sh.b.z;
    lpvVolumes[base + 3] = sh.b.w;
}

// VPL injection is a scatter over many RSM texels, so the SH coefficients are accumulated with
// float atomics (VK_EXT_shader_atomic_float, enabled by the renderer).
void LPV_AtomicAddVolumeSH(int volume, int cascade, int3 cell, LPVSH sh)
{
    uint base = LPV_VolumeFloatIndex(volume, cascade, cell, 0);
    InterlockedAdd(lpvVolumes[base + 0], sh.r.x);
    InterlockedAdd(lpvVolumes[base + 1], sh.r.y);
    InterlockedAdd(lpvVolumes[base + 2], sh.r.z);
    InterlockedAdd(lpvVolumes[base + 3], sh.r.w);
    base = LPV_VolumeFloatIndex(volume, cascade, cell, 1);
    InterlockedAdd(lpvVolumes[base + 0], sh.g.x);
    InterlockedAdd(lpvVolumes[base + 1], sh.g.y);
    InterlockedAdd(lpvVolumes[base + 2], sh.g.z);
    InterlockedAdd(lpvVolumes[base + 3], sh.g.w);
    base = LPV_VolumeFloatIndex(volume, cascade, cell, 2);
    InterlockedAdd(lpvVolumes[base + 0], sh.b.x);
    InterlockedAdd(lpvVolumes[base + 1], sh.b.y);
    InterlockedAdd(lpvVolumes[base + 2], sh.b.z);
    InterlockedAdd(lpvVolumes[base + 3], sh.b.w);
}

void LPV_AddAccumSH(int cascade, int3 cell, LPVSH sh)
{
    uint base = LPV_AccumFloatIndex(cascade, cell, 0);
    lpvAccum[base + 0] += sh.r.x;
    lpvAccum[base + 1] += sh.r.y;
    lpvAccum[base + 2] += sh.r.z;
    lpvAccum[base + 3] += sh.r.w;
    base = LPV_AccumFloatIndex(cascade, cell, 1);
    lpvAccum[base + 0] += sh.g.x;
    lpvAccum[base + 1] += sh.g.y;
    lpvAccum[base + 2] += sh.g.z;
    lpvAccum[base + 3] += sh.g.w;
    base = LPV_AccumFloatIndex(cascade, cell, 2);
    lpvAccum[base + 0] += sh.b.x;
    lpvAccum[base + 1] += sh.b.y;
    lpvAccum[base + 2] += sh.b.z;
    lpvAccum[base + 3] += sh.b.w;
}

float4 LPV_LoadAccumSH(int cascade, int3 cell, int channel)
{
    uint base = LPV_AccumFloatIndex(cascade, cell, channel);
    return float4(lpvAccum[base], lpvAccum[base + 1], lpvAccum[base + 2], lpvAccum[base + 3]);
}

// Trilinear lookup of one channel of the accumulated volume. Cell centers live at
// origin + (i + 0.5) * cellSize, therefore the texel coordinate is (p - origin) / cellSize - 0.5.
float4 LPV_SampleAccumChannel(int cascade, float3 position, int channel)
{
    int    n = LPV_GridResolution();
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
    int    n = LPV_GridResolution();
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
    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    if (gvSource == LPVGeometrySource::eLPVGeometryFromCamera)
        return float4(gvCamera[base], gvCamera[base + 1], gvCamera[base + 2], gvCamera[base + 3]);
    return float4(gvRsm[base], gvRsm[base + 1], gvRsm[base + 2], gvRsm[base + 3]);
}

void LPV_AtomicAddGv(int gvSource, int cascade, int3 cell, float4 value)
{
    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    if (gvSource == LPVGeometrySource::eLPVGeometryFromCamera)
    {
        InterlockedAdd(gvCamera[base + 0], value.x);
        InterlockedAdd(gvCamera[base + 1], value.y);
        InterlockedAdd(gvCamera[base + 2], value.z);
        InterlockedAdd(gvCamera[base + 3], value.w);
    }
    else
    {
        InterlockedAdd(gvRsm[base + 0], value.x);
        InterlockedAdd(gvRsm[base + 1], value.y);
        InterlockedAdd(gvRsm[base + 2], value.z);
        InterlockedAdd(gvRsm[base + 3], value.w);
    }
}

float4 LPV_LoadGvMerged(int cascade, int3 cell)
{
    uint base = LPV_GvFloatIndex(cascade, cell, 0);
    return float4(gvMerged[base], gvMerged[base + 1], gvMerged[base + 2], gvMerged[base + 3]);
}

// Trilinear lookup of the merged geometry volume at a world position.
float4 LPV_SampleGvMerged(int cascade, float3 position)
{
    int    n = LPV_GridResolution();
    float  s = LPV_CascadeCellSize(cascade);
    float3 o = LPV_CascadeOrigin(cascade);

    float3 c = clamp((position - o) / s, float3(0.0, 0.0, 0.0), float3(n - 1, n - 1, n - 1));
    int3   i0 = int3(floor(c));
    int3   i1 = min(i0 + int3(1, 1, 1), int3(n - 1, n - 1, n - 1));
    float3 t  = c - float3(i0.x, i0.y, i0.z);

    float4 gv000 = LPV_LoadGvMerged(cascade, int3(i0.x, i0.y, i0.z));
    float4 gv100 = LPV_LoadGvMerged(cascade, int3(i1.x, i0.y, i0.z));
    float4 gv010 = LPV_LoadGvMerged(cascade, int3(i0.x, i1.y, i0.z));
    float4 gv110 = LPV_LoadGvMerged(cascade, int3(i1.x, i1.y, i0.z));
    float4 gv001 = LPV_LoadGvMerged(cascade, int3(i0.x, i0.y, i1.z));
    float4 gv101 = LPV_LoadGvMerged(cascade, int3(i1.x, i0.y, i1.z));
    float4 gv011 = LPV_LoadGvMerged(cascade, int3(i0.x, i1.y, i1.z));
    float4 gv111 = LPV_LoadGvMerged(cascade, int3(i1.x, i1.y, i1.z));

    float4 x00 = lerp(gv000, gv100, t.x);
    float4 x10 = lerp(gv010, gv110, t.x);
    float4 x01 = lerp(gv001, gv101, t.x);
    float4 x11 = lerp(gv011, gv111, t.x);
    return lerp(lerp(x00, x10, t.y), lerp(x01, x11, t.y), t.z);
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
   paper's storage step itself has no clamp. Cell centres of the geometry volume sit on the *corners*
   of the LPV cells (paper 3.2, the GV is "shifted by half a cell" with respect to the LPV), which is
   why the index below rounds instead of flooring. That half cell shift of the *grid* is the paper's
   only displacement; the surfel itself is not moved along its normal. */
void LPV_InjectBlocking(int gvSource, int cascade, float3 position, float3 normal, float surfelArea)
{
    float s = LPV_CascadeCellSize(cascade);
    int   resolution = LPV_GridResolution();
    int3  cell = int3(floor((position - LPV_CascadeOrigin(cascade)) / s + 0.5));
    if (!LPV_CellInside(cell, resolution))
        return;

    float scale = surfelArea / (s * s);
    if (scale <= 0.0)
        return;
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

// Solid angle of one pixel, derived from the projection matrix (1/P[1][1] = tan(fovY / 2)).
float LPV_PixelSolidAngle()
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float     tanY = 1.0 / max(abs(sceneInfo.projMatrix[1][1]), 1e-6);
    float     tanX = 1.0 / max(abs(sceneInfo.projMatrix[0][0]), 1e-6);
    float     width  = float(pushConst.screenSize.x);
    float     height = float(pushConst.screenSize.y);
    return (2.0 * tanX / width) * (2.0 * tanY / height);
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
