// VXPG - shared Slang helpers of the voxel path guiding port.
//
// Layout of the guiding data (see README.md for the mapping to the reference):
//   * cubic voxel grid of voxelResolution^3 cells over the scene AABB cube
//   * per voxel: packed irradiance sum + VPL count + sub-voxel surface AABB
//   * light tree: LBVH over the Morton ordered voxel slots (leaf == voxel slot)
//   * 32 clusters = contiguous Morton ranges, used by the per superpixel top level tree
#pragma once

#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VXPG/VXPGShaderio.h"

[[vk::push_constant]] ConstantBuffer<VXPGPushConstant, ScalarDataLayout> pushConst;

[[vk::binding(VXPGStaticBindingPoints::eVoxelIrradiance)]]
RWStructuredBuffer<uint> voxelIrradiance;      // packed sum, uint(irradiance * VXPG_IRRADIANCE_SCALE)
[[vk::binding(VXPGStaticBindingPoints::eVoxelVplCount)]]
RWStructuredBuffer<uint> voxelVplCount;        // number of injected VPLs
[[vk::binding(VXPGStaticBindingPoints::eVoxelBoundMin)]]
RWStructuredBuffer<uint> voxelBoundMin;        // uint[4] per voxel: packed sub-voxel surface AABB min
[[vk::binding(VXPGStaticBindingPoints::eVoxelBoundMax)]]
RWStructuredBuffer<uint> voxelBoundMax;        // uint[4] per voxel: packed sub-voxel surface AABB max
[[vk::binding(VXPGStaticBindingPoints::ePixelData)]]
RWStructuredBuffer<VXPGPixelData> pixelData;   // one record per pixel
[[vk::binding(VXPGStaticBindingPoints::eTreeNodes)]]
RWStructuredBuffer<VXPGTreeNode> treeNodes;    // 2 * N^3 - 1 nodes
[[vk::binding(VXPGStaticBindingPoints::eClusterRoots)]]
RWStructuredBuffer<int> clusterRoots;          // VXPG_CLUSTER_COUNT entries
[[vk::binding(VXPGStaticBindingPoints::eTopLevelTree)]]
RWStructuredBuffer<float> topLevelTree;        // spixelCount * VXPG_TOP_LEVEL_STRIDE
[[vk::binding(VXPGStaticBindingPoints::eSpixelInfo)]]
RWStructuredBuffer<VXPGSpixelInfo> spixelInfo; // spixelCount entries
[[vk::binding(VXPGStaticBindingPoints::eSpixelClusterWeight)]]
RWStructuredBuffer<int> spixelClusterWeight;   // spixelCount * VXPG_CLUSTER_COUNT, fixed point 1/1000
[[vk::binding(VXPGStaticBindingPoints::eSpixelClusterTotal)]]
RWStructuredBuffer<int> spixelClusterTotal;    // spixelCount entries, fixed point 1/1000

#define VXPG_INVALID_INDEX 0xffffffffu
#define VXPG_PI 3.14159265358979323846

//------------------------------------------------------------------------------------------
// Voxel grid mapping (identical to the reference VoxelTexInfo)
//------------------------------------------------------------------------------------------
int  VXPG_VoxelResolution() { return pushConst.voxelResolution; }
uint VXPG_VoxelCount()      { return uint(pushConst.voxelResolution) * uint(pushConst.voxelResolution) * uint(pushConst.voxelResolution); }

float3 VXPG_WorldToUnit(float3 p) { return (p - pushConst.VGBStartPos) * (1.0 / (2.0 * pushConst.VGBExtent)); }

int3 VXPG_WorldToVoxelID(float3 p)
{
    int res = VXPG_VoxelResolution();
    int3 v = int3(VXPG_WorldToUnit(p) * float(res));
    return clamp(v, int3(0, 0, 0), int3(res - 1, res - 1, res - 1));
}

// The voxel domain is now a camera centred cube (the reference implementation's VXGI clipmap),
// so positions outside of it must be rejected instead of being clamped into the border voxels.
bool VXPG_PositionInDomain(float3 p)
{
    float3 lo = pushConst.VGBStartPos;
    float3 hi = lo + float3(2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent);
    return all(p >= lo) && all(p <= hi);
}

uint VXPG_FlatIndex(int3 v)
{
    int res = VXPG_VoxelResolution();
    return uint(v.x) + uint(v.y) * uint(res) + uint(v.z) * uint(res) * uint(res);
}

int3 VXPG_ReconstructIndex(uint f)
{
    int res = VXPG_VoxelResolution();
    return int3(int(f % uint(res)), int((f / uint(res)) % uint(res)), int(f / (uint(res) * uint(res))));
}

float3 VXPG_VoxelMin(int3 v) { return pushConst.VGBStartPos + float3(v) * pushConst.VGBVoxelSize; }
float3 VXPG_VoxelMax(int3 v) { return pushConst.VGBStartPos + float3(v + int3(1, 1, 1)) * pushConst.VGBVoxelSize; }

//------------------------------------------------------------------------------------------
// Morton order. The light tree is built on the Morton ordered dense voxel slots, so the
// leaf order is a spatially coherent permutation of the voxel grid and no sort is needed.
//------------------------------------------------------------------------------------------
uint VXPG_MortonEncode(int3 v, int bits)
{
    uint code = 0u;
    for (int b = 0; b < bits; ++b)
    {
        uint x = (uint(v.x) >> uint(b)) & 1u;
        uint y = (uint(v.y) >> uint(b)) & 1u;
        uint z = (uint(v.z) >> uint(b)) & 1u;
        code |= (x | (y << 1) | (z << 2)) << uint(3 * b);
    }
    return code;
}

int3 VXPG_MortonDecode(uint code, int bits)
{
    int3 v = int3(0, 0, 0);
    for (int b = 0; b < bits; ++b)
    {
        uint bits3 = (code >> uint(3 * b)) & 7u;
        v.x |= int((bits3 >> 0) & 1u) << b;
        v.y |= int((bits3 >> 1) & 1u) << b;
        v.z |= int((bits3 >> 2) & 1u) << b;
    }
    return v;
}

// Voxel slot <-> voxel id
int   VXPG_MortonBits()  { return pushConst.maxTreeLevel; }
uint  VXPG_VoxelToSlot(int3 v) { return VXPG_MortonEncode(v, VXPG_MortonBits()); }
int3  VXPG_SlotToVoxel(uint s) { return VXPG_MortonDecode(s, VXPG_MortonBits()); }

// Number of leaves of the light tree (one per voxel slot).
uint VXPG_LeafCount()          { return VXPG_VoxelCount(); }
uint VXPG_LeafStartIndex()     { return VXPG_LeafCount() - 1u; }
uint VXPG_LeafNodeIndex(uint s){ return VXPG_LeafStartIndex() + s; }
uint VXPG_NodeCount()          { return 2u * VXPG_LeafCount() - 1u; }

// Cluster id of a leaf slot: the reference groups voxels by their visibility signature
// (MRCS). The port groups them by contiguous Morton ranges, i.e. by spatial proximity.
uint VXPG_ClusterOfSlot(uint slot)
{
    uint perCluster = max(1u, VXPG_LeafCount() / uint(VXPG_CLUSTER_COUNT));
    return min(uint(VXPG_CLUSTER_COUNT) - 1u, slot / perCluster);
}

uint VXPG_ClusterOfVoxel(int3 v) { return VXPG_ClusterOfSlot(VXPG_VoxelToSlot(v)); }

//------------------------------------------------------------------------------------------
// Kernel widths / pdf helpers
//------------------------------------------------------------------------------------------
// The light tree stores, for each internal node, the sum of its children intensities, so the
// probability of a descent step is I_child / (I_left + I_right) and the probability of a
// whole root->leaf path telescopes to I_leaf / I_root.
float VXPG_ChildProbability(float leftIntensity, float rightIntensity, bool takeLeft)
{
    float sum = leftIntensity + rightIntensity;
    if (sum <= 0.0f) return 0.0f;
    return takeLeft ? leftIntensity / sum : rightIntensity / sum;
}

// Traverse the light tree from "nid" down to a leaf. Returns VXPG_INVALID_INDEX for an
// empty (zero intensity) subtree. "prob" receives the probability of the descent.
// Intensity only weighting, which is what the reference uses for its guiding tree.
uint VXPG_TraverseTree(uint nid, float rnd, out float prob)
{
    prob = 1.0f;
    uint leafStart = VXPG_LeafStartIndex();
    if (nid == VXPG_INVALID_INDEX) return VXPG_INVALID_INDEX;
    if (nid >= leafStart) return nid;   // single node cluster

    uint current = nid;
    for (int iter = 0; iter < 64; ++iter)
    {
        VXPGTreeNode node = treeNodes[current];
        const uint nodeCap = VXPG_NodeCount();
        if (node.leftIdx >= nodeCap || node.rightIdx >= nodeCap)
            return VXPG_INVALID_INDEX;
        float leftI = treeNodes[node.leftIdx].intensity;
        float rightI = treeNodes[node.rightIdx].intensity;
        float sum = leftI + rightI;
        if (sum <= 0.0f) return VXPG_INVALID_INDEX;

        float pLeft = leftI / sum;
        bool takeLeft;
        if (leftI <= 0.0f)      { takeLeft = false; }
        else if (rightI <= 0.0f) { takeLeft = true; }
        else                    { takeLeft = rnd < pLeft; }

        if (takeLeft)
        {
            prob *= (leftI <= 0.0f) ? 0.0f : pLeft;
            rnd = (pLeft > 0.0f) ? rnd / pLeft : 0.0f;
            current = node.leftIdx;
        }
        else
        {
            prob *= (rightI <= 0.0f) ? 0.0f : (1.0f - pLeft);
            rnd = (pLeft < 1.0f) ? (rnd - pLeft) / (1.0f - pLeft) : 0.0f;
            current = node.rightIdx;
        }
        if (current >= leafStart) return current;
    }
    return VXPG_INVALID_INDEX;
}

// Probability of reaching "leafNode" when traversing from "rootNode".
float VXPG_TreePathProbability(uint rootNode, uint leafNode)
{
    const uint nodeCap = VXPG_NodeCount();
    if (rootNode == VXPG_INVALID_INDEX || leafNode == VXPG_INVALID_INDEX) return 0.0f;
    if (rootNode >= nodeCap || leafNode >= nodeCap) return 0.0f;
    if (rootNode == leafNode) return 1.0f;
    float rootIntensity = treeNodes[rootNode].intensity;
    float leafIntensity = treeNodes[leafNode].intensity;
    if (rootIntensity <= 0.0f || leafIntensity <= 0.0f) return 0.0f;
    return leafIntensity / rootIntensity;
}

//------------------------------------------------------------------------------------------
// Per superpixel top level tree: a 64 float block with a 32 leaf binary heap.
//   [offset + 32 + c] = importance of cluster c, [offset + 1] = total.
// Same packing as the reference (SampleTopLevelTree).
//------------------------------------------------------------------------------------------
int VXPG_SampleTopLevelTree(int spixelId, float rnd, out float prob)
{
    prob = 0.0f;
    if (spixelId < 0) return -1;
    int offset = spixelId * VXPG_TOP_LEVEL_STRIDE;
    if (topLevelTree[offset + 1] <= 0.0f) return -1;

    int index = 1;
    float p = 1.0f;
    for (int level = 0; level < 5; ++level)
    {
        int left = index << 1;
        int right = left + 1;
        float leftI = topLevelTree[offset + left];
        float rightI = topLevelTree[offset + right];
        float sum = leftI + rightI;
        if (sum <= 0.0f) return -1;
        float pLeft = leftI / sum;
        if (rnd < pLeft)
        {
            p *= pLeft;
            rnd = (pLeft > 0.0f) ? rnd / pLeft : 0.0f;
            index = left;
        }
        else
        {
            p *= (1.0f - pLeft);
            rnd = (pLeft < 1.0f) ? (rnd - pLeft) / (1.0f - pLeft) : 0.0f;
            index = right;
        }
    }
    prob = p;
    return index - VXPG_TOP_LEVEL_LEAF_OFFSET;
}

float VXPG_TopLevelProbability(int spixelId, int cluster)
{
    if (spixelId < 0 || cluster < 0 || cluster >= VXPG_CLUSTER_COUNT) return 0.0f;
    int offset = spixelId * VXPG_TOP_LEVEL_STRIDE;
    float total = topLevelTree[offset + 1];
    if (total <= 0.0f) return 0.0f;
    return topLevelTree[offset + VXPG_TOP_LEVEL_LEAF_OFFSET + cluster] / total;
}

// Per superpixel availability of the top level tree (false when a region saw no VPL).
bool VXPG_SpixelHasImportance(int spixelId)
{
    if (spixelId < 0) return false;
    return topLevelTree[spixelId * VXPG_TOP_LEVEL_STRIDE + 1] > 0.0f;
}

//------------------------------------------------------------------------------------------
// Spherical quad sampling (from the reference / PBRT "spherical quad" formulation).
// A voxel is sampled as the union of three spherical rectangles (one per axis pair), each
// weighted by its solid angle. Sampling and pdf are exactly consistent (pdf = 1 / sum S).
//------------------------------------------------------------------------------------------
struct VXPGSphQuad
{
    float3 o, x, y, z;
    float z0, z0sq;
    float x0, y0, y0sq;
    float x1, y1, y1sq;
    float b0, b1, b0sq, k;
    float S;
};

void VXPG_SphQuadInit(float3 s, float3 ex, float3 ey, float3 o, out VXPGSphQuad squad)
{
    squad.o = o;
    float exl = length(ex);
    float eyl = length(ey);
    squad.x = ex / exl;
    squad.y = ey / eyl;
    squad.z = cross(squad.x, squad.y);
    float3 d = s - o;
    squad.z0 = dot(d, squad.z);
    if (squad.z0 > 0.0)
    {
        squad.z *= -1.0;
        squad.z0 *= -1.0;
    }
    squad.z0sq = squad.z0 * squad.z0;
    squad.x0 = dot(d, squad.x);
    squad.y0 = dot(d, squad.y);
    squad.x1 = squad.x0 + exl;
    squad.y1 = squad.y0 + eyl;
    squad.y0sq = squad.y0 * squad.y0;
    squad.y1sq = squad.y1 * squad.y1;

    float3 v00 = float3(squad.x0, squad.y0, squad.z0);
    float3 v01 = float3(squad.x0, squad.y1, squad.z0);
    float3 v10 = float3(squad.x1, squad.y0, squad.z0);
    float3 v11 = float3(squad.x1, squad.y1, squad.z0);

    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));

    float g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
    float g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
    float g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
    float g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));

    squad.b0 = n0.z;
    squad.b1 = n2.z;
    squad.b0sq = squad.b0 * squad.b0;
    squad.k = 2.0 * VXPG_PI - g2 - g3;
    squad.S = g0 + g1 - squad.k;
}

VXPGSphQuad VXPG_CreateSphQuad(float3 local, float2 extend)
{
    VXPGSphQuad squad;
    float2 e2 = extend + extend;
    VXPG_SphQuadInit(float3(-extend.x, -extend.y, 0.0), float3(e2.x, 0.0, 0.0), float3(0.0, e2.y, 0.0), local, squad);
    return squad;
}

void VXPG_SampleSphQuad(float3 pos, VXPGSphQuad squad, float2 uv, out float3 w, out float pdf)
{
    float au = uv.x * squad.S + squad.k;
    float sinAu = sin(au);
    sinAu = (abs(sinAu) < 1e-8) ? 1e-8 : sinAu;
    float fu = (cos(au) * squad.b0 - squad.b1) / sinAu;
    float cu = 1.0 / sqrt(fu * fu + squad.b0sq) * (fu > 0.0 ? 1.0 : -1.0);
    cu = clamp(cu, -1.0, 1.0);
    float xu = -(cu * squad.z0) / sqrt(max(1.0 - cu * cu, 1e-12));
    xu = clamp(xu, squad.x0, squad.x1);
    float d = sqrt(xu * xu + squad.z0sq);
    float h0 = squad.y0 / sqrt(d * d + squad.y0sq);
    float h1 = squad.y1 / sqrt(d * d + squad.y1sq);
    float hv = h0 + uv.y * (h1 - h0);
    float hv2 = hv * hv;
    float yv = (hv2 < 1.0 - 0.0001) ? (hv * d) / sqrt(1.0 - hv2) : squad.y1;

    float3 p = squad.o + xu * squad.x + yv * squad.y + squad.z0 * squad.z;
    w = normalize(p - pos);
    pdf = 1.0 / max(squad.S, 1e-12);
}

// Directions of the three spherical rectangles for a box (center, half extent) seen from
// "refPoint": returns their solid angles.
float3 VXPG_SphericalFaceSolidAngles(float3 center, float3 extend, float3 refPoint)
{
    float3 s = sign(refPoint - center);
    float3 xc = center + float3(s.x * extend.x, 0, 0);
    float3 yc = center + float3(0, s.y * extend.y, 0);
    float3 zc = center + float3(0, 0, s.z * extend.z);

    // local = R_i * s_i * (refPoint - center_i) with R_0=(y,z,x), R_1=(x,z,y), R_2=I
    float3 d0 = refPoint - xc;
    float3 d1 = refPoint - yc;
    float3 d2 = refPoint - zc;
    float3 local0 = float3(d0.y, d0.z, d0.x) * s.x;
    float3 local1 = float3(d1.x, d1.z, d1.y) * s.y;
    float3 local2 = d2 * s.z;

    VXPGSphQuad q0 = VXPG_CreateSphQuad(local0, extend.yz);
    VXPGSphQuad q1 = VXPG_CreateSphQuad(local1, extend.xz);
    VXPGSphQuad q2 = VXPG_CreateSphQuad(local2, extend.xy);

    float3 pdfs;
    pdfs.x = (s.x == 0.0 || isnan(q0.S)) ? 0.0 : q0.S;
    pdfs.y = (s.y == 0.0 || isnan(q1.S)) ? 0.0 : q1.S;
    pdfs.z = (s.z == 0.0 || isnan(q2.S)) ? 0.0 : q2.S;
    return pdfs;
}

// Sample a direction towards a box (a "spherical voxel").
float3 VXPG_SampleSphericalVoxel(float3 center, float3 extend, float3 refPoint, float3 rnds, out float pdf)
{
    float3 s = sign(refPoint - center);
    float3 xc = center + float3(s.x * extend.x, 0, 0);
    float3 yc = center + float3(0, s.y * extend.y, 0);
    float3 zc = center + float3(0, 0, s.z * extend.z);

    float3 d0 = refPoint - xc;
    float3 d1 = refPoint - yc;
    float3 d2 = refPoint - zc;
    float3 local0 = float3(d0.y, d0.z, d0.x) * s.x;
    float3 local1 = float3(d1.x, d1.z, d1.y) * s.y;
    float3 local2 = d2 * s.z;

    VXPGSphQuad q0 = VXPG_CreateSphQuad(local0, extend.yz);
    VXPGSphQuad q1 = VXPG_CreateSphQuad(local1, extend.xz);
    VXPGSphQuad q2 = VXPG_CreateSphQuad(local2, extend.xy);

    float3 pdfs;
    pdfs.x = (s.x == 0.0 || isnan(q0.S)) ? 0.0 : q0.S;
    pdfs.y = (s.y == 0.0 || isnan(q1.S)) ? 0.0 : q1.S;
    pdfs.z = (s.z == 0.0 || isnan(q2.S)) ? 0.0 : q2.S;
    float sum = pdfs.x + pdfs.y + pdfs.z;
    if (sum <= 0.0)
    {
        pdf = 0.0;
        return normalize(center - refPoint);
    }

    float r = rnds.x * sum;
    int face = 2;
    float facePdf = pdfs.z / sum;
    float accum = pdfs.x;
    if (r <= accum) { face = 0; facePdf = pdfs.x / sum; }
    else
    {
        accum += pdfs.y;
        if (r <= accum) { face = 1; facePdf = pdfs.y / sum; }
    }

    float3 localDir;
    float quadPdf;
    if (face == 0) VXPG_SampleSphQuad(local0, q0, rnds.yz, localDir, quadPdf);
    else if (face == 1) VXPG_SampleSphQuad(local1, q1, rnds.yz, localDir, quadPdf);
    else VXPG_SampleSphQuad(local2, q2, rnds.yz, localDir, quadPdf);

    pdf = facePdf * quadPdf;
    float3 worldDir;
    if (face == 0) worldDir = float3(localDir.z, localDir.x, localDir.y) * s.x;
    else if (face == 1) worldDir = float3(localDir.x, localDir.z, localDir.y) * s.y;
    else worldDir = localDir * s.z;
    return normalize(worldDir);
}

float VXPG_PdfSampleSphericalVoxel(float3 center, float3 extend, float3 refPoint)
{
    float3 pdfs = VXPG_SphericalFaceSolidAngles(center, extend, refPoint);
    float sum = pdfs.x + pdfs.y + pdfs.z;
    return (sum > 0.0) ? 1.0 / sum : 0.0;
}

//------------------------------------------------------------------------------------------
// Voxel query helpers
//------------------------------------------------------------------------------------------
// Unpack the sub-voxel surface AABB of a voxel (world space). Falls back to the full voxel
// when the geometry injection did not write anything (packed min = 0, max = 0xffffffff).
void VXPG_VoxelSurfaceBounds(int3 v, out float3 bmin, out float3 bmax)
{
    uint flat = VXPG_FlatIndex(v);
    float3 vmin = VXPG_VoxelMin(v);
    float3 vmax = VXPG_VoxelMax(v);
    float3 ext = vmax - vmin;
    float3 pmin = float3(voxelBoundMin[flat * 4 + 0], voxelBoundMin[flat * 4 + 1], voxelBoundMin[flat * 4 + 2]) / float(0xffffffffu);
    float3 pmax = float3(voxelBoundMax[flat * 4 + 0], voxelBoundMax[flat * 4 + 1], voxelBoundMax[flat * 4 + 2]) / float(0xffffffffu);
    bmin = vmin + saturate(pmin) * ext;
    bmax = vmin + saturate(pmax) * ext;
    // degenerate / unwritten bounds collapse to an empty box: use the full voxel.
    if (any(bmax - bmin <= 1e-7 * ext))
    {
        bmin = vmin;
        bmax = vmax;
    }
}

// Average irradiance of a voxel (0 when nothing was injected there).
float VXPG_VoxelIrradiance(uint flat)
{
    uint count = voxelVplCount[flat];
    if (count == 0u) return 0.0f;
    return (float(voxelIrradiance[flat]) / VXPG_IRRADIANCE_SCALE) / float(count);
}

float VXPG_VoxelImportance(uint flat) { return VXPG_VoxelIrradiance(flat); }

//------------------------------------------------------------------------------------------
// Ray tracing / payload helpers (RayQuery based, like the VRPG renderer of this project)
//------------------------------------------------------------------------------------------
struct VXPGHit
{
    bool  hit;
    bool  frontFacing;
    bool  hasTexCoords;
    float3 position;
    float3 normal;
    float2 texCoords;
    int   materialIndex;
};

VXPGHit VXPG_TraceRay(RayDesc ray)
{
    VXPGHit result;
    result.hit = false;
    result.frontFacing = true;
    result.hasTexCoords = false;
    result.position = float3(0.0);
    result.normal = float3(0.0, 1.0, 0.0);
    result.texCoords = float2(0.0);
    result.materialIndex = 0;

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
    float3 bary = float3(1.0 - bary2.x - bary2.y, bary2.x, bary2.y);
    float4x3 worldToObject = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
    int instanceIndex = q.CommittedInstanceIndex();
    uint primitiveIndex = q.CommittedPrimitiveIndex();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    float3 localPos = triPos[0] * bary.x + triPos[1] * bary.y + triPos[2] * bary.z;
    result.position = float3(mul(float4(localPos, 1.0), objectToWorld));
    float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    result.normal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    float cosineON = dot(result.normal, -ray.Direction);
    result.frontFacing = cosineON > 0.0;
    if (cosineON <= 0.0) result.normal = -result.normal;

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0)
    {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, int(primitiveIndex));
        result.texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, bary);
        result.texCoords.y = 1.0 - result.texCoords.y;
        result.hasTexCoords = true;
    }
    result.materialIndex = int(instance.materialIndex);
    result.hit = true;
    return result;
}

float3 VXPG_RayMiss(float3 rayDirection)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    return sceneInfo.backgroundColor;
}

// Build a shader payload from an explicit shading point (used for the second vertices where
// the tracer only returns the hit record).
void VXPG_MakePayload(inout CallablePayload payload, VXPGHit hit, float3 outgoing, uint seed, SceneInfo sceneInfo)
{
    payload.material = sceneInfo.materials[hit.materialIndex];
    payload.hitPos = hit.position;
    payload.hitNormal = hit.normal;
    payload.outgoing = outgoing;
    payload.isExt = hit.frontFacing;
    payload.randomSeed = seed;
    payload.bsdf = float3(0.0, 0.0, 0.0);
    payload.pdf = 1.0;
    payload.incidence = float3(0.0, 0.0, 0.0);
    payload.directionLightDir = float3(0.0, 0.0, 0.0);
    payload.directionLightRadiance = float3(0.0, 0.0, 0.0);
    orthonormalBasis(payload.hitNormal, payload.TBN[0], payload.TBN[1]);
    payload.TBN[2] = payload.hitNormal;
    if (hit.hasTexCoords)
        getTextueData(payload, hit.texCoords);
    if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
        payload.material.emissive *= hit.frontFacing ? 1.0 : 0.0;
}

//------------------------------------------------------------------------------------------
// Guide sampling (sample mode 4 uses the "spixel + light tree" distribution of VXPG).
//------------------------------------------------------------------------------------------
struct VXPGGuideSample
{
    bool   valid;
    float3 direction;
    float3 aabbMin;
    float3 aabbMax;
    int3   voxel;
    float  pdf;
};

// Select the superpixel of a pixel. The reference also offers a "fuzzy" mode that blends the
// 4 nearest superpixels; the port keeps the exact-pdf variant only (see README.md).
int VXPG_SelectSpixel(int2 pixel)
{
    int2 res = int2(pushConst.screenSize);
    int2 p = clamp(pixel, int2(0, 0), res - int2(1, 1));
    return int(pixelData[p.y * res.x + p.x].info.x);
}

// Sample the VXPG distribution at a shading point.
VXPGGuideSample VXPG_SampleVoxelGuiding(int2 pixel, float3 shadingPos, int3 currentVoxel, inout uint seed)
{
    VXPGGuideSample result;
    result.valid = false;
    result.direction = float3(0.0, 0.0, 1.0);
    result.aabbMin = float3(1e30);
    result.aabbMax = float3(-1e30);
    result.pdf = 0.0;

    // Jitter the shading point (SP) by up to half a voxel and the superpixel lookup by a few
    // pixels. Every sample therefore evaluates the guiding distribution at a slightly different
    // position, which dithers the per voxel / per superpixel weight steps instead of freezing them
    // on the grid. Sampling and the returned pdf use the same jittered SP, so the density is exact.
    const float3 spJitter = (float3(rand(seed), rand(seed), rand(seed)) - 0.5) * pushConst.VGBVoxelSize
                          * pushConst.jitterRadius;
    const float3 shadingPosJittered = shadingPos + spJitter;
    const int spixelJitterRange = max(int(pushConst.padding0), 0);
    const int2 pixelJittered = pixel + int2(int(rand(seed) * float(spixelJitterRange)),
                                            int(rand(seed) * float(spixelJitterRange)));

    int spixel = VXPG_SelectSpixel(pixelJittered);

    // 1) pick a cluster with the per superpixel top level tree
    float topProb = 0.0f;
    int cluster = VXPG_SampleTopLevelTree(spixel, rand(seed), topProb);
    if (cluster < 0 || topProb <= 0.0f)
        return result;

    int clusterRoot = clusterRoots[cluster];
    if (clusterRoot < 0)
        return result;

    // 2) descend the light tree of the cluster
    float bottomProb = 0.0f;
    uint leaf = VXPG_TraverseTree(uint(clusterRoot), rand(seed), bottomProb);
    if (leaf == VXPG_INVALID_INDEX || bottomProb <= 0.0f)
        return result;

    if (leaf >= VXPG_NodeCount())
        return result;
    if (leaf >= VXPG_NodeCount())
        return result;
    uint slot = treeNodes[leaf].voxelSlot;
    int3 voxel = VXPG_SlotToVoxel(slot);
    if (all(voxel == currentVoxel))
        return result;

    // 3) sample a direction within the surface sub-voxel.
    // The target box is jittered inside the voxel: the aimed surface patch then moves from sample
    // to sample (and therefore from frame to frame), so a systematically occluded patch can no
    // longer imprint a fixed voxel sized patch on the shaded surfaces. Sampling and pdf use the
    // same jittered box, so the density stays exact.
    float3 bmin, bmax;
    VXPG_VoxelSurfaceBounds(voxel, bmin, bmax);
    float3 center = 0.5 * (bmin + bmax);
    float3 extend = max(0.5 * (bmax - bmin), float3(1e-6));
    // Jitter the target box inside the voxel only as far as it still fits, so the aimed point
    // moves between samples (dithering the voxel grid) while the direction stays inside the voxel
    // that the semi-NEE acceptance test uses. This keeps the brightness unchanged.
    const float3 boxExtent = 2.0 * extend;
    const float3 freeSpace = max(pushConst.VGBVoxelSize - boxExtent, float3(0.0));
    const float3 boxShift = (float3(rand(seed), rand(seed), rand(seed)) - 0.5) * freeSpace * pushConst.jitterRadius;
    float sphPdf = 0.0f;
    float rnd0 = rand(seed);
    float rnd1 = rand(seed);
    float rnd2 = rand(seed);
    // Sampled from the *real* shading point: the jitter only moved the lookup position, so the
    // aimed solid angle and the returned density stay consistent with the acceptance test.
    center += boxShift;
    float3 dir = VXPG_SampleSphericalVoxel(center, extend, shadingPos, float3(rnd0, rnd1, rnd2), sphPdf);
    if (sphPdf <= 0.0f)
        return result;

    result.valid = true;
    result.direction = dir;
    result.aabbMin = VXPG_VoxelMin(voxel);
    result.aabbMax = VXPG_VoxelMax(voxel);
    result.voxel = voxel;
    result.pdf = topProb * bottomProb * sphPdf;
    return result;
}

// Probability of the guidance generating a direction that lands in "voxel" (i.e. the
// probability the guided technique would have sampled this voxel and a direction inside it).
float VXPG_PdfVoxelGuiding(int2 pixel, int3 voxel, int3 currentVoxel, float3 shadingPos)
{
    if (all(voxel == currentVoxel)) return 0.0f;
    if (voxel.x < 0 || voxel.y < 0 || voxel.z < 0) return 0.0f;
    int res = VXPG_VoxelResolution();
    if (voxel.x >= res || voxel.y >= res || voxel.z >= res) return 0.0f;

    uint slot = VXPG_VoxelToSlot(voxel);
    uint leaf = VXPG_LeafNodeIndex(slot);
    if (leaf >= VXPG_NodeCount()) return 0.0f;
    if (treeNodes[leaf].intensity <= 0.0f) return 0.0f;

    int spixel = VXPG_SelectSpixel(pixel);
    int cluster = int(VXPG_ClusterOfSlot(slot));
    int clusterRoot = clusterRoots[cluster];
    if (clusterRoot < 0) return 0.0f;

    float clusterProb = VXPG_TopLevelProbability(spixel, cluster);
    if (clusterProb <= 0.0f) return 0.0f;

    float voxelProb = VXPG_TreePathProbability(uint(clusterRoot), leaf);
    if (voxelProb <= 0.0f) return 0.0f;

    float3 bmin, bmax;
    VXPG_VoxelSurfaceBounds(voxel, bmin, bmax);
    float3 center = 0.5 * (bmin + bmax);
    float3 extend = max(0.5 * (bmax - bmin), float3(1e-6));
    float sphPdf = VXPG_PdfSampleSphericalVoxel(center, extend, shadingPos);

    return clusterProb * voxelProb * sphPdf;
}

//------------------------------------------------------------------------------------------
// Small integration helpers
//------------------------------------------------------------------------------------------
RayDesc VXPG_SpawnRay(float3 origin, float3 normal, float3 direction)
{
    RayDesc ray;
    ray.Origin = offsetRay(origin, normal);
    ray.Direction = normalize(direction);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;
    return ray;
}

// Direct lighting of a shading point: punctual lights through NEE plus the emissive term of
// the surface itself. The reference only supports analytic lights in its injection, the port
// also adds the emissive term so that emissive mesh scenes (e.g. "staircase") drive the
// guiding distribution. getDirectionLight() must run before BSDF_SAMPLE(), which multiplies
// its result by the BRDF.
float3 VXPG_DirectLight(inout CallablePayload payload, SceneInfo sceneInfo)
{
    // A back facing vertex (the geometric normal points away from the incoming ray) has no
    // direct light on the side the path sees it from; evaluating NEE with the flipped shading
    // normal would create light where there is none.
    if (!payload.isExt)
    {
        payload.directionLightRadiance = float3(0.0, 0.0, 0.0);
        return payload.material.emissive;
    }
    getDirectionLight(payload, sceneInfo);
    float3 emissive = payload.material.emissive;
    BSDF_SAMPLE(payload);
    return emissive + payload.directionLightRadiance;
}

//------------------------------------------------------------------------------------------
// Smooth guide density used for the MIS weights. The exact density (VXPG_PdfVoxelGuiding)
// contains the per voxel intensity ratio I_leaf / I_clusterRoot, which jumps between 0 and a
// large value for neighbouring voxels; using it directly as a MIS weight paints the BSDF
// contribution with the voxel grid (the visible squares). The ratio is therefore clamped to a
// narrow range here. Both techniques use this same weight function while their contributions are
// still divided by the exact densities, so the weights sum to one per direction and the combined
// estimator stays consistent; only the split between the two techniques changes.
float VXPG_PdfVoxelGuidingSmooth(int2 pixel, int3 voxel, int3 currentVoxel, float3 shadingPos)
{
    if (all(voxel == currentVoxel))
        return 0.0f;
    const int res = VXPG_VoxelResolution();
    if (any(voxel < int3(0, 0, 0)) || any(voxel >= int3(res, res, res)))
        return 0.0f;

    const uint slot = VXPG_VoxelToSlot(voxel);
    const uint leaf = VXPG_LeafNodeIndex(slot);
    if (leaf >= VXPG_NodeCount())
        return 0.0f;

    const int spixel = VXPG_SelectSpixel(pixel);
    const int cluster = int(VXPG_ClusterOfSlot(slot));
    const int clusterRoot = clusterRoots[cluster];
    if (clusterRoot < 0)
        return 0.0f;

    const float clusterProb = VXPG_TopLevelProbability(spixel, cluster);
    if (clusterProb <= 0.0f)
        return 0.0f;

    float ratio = 0.0f;
    const float rootIntensity = treeNodes[uint(clusterRoot)].intensity;
    if (rootIntensity > 0.0f)
        ratio = clamp(treeNodes[leaf].intensity / rootIntensity, 0.05, 1.0);

    float3 bmin, bmax;
    VXPG_VoxelSurfaceBounds(voxel, bmin, bmax);
    const float3 center = 0.5 * (bmin + bmax);
    const float3 extend = max(0.5 * (bmax - bmin), float3(1e-6));
    return clusterProb * ratio * VXPG_PdfSampleSphericalVoxel(center, extend, shadingPos);
}