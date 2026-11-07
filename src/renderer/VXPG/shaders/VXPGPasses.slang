// VXPG - all compute passes except the integrator (see VXPG.slang).
//
// One module with several entry points, like the reference implementation which has one
// shader per pass. All entry points share the descriptor set / push constant declared in
// VXPGCommon.slang.
#include "renderer/VXPG/shaders/VXPGCommon.slang"

//------------------------------------------------------------------------------------------
// 1) VBuffer: trace the primary ray at the pixel centre and store the shading point.
//------------------------------------------------------------------------------------------
[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgVBufferMain(uint2 threadIndex : SV_DispatchThreadID){
    if (any(threadIndex >= pushConst.screenSize)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float2 launchID = float2(threadIndex) + 0.5;
    float2 launchSize = float2(pushConst.screenSize);
    float2 clipCoords = launchID / launchSize * 2.0 - 1.0;

    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    RayDesc ray;
    ray.Origin = sceneInfo.cameraPosition;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    VXPGHit hit = VXPG_TraceRay(ray);

    // Keep the superpixel id of the previous frame: when the guiding structure is not rebuilt
    // every frame, this is the only place the assignment survives.
    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    const VXPGPixelData previous = pixelData[index];
    const uint previousSpixel = previous.info.x;
    // The VPL voxel and the throughput were produced by the injection pass that ran just before
    // this pass (from the previous frame's samples), so they have to survive this overwrite.
    const uint previousVplVoxel = previous.info.z;
    const float previousVplThroughput = previous.vplPos.w;

    VXPGPixelData pd;
    pd.position = float4(0.0);
    pd.normalMaterial = float4(0.0);
    pd.vplPos = float4(0.0, 0.0, 0.0, previousVplThroughput);
    pd.info = uint4(previousSpixel, VXPG_INVALID_INDEX, previousVplVoxel, 0u);
    pd.texCoordsIsExt = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (hit.hit){
        pd.position = float4(hit.position, 1.0);
        pd.normalMaterial = float4(hit.normal, asfloat(uint(hit.materialIndex)));
        pd.info.y = VXPG_FlatIndex(VXPG_WorldToVoxelID(hit.position));
        pd.info.w = 1u;
        pd.texCoordsIsExt.xy = hit.texCoords;
        pd.texCoordsIsExt.z = hit.hasTexCoords ? 1.0f : 0.0f;
        pd.texCoordsIsExt.w = hit.frontFacing ? 1.0f : 0.0f;
    }

    pixelData[index] = pd;
}

//------------------------------------------------------------------------------------------
// 2) Light injection: place one virtual point light at the 1st indirect vertex of a camera
//    path and accumulate its irradiance into the voxel grid (reference "ImportonInjection").
//------------------------------------------------------------------------------------------
[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgLightInjectMain(uint2 threadIndex : SV_DispatchThreadID){
    if (any(threadIndex >= pushConst.screenSize)) return;

    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;

    // The injection runs BEFORE vBuffer and therefore reads the BSDF samples of the previous frame
    // (supplemental Eq. 8): the guiding distribution must not be conditioned on the samples that
    // the MIS estimator uses this frame, otherwise the balance heuristic is biased. The results are
    // stored in the record so vBuffer can carry them into this frame's record.
    {
        VXPGPixelData previous = pixelData[index];
        previous.info.z = VXPG_INVALID_INDEX;
        previous.vplPos.w = 0.0;
        pixelData[index] = previous;
    }

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    // Independent random stream (supplemental Eq. 8): the injected BSDF sample has to be an
    // independent draw from p0, not the very sample the MIS estimator uses for this pixel. With a
    // shared seed the guide distribution is fitted to the estimator's own sample and the balance
    // heuristic becomes biased. This is the cheap variant of "trace two sets of BSDF samples".
    uint seed = xxhash32(uint3(threadIndex, uint(pushConst.randomFrameIndex) ^ 0x9E3779B9u));
    // Reuses the hit record of the previous frame (position / shading normal / material / texcoords /
    // front facing); the camera ray is only needed for the exact outgoing direction.
    float2 launchID = float2(threadIndex) + 0.5;
    float2 launchSize = float2(pushConst.screenSize);
    float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    RayDesc cameraRay;
    cameraRay.Origin = sceneInfo.cameraPosition;
    cameraRay.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    cameraRay.TMin = 0.001;
    cameraRay.TMax = INFINITE;

    //VXPGHit primary = VXPG_TraceRay(cameraRay);
    //if (!primary.hit) return;

    VXPGPixelData pd = pixelData[index];
    if (pd.info.w == 0u) return;

    CallablePayload payload;
    VXPG_MakePayload(payload, pd, -cameraRay.Direction, seed, sceneInfo);

    BSDF_SAMPLE(payload);
    float3 bsdfCosine = payload.bsdf * abs(dot(payload.incidence, payload.hitNormal));
    if (!(payload.pdf > 0.0) || all(bsdfCosine <= 0.0)) return;

    // Camera path throughput up to the virtual light: bsdf * cos / pdf. The paper's supervoxel
    // weight is T_j * sum(Phi), where T_j is the average throughput of the cluster for a shading
    // point; each camera path sample is such a (shading point, voxel) pair, so the throughput is
    // measured here and combined with the voxel power in vxpgSpixelAccumMain.
    const float3 pathThroughput = bsdfCosine / payload.pdf;

    RayDesc ray = VXPG_SpawnRay(payload.hitPos, payload.hitNormal, payload.incidence);
    VXPGHit hit = VXPG_TraceRay(ray);
    if (!hit.hit) return;
    // A VPL on a back facing surface carries no physical light on the side the path sees it
    // from, and a VPL far outside the camera volume is either unreachable or would waste the
    // voxel grid (the reference implementation uses a camera centred clipmap for this).
    if (!hit.frontFacing) return;
    if (!VXPG_PositionInDomain(hit.position)) return;

    VXPG_MakePayload(payload, hit, -payload.incidence, seed, sceneInfo);
    float3 radiance = VXPG_DirectLight(payload, sceneInfo);

    float irradiance = max(max(radiance.x, radiance.y), radiance.z);
    if (!(irradiance > 0.0)) return;

    // Jittered injection: the VPL position is jittered by up to half a voxel (SV Injection Jitter
    // in the panel) before it is quantised, so the per voxel importance becomes a smooth function
    // of the position instead of a hard per voxel count.
    const float3 jitter = float3(rand(seed) - 0.5, rand(seed) - 0.5, rand(seed) - 0.5) * pushConst.VGBVoxelSize * pushConst.padding1;
    int3 voxel = VXPG_WorldToVoxelID(hit.position + jitter);
    uint flat = VXPG_FlatIndex(voxel);
    uint packedIrradiance = uint(irradiance * VXPG_IRRADIANCE_SCALE);
    if (packedIrradiance == 0u) packedIrradiance = 1u;

    InterlockedAdd(voxelIrradiance[flat], packedIrradiance);
    InterlockedAdd(voxelVplCount[flat], 1u);

    // w stores the luminance of the contribution this VPL makes to this pixel (throughput *
    // radiance); it is the per sample throughput estimate used by the supervoxel weights.
    const float3 vplContribution = pathThroughput * radiance;
    pd.vplPos = float4(hit.position, dot(vplContribution, float3(0.2126, 0.7152, 0.0722)));
    pd.info.z = flat;
    pixelData[index] = pd;
}

//------------------------------------------------------------------------------------------
// 3) Geometry injection: conservative per triangle voxelization. For every voxel touched by
//    a triangle the AABB of the (triangle AABB clipped to the voxel) is merged into the
//    per voxel surface bounds, which the guide sampler uses as the spherical voxel shape.
//------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
[shader("compute")]
void vxpgGeometryInjectMain(uint threadIndex : SV_DispatchThreadID)
{
    if (threadIndex >= uint(pushConst.triangleCount)) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    if (mesh.dataBuffer == nullptr) return;
    const uint vertexCount = mesh.triMesh.positions.count;
    if (vertexCount == 0u || mesh.triMesh.indices.count == 0u) return;

    int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, int(threadIndex));
    if (indices.x < 0 || indices.y < 0 || indices.z < 0) return;
    if (uint(indices.x) >= vertexCount || uint(indices.y) >= vertexCount || uint(indices.z) >= vertexCount) return;

    float3 v0 = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, uint(indices.x));
    float3 v1 = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, uint(indices.y));
    float3 v2 = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, uint(indices.z));
    float3 p0 = mul(float4(v0, 1.0), instance.transform).xyz;
    float3 p1 = mul(float4(v1, 1.0), instance.transform).xyz;
    float3 p2 = mul(float4(v2, 1.0), instance.transform).xyz;

    float3 lo = min(p0, min(p1, p2));
    float3 hi = max(p0, max(p1, p2));

    // Reject triangles that do not overlap the (camera centred) voxel domain: clamping them
    // into the border voxels would write bogus surface bounds there.
    {
        const float3 domainMin = pushConst.VGBStartPos;
        const float3 domainMax = domainMin + float3(2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent);
        if (any(hi < domainMin) || any(lo > domainMax)) return;
    }
    int3 vlo = VXPG_WorldToVoxelID(lo);
    int3 vhi = VXPG_WorldToVoxelID(hi);

    // Conservative lattice sampling: large triangles (relative to a voxel) are subsampled to
    // at most 4 voxels per axis, which bounds the work per triangle (a huge floor/wall
    // triangle must not turn into a multi second dispatch). Only voxels that received a VPL
    // are touched, exactly like the reference dynamic geometry injection, so the cost scales
    // with the number of lit voxels and not with the scene resolution.
    int3 span = vhi - vlo + int3(1, 1, 1);
    int3 stepSize = max(int3(1, 1, 1), (span + int3(3, 3, 3)) / 4);
    int3 gridCount = int3((vhi - vlo) / stepSize) + int3(1, 1, 1);
    int cellCount = min(gridCount.x * gridCount.y * gridCount.z, 64);

    for (int cell = 0; cell < cellCount; ++cell)
    {
        int3 offset = int3(cell % gridCount.x, (cell / gridCount.x) % gridCount.y,
                           cell / (gridCount.x * gridCount.y));
        int3 voxel = vlo + offset * stepSize;
        uint flat = VXPG_FlatIndex(voxel);
        if (voxelVplCount[flat] == 0u) continue;

        float3 vmin = VXPG_VoxelMin(voxel);
        float3 vmax = VXPG_VoxelMax(voxel);
        float3 ext = max(vmax - vmin, float3(1e-9));

        float3 bmin = clamp(lo, vmin, vmax);
        float3 bmax = clamp(hi, vmin, vmax);
        float3 qmin = saturate((bmin - vmin) / ext);
        float3 qmax = saturate((bmax - vmin) / ext);
        const float kMaxQuant = 4294967040.0f;
        uint3 qmn = uint3(saturate(qmin) * kMaxQuant);
        uint3 qmx = uint3(saturate(qmax) * kMaxQuant);

        InterlockedMin(voxelBoundMin[flat * 4 + 0], qmn.x);
        InterlockedMin(voxelBoundMin[flat * 4 + 1], qmn.y);
        InterlockedMin(voxelBoundMin[flat * 4 + 2], qmn.z);
        InterlockedMax(voxelBoundMax[flat * 4 + 0], qmx.x);
        InterlockedMax(voxelBoundMax[flat * 4 + 1], qmx.y);
        InterlockedMax(voxelBoundMax[flat * 4 + 2], qmx.z);
    }
}

//------------------------------------------------------------------------------------------
// 4) Light tree initialization: default internal nodes and leaf nodes (leaf == voxel slot).
//------------------------------------------------------------------------------------------
[numthreads(VXPG_TREE_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void vxpgTreeInitMain(uint threadIndex : SV_DispatchThreadID)
{
    if (threadIndex < uint(VXPG_CLUSTER_COUNT)) clusterRoots[threadIndex] = -1;

    uint leafCount = VXPG_LeafCount();
    uint leafStart = VXPG_LeafStartIndex();

    //NOTE: leafVoxel / voxelLeaf / clusterRange are filled by vxpgLeafMapInitMain (separate pass,
    // because this pass reads them).
    if (threadIndex >= VXPG_NodeCount())
        return;

    VXPGTreeNode node;
    node.intensity = 0.0f;
    node.flag = 0u;
    node.parentIdx = VXPG_INVALID_INDEX;
    node.leftIdx = VXPG_INVALID_INDEX;
    node.rightIdx = VXPG_INVALID_INDEX;
    node.voxelSlot = VXPG_INVALID_INDEX;

    if (threadIndex < leafStart)
    {
        treeNodes[threadIndex] = node;
        return;
    }

    uint slot = threadIndex - leafStart;
    uint flat = leafVoxel[slot];
    if (flat == VXPG_INVALID_INDEX)
    {
        // Leaf that no lit voxel was sorted into (the cluster sorted order only fills the first
        // litVoxelCount leaves): keep it empty so the sampler never descends into it.
        treeNodes[threadIndex] = node;
        return;
    }

    node.intensity = VXPG_VoxelImportance(flat);
    node.flag = VXPG_ClusterOfSlot(slot);
    node.voxelSlot = slot;
    treeNodes[threadIndex] = node;
}

//------------------------------------------------------------------------------------------
// 5) Light tree topology (Karras style binary radix tree over the Morton ordered slots, i.e.
//    exactly the reference "TreeInternal" pass, but the codes are the slot order itself so
//    no bitonic sort is needed).
//------------------------------------------------------------------------------------------
int VXPG_Clz(uint x)
{
    return (x == 0u) ? 32 : (31 - firstbithigh(x));
}

int VXPG_CodeCommonBits(uint a, uint b)
{
    // code(i) = i + 1
    return VXPG_Clz((a + 1u) ^ (b + 1u));
}

uint2 VXPG_DetermineRange(uint numLeaves, uint idx)
{
    if (idx == 0u)
        return uint2(0u, numLeaves - 1u);

    uint self = idx + 1u;
    int leftDelta = VXPG_Clz(self ^ idx);           // code(idx-1) == idx
    int rightDelta = VXPG_Clz(self ^ (idx + 2u));   // code(idx+1) == idx+2
    int direction = (rightDelta > leftDelta) ? 1 : -1;
    int deltaMin = min(leftDelta, rightDelta);

    int lMax = 2;
    int delta = -1;
    int probe = int(idx) + direction * lMax;
    if (probe >= 0 && probe < int(numLeaves))
        delta = VXPG_CodeCommonBits(idx, uint(probe));

    while (delta > deltaMin)
    {
        lMax <<= 1;
        probe = int(idx) + direction * lMax;
        delta = -1;
        if (probe >= 0 && probe < int(numLeaves))
            delta = VXPG_CodeCommonBits(idx, uint(probe));
    }

    int l = 0;
    int t = lMax >> 1;
    while (t > 0)
    {
        probe = int(idx) + (l + t) * direction;
        delta = -1;
        if (probe >= 0 && probe < int(numLeaves))
            delta = VXPG_CodeCommonBits(idx, uint(probe));
        if (delta > deltaMin)
            l += t;
        t >>= 1;
    }

    uint other = idx + uint(l * direction);
    uint first = min(idx, other);
    uint last = max(idx, other);
    return uint2(first, last);
}

uint VXPG_FindSplit(uint first, uint last)
{
    if (first == last)
        return (first + last) >> 1;

    int split = int(first);
    int stride = int(last - first);
    int deltaNode = VXPG_CodeCommonBits(first, last);
    do
    {
        stride = (stride + 1) >> 1;
        int middle = split + stride;
        if (middle < int(last))
        {
            int delta = VXPG_CodeCommonBits(first, uint(middle));
            if (delta > deltaNode)
                split = middle;
        }
    } while (stride > 1);

    return uint(split);
}

[numthreads(VXPG_TREE_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void vxpgTreeInternalMain(uint threadIndex : SV_DispatchThreadID)
{
    uint numLeaves = VXPG_LeafCount();
    uint leafStart = VXPG_LeafStartIndex();
    if (threadIndex >= leafStart) return;

    uint2 range = VXPG_DetermineRange(numLeaves, threadIndex);
    uint split = VXPG_FindSplit(range.x, range.y);

    uint left = split;
    uint right = split + 1u;
    if (min(range.x, range.y) == split)
        left += leafStart;
    if (max(range.x, range.y) == split + 1u)
        right += leafStart;

    const uint nodeCap = VXPG_NodeCount();
    if (left >= nodeCap || right >= nodeCap)
        return;
    treeNodes[threadIndex].leftIdx = left;
    treeNodes[threadIndex].rightIdx = right;
    treeNodes[left].parentIdx = threadIndex;
    treeNodes[right].parentIdx = threadIndex;
}

//------------------------------------------------------------------------------------------
// 6) Light tree intensities: every leaf adds its importance to all of its ancestors. This
//    replaces the intensity computation of the reference merge pass, which relies on a
//    lock step handshake between the two children threads; that handshake is not guaranteed
//    to publish the child intensity before the parent reads it, which left internal nodes at
//    zero (the guide sampler then aborted most descents). Summing from the leaves with
//    float atomics is order independent and always exact.
//------------------------------------------------------------------------------------------
[numthreads(VXPG_TREE_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void vxpgTreeAccumulateMain(uint threadIndex : SV_DispatchThreadID)
{
    uint leafCount = VXPG_LeafCount();
    uint leafStart = VXPG_LeafStartIndex();
    if (threadIndex >= leafCount)
        return;

    uint leafNode = leafStart + threadIndex;
    float intensity = treeNodes[leafNode].intensity;
    if (!(intensity > 0.0))
        return;

    const uint nodeCap = VXPG_NodeCount();
    uint node = treeNodes[leafNode].parentIdx;
    for (int iter = 0; iter < 64; ++iter)
    {
        if (node == VXPG_INVALID_INDEX || node >= nodeCap)
            break;
        InterlockedAdd(treeNodes[node].intensity, intensity);
        node = treeNodes[node].parentIdx;
    }
}
//------------------------------------------------------------------------------------------
// 7) Cluster roots: for every cluster (a contiguous range of Morton ordered leaves) find the
//    lowest common ancestor of its leaves, i.e. the highest tree node that is entirely inside
//    the cluster. This is done by a single thread per cluster, so it is fully deterministic:
//    the reference's "two children handshake" merge is not, because it relies on cross thread
//    memory ordering and on the two siblings arriving in a specific order.
//------------------------------------------------------------------------------------------
uint2 VXPG_NodeRange(uint node)
{
    uint leafStart = VXPG_LeafStartIndex();
    if (node >= leafStart)
    {
        uint slot = node - leafStart;
        return uint2(slot, slot);
    }
    return VXPG_DetermineRange(VXPG_LeafCount(), node);
}

[numthreads(VXPG_CLUSTER_COUNT, 1, 1)]
[shader("compute")]
void vxpgClusterRootsMain(uint clusterIndex : SV_DispatchThreadID)
{
    if (clusterIndex >= uint(VXPG_CLUSTER_COUNT))
        return;

    const uint leafCount = VXPG_LeafCount();
    const uint leafStart = VXPG_LeafStartIndex();
    const uint2 leafRange = clusterRange[clusterIndex];
    const uint lo = leafRange.x;
    const uint hi = leafRange.y;
    if (hi < lo)
    {
        // Empty cluster (no lit voxel was assigned to it): no root, no weight.
        clusterRoots[clusterIndex] = -1;
        return;
    }

    uint node = 0;
    for (int iter = 0; iter < 64; ++iter)
    {
        if (node >= leafStart)
            break;

        const uint left = treeNodes[node].leftIdx;
        const uint right = treeNodes[node].rightIdx;
        if (left >= VXPG_NodeCount() || right >= VXPG_NodeCount())
            break;
        const uint2 leftRange = VXPG_NodeRange(left);
        const uint2 rightRange = VXPG_NodeRange(right);

        if (leftRange.x <= lo && hi <= leftRange.y)
            node = left;
        else if (rightRange.x <= lo && hi <= rightRange.y)
            node = right;
        else
            break;
    }

    if (node >= VXPG_NodeCount())
    {
        clusterRoots[clusterIndex] = -1;
        return;
    }
    const uint2 range = VXPG_NodeRange(node);
    clusterRoots[clusterIndex] = (range.x <= lo && hi <= range.y) ? int(node) : -1;
}

//------------------------------------------------------------------------------------------
// 7) Superpixel centres (reference gSLICr "InitClusterCenter", without the SLIC iterations).
//------------------------------------------------------------------------------------------
int2 VXPG_SpixelGridSize()
{
    int2 res = int2(pushConst.screenSize);
    int s = max(pushConst.spixelSize, 1);
    return (res + int2(s - 1, s - 1)) / s;
}

//------------------------------------------------------------------------------------------
// 9) Superpixel segmentation, following the SLIC algorithm (Achanta et al. 2012, Algorithm 1),
//    which is what VXPG uses to build the shading point slabs:
//      init     : grid centres, moved to the lowest gradient position in a 3x3 neighbourhood
//      assign   : 2S x 2S search (3x3 cells), SLIC distance, accumulate cluster sums
//      update   : centre = mean of the assigned pixels
//    assign/update are iterated a few times (the paper iterates to convergence; two iterations
//    already capture most of the boundary adherence at a fraction of the cost).
//------------------------------------------------------------------------------------------
float VXPG_SpixelGradient(int2 p, int2 res)
{
    float g = 0.0;
    for (int dir = 0; dir < 2; ++dir)
    {
        const int2 axis = (dir == 0) ? int2(1, 0) : int2(0, 1);
        const int2 a = clamp(p + axis, int2(0, 0), res - int2(1, 1));
        const int2 c = clamp(p - axis, int2(0, 0), res - int2(1, 1));
        const VXPGPixelData pa = pixelData[a.y * res.x + a.x];
        const VXPGPixelData pc = pixelData[c.y * res.x + c.x];
        if (pa.info.w == 0u || pc.info.w == 0u)
            continue;
        g += length(pa.position.xyz - pc.position.xyz);
        g += 1.0 - clamp(dot(normalize(pa.normalMaterial.xyz), normalize(pc.normalMaterial.xyz)), -1.0, 1.0);
    }
    return g;
}

[numthreads(64, 1, 1)]
[shader("compute")]
void vxpgSpixelCentersMain(uint2 threadIndex : SV_DispatchThreadID)
{
    const int2 grid = VXPG_SpixelGridSize();
    if (any(int2(threadIndex) >= grid)) return;

    const int2 res = int2(pushConst.screenSize);
    const int s = max(pushConst.spixelSize, 1);
    const int2 base = int2(threadIndex) * s;
    int2 center = min(base + int2(s / 2, s / 2), res - int2(1, 1));

    // SLIC initialisation: move the centre to the lowest gradient position in its 3x3 window.
    float bestGradient = 1e30;
    int2 best = center;
    for (int dy = -1; dy <= 1; ++dy)
    {
        for (int dx = -1; dx <= 1; ++dx)
        {
            const int2 p = clamp(center + int2(dx, dy), int2(0, 0), res - int2(1, 1));
            if (pixelData[p.y * res.x + p.x].info.w == 0u) continue;
            const float g = VXPG_SpixelGradient(p, res);
            if (g < bestGradient){
                bestGradient = g;
                best = p;
            }
        }
    }

    const int id = int(threadIndex.y) * grid.x + int(threadIndex.x);
    VXPGSpixelInfo info;
    info.pos = float4(0.0);
    info.normal = float4(0.0);
    info.center = float2(best);
    info.noPixels = 0;
    info.id = id;

    VXPGPixelData pd = pixelData[best.y * res.x + best.x];
    int2 chosen = best;
    if (pd.info.w == 0u)
    {
        // background centre: use the first valid pixel inside the cell (coarse scan)
        [loop]
        for (int yy = base.y; yy < min(base.y + s, res.y); yy += 4)
        {
            [loop]
            for (int xx = base.x; xx < min(base.x + s, res.x); xx += 4)
            {
                const VXPGPixelData candidate = pixelData[yy * res.x + xx];
                if (candidate.info.w != 0u)
                {
                    pd = candidate;
                    chosen = int2(xx, yy);
                    break;
                }
            }
            if (pd.info.w != 0u)
                break;
        }
    }
    if (pd.info.w != 0u)
    {
        info.center = float2(chosen);
        info.pos = float4(pd.position.xyz, 1.0);          // w = sample count
        info.normal = float4(normalize(pd.normalMaterial.xyz), 0.0);
        info.noPixels = 1;
    }
    spixelInfo[id] = info;
}

// The 3x3 candidate superpixels are the same for every thread of a workgroup (a 16x16 group fits
// inside one cell whenever the superpixel size is a multiple of 16), so they are staged once per
// group in shared memory. That removes the 9 global loads of the 48 byte VXPGSpixelInfo per pixel
// which dominated the Nsight capture (LSU global load 81% + LG throttle 40%).
groupshared VXPGSpixelInfo sCandidates[9];

[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgSpixelAssignMain(uint2 threadIndex : SV_DispatchThreadID, uint2 groupIndex : SV_GroupID,
                          uint groupThreadIndex : SV_GroupIndex)
{
    const int2 grid = VXPG_SpixelGridSize();
    const int s = max(pushConst.spixelSize, 1);
    const int2 groupOrigin = int2(groupIndex) * int2(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y);
    const int2 groupCell = clamp(groupOrigin / s, int2(0, 0), grid - int2(1, 1));

    // Stage the 3x3 neighbourhood of the group's cell (nine threads, one each).
    if (groupThreadIndex < 9u)
    {
        const int2 offset = int2(int(groupThreadIndex % 3u) - 1, int(groupThreadIndex / 3u) - 1);
        const int2 tile = clamp(groupCell + offset, int2(0, 0), grid - int2(1, 1));
        sCandidates[groupThreadIndex] = spixelInfo[tile.y * grid.x + tile.x];
    }
    GroupMemoryBarrierWithGroupSync();

    if (any(threadIndex >= pushConst.screenSize))
        return;

    const uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    VXPGPixelData pd = pixelData[index];
    if (pd.info.w == 0u)
        return;

    const int2 cell = clamp(int2(threadIndex) / s, int2(0, 0), grid - int2(1, 1));
    const float3 pixPos = pd.position.xyz;
    const float3 pixNormal = normalize(pd.normalMaterial.xyz);
    const float2 pixXY = float2(threadIndex);

    // The staged candidates are only valid for pixels inside the group's cell; the (rare) pixels of
    // a group that straddles a cell boundary fall back to the global buffer.
    const bool useStage = all(cell == groupCell);

    float best = 1e30;
    int bestId = -1;
    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            const int2 tile = clamp(cell + int2(i, j), int2(0, 0), grid - int2(1, 1));
            const int id = tile.y * grid.x + tile.x;
            const VXPGSpixelInfo info = useStage ? sCandidates[uint((j + 1) * 3 + (i + 1))] : spixelInfo[id];
            if (info.pos.w <= 0.0)
                continue;
            const float distance = VXPG_SpixelDistance(pixPos, pixNormal, pixXY, info);
            if (distance < best)
            {
                best = distance;
                bestId = id;
            }
        }
    }

    if (bestId < 0)
    {
        pd.info.x = VXPG_INVALID_INDEX;
        pixelData[index] = pd;
        return;
    }

    // accumulate into the dedicated SLIC sums (float atomics, feature is enabled)
    InterlockedAdd(spixelAccum[bestId].posSum.x, pixPos.x);
    InterlockedAdd(spixelAccum[bestId].posSum.y, pixPos.y);
    InterlockedAdd(spixelAccum[bestId].posSum.z, pixPos.z);
    InterlockedAdd(spixelAccum[bestId].posSum.w, 1.0);
    InterlockedAdd(spixelAccum[bestId].normalSum.x, pixNormal.x);
    InterlockedAdd(spixelAccum[bestId].normalSum.y, pixNormal.y);
    InterlockedAdd(spixelAccum[bestId].normalSum.z, pixNormal.z);
    InterlockedAdd(spixelAccum[bestId].xySum.x, pixXY.x);
    InterlockedAdd(spixelAccum[bestId].xySum.y, pixXY.y);

    pd.info.x = uint(bestId);
    pixelData[index] = pd;
}

// Centre = mean of the assigned pixels; the sums are cleared for the next iteration.
[numthreads(64, 1, 1)]
[shader("compute")]
void vxpgSpixelUpdateMain(uint2 threadIndex : SV_DispatchThreadID)
{
    const int2 grid = VXPG_SpixelGridSize();
    if (any(int2(threadIndex) >= grid))
        return;

    const int id = int(threadIndex.y) * grid.x + int(threadIndex.x);
    VXPGSpixelAccum acc = spixelAccum[id];
    VXPGSpixelInfo info = spixelInfo[id];
    const int count = int(acc.posSum.w);
    if (count > 0)
    {
        const float invCount = 1.0 / float(count);
        info.pos = float4(acc.posSum.xyz * invCount, 1.0);
        info.normal = float4(normalize(acc.normalSum.xyz), 0.0);
        info.center = acc.xySum * invCount;
        info.noPixels = count;
    }
    else
    {
        // Keep the previous centre: an emptied cluster must be able to attract pixels again on the
        // next iteration (zeroing it made it permanently dead).
        info.noPixels = max(info.noPixels, 0);
    }
    spixelInfo[id] = info;

    // reset the sums for the next iteration
    acc.posSum = float4(0.0);
    acc.normalSum = float4(0.0);
    acc.xySum = float2(0.0);
    spixelAccum[id] = acc;
}


//------------------------------------------------------------------------------------------
// 9) Per superpixel cluster importance: every pixel adds the power of the VPL it found to the
//    cluster of that VPL. This is the visibility aware replacement of the reference's
//    stochastic "SPixelAvgVisibility" pass (see README.md).
//------------------------------------------------------------------------------------------
[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgSpixelAccumMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    VXPGPixelData pd = pixelData[index];
    if (pd.info.w == 0u || pd.info.x == VXPG_INVALID_INDEX || pd.info.z == VXPG_INVALID_INDEX)
        return;

    int spixel = int(pd.info.x);
    if (spixel < 0 || spixel >= pushConst.spixelCount)
        return;

    uint flat = pd.info.z;
    float importance = VXPG_VoxelImportance(flat);
    if (!(importance > 0.0))
        return;

    // Supervoxel weight of the paper: average throughput times the total power of the cluster.
    // Here the throughput is the luminance of this pixel's VPL contribution, so the accumulated
    // cluster weight is sum(throughput * Phi) over the samples of this superpixel, i.e. the
    // contribution aware term (visibility included, because the VPL came from this very path).
    const float throughput = max(pd.vplPos.w, 0.0);

    // The cluster id must match the one the light tree uses: the visibility clustering (KMPP)
    // when it is enabled, the Morton chunk otherwise. Mixing the two made the top level
    // distribution point at clusters whose roots belong to a different voxel group.
    uint cluster;
    if ((pushConst.flags & 512) != 0)
    {
        cluster = voxelCluster[flat];
        if (cluster == VXPG_INVALID_INDEX)
            cluster = VXPG_ClusterOfVoxel(VXPG_ReconstructIndex(flat));
    }
    else
    {
        cluster = VXPG_ClusterOfVoxel(VXPG_ReconstructIndex(flat));
    }
    cluster = min(cluster, uint(VXPG_CLUSTER_COUNT) - 1u);
    // Eq. (7) made the weight area dependent; dividing by the reference (full voxel) face area
    // keeps the fixed point accumulation in the same numeric range as before. The ratio is a
    // constant of the frame, so the relative cluster weights are unchanged by it. The clamped
    // value keeps a single firefly sample from saturating the int32 accumulator.
    float weight = importance / VXPG_VoxelReferenceArea() * throughput;
    int fixed = int(clamp(weight * 100.0, 1.0, 1.0e6));

    InterlockedAdd(spixelClusterWeight[spixel * VXPG_CLUSTER_COUNT + int(cluster)], fixed);
    InterlockedAdd(spixelClusterTotal[spixel], fixed);
}

//------------------------------------------------------------------------------------------
// 10) Per superpixel top level tree (32 leaves, same 64 float heap layout as the reference).
//     Regions without any VPL fall back to the global cluster importance.
//------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
[shader("compute")]
void vxpgTopLevelMain(uint threadIndex : SV_DispatchThreadID)
{
    if (int(threadIndex) >= pushConst.spixelCount)
        return;

    int spixel = int(threadIndex);
    int offset = spixel * VXPG_TOP_LEVEL_STRIDE;

    float local[VXPG_TOP_LEVEL_STRIDE];
    for (int i = 0; i < VXPG_TOP_LEVEL_STRIDE; ++i)
        local[i] = 0.0;

    // The leaf weight of a cluster is the power that the pixels of this region actually found
    // in it (the visibility aware term of the reference), plus a small fraction of the global
    // cluster importance. The global term guarantees that a region never ends up with an
    // empty distribution (e.g. a region that only sees emissive panels), while the local term
    // dominates whenever the region contributed anything.
    for (int c = 0; c < VXPG_CLUSTER_COUNT; ++c)
    {
        float localValue = max(float(spixelClusterWeight[spixel * VXPG_CLUSTER_COUNT + c]) / 1000.0, 0.0);
        int root = clusterRoots[c];
        // The cluster intensity now carries the Eq. (7) area factor; normalise it the same way as
        // the local term above so the 1e-4 fallback weight keeps its previous meaning.
        float globalValue = (root >= 0) ? max(treeNodes[uint(root)].intensity, 0.0) : 0.0;
        globalValue /= VXPG_VoxelReferenceArea();
        local[VXPG_TOP_LEVEL_LEAF_OFFSET + c] = localValue + globalValue * 1e-4;
    }

    // Bottom up sums of the complete binary heap: the 32 leaves live at indices
    // VXPG_TOP_LEVEL_LEAF_OFFSET .. +31, so the first level to reduce is 31.
    for (int i = VXPG_TOP_LEVEL_LEAF_OFFSET - 1; i >= 1; --i)
        local[i] = local[i * 2] + local[i * 2 + 1];

    for (int i = 0; i < VXPG_TOP_LEVEL_STRIDE; ++i)
        topLevelTree[offset + i] = local[i];
}

//------------------------------------------------------------------------------------------
// Leaf order indirection: identity mapping + Morton contiguous cluster ranges. Step 1 of the
// visibility clustering work; everything downstream (light tree init, cluster roots, guide
// sampling, pdf) reads these buffers, so exchanging the identity order for a KMPP order later
// does not touch the samplers. Separate pass because the tree init reads what this writes.
//------------------------------------------------------------------------------------------
[numthreads(64, 1, 1)]
[shader("compute")]
void vxpgLeafMapInitMain(uint threadIndex : SV_DispatchThreadID)
{
    const uint leafCount = VXPG_LeafCount();
    if (threadIndex < leafCount)
    {
        const uint flatId = VXPG_FlatIndex(VXPG_SlotToVoxel(threadIndex));
        leafVoxel[threadIndex] = flatId;
        voxelLeaf[flatId] = threadIndex;
    }
    if (threadIndex < uint(VXPG_CLUSTER_COUNT))
    {
        const uint perCluster = max(1u, leafCount / uint(VXPG_CLUSTER_COUNT));
        const uint lo = threadIndex * perCluster;
        clusterRange[threadIndex] = uint2(lo, min(lo + perCluster - 1u, leafCount - 1u));
    }
}

//------------------------------------------------------------------------------------------
// Visibility clustering (paper Section 5.2, MRCS): 128 stratified jittered representative
// shading points, one 128 bit visibility signature per lit voxel, KMPP clustering into
// VXPG_CLUSTER_COUNT clusters, and a cluster sorted leaf order so that every cluster is a
// contiguous leaf range (which keeps the existing cluster root / pdf code valid).
//------------------------------------------------------------------------------------------
[numthreads(128, 1, 1)]
[shader("compute")]
void vxpgRepSPMain(uint threadIndex : SV_DispatchThreadID)
{
    if (threadIndex >= uint(VXPG_REP_SP_COUNT)) return;

    const uint2 cells = uint2(16, 8);
    const uint2 cell = uint2(threadIndex % cells.x, threadIndex / cells.x);
    uint seed = xxhash32(uint3(threadIndex, uint(pushConst.randomFrameIndex), 0x5eed1234u));
    const float2 cellSize = float2(pushConst.screenSize) / float2(cells);
    const float2 pos = cellSize * (float2(cell) + float2(rand(seed), rand(seed)));
    const uint2 px = min(uint2(pos), max(pushConst.screenSize - uint2(1, 1), uint2(0, 0)));

    VXPGPixelData pd = pixelData[px.y * pushConst.screenSize.x + px.x];
    repSP[threadIndex] = float4(pd.position.xyz, (pd.info.w != 0u) ? 1.0 : 0.0);
}

groupshared uint groupLitVoxelCount;
groupshared uint groupLitVoxelOffset;
[numthreads(1024, 1, 1)]
[shader("compute")]
void vxpgLitVoxelSortMain(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupLitVoxelCount = 0u;
    GroupMemoryBarrierWithGroupSync();

    bool lit = threadIndex >= VXPG_VoxelCount() ? false : voxelVplCount[threadIndex] > 0u;
    uint listIndex = 0u;
    {
        uint warpLitVoxelCount = WaveActiveCountBits(lit);
        uint warpLitVoxelOffset = 0u;
        if ((groupThreadIndex & 31) == 0) InterlockedAdd(groupLitVoxelCount, warpLitVoxelCount, warpLitVoxelOffset);
        GroupMemoryBarrierWithGroupSync();
        if (groupThreadIndex == 0) InterlockedAdd(litVoxelCount[0], groupLitVoxelCount, groupLitVoxelOffset);
        GroupMemoryBarrierWithGroupSync();

        listIndex = WavePrefixCountBits(lit);
        listIndex += WaveReadLaneFirst(warpLitVoxelOffset);
        listIndex += groupLitVoxelOffset;
    }
    if (listIndex < uint(VXPG_KMPP_MAX_VOXELS) && lit) litVoxelList[listIndex] = threadIndex;
}

groupshared uint groupWords[VXPG_SIGNATURE_WORDS];
[numthreads(128, 1, 1)]
[shader("compute")]
void vxpgSignatureMain(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < 4) groupWords[groupThreadIndex] = 0u;
    GroupMemoryBarrierWithGroupSync();

    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    if (threadIndex == 0) {
        globalInfo[0].cmd.x = 1;   // the mean stage runs as a single 1024 thread group
        globalInfo[0].cmd.y = 1; globalInfo[0].cmd.z = 1;
    }
    if (threadGroupIndex >= litCount) return;      // whole group leaves together, no barrier after

    // threadGroupIndex is the index of the lit voxel *in the compact list*, not a flat voxel index
    const uint voxelFlat = litVoxelList[threadGroupIndex];
    const int3 voxel = VXPG_ReconstructIndex(voxelFlat);
    const float3 center = 0.5 * (VXPG_VoxelMin(voxel) + VXPG_VoxelMax(voxel));

    // One thread per representative shading point; rays are traced inside the branch so that no
    // thread returns before the group barrier below (a divergent return would deadlock the group).
    const int repSpIndex = groupThreadIndex;
    const float4 sp = repSP[repSpIndex];
    if (sp.w != 0.0)
    {
        const float3 toVoxel = center - sp.xyz;
        const float dist = length(toVoxel);
        if (dist > 1e-4)
        {
            RayDesc ray;
            ray.Direction = toVoxel / dist;
            ray.Origin = sp.xyz + ray.Direction * 1e-3;
            ray.TMin = 0.0;
            ray.TMax = dist * 0.999;
            if (!VXPG_TraceRay(ray).hit)
                groupWords[repSpIndex >> 5u] |= (1u << (repSpIndex & 31u));
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0)
        voxelSignature[voxelFlat] = uint4(groupWords[0], groupWords[1], groupWords[2], groupWords[3]);


}

groupshared float3 groupMean;
groupshared float3 warpMeans[32];
[numthreads(1024, 1, 1)]
[shader("compute")]
void vxpgClusterSeedMain_1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID){
    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    if (litCount == 0u) {
        for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c) clusterCentroid[c] = uint4(0u, 0u, 0u, 0u);
        return;
    }
    if (groupThreadIndex == 0) groupMean = float3(0.0);
    GroupMemoryBarrierWithGroupSync();

    // Sum the lit voxel centres. A single 1024 thread group loops over the list (at most
    // VXPG_KMPP_MAX_VOXELS / 1024 iterations per thread), so no (unsupported) floating point
    // atomics are needed: the reduction is a wave sum plus a groupshared warp table.
    float3 local = float3(0.0);
    for (uint i = threadIndex; i < litCount; i += 1024u) {
        const int3 v = VXPG_ReconstructIndex(litVoxelList[i]);
        local += 0.5 * (VXPG_VoxelMin(v) + VXPG_VoxelMax(v));
    }
    local = WaveActiveSum(local);
    if ((groupThreadIndex & 31) == 0)
        warpMeans[groupThreadIndex >> 5] = local;
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) {
        float3 total = float3(0.0);
        for (uint w = 0; w < 32u; ++w)
            total += warpMeans[w];
        globalInfo[0].globalMean = total;
        globalInfo[0].cmd.x = 1;   // the seed selection stage runs as a single 32 thread group
        globalInfo[0].cmd.y = 1;
        globalInfo[0].cmd.z = 1;
    }
}

// Supplemental Eq. (4): distR(p, q) = countbits(R_p ^ R_q) + w_E * |E_p - E_q|. The irradiance
// term keeps the clustering informative when the signatures are degenerate (every lit voxel is
// invisible from the 128 representative screen points, so every XOR is 0): the voxels still differ
// in brightness.
float VXPG_ClusterDistance(uint4 sigA, float eA, uint4 sigB, float eB)
{
    return float(VXPG_HammingWeight(sigA, sigB)) + max(pushConst.kmppWeightE, 0.0f) * abs(eA - eB);
}

uint VXPG_HammingWeight(uint4 a, uint4 b)
{
    return countbits(a.x ^ b.x) + countbits(a.y ^ b.y) + countbits(a.z ^ b.z) + countbits(a.w ^ b.w);
}

groupshared uint4 groupCentroid[VXPG_CLUSTER_COUNT];
groupshared float groupCentroidE[VXPG_CLUSTER_COUNT];
[numthreads(32, 1, 1)]
[shader("compute")]
void vxpgClusterSeedMain_2(uint threadIndex : SV_DispatchThreadID)
{
    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    if (litCount == 0u){
        for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c) clusterCentroid[c] = uint4(0u, 0u, 0u, 0u);
        return;
    }

    // First seed: the lit voxel closest to the mean position (deterministic starting point).
    float3 mean = globalInfo[0].globalMean;
    mean /= float(litCount);

    float best = 1e30;
    uint firstVoxel = litVoxelList[0];
    for (uint i = threadIndex; i < litCount; i += 32)
    {
        const int3 v = VXPG_ReconstructIndex(litVoxelList[i]);
        const float d = distance(0.5 * (VXPG_VoxelMin(v) + VXPG_VoxelMax(v)), mean);
        if (d < best) { best = d; firstVoxel = litVoxelList[i]; }
    }
    float warpBest = WaveActiveMin(best);
    if (best == warpBest && WavePrefixCountBits(best == warpBest) == 0)
    {
        groupCentroid[0]  = voxelSignature[firstVoxel];
        groupCentroidE[0] = VXPG_VoxelIrradiance(firstVoxel);
    }
    GroupMemoryBarrierWithWaveSync();

    // Remaining seeds: K-means++ style, farthest signature from the chosen centroids.
    for (uint c = 1; c < uint(VXPG_CLUSTER_COUNT); ++c)
    {
        float bestD = -1.0;
        uint bestVoxel = firstVoxel;
        for (uint i = threadIndex; i < litCount; i += 32)
        {
            const uint voxel = litVoxelList[i];
            const uint4 sig = voxelSignature[voxel];
            const float irradiance = VXPG_VoxelIrradiance(voxel);
            float nearest = 1e30;
            for (uint j = 0; j < c; ++j)
                nearest = min(nearest, VXPG_ClusterDistance(sig, irradiance, groupCentroid[j], groupCentroidE[j]));
            // K-means++: score by the squared distance to the nearest centroid. The irradiance term
            // is part of the distance now (Eq. 4), so it is no longer multiplied in separately.
            const float score = nearest * nearest;
            if (score > bestD) { bestD = score; bestVoxel = voxel; }
        }
        float warpBestD = WaveActiveMax(bestD);
        if (bestD == warpBestD && WavePrefixCountBits(bestD == warpBestD) == 0)
        {
            groupCentroid[c]  = voxelSignature[bestVoxel];
            groupCentroidE[c] = VXPG_VoxelIrradiance(bestVoxel);
        }
        GroupMemoryBarrierWithWaveSync();
    }

    for (int i = threadIndex; i < VXPG_CLUSTER_COUNT; i += 32)
    {
        clusterCentroid[i]  = groupCentroid[i];
        clusterCentroidE[i] = groupCentroidE[i];
    }
}

[numthreads(512, 1, 1)]
[shader("compute")]
void vxpgClusterAssignMain(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID)
{
    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    int litVoxelIndex = threadIndex / 32;
    if (litVoxelIndex >= litCount) return;

    const uint voxel = litVoxelList[litVoxelIndex];
    int warpLane = groupThreadIndex & 31;
    uint bestCluster = 0u;
    float bestD = 1e30;
    const uint4 sig = voxelSignature[voxel];
    const float irradiance = VXPG_VoxelIrradiance(voxel);
    for (uint c = warpLane; c < uint(VXPG_CLUSTER_COUNT); c += 32)
    {
        const float d = VXPG_ClusterDistance(sig, irradiance, clusterCentroid[c], clusterCentroidE[c]);
        if (d < bestD) { bestD = d; bestCluster = c; }
    }
    float warpBestD = WaveActiveMin(bestD);
    if (bestD == warpBestD && WavePrefixCountBits(bestD == warpBestD) == 0) voxelCluster[voxel] = bestCluster;
}

[numthreads(VXPG_CLUSTER_COUNT, 1, 1)]
[shader("compute")]
void vxpgClusterUpdateMain(uint threadIndex : SV_DispatchThreadID)
{
    if (threadIndex >= uint(VXPG_CLUSTER_COUNT)) return;

    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    uint ones[VXPG_REP_SP_COUNT];
    for (uint k = 0; k < uint(VXPG_REP_SP_COUNT); ++k) ones[k] = 0u;

    uint members = 0u;
    float sumIrradiance = 0.0f;
    for (uint i = 0; i < litCount; ++i)
    {
        const uint voxel = litVoxelList[i];
        if (voxelCluster[voxel] != threadIndex) continue;
        ++members;
        sumIrradiance += VXPG_VoxelIrradiance(voxel);
        const uint4 sig = voxelSignature[voxel];
        for (uint k = 0; k < uint(VXPG_REP_SP_COUNT); ++k) ones[k] += (sig[k >> 5u] >> (k & 31u)) & 1u;
    }
    if (members == 0u)  return;   // keep the seeded centroid for an empty cluster

    uint words[VXPG_SIGNATURE_WORDS] = { 0u, 0u, 0u, 0u };
    for (uint k = 0; k < uint(VXPG_REP_SP_COUNT); ++k)
        if (ones[k] * 2u > members) words[k >> 5u] |= (1u << (k & 31u));
    clusterCentroid[threadIndex]  = uint4(words[0], words[1], words[2], words[3]);
    clusterCentroidE[threadIndex] = sumIrradiance / float(members);
}

[numthreads(1, 1, 1)]
[shader("compute")]
void vxpgClusterSortMain(uint threadIndex : SV_DispatchThreadID)
{
    const uint litCount = min(litVoxelCount[0], uint(VXPG_KMPP_MAX_VOXELS));
    const uint leafCount = VXPG_LeafCount();
    // Each cluster lives in its own aligned power of two block of leaves, because the light tree is
    // a radix tree over the leaf index: an aligned block is a subtree, so the lowest common
    // ancestor of a cluster never covers leaves of another cluster. With arbitrary ranges the
    // cluster root would include foreign leaves, I_root would not match the cluster and p_leaf
    // would stop summing to one (that made the guided estimate too dark).
    const uint perBlock = max(1u, leafCount / uint(VXPG_CLUSTER_COUNT));

    uint counts[VXPG_CLUSTER_COUNT];
    for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c) counts[c] = 0u;
    for (uint i = 0; i < litCount; ++i){
        const uint cl = min(voxelCluster[litVoxelList[i]], uint(VXPG_CLUSTER_COUNT) - 1u);
        ++counts[cl];
    }

    uint nonEmpty = 0u;
    uint largest = 0u;
    for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c)
    {
        if (counts[c] > 0u)  ++nonEmpty;
        largest = max(largest, counts[c]);
    }

    // A KMPP run on nearly identical signatures produces a wildly unbalanced partition (almost all
    // lit voxels in one cluster). The top level distribution then collapses onto a single cluster
    // and the guided estimate degenerates, so such a partition is rejected in favour of the
    // spatial one; the clustering only takes over when it actually separates the voxels.
    // Relaxed test: only reject the partition when it carries no information at all (a single non
    // empty cluster) or when a cluster no longer fits its aligned block (perBlock leaves). A merely
    // skewed partition (e.g. one cluster with 60% of the voxels and 31 small ones) is still useful:
    // every cluster keeps a non zero weight in the top level distribution, and rejecting it would
    // throw the clustering away and pay for the single threaded leafCount iteration Morton rewrite.
    const bool unbalanced = (nonEmpty <= 1u) || (largest > perBlock);

    //if (threadIndex == 0) printf("nonEmpty: %d, unbalanced: %d£¬ largest: %d, litCount: %d\n", nonEmpty, unbalanced, largest, litCount);
    if (nonEmpty <= 1u || unbalanced){
        // Degenerate signatures (every representative point sees the same lit set): the clustering
        // carries no information, so fall back to the Morton partition, which is byte for byte the
        // behaviour without visibility clustering.
        const uint perCluster = max(1u, leafCount / uint(VXPG_CLUSTER_COUNT));
        for (uint j = 0; j < leafCount; ++j)
        {
            const uint flatId = VXPG_FlatIndex(VXPG_SlotToVoxel(j));
            leafVoxel[j] = flatId;
            voxelLeaf[flatId] = j;
        }
        for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c)
        {
            const uint lo = c * perCluster;
            clusterRange[c] = uint2(lo, min(lo + perCluster - 1u, leafCount - 1u));
        }
        return;
    }

    uint cursor[VXPG_CLUSTER_COUNT];
    for (uint c = 0; c < uint(VXPG_CLUSTER_COUNT); ++c)
    {
        const uint blockStart = min(c * perBlock, leafCount - 1u);
        const uint capacity = min(perBlock, leafCount - blockStart);
        const uint used = min(counts[c], capacity);
        cursor[c] = blockStart;
        clusterRange[c] = (used > 0u) ? uint2(blockStart, blockStart + used - 1u) : uint2(1u, 0u);
    }

    for (uint i = 0; i < litCount; ++i)
    {
        const uint voxel = litVoxelList[i];
        const uint cl = min(voxelCluster[voxel], uint(VXPG_CLUSTER_COUNT) - 1u);
        const uint blockStart = min(cl * perBlock, leafCount - 1u);
        const uint capacity = min(perBlock, leafCount - blockStart);
        const uint blockEnd = blockStart + capacity;
        if (cursor[cl] >= blockEnd) continue;   // block full: the voxel stays without a leaf (bounce falls back to BSDF)
        const uint slot = cursor[cl]++;
        leafVoxel[slot] = voxel;
        voxelLeaf[voxel] = slot;
    }
}

//------------------------------------------------------------------------------------------
// PDF consistency test (VXPG_PDFTEST=1 -> flags bit 4096): is 1/p_sph really the solid angle of
// the patch that the sampler targets? For a fixed lit voxel and a fixed shading point we
//   * measure the solid angle of the patch empirically (uniform sphere sampling + ray/AABB),
//   * ask the sampler for its pdf,
// and write both into pixel 0's record so the dump can print them.
//------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void vxpgPdfTestMain(uint threadIndex : SV_DispatchThreadID)
{
    // The CPU dump reads pixelData from row 360 onwards, so the probe result has to be stored in
    // the record of that row (pixel 0 of the dump window).
    const uint probeIndex = pushConst.screenSize.x * 360u;

    // sentinel so the dump can tell "ran" from "did not run"
    {
        VXPGPixelData stamp = pixelData[probeIndex];
        stamp.vplPos = float4(-1.0, -1.0, -1.0, -1.0);
        pixelData[probeIndex] = stamp;
    }

    // first lit voxel (prefer one that already has written surface bounds)
    uint flat = VXPG_INVALID_INDEX;
    uint litCount = 0u;
    uint boundedCount = 0u;
    for (uint f = 0; f < VXPG_VoxelCount(); ++f)
    {
        if (voxelVplCount[f] == 0u)
            continue;
        ++litCount;
        const bool bounded = (voxelBoundMax[f * 4 + 0] | voxelBoundMax[f * 4 + 1] | voxelBoundMax[f * 4 + 2]) != 0u;
        if (bounded) ++boundedCount;
        if (bounded && flat == VXPG_INVALID_INDEX)
            flat = f;
        else if (flat == VXPG_INVALID_INDEX)
            flat = f;
    }
    if (flat == VXPG_INVALID_INDEX)
    {
        VXPGPixelData none = pixelData[probeIndex];
        none.vplPos = float4(-2.0, -2.0, float(litCount), float(boundedCount));
        pixelData[probeIndex] = none;
        return;
    }

    int3 voxel = VXPG_ReconstructIndex(flat);
    float3 bmin, bmax;
    VXPG_VoxelSurfaceBounds(voxel, bmin, bmax);
    float3 center = 0.5 * (bmin + bmax);
    float3 extend = max(0.5 * (bmax - bmin), float3(1e-6));

    // fixed shading point at a realistic distance from the patch
    const float3 sp = center + float3(0.6, 0.35, 2.2);

    uint seed = 0x1234u;
    float sphPdf = 0.0;
    VXPG_SampleSphericalVoxel(center, extend, sp, float3(rand(seed), rand(seed), rand(seed)), sphPdf);

    // empirical solid angle of the patch: uniform sphere sampling, ray vs patch AABB
    const uint M = 262144u;   // 256k directions: ~60 hits for a 1e-3 sr patch
    uint hits = 0u;
    for (uint i = 0; i < M; ++i)
    {
        float z = 1.0 - 2.0 * rand(seed);
        float r = sqrt(max(1.0 - z * z, 0.0));
        float phi = 6.28318530718 * rand(seed);
        float3 dir = float3(r * cos(phi), r * sin(phi), z);

        float tmin = 0.0;
        float tmax = 1e30;
        bool miss = false;
        for (int axis = 0; axis < 3; ++axis)
        {
            float d = dir[axis];
            float o = sp[axis];
            float lo = bmin[axis];
            float hi = bmax[axis];
            if (abs(d) < 1e-12)
            {
                if (o < lo || o > hi) { miss = true; break; }
            }
            else
            {
                float t1 = (lo - o) / d;
                float t2 = (hi - o) / d;
                if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
                tmin = max(tmin, t1);
                tmax = min(tmax, t2);
                if (tmin > tmax) { miss = true; break; }
            }
        }
        if (!miss && tmax > 0.0)
            ++hits;
    }

    const float omegaEmpirical = 12.5663706144 * float(hits) / float(M);
    VXPGPixelData rec = pixelData[probeIndex];
    rec.vplPos = float4(omegaEmpirical, (sphPdf > 0.0) ? 1.0 / sphPdf : 0.0,
                        float(hits) / float(M), sphPdf);
    pixelData[probeIndex] = rec;
}

//------------------------------------------------------------------------------------------
// Cluster visualisation (UI: "Show Cluster Colors"). One primary ray per pixel; the first-hit
// voxel is drawn as a square tile coloured by its supervoxel (cluster) id, with a dark border at
// the voxel faces so the voxel grid itself stays readable. Unlit geometry is dark grey and the
// background nearly black. This mirrors computeMain_GetDebugClusterG of the FzbPathGuiding
// renderer (primary ray -> voxel of the hit -> cluster id -> colour).
//------------------------------------------------------------------------------------------
// Colour key that depends on the cluster's *content* instead of its index. The cluster indices are
// permuted whenever the compact lit voxel list comes out in a different order (it is filled with
// atomics), so an index based colour flickers every frame; the centroid signature and irradiance
// are permutation invariant, hence a cluster keeps its colour for a static scene.
uint VXPG_ClusterDebugColorKey(uint cluster)
{
    const uint4 sig = clusterCentroid[cluster];
    uint h = (sig.x * 73856093u) ^ (sig.y * 19349663u) ^ (sig.z * 83492791u) ^ (sig.w * 2654435761u);
    h ^= asuint(clusterCentroidE[cluster]) * 2246822519u;
    h ^= h >> 13u;
    return h;
}

float3 VXPG_ClusterDebugColor(uint cluster)
{
    // Golden ratio hue stepping: 32 ids land far apart on the colour wheel. The slight brightness
    // variation keeps neighbouring ids distinguishable in greyscale as well.
    const float h = frac(float(cluster) * 0.6180339887 + 0.08);
    const float3 k = float3(1.0, 2.0 / 3.0, 1.0 / 3.0);
    const float3 rgb = clamp(abs(frac(h + k) * 6.0 - 3.0) - 1.0, 0.0, 1.0);
    return rgb * (0.75 + 0.25 * frac(float(cluster) * 0.37));
}

//------------------------------------------------------------------------------------------
static const int3 kCubeCorners[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(0, 1, 0), int3(1, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(0, 1, 1), int3(1, 1, 1)
};
// 12 edges of the unit cube, two vertices per edge (LINE_LIST)
static const int kCubeEdges[24] = {
    0, 1, 1, 3, 3, 2, 2, 0,
    4, 5, 5, 7, 7, 6, 6, 4,
    0, 4, 1, 5, 2, 6, 3, 7
};
struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float4 color;
};
struct PSout
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout vxpgDebugClusterMain_vert(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout output;
    output.color = float4(0.30, 0.30, 0.32, 1.0);
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    if (vertexIndex >= 24u || instanceIndex < 0)
        return output;
    if (voxelVplCount[instanceIndex] == 0u) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    const int3 voxel = VXPG_ReconstructIndex(instanceIndex);
    const float3 vmin = float3(VXPG_VoxelMin(voxel));
    const float3 size = float3(VXPG_VoxelMax(voxel)) - vmin;

    // Cluster colour: resolved through the leaf slot so the Metzner fallback (KMPP off) works too.
    const uint slot = VXPG_VoxelLeafSlot(voxel);
    const int  cluster = (slot != VXPG_INVALID_INDEX) ? VXPG_ClusterOfLeafSlot(slot) : -1;
    output.color = (cluster < 0)
        ? float4(0.30, 0.30, 0.32, 1.0)
        : float4(VXPG_ClusterDebugColor(VXPG_ClusterDebugColorKey(uint(cluster)) & 31u), 1.0f);

    // Cube wireframe from the 8 corners; vertexIndex selects the corner through a bit pattern.
    const uint vi = vertexIndex & 7u;
    output.sv_position = float4( vmin + float3((vi & 1u) == 1 ? 1.0 : 0.0, (vi & 2u) == 2 ? 1.0 : 0.0, (vi & 4u) == 4 ? 1.0 : 0.0) * size, 1.0f);
    output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    return output;
}

[shader("pixel")]
PSout vxpgDebugClusterMain_frag(VSout input) {
    return { input.color };
}
//------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void vxpgSuperPixelDebugMain(uint2 threadIndex: SV_DispatchThreadID) {
    if(any(threadIndex >= pushConst.screenSize)) return;

    const uint pixelIndex = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    int superPixelIndex = pixelData[pixelIndex].info.x;
    outImage[threadIndex] = float4(VXPG_ClusterDebugColor(superPixelIndex), 1.0f);
}