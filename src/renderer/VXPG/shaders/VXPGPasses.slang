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
void vxpgVBufferMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

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
    const uint previousSpixel = pixelData[threadIndex.y * pushConst.screenSize.x + threadIndex.x].info.x;

    VXPGPixelData pd;
    pd.position = float4(0.0);
    pd.normalMaterial = float4(0.0);
    pd.vplPos = float4(0.0);
    pd.info = uint4(previousSpixel, VXPG_INVALID_INDEX, VXPG_INVALID_INDEX, 0u);

    if (hit.hit)
    {
        pd.position = float4(hit.position, 1.0);
        pd.normalMaterial = float4(hit.normal, asfloat(uint(hit.materialIndex)));
        pd.info.y = VXPG_FlatIndex(VXPG_WorldToVoxelID(hit.position));
        pd.info.w = 1u;
    }

    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    pixelData[index] = pd;
}

//------------------------------------------------------------------------------------------
// 2) Light injection: place one virtual point light at the 1st indirect vertex of a camera
//    path and accumulate its irradiance into the voxel grid (reference "ImportonInjection").
//------------------------------------------------------------------------------------------
[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgLightInjectMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    uint seed = xxhash32(uint3(threadIndex, uint(pushConst.randomFrameIndex)));

    // The injection pass traces its own primary ray (pixel centre, like the VBuffer pass) so
    // that the payload is built from a real hit record: geometry normal, texture coordinates
    // and material index all come from the same intersection.
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

    VXPGHit primary = VXPG_TraceRay(cameraRay);
    if (!primary.hit)
        return;

    VXPGPixelData pd = pixelData[index];
    if (pd.info.w == 0u)
        return;

    CallablePayload payload;
    VXPG_MakePayload(payload, primary, -cameraRay.Direction, seed, sceneInfo);

    BSDF_SAMPLE(payload);
    float3 bsdfCosine = payload.bsdf * abs(dot(payload.incidence, payload.hitNormal));
    if (!(payload.pdf > 0.0) || all(bsdfCosine <= 0.0))
        return;

    RayDesc ray = VXPG_SpawnRay(primary.position, primary.normal, payload.incidence);
    VXPGHit hit = VXPG_TraceRay(ray);
    if (!hit.hit)
        return;
    // A VPL on a back facing surface carries no physical light on the side the path sees it
    // from, and a VPL far outside the camera volume is either unreachable or would waste the
    // voxel grid (the reference implementation uses a camera centred clipmap for this).
    if (!hit.frontFacing)
        return;
    if (!VXPG_PositionInDomain(hit.position))
        return;

    CallablePayload payload2;
    VXPG_MakePayload(payload2, hit, -payload.incidence, seed, sceneInfo);
    float3 radiance = VXPG_DirectLight(payload2, sceneInfo);

    float irradiance = max(max(radiance.x, radiance.y), radiance.z);
    if (!(irradiance > 0.0))
        return;

    // Jittered injection: the VPL position is jittered by up to half a voxel (SV Injection Jitter
    // in the panel) before it is quantised, so the per voxel importance becomes a smooth function
    // of the position instead of a hard per voxel count.
    const float3 jitter = float3(rand(seed) - 0.5, rand(seed) - 0.5, rand(seed) - 0.5) * pushConst.VGBVoxelSize
                        * pushConst.padding1;
    int3 voxel = VXPG_WorldToVoxelID(hit.position + jitter);
    uint flat = VXPG_FlatIndex(voxel);
    uint packedIrradiance = uint(irradiance * VXPG_IRRADIANCE_SCALE);
    if (packedIrradiance == 0u)
        packedIrradiance = 1u;

    InterlockedAdd(voxelIrradiance[flat], packedIrradiance);
    InterlockedAdd(voxelVplCount[flat], 1u);

    pd.vplPos = float4(hit.position, 0.0);
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
    if (threadIndex >= uint(pushConst.triangleCount))
        return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    if (mesh.dataBuffer == nullptr)
        return;
    const uint vertexCount = mesh.triMesh.positions.count;
    if (vertexCount == 0u || mesh.triMesh.indices.count == 0u)
        return;

    int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, int(threadIndex));
    if (indices.x < 0 || indices.y < 0 || indices.z < 0)
        return;
    if (uint(indices.x) >= vertexCount || uint(indices.y) >= vertexCount || uint(indices.z) >= vertexCount)
        return;

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
        if (any(hi < domainMin) || any(lo > domainMax))
            return;
    }
        {
        const float3 domainMin = pushConst.VGBStartPos;
        const float3 domainMax = domainMin + float3(2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent, 2.0 * pushConst.VGBExtent);
        if (any(hi < domainMin) || any(lo > domainMax))
            return;
    }int3 vlo = VXPG_WorldToVoxelID(lo);
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
        if (voxelVplCount[flat] == 0u)
            continue;

        float3 vmin = VXPG_VoxelMin(voxel);
        float3 vmax = VXPG_VoxelMax(voxel);
        float3 ext = max(vmax - vmin, float3(1e-9));

        float3 bmin = clamp(lo, vmin, vmax);
        float3 bmax = clamp(hi, vmin, vmax);
        float3 qmin = saturate((bmin - vmin) / ext);
        float3 qmax = saturate((bmax - vmin) / ext);
        uint3 qmn = uint3(qmin * 4294967295.0);
        uint3 qmx = uint3(qmax * 4294967295.0);

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
    if (threadIndex < uint(VXPG_CLUSTER_COUNT))
        clusterRoots[threadIndex] = -1;

    uint leafCount = VXPG_LeafCount();
    uint leafStart = VXPG_LeafStartIndex();
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
    int3 voxel = VXPG_SlotToVoxel(slot);
    uint flat = VXPG_FlatIndex(voxel);

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
    if (threadIndex >= leafStart)
        return;

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
    const uint perCluster = max(1u, leafCount / uint(VXPG_CLUSTER_COUNT));
    const uint lo = clusterIndex * perCluster;
    const uint hi = min(lo + perCluster - 1u, leafCount - 1u);

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

[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgSpixelCentersMain(uint2 threadIndex : SV_DispatchThreadID)
{
    int2 grid = VXPG_SpixelGridSize();
    if (any(int2(threadIndex) >= grid))
        return;

    int2 res = int2(pushConst.screenSize);
    int s = pushConst.spixelSize;
    int2 base = int2(threadIndex) * s;
    int2 center = base + int2(s / 2, s / 2);
    center = min(center, res - int2(1, 1));

    VXPGPixelData pd = pixelData[center.y * res.x + center.x];

    // fall back to the first valid pixel inside the tile
    if (pd.info.w == 0u)
    {
        [loop]
        for (int y = base.y; y < min(base.y + s, res.y); y += 4)
        {
            [loop]
            for (int x = base.x; x < min(base.x + s, res.x); x += 4)
            {
                VXPGPixelData candidate = pixelData[y * res.x + x];
                if (candidate.info.w != 0u)
                {
                    pd = candidate;
                    center = int2(x, y);
                    break;
                }
            }
            if (pd.info.w != 0u)
                break;
        }
    }

    VXPGSpixelInfo info;
    info.pos = float4(0.0);
    info.normal = float4(0.0, 0.0, 1.0, 0.0);
    info.center = float2(center);
    info.noPixels = 0;
    info.id = int(threadIndex.y) * grid.x + int(threadIndex.x);
    if (pd.info.w != 0u)
    {
        info.pos = float4(pd.position.xyz, 1.0);
        info.normal = float4(normalize(pd.normalMaterial.xyz), 0.0);
    }
    spixelInfo[info.id] = info;
}

//------------------------------------------------------------------------------------------
// 8) Superpixel association (reference gSLICr "FindCenterAssociation"): SLIC distance in
//    position space with a normal gate, evaluated over the 3x3 neighbouring tiles.
//------------------------------------------------------------------------------------------
[numthreads(VXPG_THREADGROUP_SIZE_X, VXPG_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void vxpgSpixelAssignMain(uint2 threadIndex : SV_DispatchThreadID)
{
    if (any(threadIndex >= pushConst.screenSize))
        return;

    uint index = threadIndex.y * pushConst.screenSize.x + threadIndex.x;
    VXPGPixelData pd = pixelData[index];
    if (pd.info.w == 0u)
        return;

    int2 grid = VXPG_SpixelGridSize();
    int s = pushConst.spixelSize;
    int2 cell = clamp(int2(threadIndex) / s, int2(0, 0), grid - int2(1, 1));

    float3 pixPos = pd.position.xyz;
    float3 pixNormal = normalize(pd.normalMaterial.xyz);
    float2 pixXY = float2(threadIndex);

    float maxXYDistance = 1.0 / (1.4242 * float(s));
    maxXYDistance *= maxXYDistance;
    float maxColorDistance = 5.0 / 1.7321;
    maxColorDistance *= maxColorDistance;

    float best = 1e30;
    int bestId = -1;
    for (int j = -1; j <= 1; ++j)
    {
        for (int i = -1; i <= 1; ++i)
        {
            int2 tile = clamp(cell + int2(i, j), int2(0, 0), grid - int2(1, 1));
            int id = tile.y * grid.x + tile.x;
            VXPGSpixelInfo info = spixelInfo[id];
            if (info.pos.w == 0.0)
                continue;

            float3 delta = pixPos - info.pos.xyz;
            float2 deltaXY = pixXY - info.center;
            float distance = dot(delta, delta) * maxColorDistance + 0.6 * dot(deltaXY, deltaXY) * maxXYDistance;
            if (dot(pixNormal, info.normal.xyz) <= 0.01)
                distance = 1e6;
            if (distance < best)
            {
                best = distance;
                bestId = id;
            }
        }
    }

    if (bestId >= 0)
    {
        pd.info.x = uint(bestId);
        pixelData[index] = pd;
    }
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

    int3 voxel = VXPG_ReconstructIndex(flat);
    uint cluster = min(VXPG_ClusterOfVoxel(voxel), uint(VXPG_CLUSTER_COUNT) - 1u);
    int fixed = max(1, int(importance * 1000.0));

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
        float globalValue = (root >= 0) ? max(treeNodes[uint(root)].intensity, 0.0) : 0.0;
        local[VXPG_TOP_LEVEL_LEAF_OFFSET + c] = localValue + globalValue * 1e-4;
    }

    // Bottom up sums of the complete binary heap: the 32 leaves live at indices
    // VXPG_TOP_LEVEL_LEAF_OFFSET .. +31, so the first level to reduce is 31.
    for (int i = VXPG_TOP_LEVEL_LEAF_OFFSET - 1; i >= 1; --i)
        local[i] = local[i * 2] + local[i * 2 + 1];

    for (int i = 0; i < VXPG_TOP_LEVEL_STRIDE; ++i)
        topLevelTree[offset + i] = local[i];
}
