#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<RISPushConstant_VRPG2> pushConst;
[[vk::binding(RISBindingPoints_VRPG2::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];

[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_SVO)]] RWStructuredBuffer<OctreeNodeInfo_VRPG2> ClusterInfo_SVO;

[[vk::binding(RISBindingPoints_VRPG2::eRISGlobalInfo)]] RWStructuredBuffer<RISGlobalInfo_VRPG2, ScalarDataLayout> risGlobalInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo_VRPG2, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eClusterPairHitInfo)]] RWStructuredBuffer<ClusterPairHitInfo_VRPG2, ScalarDataLayout> clusterPairHitTestInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG2, ScalarDataLayout> candidateSubTreeInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData_VRPG2, ScalarDataLayout> candidateLeafNodeDataBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_VRPG2, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;
//-------------------------------------------------------------------------------------------------------------------------
struct HitPayload_FzbWeight {
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
    int materialIndex;
};
void traceRay(RayDesc ray, inout HitPayload_FzbWeight payload)
{
    // Initialize inline ray query with no special flags
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    // Process all potential intersections along the ray
    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept all intersections as solid
    }

    // Check if ray hit any geometry
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Extract intersection data from the committed hit
        float2 barycentricCoords = q.CommittedTriangleBarycentrics(); // (u,v) coordinates on triangle
        float4x3 worldToObject = q.CommittedWorldToObject4x3();       // Transform matrix
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix
        int instanceIndex = q.CommittedInstanceIndex();               // Instance index in scene

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex];         // Instance data
        // payload.material = sceneInfo.materials[instance.materialIndex]; // Material properties
        payload.materialIndex = instance.materialIndex;

        // Convert built-in barycentric coordinates to full barycentric coordinates
        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        // Process intersection data into world-space shading information
        // HitState hitState = getHitStateWithPositionFetch(barycentrics);
        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));
        payload.hit = 1.0f;

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        // payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        payload.normal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        // Make the geometric normal face the incoming ray. glTF triangle
        // winding is not guaranteed to match the normal partition; this is
        // shading orientation, not a post-hit rejection flip.
        if (dot(payload.normal, ray.Direction) > 0.0f)
            payload.normal = -payload.normal;
    }
    else payload.hit = 0.0f;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_dispatchGetCandidateLeafNodes() {
    GlobalInfoBuffer[0].cmd.x = (risGlobalInfoBuffer[0].clusterPairCount * 8 + 255) / 256;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
struct LightLeafNode {
    AABB aabb;
    float3 radiance;
    float3 normal;

    float pdf;
    int nodeIndex;
};

groupshared uint groupClusterPairCount;

LightLeafNode getLightLeafNode(int warpLane, int layer, uint subTreeNodeIndex, float3 pos, uint randomSeed) {
    LightLeafNode leafNode;
    leafNode.pdf = 1.0f;
    leafNode.nodeIndex = -1;

    LightNodeData_Octree_VRPG node = LightNodeData_Octree[layer][subTreeNodeIndex];
    if (all(node.radiance == 0.0f)) {
        leafNode.aabb = node.aabb;
        leafNode.radiance = node.radiance;
        leafNode.normal = normalize(node.meanNormal.xyz);
        return leafNode;
    }
    if (layer == pushConst.lightOctreeMaxLayer) {
        leafNode.nodeIndex = subTreeNodeIndex;
        leafNode.aabb = node.aabb;
        leafNode.radiance = node.radiance;
        leafNode.normal = normalize(node.meanNormal.xyz);
        return leafNode;
    }

    int nodeIndex = subTreeNodeIndex;
    leafNode.pdf = 1.0f;
    for (; layer < pushConst.lightOctreeMaxLayer; ++layer) {
        node = LightNodeData_Octree[layer + 1][nodeIndex * 8 + (warpLane & 7)];

        float weight = 0.0f;
        if (any(node.radiance > 0.0f)) {
            float M = 1.0f;
            float G = 1.0f;
            float diagonalLength = length(node.aabb.maximum - node.aabb.minimum);
            float3 center = (node.aabb.maximum + node.aabb.minimum) * 0.5f;
            float d = max(length(pos - center), diagonalLength + 1e-3f);
            weight = dot(node.radiance, 1.0f);  //M * G * dot(node.radiance, 1.0f) / (d * d);
        }

        int blockLane = warpLane & 7;
        int blockFirstWarpLane = warpLane & ~7;

        float incl = weight;
        [unroll]
        for (int off = 1; off < 8; off *= 2) {
            float other = WaveReadLaneAt(incl, warpLane - off);
            if (blockLane >= off) incl += other;
        }

        float weightSum = WaveReadLaneAt(incl, blockFirstWarpLane + 7);
        if (weightSum == 0.0f) { leafNode.nodeIndex = -1; return leafNode; }

        float target = WaveReadLaneAt(rand(randomSeed), blockFirstWarpLane) * weightSum;
        uint belowBits = (WaveActiveBallot(incl <= target).x >> blockFirstWarpLane) & 0xFFu;
        int chosen = min((int)countbits(belowBits), 7);
        float chosenWeight = WaveReadLaneAt(weight, blockFirstWarpLane + chosen);

        nodeIndex = nodeIndex * 8 + chosen;
        leafNode.pdf *= chosenWeight / weightSum;
    }

    node = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][nodeIndex];
    leafNode.aabb = node.aabb;
    leafNode.radiance = node.radiance;
    leafNode.normal = normalize(node.meanNormal.xyz);
    leafNode.nodeIndex = nodeIndex;
    return leafNode;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void computeMain_getCandidateLeafNodes(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterPairCount = risGlobalInfoBuffer[0].clusterPairCount;
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    if (threadIndex >= clusterPairCount * 8) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = (groupClusterPairCount * HITTEST_COUNT_VRPG2 + RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE - 1) / RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE;

    int clusterPairIndex = threadIndex / 8;
    ClusterPairInfo_VRPG2 clusterPairInfo = clusterPairInfoBuffer[clusterPairIndex];
    ClusterData_Octree_VRPG nodeData_G = ClusterData_Octree[clusterPairInfo.layerIndex_G][clusterPairInfo.nodeIndex_G];
    float3 centerPos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    uint seed = xxhash32(uint3(clusterPairIndex, groupThreadIndex, (uint)pushConst.randomFrameIndex));
    LightLeafNode nodeData_E = getLightLeafNode(groupThreadIndex & 31, clusterPairInfo.layerIndex_E, clusterPairInfo.nodeIndex_E, centerPos_G, seed);
    if ((groupThreadIndex & 7) == 0) {
        clusterPairInfoBuffer[clusterPairIndex].leafNodeIndex_E = nodeData_E.nodeIndex;
        clusterPairInfoBuffer[clusterPairIndex].leafNodePdf = nodeData_E.pdf;

        CandidateSubTreeInfo_VRPG2 subTreeData;
        int candidateNodeIndex = clusterPairInfo.clusterIndex * Candidate_Samples_Count + clusterPairInfo.candidateNodeIndex;
        subTreeData.subTreeNodeIndex_layer = (clusterPairInfo.nodeIndex_E << 4) | clusterPairInfo.layerIndex_E;
        subTreeData.ESum = dot(LightNodeData_Octree[clusterPairInfo.layerIndex_E][clusterPairInfo.nodeIndex_E].radiance, 1.0f);
        candidateSubTreeInfoBuffer[candidateNodeIndex] = subTreeData;

        CandidateLeafNodeData_VRPG2 leafNodeData;
        leafNodeData.radiance = nodeData_E.radiance;
        leafNodeData.aabb = nodeData_E.aabb;
        candidateLeafNodeDataBuffer[candidateNodeIndex] = leafNodeData;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared ClusterPairInfo_VRPG2 groupClusterPairInfo_VRPG2s[RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_VRPG2];
groupshared ClusterData_Octree_VRPG groupNodeData_G[RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_VRPG2];

[numthreads(RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_clusterPairHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterPairCount = risGlobalInfoBuffer[0].clusterPairCount;
    GroupMemoryBarrierWithGroupSync();

    int clusterPairCount = groupClusterPairCount;
    if (groupThreadIndex < RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_VRPG2) {
        int groupClusterPairStartIndex = threadGroupIndex * RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_VRPG2;
        if (groupThreadIndex + groupClusterPairStartIndex < clusterPairCount) {
            ClusterPairInfo_VRPG2 clusterPairInfo = clusterPairInfoBuffer[groupClusterPairStartIndex + groupThreadIndex];
            groupClusterPairInfo_VRPG2s[groupThreadIndex] = clusterPairInfo;
            groupNodeData_G[groupThreadIndex] = ClusterData_Octree[clusterPairInfo.layerIndex_G][clusterPairInfo.nodeIndex_G];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= clusterPairCount * HITTEST_COUNT_VRPG2) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = clusterPairCount;

    int clusterPairIndex = threadIndex / HITTEST_COUNT_VRPG2;
    int hitTestIndex = threadIndex & (HITTEST_COUNT_VRPG2 - 1);
    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.randomFrameIndex));

    ClusterPairInfo_VRPG2 clusterPairInfo = groupClusterPairInfo_VRPG2s[groupThreadIndex / HITTEST_COUNT_VRPG2];
    if (clusterPairInfo.leafNodeIndex_E == -1) {
        clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_VRPG2 + hitTestIndex].isHit = -1.0f;
        return;
    }
    ClusterData_Octree_VRPG nodeData_G = groupNodeData_G[groupThreadIndex / HITTEST_COUNT_VRPG2];
    LightNodeData_Octree_VRPG nodeData_E = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
    bool isPrintf = pushConst.frameIndex == 0 && clusterPairInfo.clusterIndex == 1459;

    float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

    uint octreeLayerNode = 1 << (3 * clusterPairInfo.layerIndex_G);
    uint normalIndex = clusterPairInfo.nodeIndex_G / octreeLayerNode;
    float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
    float3 normal_G;
    normal_G.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
    normal_G.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
    normal_G.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

    randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 samplePoint_E = randXYZ * (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) + nodeData_E.aabb.minimum;
    float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
    samplePoint_E = getAABBPoint(nodeData_E.aabb, samplePoint_E, sampleDir);

    HitPayload_FzbWeight hitPayload;
    hitPayload.hit = -1.0f;
    hitPayload.sampleDir = sampleDir;

    bool hitTestAble = dot(nodeData_E.meanNormal.xyz, sampleDir) > 0.0f;
    hitTestAble &= dot(-sampleDir, normal_G) > 0.0f;
    if (hitTestAble) {
        RayDesc ray;
        ray.Direction = sampleDir;
        ray.Origin = samplePoint_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        traceRay(ray, hitPayload);
        if (hitPayload.hit > 0.0f) {
            bool check = true;  //checkOcclusion(samplePoint_E, hitPayload.hitPos, nodeData_G.aabb);
            check &= checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb);
            check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
            check &= all(GetPrincipalAxis(hitPayload.normal) == GetPrincipalAxis(normal_G)); // dot(hitPayload.normal, normal_G) > 0.707f;
            if (!check) hitPayload.hit = -2.0f;
        }
    }

    ClusterPairHitInfo_VRPG2 hitInfo;
    hitInfo.pos_G = hitPayload.hitPos;
    hitInfo.isHit = hitPayload.hit;
    hitInfo.normal_G = hitPayload.normal;
    hitInfo.materialIndex = hitPayload.materialIndex;
    hitInfo.sampleDir = hitPayload.sampleDir;
    hitInfo.d = distance(samplePoint_E, samplePoint_G);
    hitInfo.cosine = max(dot(-hitPayload.sampleDir, hitPayload.normal), 0.0f);
    hitInfo.cosine_Light = max(dot(hitPayload.sampleDir, nodeData_E.meanNormal.xyz), 0.0f);
    clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_VRPG2 + hitTestIndex] = hitInfo;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared uint groupClusterCount;
groupshared SceneInfo groupSceneInfo;
groupshared ClusterPairInfo_VRPG2 groupClusterPairInfo_VRPG2;
groupshared ClusterPairHitInfo_VRPG2 groupClusterPairHitInfo_VRPG2[HITTEST_COUNT_VRPG2];
groupshared LightNodeData_Octree_VRPG groupLeafNodeData_E;

[numthreads(RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodeWeight(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int clusterPairIndex = threadGroupIndex;
    int hitTestIndex = groupThreadIndex & (HITTEST_COUNT_VRPG2 - 1);

    int warpIndex = groupThreadIndex / 32;
    int warpLane = groupThreadIndex & 31;
    #if RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE > 160
    if (warpIndex < 4 && warpLane == 0) {
        if (warpIndex == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
        else if (warpIndex == 1) groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
        else if (warpIndex == 2) groupClusterPairCount = risGlobalInfoBuffer[0].clusterPairCount;
        else if (warpIndex == 3) {
            ClusterPairInfo_VRPG2 clusterPairInfo = clusterPairInfoBuffer[threadGroupIndex];
            groupClusterPairInfo_VRPG2 = clusterPairInfo;
            if (clusterPairInfo.leafNodeIndex_E >= 0)
                groupLeafNodeData_E = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
            else groupLeafNodeData_E.radiance = 0.0f;
        }
    }
    if (warpIndex == 4 && warpLane < HITTEST_COUNT_VRPG2) {
        int hitInfoIndex = clusterPairIndex * HITTEST_COUNT_VRPG2 + warpLane;
        groupClusterPairHitInfo_VRPG2[warpLane] = clusterPairHitTestInfoBuffer[hitInfoIndex];
    }
    #else
    if (groupThreadIndex == 0) {
        groupSceneInfo = pushConst.sceneInfoAddress[0];
        groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
        ClusterPairInfo_VRPG2 clusterPairInfo = clusterPairInfoBuffer[threadGroupIndex];
        groupClusterPairInfo_VRPG2 = clusterPairInfo;
        if (clusterPairInfo.leafNodeIndex_E >= 0)
            groupLeafNodeData_E = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
        else groupLeafNodeData_E.radiance = 0.0f;
    }
    if (groupThreadIndex < HITTEST_COUNT_VRPG2) {
        int hitInfoIndex = clusterPairIndex * HITTEST_COUNT_VRPG2 + groupThreadIndex;
        groupClusterPairHitInfo_VRPG2[groupThreadIndex] = clusterPairHitTestInfoBuffer[hitInfoIndex];
    }
    #endif
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    int outgoingIndex = groupThreadIndex / HITTEST_COUNT_VRPG2;
    if (threadIndex >= clusterPairCount * OUTGOING_COUNT_VRPG2 * HITTEST_COUNT_VRPG2) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_VRPG2 * groupClusterCount;
    if (all(groupLeafNodeData_E.radiance == 0.0f)) return;

    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.randomFrameIndex));

    ClusterPairHitInfo_VRPG2 hitInfo = groupClusterPairHitInfo_VRPG2[hitTestIndex];
    float3 f = 0.0f;
    if (hitInfo.isHit > 0) {
        LightNodeData_Octree_VRPG leafNodeData_E = groupLeafNodeData_E;

        float3 outgoing = mul(getOutgoing_VRPG2(outgoingIndex, OUTGOING_COUNT_VRPG2), pushConst.randomRotateMatrix);

        float3 tangent, bitangent;
        orthonormalBasis(hitInfo.normal_G, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitInfo.normal_G);

        BSDFMaterial material = groupSceneInfo.materials[hitInfo.materialIndex];
        float3 bsdf = getBSDF(material, -hitInfo.sampleDir, hitInfo.normal_G, outgoing, TBN_G, true);
        float d = hitInfo.d;
        f = leafNodeData_E.radiance * bsdf * hitInfo.cosine * hitInfo.cosine_Light / max(d * d, 1e-3f);
    }

    for (int offset = HITTEST_COUNT_VRPG2 / 2; offset > 0; offset /= 2) {
        uint targetLane = warpLane + offset;
        f += WaveReadLaneAt(f, targetLane);
    }

    int clusterPairFirstWarpLane = ((groupThreadIndex / HITTEST_COUNT_VRPG2) * HITTEST_COUNT_VRPG2) & 31;
    if (warpLane == clusterPairFirstWarpLane) {
        f /= HITTEST_COUNT_VRPG2;
        float p_hat = dot(f, 1.0f);
        if (p_hat == 0.0f) return;

        float p = groupClusterPairInfo_VRPG2.leafNodePdf;
        float m = 1.0f;
        float w = m * p_hat / max(p, 1e-6f);

        CandidateSampleData_VRPG2 candidateNodeData;
        candidateNodeData.w = w;
        candidateNodeData.p_hat = p_hat;
        int candidateNodeIndex = outgoingIndex * groupClusterCount * Candidate_Samples_Count +
                                 groupClusterPairInfo_VRPG2.clusterIndex * Candidate_Samples_Count +
                                 groupClusterPairInfo_VRPG2.candidateNodeIndex;
        candidateSampleDataBuffer[candidateNodeIndex] = candidateNodeData;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared float groupWSums[Candidate_Samples_Count / 32];

[numthreads(Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_createCandidatePrefixWSum(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
    GroupMemoryBarrierWithGroupSync();
    int clusterCount = groupClusterCount;

    int clusterIndex = threadGroupIndex % clusterCount;
    int outgoingIndex = threadGroupIndex / clusterCount;
    int candidateNodeIndex = groupThreadIndex;

    int candidateNodeGlobalIndex = outgoingIndex * groupClusterCount * Candidate_Samples_Count + clusterIndex * Candidate_Samples_Count + candidateNodeIndex;
    float w = candidateSampleDataBuffer[candidateNodeGlobalIndex].w;

    int warpIndex = groupThreadIndex / 32;
    int warpLane = groupThreadIndex & 31;
    float incl = w;
    [unroll]
    for (int off = 1; off < 32; off *= 2) {
        float other = WaveReadLaneAt(incl, warpLane - off);
        if (warpLane >= off) incl += other;
    }
    if (warpLane == 31) groupWSums[warpIndex] = incl;
    GroupMemoryBarrierWithGroupSync();

    for (int i = 0; i < warpIndex; ++i) incl += groupWSums[i];
    candidateSamplePrefixWSumBuffer[candidateNodeGlobalIndex] = incl;
}