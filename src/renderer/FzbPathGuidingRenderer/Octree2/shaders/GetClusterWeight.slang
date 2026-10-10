#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_FzbPG::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eClusterPairGlobalInfo)]] RWStructuredBuffer<ClusterPairGlobalInfo, ScalarDataLayout> clusterPairGlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eClusetPairHitTestInfo)]] RWStructuredBuffer<ClusterPairHitInfo, ScalarDataLayout> clusterPairHitTestInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateNodeData_E)]] RWStructuredBuffer<CandidateNodeData_E_FzbPG, ScalarDataLayout> candidateNodeDataBuffer_E;
//----------------------------------------------getNodePairData-------------------------------------------------
groupshared uint groupIndivisibleNodeCount_G;

[numthreads(INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initCandidateNodes(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    if (indivisibleNodeCount_G == 0) {
        if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = 0;
        return;
    }
    int threadTotalCount = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G * Candidate_Samples_Count;
    if (threadIndex >= threadTotalCount) return;
    candidateNodeDataBuffer_E[threadIndex].w = 0.0f;

    if (threadIndex == 0) {
        clusterPairGlobalInfoBuffer[0].clusterPairCount = 0;
        GlobalInfoBuffer[0].cmd.x = (indivisibleNodeCount_G + GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE;
        bool hasLightNode = false;
        for (int i = 0; i < 6; ++i) {
            if (any(OctreeNodeDataBuffer_E[0][i].radiance > 0.0f)) {
                hasLightNode = true;
                break;
            }
        }
        if (!hasLightNode) GlobalInfoBuffer[0].cmd.x = 0;
    } 
}
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
    }
    else payload.hit = 0.0f;
}
//-------------------------------------------------------------------------------------------------------------------------
float3 getLightcutMUpperBound(OctreeNodeData_E_FzbPG lightNode, BSDFMaterial material, float3 pos, float3 normal, float3 outgoing) {
    float3 maxCosine = MaxCosToAABB(pos, lightNode.meanNormal.xyz, lightNode.aabb.minimum, lightNode.aabb.maximum);

    float3 tangent, bitangent;
    orthonormalBasis(normal, tangent, bitangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    float3 maxBSDF_Cosine = -1.0f;
    for (int i = 0; i < 9; ++i) {
        float3 lightPos = lightNode.aabb.minimum + float3(
            (i & 1) == 1 ? lightNode.aabb.maximum.x : lightNode.aabb.minimum.x,
            (i & 2) == 2 ? lightNode.aabb.maximum.y : lightNode.aabb.minimum.y,
            (i & 4) == 4 ? lightNode.aabb.maximum.z : lightNode.aabb.minimum.z
        );
        if(i == 8) lightPos = (lightNode.aabb.maximum + lightNode.aabb.minimum) * 0.5f;
        float3 incidence = normalize(lightPos - pos);
        float3 brdf = getBSDF(material, incidence, normal, outgoing, TBN, true);
        maxBSDF_Cosine = max(maxBSDF_Cosine, brdf * maxCosine);
    }

    return maxBSDF_Cosine;
}
float getLightGUpperBound(OctreeNodeData_E_FzbPG lightNode, float3 pos) {
    float3 minimum = pos - lightNode.aabb.maximum;
    float3 maximum = pos - lightNode.aabb.minimum;

    float cosThetaMax = MaxCosToAABB(float3(0, 0, 0), lightNode.meanNormal.xyz, minimum, maximum);
    cosThetaMax = clamp(cosThetaMax, -1.0, 1.0);

    float thetaMin = acos(cosThetaMax);
    if (thetaMin <= lightNode.halfAngle) return 1.0;
    else return max(0.0, cos(thetaMin - lightNode.halfAngle));
}

#define LIGHTCUT_RELATIVE_ERROR 0.02f
struct LightCutNode {
    int layerIndex;
    int nodeIndex;
    float error;
    float I;
};

groupshared uint groupCandidateNodeCount;
groupshared uint groupOffsetInGlobal;

[numthreads(GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupCandidateNodeCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    if (threadIndex >= indivisibleNodeCount_G) return;

    uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[threadIndex];
    OctreeNodeData_G_FzbPG nodeData_G = OctreeNodeDataBuffer_G[indivisibleNodeInfo.x][indivisibleNodeInfo.y];
    float3 nodePos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    LightCutNode cutNodes[Candidate_Samples_Count];
    cutNodes[0].layerIndex = -1;
    cutNodes[0].nodeIndex = 0;

    uint cutNodeCount = 1;
    float cutNodeISum = 0.0f;
    int maxErrorCutNodeIndex = 0;
    float maxError = 0.0f;

    int iteration = 20;
    while (iteration > 0) {
        bool firstHasISubNode = true;
        int nodeStartIndex = cutNodes[maxErrorCutNodeIndex].nodeIndex * 8;
        int layerIndex = cutNodes[maxErrorCutNodeIndex].layerIndex + 1;
        for (int i = 0; i < 8; ++i) {
            if (layerIndex == 0 && i >= 6) break;

            int subNodeIndex = nodeStartIndex + i;
            OctreeNodeData_E_FzbPG node_E = OctreeNodeDataBuffer_E[layerIndex][subNodeIndex];
            if (all(node_E.radiance == 0.0f)) continue;

            int cutNodeIndex = maxErrorCutNodeIndex;
            if (!firstHasISubNode) {
                cutNodeIndex = cutNodeCount;
                ++cutNodeCount;
            }
            firstHasISubNode = false;

            float I = dot(node_E.radiance, 1.0f);
            float3 MUpperBound = MaxCosToAABB(nodePos_G, nodeData_G.meanNormal.xyz, node_E.aabb.minimum, node_E.aabb.maximum);
            float GUpperBound = getLightGUpperBound(node_E, nodePos_G);
            float diagonalLength = length(node_E.aabb.maximum - node_E.aabb.minimum);
            float3 center = (node_E.aabb.maximum + node_E.aabb.minimum) * 0.5f;
            float d = max(length(nodePos_G - center), diagonalLength + 1e-3f);

            cutNodes[cutNodeIndex].layerIndex = layerIndex;
            cutNodes[cutNodeIndex].nodeIndex = subNodeIndex;
            cutNodes[cutNodeIndex].error = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            cutNodes[cutNodeIndex].I = I;

            cutNodeISum += I;
        }

        float errorThreshold = LIGHTCUT_RELATIVE_ERROR * cutNodeISum;
        maxError = -3.402823466e+38F;
        bool canContinue = false;
        for (int i = 0; i < cutNodeCount; ++i) {
            if (cutNodes[i].layerIndex == pushConst.octreeMaxLayer) continue;
            if (maxError < cutNodes[i].error) {
                maxError = cutNodes[i].error;
                maxErrorCutNodeIndex = i;
                canContinue = true;
            }
        }
        if (maxError <= errorThreshold || !canContinue) break;
        if (cutNodeCount + 7 > Candidate_Samples_Count) break;
        cutNodeISum -= cutNodes[maxErrorCutNodeIndex].I;

        --iteration;
    }

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    int candidateNodeCount = WaveActiveSum(cutNodeCount);
    int offsetInWarp = WavePrefixSum(cutNodeCount);

    int offsetInGroup;
    if (warpLane == 0) InterlockedAdd(groupCandidateNodeCount, candidateNodeCount, offsetInGroup);
    offsetInGroup = WaveReadLaneFirst(offsetInGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd(clusterPairGlobalInfoBuffer[0].clusterPairCount, groupCandidateNodeCount, groupOffsetInGlobal);
    GroupMemoryBarrierWithGroupSync();

    int clusterPairStartIndex = offsetInWarp + offsetInGroup + groupOffsetInGlobal;

    for (int i = 0; i < cutNodeCount; ++i) {
        ClusterPairInfo clusterPairInfo;
        clusterPairInfo.layerIndex_G = indivisibleNodeInfo.x;
        clusterPairInfo.nodeIndex_G = indivisibleNodeInfo.y;
        clusterPairInfo.indivisibleNodeLabel = threadIndex;
        clusterPairInfo.layerIndex_E = cutNodes[i].layerIndex;
        clusterPairInfo.nodeIndex_E = cutNodes[i].nodeIndex;
        clusterPairInfo.candidateNodeIndex = i;
        clusterPairInfoBuffer[clusterPairStartIndex + i] = clusterPairInfo;
    }

    if (threadIndex == 0)  GlobalInfoBuffer[0].cmd.x = 1;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_dispatchGetWeight() {
    GlobalInfoBuffer[0].cmd.x = (clusterPairGlobalInfoBuffer[0].clusterPairCount * HITTEST_COUNT_FZBPG + CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE - 1) / CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
struct LightLeafNode {
    AABB aabb;
    float3 normal;
    float3 radiance;
    float pdf;
    int nodeIndex;
};

groupshared uint groupClusterPairCount;
groupshared ClusterPairInfo groupClusterPairInfos[CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG];
groupshared OctreeNodeData_G_FzbPG groupNodeData_G[CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG];
groupshared OctreeNodeData_E_FzbPG groupNodeData_E[CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG];

LightLeafNode getLightLeafNode(int groupClusterPairIndex, int warpLane, int layer, uint subTreeNodeIndex, float3 pos, uint randomSeed) {
    LightLeafNode leafNode;
    leafNode.radiance = 0.0f;
    leafNode.pdf = 1.0f;

    OctreeNodeData_E_FzbPG node = groupNodeData_E[groupClusterPairIndex];
    if (all(node.radiance == 0.0f)) return leafNode;
    if (layer == pushConst.octreeMaxLayer) {
        leafNode.aabb = node.aabb;
        leafNode.normal = node.meanNormal.xyz;
        leafNode.radiance = node.radiance;
        leafNode.nodeIndex = subTreeNodeIndex;
        return leafNode;
    }

    int nodeIndex = subTreeNodeIndex;
    leafNode.pdf = 1.0f;
    for (; layer < pushConst.octreeMaxLayer; ++layer) {
        node = OctreeNodeDataBuffer_E[layer + 1][nodeIndex * 8 + (warpLane & 7)];

        float weight = 0.0f;
        if (any(node.radiance > 0.0f)) {
            float M = 1.0f;
            float G = 1.0f;
            float diagonalLength = length(node.aabb.maximum - node.aabb.minimum);
            float3 center = (node.aabb.maximum + node.aabb.minimum) * 0.5f;
            float d = max(length(pos - center), diagonalLength + 1e-3f);
            weight = M * G * dot(node.radiance, 1.0f) / (d * d);
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
        if (weightSum == 0.0f) { leafNode.radiance = 0.0f; return leafNode; }

        float target = WaveReadLaneAt(rand(randomSeed), blockFirstWarpLane) * weightSum;
        uint belowBits = (WaveActiveBallot(incl <= target).x >> blockFirstWarpLane) & 0xFFu;
        int chosen = (int)countbits(belowBits);
        float chosenWeight = WaveReadLaneAt(weight, blockFirstWarpLane + chosen);

        nodeIndex = nodeIndex * 8 + chosen;
        leafNode.pdf *= chosenWeight / weightSum;
    }

    node = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][nodeIndex];
    leafNode.aabb = node.aabb; leafNode.normal = node.meanNormal.xyz; leafNode.radiance = node.radiance; leafNode.nodeIndex = nodeIndex;
    return leafNode;
}

[numthreads(CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_clusterPairHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterPairCount = clusterPairGlobalInfoBuffer[0].clusterPairCount;
    GroupMemoryBarrierWithGroupSync();

    int clusterPairCount = groupClusterPairCount;
    if (groupThreadIndex < CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG) {
        int groupClusterPairStartIndex = threadGroupIndex * CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG;
        if (groupThreadIndex + groupClusterPairStartIndex < clusterPairCount) {
            ClusterPairInfo clusterPairInfo = clusterPairInfoBuffer[groupClusterPairStartIndex + groupThreadIndex];
            groupClusterPairInfos[groupThreadIndex] = clusterPairInfo;
            groupNodeData_G[groupThreadIndex] = OctreeNodeDataBuffer_G[clusterPairInfo.layerIndex_G][clusterPairInfo.nodeIndex_G];
            groupNodeData_E[groupThreadIndex] = OctreeNodeDataBuffer_E[clusterPairInfo.layerIndex_E][clusterPairInfo.nodeIndex_E];
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= clusterPairCount * HITTEST_COUNT_FZBPG) return;

    int clusterPairIndex = threadIndex / HITTEST_COUNT_FZBPG;
    int hitTestIndex = threadIndex & (HITTEST_COUNT_FZBPG - 1);
    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.frameIndex));

    ClusterPairInfo clusterPairInfo = groupClusterPairInfos[groupThreadIndex / HITTEST_COUNT_FZBPG];
    OctreeNodeData_G_FzbPG nodeData_G = groupNodeData_G[groupThreadIndex / HITTEST_COUNT_FZBPG];
    float3 centerPos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    #if HITTEST_COUNT_FZBPG == 8
    LightLeafNode nodeData_E = getLightLeafNode(groupThreadIndex / HITTEST_COUNT_FZBPG, groupThreadIndex & 31, clusterPairInfo.layerIndex_E, clusterPairInfo.nodeIndex_E, centerPos_G, seed);
    #else
    LightLeafNode nodeData_E;
    if (hitTestIndex < 8)
        nodeData_E = getLightLeafNode(groupThreadIndex / HITTEST_COUNT_FZBPG, groupThreadIndex & 31, clusterPairInfo.layerIndex_E, clusterPairInfo.nodeIndex_E, centerPos_G, seed);
    int hitTestFirstWarpLane = (groupThreadIndex & 31) & ~15;
    nodeData_E.aabb.maximum = WaveReadLaneAt(nodeData_E.aabb.maximum, hitTestFirstWarpLane);
    nodeData_E.aabb.minimum = WaveReadLaneAt(nodeData_E.aabb.minimum, hitTestFirstWarpLane);
    nodeData_E.normal = WaveReadLaneAt(nodeData_E.normal, hitTestFirstWarpLane);
    nodeData_E.radiance = WaveReadLaneAt(nodeData_E.radiance, hitTestFirstWarpLane);
    nodeData_E.pdf = WaveReadLaneAt(nodeData_E.pdf, hitTestFirstWarpLane);
    #endif
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = groupClusterPairCount;
    if (all(nodeData_E.radiance == 0.0f)) {
        clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_FZBPG + hitTestIndex].isHit = -1.0f;
        return;
    }

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

    bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
    hitTestAble &= dot(-sampleDir, normal_G) > 0.0f;
    if (hitTestAble) {
        RayDesc ray;
        ray.Direction = sampleDir;
        ray.Origin = samplePoint_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        traceRay(ray, hitPayload);
        if (hitPayload.hit > 0.0f) {
            bool check = checkOcclusion(samplePoint_E, hitPayload.hitPos, nodeData_G.aabb);
            if (checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb)) {
                check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                check &= dot(hitPayload.normal, normal_G) > 0.707f;
            }
            if (!check) hitPayload.hit = -2.0f;
        }
    }

    ClusterPairHitInfo hitInfo;
    hitInfo.pos_G = hitPayload.hitPos;
    hitInfo.isHit = hitPayload.hit;
    hitInfo.normal_G = hitPayload.normal;
    hitInfo.materialIndex = hitPayload.materialIndex;
    hitInfo.sampleDir = hitPayload.sampleDir;
    hitInfo.d = distance(samplePoint_E, samplePoint_G);
    hitInfo.cosine = max(dot(-hitPayload.sampleDir, hitPayload.normal), 0.0f);
    hitInfo.cosine_Light = max(dot(hitPayload.sampleDir, nodeData_E.normal), 0.0f);
    clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_FZBPG + hitTestIndex] = hitInfo;

    if (hitTestIndex == 0) {
        clusterPairInfoBuffer[clusterPairIndex].nodeIndex_E = nodeData_E.nodeIndex;
        clusterPairInfoBuffer[clusterPairIndex].leafNodePdf = nodeData_E.pdf;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared SceneInfo groupSceneInfo;
groupshared ClusterPairInfo groupClusterPairInfo;
groupshared ClusterPairHitInfo groupClusterPairHitInfo[HITTEST_COUNT_FZBPG];
groupshared OctreeNodeData_E_FzbPG groupLeafNodeData_E;

[numthreads(GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodeWeight(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int clusterPairIndex = threadGroupIndex;
    int outgoingIndex = groupThreadIndex / HITTEST_COUNT_FZBPG;
    int hitTestIndex = groupThreadIndex & (HITTEST_COUNT_FZBPG - 1);

    int warpIndex = groupThreadIndex / 32;
    int warpLane = groupThreadIndex & 31;
    if (warpIndex < 4 && warpLane == 0) {
        if (warpIndex == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
        else if (warpIndex == 1) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        else if (warpIndex == 2) groupClusterPairCount = clusterPairGlobalInfoBuffer[0].clusterPairCount;
        else if (warpIndex == 3) {
            ClusterPairInfo clusterPairInfo = clusterPairInfoBuffer[threadGroupIndex];
            groupClusterPairInfo = clusterPairInfo;
            groupLeafNodeData_E = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][clusterPairInfo.nodeIndex_E];
        }
    }
    if (warpIndex == 4 && warpLane < HITTEST_COUNT_FZBPG) {
        int hitInfoIndex = clusterPairIndex * HITTEST_COUNT_FZBPG + warpLane;
        groupClusterPairHitInfo[warpLane] = clusterPairHitTestInfoBuffer[hitInfoIndex];
    }
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    if (threadIndex >= clusterPairCount * OUTGOING_COUNT_FZBPG * HITTEST_COUNT_FZBPG) return;

    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.frameIndex));

    ClusterPairHitInfo hitInfo = groupClusterPairHitInfo[hitTestIndex];
    if (hitInfo.isHit <= 0) return;
    OctreeNodeData_E_FzbPG leafNodeData_E = groupLeafNodeData_E;

    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);

    float3 tangent, bitangent;
    orthonormalBasis(hitInfo.normal_G, tangent, bitangent);
    float3x3 TBN_G = float3x3(tangent, bitangent, hitInfo.normal_G);

    BSDFMaterial material = groupSceneInfo.materials[hitInfo.materialIndex];
    float3 bsdf = getBSDF(material, -hitInfo.sampleDir, hitInfo.normal_G, outgoing, TBN_G, true);
    float d = hitInfo.d;
    float3 f = leafNodeData_E.radiance * bsdf * hitInfo.cosine * hitInfo.cosine_Light / max(d * d, 0.01f);

    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = warpLane + offset;
        f += WaveReadLaneAt(f, targetLane);
    }

    int clusterPairFirstWarpLane = ((groupThreadIndex / HITTEST_COUNT_FZBPG) * HITTEST_COUNT_FZBPG) & 31;
    if (warpLane == clusterPairFirstWarpLane) {
        f /= HITTEST_COUNT_FZBPG;
        float p_hat = dot(f, 1.0f);
        float p = groupClusterPairInfo.leafNodePdf;
        float m = 1.0f;
        float w = m * p_hat / max(p, 1e-6f);

        CandidateNodeData_E_FzbPG candidateNodeData;
        candidateNodeData.w = w;
        candidateNodeData.p_hat = p_hat;
        candidateNodeData.aabb = leafNodeData_E.aabb;

        int candidateNodeIndex = outgoingIndex * groupIndivisibleNodeCount_G * Candidate_Samples_Count +
                                 groupClusterPairInfo.indivisibleNodeLabel * Candidate_Samples_Count +
                                 groupClusterPairInfo.candidateNodeIndex;
        candidateNodeDataBuffer_E[candidateNodeIndex] = candidateNodeData;
    }
}