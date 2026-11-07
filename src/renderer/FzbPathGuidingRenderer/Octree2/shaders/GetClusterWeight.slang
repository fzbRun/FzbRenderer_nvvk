#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_FzbPG::eAdjClusterInfo_G)]] RWStructuredBuffer<uint> AdjClusterInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eClusterPairGlobalInfo)]] RWStructuredBuffer<ClusterPairGlobalInfo, ScalarDataLayout> clusterPairGlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eClusetPairHitTestInfo)]] RWStructuredBuffer<ClusterPairHitInfo, ScalarDataLayout> clusterPairHitTestInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo, ScalarDataLayout> candidateSubTreeInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData, ScalarDataLayout> candidateLeafNodeDataBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_FzbPG, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;

#ifdef SpatialReuse_FzbPG
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleIndex)]] RWStructuredBuffer<int> candidateSampleIndexBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleAABB)]] RWStructuredBuffer<AABB, ScalarDataLayout> candidateSampleAABBBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleData2)]] RWStructuredBuffer<CandidateSampleData_FzbPG, ScalarDataLayout> candidateSampleData2Buffer;
#endif
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

    CandidateSampleData_FzbPG candidateData;
    candidateData.w = 0.0f;
    candidateData.p_hat = 0.0f;
    candidateSampleDataBuffer[threadIndex] = candidateData;
    #ifdef SpatialReuse_FzbPG
    candidateSampleData2Buffer[threadIndex] = candidateData;
    candidateSampleAABBBuffer[threadIndex] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    #endif
    if (threadIndex < indivisibleNodeCount_G * Candidate_Samples_Count) candidateSubTreeInfoBuffer[threadIndex].subTreeNodeIndex_layer = -1;

    if (threadIndex == 0) {
        clusterPairGlobalInfoBuffer[0].clusterPairCount = 0;
        DispatchIndirectCommand cmd;
        #ifdef NoLightcuts
        cmd.x = indivisibleNodeCount_G;
        #else
        cmd.x = (indivisibleNodeCount_G + GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE;
        #endif
        cmd.y = 1; cmd.z = 1;
        GlobalInfoBuffer[0].cmd = cmd;
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
//-------------------------------------------------------------------------------------------------------------------------
groupshared uint groupCandidateNodeCount;
groupshared uint groupOffsetInGlobal;

groupshared uint2 groupNodeInfoG;
groupshared OctreeNodeData_G_FzbPG groupNodeDataG;

[numthreads(Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_NoLightcuts(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0){
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;

        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[threadGroupIndex];
        groupNodeInfoG = nodeInfo;
        groupNodeDataG = OctreeNodeDataBuffer_G[nodeInfo.x][nodeInfo.y];

        groupCandidateNodeCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeIndex = threadGroupIndex;
    //if (indivisibleNodeIndex >= groupIndivisibleNodeCount_G) return;
    int candidateNodeIndex = groupThreadIndex;

    OctreeNodeData_G_FzbPG node_G = groupNodeDataG;
    float3 nodePos_G = (node_G.aabb.maximum + node_G.aabb.minimum) * 0.5;

    int seed = xxhash32(uint3(indivisibleNodeIndex, candidateNodeIndex, uint(pushConst.frameIndex)));
    int layerIndex = 0; int nodeIndex_E = 0; float p = 1.0f;
    for (; layerIndex <= pushConst.octreeMaxLayer; ++layerIndex) {
        float ws[8]; float wSum = 0.0f;
        for (int subNodeIndex = 0; subNodeIndex < 8; ++subNodeIndex) {
            ws[subNodeIndex] = 0.0f;
            if (layerIndex == 0 && subNodeIndex >= 6) break;

            int nodeIndex = nodeIndex_E * 8 + subNodeIndex;
            OctreeNodeData_E_FzbPG node_E = OctreeNodeDataBuffer_E[layerIndex][nodeIndex];

            float I = dot(node_E.radiance, 1.0f);
            float3 MUpperBound = MaxCosToAABB(nodePos_G, node_G.meanNormal.xyz, node_E.aabb.minimum, node_E.aabb.maximum);
            float GUpperBound = getLightGUpperBound(node_E, nodePos_G);
            float3 center = (node_E.aabb.maximum + node_E.aabb.minimum) * 0.5f;
            float diagonalLength = length(node_E.aabb.maximum - node_E.aabb.minimum);
            float d = max(length(nodePos_G - center), diagonalLength + 1e-3f);
            // float d = max(length(nodePos_G - center), 1.0f);

            ws[subNodeIndex] = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            wSum += ws[subNodeIndex];

            //if (pushConst.frameIndex == 0 && indivisibleNodeIndex == 0 && candidateNodeIndex == 0)
            //    printf("layerIndex: %d, nodeIndex: %d, w: %f, radiance: %f %f %f\n",
            //        layerIndex, nodeIndex,
            //        ws[subNodeIndex],
            //        node_E.radiance.x, node_E.radiance.y, node_E.radiance.z
            //    );
        }
        if (wSum == 0.0f) break;

        float randomNumber = rand(seed);
        for (int subNodeIndex = 0; subNodeIndex < 8; ++subNodeIndex) {
            float pSelect = ws[subNodeIndex] / wSum;
            if (pSelect >= randomNumber) {
                nodeIndex_E = nodeIndex_E * 8 + subNodeIndex;
                p *= pSelect;
                break;
            }
            randomNumber -= pSelect;
        }
    }
    bool getCandidate = layerIndex == pushConst.octreeMaxLayer + 1;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    int candidateNodeCount = WaveActiveSum(getCandidate);
    int offsetInWarp = WavePrefixSum(getCandidate);

    int offsetInGroup;
    if (warpLane == 0) InterlockedAdd(groupCandidateNodeCount, candidateNodeCount, offsetInGroup);
    offsetInGroup = WaveReadLaneFirst(offsetInGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd(clusterPairGlobalInfoBuffer[0].clusterPairCount, groupCandidateNodeCount, groupOffsetInGlobal);
    GroupMemoryBarrierWithGroupSync();

    if (getCandidate) {
        int clusterPairStartIndex = offsetInWarp + offsetInGroup + groupOffsetInGlobal;

        ClusterPairInfo pairInfo;
        pairInfo.layerIndex_G = groupNodeInfoG.x;
        pairInfo.nodeIndex_G = groupNodeInfoG.y;
        pairInfo.indivisibleNodeLabel = indivisibleNodeIndex;
        pairInfo.candidateNodeIndex = candidateNodeIndex;
        pairInfo.leafNodeIndex_E = nodeIndex_E;
        pairInfo.leafNodePdf = p;
        clusterPairInfoBuffer[clusterPairStartIndex] = pairInfo;

        OctreeNodeData_E_FzbPG node_E = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][nodeIndex_E];
        CandidateLeafNodeData leafNodeData;
        leafNodeData.radiance = node_E.radiance;
        leafNodeData.aabb = node_E.aabb;
        #ifdef SpatialReuse_FzbPG
        leafNodeData.normal = node_E.meanNormal.xyz;
        #endif

        candidateLeafNodeDataBuffer[indivisibleNodeIndex * Candidate_Samples_Count + candidateNodeIndex] = leafNodeData;
    }
    //if (pushConst.frameIndex == 0) printf("getCandidate = %d\n", getCandidate);

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = (groupIndivisibleNodeCount_G * Candidate_Samples_Count * HITTEST_COUNT_FZBPG + CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE - 1) / CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE;
}
//-------------------------------------------------------------------------------------------------------------------------

#define LIGHTCUT_RELATIVE_ERROR 0.001f
struct LightCutNode {
    int layerIndex;
    int nodeIndex;
    float error;
    float I;
};

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
    // float3 nodePos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;
    uint seed = xxhash32(uint3(threadIndex, 0, (uint)pushConst.frameIndex));
    float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 nodePos_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

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

            float I = dot(node_E.radiance, 1.0f);
            if (layerIndex == 0) cutNodeISum += I;

            float3 MUpperBound = MaxCosToAABB(nodePos_G, nodeData_G.meanNormal.xyz, node_E.aabb.minimum, node_E.aabb.maximum);
            float GUpperBound = getLightGUpperBound(node_E, nodePos_G);
            float3 center = (node_E.aabb.maximum + node_E.aabb.minimum) * 0.5f;
            float diagonalLength = length(node_E.aabb.maximum - node_E.aabb.minimum);
            float d = max(length(nodePos_G - center), diagonalLength + 1e-3f);
            //float d = max(length(nodePos_G - center), 1.0f);

            float error = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            if (error == 0.0f) continue;

            int cutNodeIndex = maxErrorCutNodeIndex;
            if (!firstHasISubNode) {
                cutNodeIndex = cutNodeCount;
                ++cutNodeCount;
            }
            firstHasISubNode = false;

            cutNodes[cutNodeIndex].layerIndex = layerIndex;
            cutNodes[cutNodeIndex].nodeIndex = subNodeIndex;
            cutNodes[cutNodeIndex].error = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            cutNodes[cutNodeIndex].I = I;
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
void computeMain_dispatchGetCandidateLeafNodes() {
    GlobalInfoBuffer[0].cmd.x = (clusterPairGlobalInfoBuffer[0].clusterPairCount * 8 + 255) / 256;
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

    OctreeNodeData_E_FzbPG node = OctreeNodeDataBuffer_E[layer][subTreeNodeIndex];
    if (all(node.radiance == 0.0f)) {
        leafNode.aabb = node.aabb;
        leafNode.radiance = node.radiance;
        leafNode.normal = normalize(node.meanNormal.xyz);
        return leafNode;
    }
    if (layer == pushConst.octreeMaxLayer) {
        leafNode.nodeIndex = subTreeNodeIndex;
        leafNode.aabb = node.aabb;
        leafNode.radiance = node.radiance;
        leafNode.normal = normalize(node.meanNormal.xyz);
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

    node = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][nodeIndex];
    leafNode.aabb = node.aabb;
    leafNode.radiance = node.radiance;
    leafNode.normal = normalize(node.meanNormal.xyz);
    leafNode.nodeIndex = nodeIndex;
    return leafNode;
}

[numthreads(256, 1, 1)]
[shader("compute")]
void computeMain_getCandidateLeafNodes(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterPairCount = clusterPairGlobalInfoBuffer[0].clusterPairCount;
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    if (threadIndex >= clusterPairCount * 8) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = (groupClusterPairCount * HITTEST_COUNT_FZBPG + CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE - 1) / CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE;

    int clusterPairIndex = threadIndex / 8;
    ClusterPairInfo clusterPairInfo = clusterPairInfoBuffer[clusterPairIndex];
    OctreeNodeData_G_FzbPG nodeData_G = OctreeNodeDataBuffer_G[clusterPairInfo.layerIndex_G][clusterPairInfo.nodeIndex_G];
    float3 centerPos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    uint seed = xxhash32(uint3(clusterPairIndex, groupThreadIndex, (uint)pushConst.frameIndex));
    LightLeafNode nodeData_E = getLightLeafNode(groupThreadIndex & 31, clusterPairInfo.layerIndex_E, clusterPairInfo.nodeIndex_E, centerPos_G, seed);
    if ((groupThreadIndex & 7) == 0) {
        clusterPairInfoBuffer[clusterPairIndex].leafNodeIndex_E = nodeData_E.nodeIndex;
        clusterPairInfoBuffer[clusterPairIndex].leafNodePdf = nodeData_E.pdf;

        CandidateSubTreeInfo subTreeData;
        int candidateNodeIndex = clusterPairInfo.indivisibleNodeLabel * Candidate_Samples_Count + clusterPairInfo.candidateNodeIndex;
        subTreeData.subTreeNodeIndex_layer = (clusterPairInfo.nodeIndex_E << 4) | clusterPairInfo.layerIndex_E;
        subTreeData.ESum = dot(OctreeNodeDataBuffer_E[clusterPairInfo.layerIndex_E][clusterPairInfo.nodeIndex_E].radiance, 1.0f);
        candidateSubTreeInfoBuffer[candidateNodeIndex] = subTreeData;

        CandidateLeafNodeData leafNodeData;
        leafNodeData.radiance = nodeData_E.radiance;
        leafNodeData.aabb = nodeData_E.aabb;
        #ifdef SpatialReuse_FzbPG
        leafNodeData.normal = nodeData_E.normal;
        #endif
        candidateLeafNodeDataBuffer[candidateNodeIndex] = leafNodeData;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared ClusterPairInfo groupClusterPairInfos[CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG];
groupshared OctreeNodeData_G_FzbPG groupNodeData_G[CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE / HITTEST_COUNT_FZBPG];

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
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= clusterPairCount * HITTEST_COUNT_FZBPG) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = clusterPairCount;

    int clusterPairIndex = threadIndex / HITTEST_COUNT_FZBPG;
    int hitTestIndex = threadIndex & (HITTEST_COUNT_FZBPG - 1);
    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.frameIndex));

    ClusterPairInfo clusterPairInfo = groupClusterPairInfos[groupThreadIndex / HITTEST_COUNT_FZBPG];
    if (clusterPairInfo.leafNodeIndex_E == -1) {
        clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_FZBPG + hitTestIndex].isHit = -1.0f;
        return;
    }
    OctreeNodeData_G_FzbPG nodeData_G = groupNodeData_G[groupThreadIndex / HITTEST_COUNT_FZBPG];
    OctreeNodeData_E_FzbPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
    //if (all(nodeData_E.radiance == 0.0f)) {
    //    clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_FZBPG + hitTestIndex].isHit = -1.0f;
    //    return;
    //}
    bool isPrintf = pushConst.frameIndex == 65 && clusterPairInfo.indivisibleNodeLabel == 49;

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
            bool check = checkOcclusion(samplePoint_E, hitPayload.hitPos, nodeData_G.aabb);
            check &= checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb);
            check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
            check &= dot(hitPayload.normal, normal_G) > 0.707f;
            if (!check) hitPayload.hit = -2.0f;
        }
    }

    //if (pushConst.frameIndex == 0) printf("%f %f\n", samplePoint_E.x, samplePoint_G.x);

    ClusterPairHitInfo hitInfo;
    hitInfo.pos_G = hitPayload.hitPos;
    hitInfo.isHit = hitPayload.hit;
    hitInfo.normal_G = hitPayload.normal;
    hitInfo.materialIndex = hitPayload.materialIndex;
    hitInfo.sampleDir = hitPayload.sampleDir;
    hitInfo.d = distance(samplePoint_E, samplePoint_G);
    hitInfo.cosine = max(dot(-hitPayload.sampleDir, hitPayload.normal), 0.0f);
    hitInfo.cosine_Light = max(dot(hitPayload.sampleDir, nodeData_E.meanNormal.xyz), 0.0f);
    clusterPairHitTestInfoBuffer[clusterPairIndex * HITTEST_COUNT_FZBPG + hitTestIndex] = hitInfo;

    //bool isPrintf = pushConst.frameIndex == 65 && clusterPairInfo.indivisibleNodeLabel == 49;
    //if (isPrintf) {
    //    float3 normalE = nodeData_E.meanNormal.xyz;
    //    printf("leaftNodeIndexE: %d, hitTesIndex: %d, hit: %f\n normalE: %f %f %f, samplePosE: %f %f %f\n normalG: %f %f %f, samplePosG: %f %f %f\n hitNormalG: %f %f %f, hitPos: %f %f %f\nAABB_E: %f %f %f %f %f %f\nAABB_G: %f %f %f %f %f %f\n\n", 
    //        clusterPairInfo.leafNodeIndex_E, hitTestIndex, hitInfo.isHit, 
    //        normalE.x, normalE.y, normalE.z,
    //        samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
    //        normal_G.x, normal_G.y, normal_G.z,
    //        samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
    //        hitPayload.normal.x, hitPayload.normal.y, hitPayload.normal.z,
    //        hitPayload.hitPos.x, hitPayload.hitPos.y, hitPayload.hitPos.z,
    //        nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
    //        nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
    //        nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
    //        nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
    //    );
    //}
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
    #if GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE > 160
    if (warpIndex < 4 && warpLane == 0) {
        if (warpIndex == 0) groupSceneInfo = pushConst.sceneInfoAddress[0];
        else if (warpIndex == 1) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        else if (warpIndex == 2) groupClusterPairCount = clusterPairGlobalInfoBuffer[0].clusterPairCount;
        else if (warpIndex == 3) {
            ClusterPairInfo clusterPairInfo = clusterPairInfoBuffer[threadGroupIndex];
            groupClusterPairInfo = clusterPairInfo;
            if (clusterPairInfo.leafNodeIndex_E >= 0)
                groupLeafNodeData_E = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
            else groupLeafNodeData_E.radiance = 0.0f;
        }
    }
    if (warpIndex == 4 && warpLane < HITTEST_COUNT_FZBPG) {
        int hitInfoIndex = clusterPairIndex * HITTEST_COUNT_FZBPG + warpLane;
        groupClusterPairHitInfo[warpLane] = clusterPairHitTestInfoBuffer[hitInfoIndex];
    }
    #else
    if (groupThreadIndex == 0) {
        groupSceneInfo = pushConst.sceneInfoAddress[0];
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        ClusterPairInfo clusterPairInfo = clusterPairInfoBuffer[threadGroupIndex];
        groupClusterPairInfo = clusterPairInfo;
        if (clusterPairInfo.leafNodeIndex_E >= 0)
            groupLeafNodeData_E = OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][clusterPairInfo.leafNodeIndex_E];
        else groupLeafNodeData_E.radiance = 0.0f;
    }
    if (groupThreadIndex < HITTEST_COUNT_FZBPG) {
        int hitInfoIndex = clusterPairIndex * HITTEST_COUNT_FZBPG + groupThreadIndex;
        groupClusterPairHitInfo[groupThreadIndex] = clusterPairHitTestInfoBuffer[hitInfoIndex];
    }
    #endif
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    if (threadIndex >= clusterPairCount * OUTGOING_COUNT_FZBPG * HITTEST_COUNT_FZBPG) return;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_FZBPG * groupIndivisibleNodeCount_G;
    if (all(groupLeafNodeData_E.radiance == 0.0f)) return;

    uint seed = xxhash32(uint3(clusterPairIndex, hitTestIndex, (uint)pushConst.frameIndex));

    ClusterPairHitInfo hitInfo = groupClusterPairHitInfo[hitTestIndex];
    float3 f = 0.0f;
    if (hitInfo.isHit > 0) {
        OctreeNodeData_E_FzbPG leafNodeData_E = groupLeafNodeData_E;

        float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);

        float3 tangent, bitangent;
        orthonormalBasis(hitInfo.normal_G, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitInfo.normal_G);

        BSDFMaterial material = groupSceneInfo.materials[hitInfo.materialIndex];
        float3 bsdf = getBSDF(material, -hitInfo.sampleDir, hitInfo.normal_G, outgoing, TBN_G, true);
        float d = hitInfo.d;
        f = leafNodeData_E.radiance * bsdf * hitInfo.cosine * hitInfo.cosine_Light / max(d * d, 1e-3f);

        //bool isPrintf = pushConst.frameIndex == 65 && groupClusterPairInfo.indivisibleNodeLabel == 49 && outgoingIndex == 0;
        //if (isPrintf) {
        //    printf("leaftNodeE: %d, hitTestIndex: %d, hit: %f\n incidence: %f %f %f, normal: %f %f %f, outgoing: %f %f %f\n\n", 
        //        groupClusterPairInfo.leafNodeIndex_E, hitTestIndex, hitInfo.isHit,
        //        -hitInfo.sampleDir.x, -hitInfo.sampleDir.y, -hitInfo.sampleDir.z,
        //        hitInfo.normal_G.x, hitInfo.normal_G.y, hitInfo.normal_G.z,
        //        outgoing.x, outgoing.y, outgoing.z,
        //    );
        //}
    }

    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = warpLane + offset;
        f += WaveReadLaneAt(f, targetLane);
    }

    int clusterPairFirstWarpLane = ((groupThreadIndex / HITTEST_COUNT_FZBPG) * HITTEST_COUNT_FZBPG) & 31;
    if (warpLane == clusterPairFirstWarpLane) {
        f /= HITTEST_COUNT_FZBPG;
        float p_hat = dot(f, 1.0f);
        if (p_hat == 0.0f) return;

        float p = groupClusterPairInfo.leafNodePdf;
        #ifdef NoLightcuts
        float m = 1.0f / Candidate_Samples_Count;
        #else
        float m = 1.0f;
        #endif
        float w = m * p_hat / max(p, 1e-6f);

        CandidateSampleData_FzbPG candidateNodeData;
        candidateNodeData.w = w;
        candidateNodeData.p_hat = p_hat;
        int candidateNodeIndex = outgoingIndex * groupIndivisibleNodeCount_G * Candidate_Samples_Count +
                                 groupClusterPairInfo.indivisibleNodeLabel * Candidate_Samples_Count +
                                 groupClusterPairInfo.candidateNodeIndex;
        candidateSampleDataBuffer[candidateNodeIndex] = candidateNodeData;
    }
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
groupshared float groupWSums[Candidate_Samples_Count / 32];

[numthreads(Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_createCandidatePrefixWSum(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;

    int indivisibleNodeIndex = threadGroupIndex % indivisibleNodeCount;
    int outgoingIndex = threadGroupIndex / indivisibleNodeCount;
    int candidateNodeIndex = groupThreadIndex;

    int candidateNodeGlobalIndex = outgoingIndex * groupIndivisibleNodeCount_G * Candidate_Samples_Count + indivisibleNodeIndex * Candidate_Samples_Count + candidateNodeIndex;
    #ifdef SpatialReuse_FzbPG
    float w = pushConst.candidateSampleBufferIndex == 0 ? candidateSampleDataBuffer[candidateNodeGlobalIndex].w : candidateSampleData2Buffer[candidateNodeGlobalIndex].w;
    #else
    float w = candidateSampleDataBuffer[candidateNodeGlobalIndex].w;
    #endif

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

#ifdef SpatialReuse_FzbPG
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = groupIndivisibleNodeCount_G;
#endif
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef SpatialReuse_FzbPG
[numthreads(OUTGOING_COUNT_FZBPG * Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_reservoirSample(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int indivisibleNodeCount = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = indivisibleNodeCount * OUTGOING_COUNT_FZBPG;

    int indivisibleNodeIndex = threadGroupIndex;
    int outgoingIndex = groupThreadIndex / Candidate_Samples_Count;
    int candidateSampleIndex = groupThreadIndex & (Candidate_Samples_Count - 1);

    int candidateNodeStartIndex = outgoingIndex * indivisibleNodeCount * Candidate_Samples_Count + indivisibleNodeIndex * Candidate_Samples_Count;
    float wSum = candidateSamplePrefixWSumBuffer[candidateNodeStartIndex + Candidate_Samples_Count - 1];
    if (wSum == 0.0f) {
        if (candidateSampleIndex == 0) candidateSampleIndexBuffer[indivisibleNodeIndex * OUTGOING_COUNT_FZBPG + outgoingIndex] = 0;
        return;
    }

    uint seed = xxhash32(uint3(candidateNodeStartIndex, candidateSampleIndex, (uint)pushConst.frameIndex));
    float randomNumber = rand(seed) * wSum;
    int lo = 0;
    int hi = Candidate_Samples_Count - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (randomNumber < candidateSamplePrefixWSumBuffer[candidateNodeStartIndex + mid]) hi = mid;
        else lo = mid + 1;
    }
    if (candidateSampleIndex == 0) candidateSampleIndexBuffer[indivisibleNodeIndex * OUTGOING_COUNT_FZBPG + outgoingIndex] = lo;
}

#ifdef Pairwise_MIS
[numthreads(Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_SpatialReuse(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
#if Candidate_Samples_Count == 32
    int indivisibleNodeCount = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = indivisibleNodeCount;

    int indivisibleNodeIndex = threadGroupIndex / OUTGOING_COUNT_FZBPG;
    int candidateLocalIndex = groupThreadIndex;
    int outgoingIndex = threadGroupIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);

    int canonicalSampleLocalIndex = candidateSampleIndexBuffer[indivisibleNodeIndex * OUTGOING_COUNT_FZBPG + outgoingIndex];
    CandidateLeafNodeData canonicalSampleData = candidateLeafNodeDataBuffer[indivisibleNodeIndex * Candidate_Samples_Count + canonicalSampleLocalIndex];
    //--------------------------------------------
    uint adjClusterIndex = AdjClusterInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + candidateLocalIndex];
    int adjClusterCandidateLocalIndex = 0;
    OctreeNodeData_G_FzbPG adjClusterData;
    CandidateSampleData_FzbPG yData_i = { 0.0f, 0.0f };
    if (adjClusterIndex < indivisibleNodeCount) {
        uint2 adjClusterInfo = IndivisibleNodeInfoBuffer_G[adjClusterIndex];
        adjClusterData = OctreeNodeDataBuffer_G[adjClusterInfo.x][adjClusterInfo.y];

        adjClusterCandidateLocalIndex = candidateSampleIndexBuffer[adjClusterIndex * OUTGOING_COUNT_FZBPG + outgoingIndex];
        int adjClusterCandidateGlobalStartIndex = outgoingIndex * indivisibleNodeCount * Candidate_Samples_Count + adjClusterIndex * Candidate_Samples_Count;
        yData_i.p_hat = candidateSampleDataBuffer[adjClusterCandidateGlobalStartIndex + adjClusterCandidateLocalIndex].p_hat;
        yData_i.w = candidateSamplePrefixWSumBuffer[adjClusterCandidateGlobalStartIndex + Candidate_Samples_Count - 1]; // wSum
    }
    //-------------------------------------------
    float validCandidateCount = WaveActiveCountBits(adjClusterIndex < indivisibleNodeCount);
    uint seed = xxhash32(uint3(threadGroupIndex, candidateLocalIndex, (uint)pushConst.frameIndex));

    const int sampleCount = 1;

    float pj_hat = 0.0f;
    if (adjClusterIndex < indivisibleNodeCount && groupThreadIndex > 0) {
        float3 f = 0.0f;

        HitPayload_FzbWeight hitPayload;
        float3 normal_G = SnapToDominantAxis(adjClusterData.meanNormal.xyz);
        for (int i = 0; i < sampleCount; ++i) {
            // float3 samplePos_G = (adjClusterData.aabb.maximum + adjClusterData.aabb.minimum) * 0.5f;
            float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
            float3 samplePos_G = randXYZ * (adjClusterData.aabb.maximum - adjClusterData.aabb.minimum) + adjClusterData.aabb.minimum;
            
            // float3 samplePos_E = (canonicalSampleData.aabb.maximum + canonicalSampleData.aabb.minimum) * 0.5f;
            randXYZ = float3(rand(seed), rand(seed), rand(seed));
            float3 samplePos_E = randXYZ * (canonicalSampleData.aabb.maximum - canonicalSampleData.aabb.minimum) + canonicalSampleData.aabb.minimum;
            float3 sampleDir = normalize(samplePos_G - samplePos_E);
            samplePos_E = getAABBPoint(canonicalSampleData.aabb, samplePos_E, sampleDir);

            hitPayload.hit = -1.0f;
            hitPayload.sampleDir = sampleDir;

            bool hitTestAble = dot(canonicalSampleData.normal, sampleDir) > 0.0f;
            hitTestAble &= dot(-sampleDir, normal_G) > 0.0f;
            if (hitTestAble) {
                RayDesc ray;
                ray.Direction = sampleDir;
                ray.Origin = samplePos_E + sampleDir * 0.00001f;
                ray.TMin = 0.001;
                ray.TMax = INFINITE;
                traceRay(ray, hitPayload);
                if (hitPayload.hit > 0.0f) {
                    bool check = checkOcclusion(samplePos_E, hitPayload.hitPos, adjClusterData.aabb);
                    if (checkPointInAABB2(hitPayload.hitPos, adjClusterData.aabb)) {
                        check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                        check &= dot(hitPayload.normal, normal_G) > 0.707f;
                    }
                    if (!check) hitPayload.hit = -2.0f;
                }
            }

            if (hitPayload.hit > 0.0f) {
                float3 tangent, bitangent;
                orthonormalBasis(normal_G, tangent, bitangent);
                float3x3 TBN_G = float3x3(tangent, bitangent, normal_G);

                BSDFMaterial material = pushConst.sceneInfoAddress.materials[hitPayload.materialIndex];
                float3 bsdf = getBSDF(material, -sampleDir, normal_G, outgoing, TBN_G, true);
                float d = distance(samplePos_E, hitPayload.hitPos);
                float cosine = dot(normal_G, -sampleDir);
                float cosine_Light = dot(canonicalSampleData.normal, sampleDir);
                f += canonicalSampleData.radiance * bsdf * cosine * cosine_Light / max(d * d, 0.01f);
            }
        }

        f /= sampleCount;
        pj_hat = dot(f, 1.0f);
    }
    float canonicalSampleP_hat = WaveReadLaneFirst(yData_i.p_hat);
    float param = canonicalSampleP_hat / (canonicalSampleP_hat + (validCandidateCount - 1) * pj_hat);
    param = WaveActiveSum(param) - (Candidate_Samples_Count - validCandidateCount);       //1 + sum(j not in R)

    float mi = 0.0f;
    if (groupThreadIndex == 0) mi = param / validCandidateCount;
    //-------------------------------------------
    int candidateGlobalIndex = outgoingIndex * indivisibleNodeCount * Candidate_Samples_Count + indivisibleNodeIndex * Candidate_Samples_Count + candidateLocalIndex;
    if (adjClusterIndex >= indivisibleNodeCount || yData_i.w == 0.0f) {
        candidateSampleData2Buffer[candidateGlobalIndex] = { 0.0f, 0.0f };
        candidateSampleAABBBuffer[candidateGlobalIndex] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        return;
    }
    //-------------------------------------------
    float pc_hat = yData_i.p_hat;
    CandidateLeafNodeData nonCanonicalSampleData = canonicalSampleData;
    if (adjClusterIndex < indivisibleNodeCount && groupThreadIndex > 0) {
        nonCanonicalSampleData = candidateLeafNodeDataBuffer[adjClusterIndex * Candidate_Samples_Count + adjClusterCandidateLocalIndex];

        uint2 curClusterInfo = IndivisibleNodeInfoBuffer_G[indivisibleNodeIndex];
        OctreeNodeData_G_FzbPG curClusterData = OctreeNodeDataBuffer_G[curClusterInfo.x][curClusterInfo.y];
        float3 normal_G = SnapToDominantAxis(curClusterData.meanNormal.xyz);

        HitPayload_FzbWeight hitPayload;
        float3 f = 0.0f;
        for (int i = 0; i < sampleCount; ++i) {
            float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
            float3 samplePos_G = randXYZ * (curClusterData.aabb.maximum - curClusterData.aabb.minimum) + curClusterData.aabb.minimum;

            randXYZ = float3(rand(seed), rand(seed), rand(seed));
            float3 samplePos_E = randXYZ * (nonCanonicalSampleData.aabb.maximum - nonCanonicalSampleData.aabb.minimum) + nonCanonicalSampleData.aabb.minimum;
            float3 sampleDir = normalize(samplePos_G - samplePos_E);
            samplePos_E = getAABBPoint(nonCanonicalSampleData.aabb, samplePos_E, sampleDir);

            hitPayload.hit = -1.0f;
            hitPayload.sampleDir = sampleDir;

            bool hitTestAble = dot(nonCanonicalSampleData.normal, sampleDir) > 0.0f;
            hitTestAble &= dot(-sampleDir, normal_G) > 0.0f;
            if (hitTestAble) {
                RayDesc ray;
                ray.Direction = sampleDir;
                ray.Origin = samplePos_E + sampleDir * 0.00001f;
                ray.TMin = 0.001;
                ray.TMax = INFINITE;
                traceRay(ray, hitPayload);
                if (hitPayload.hit > 0.0f) {
                    bool check = checkOcclusion(samplePos_E, hitPayload.hitPos, curClusterData.aabb);
                    check &= checkPointInAABB2(hitPayload.hitPos, curClusterData.aabb);
                    check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                    check &= dot(hitPayload.normal, normal_G) > 0.707f;
                    if (!check) hitPayload.hit = -2.0f;
                }
            }

            if (hitPayload.hit > 0.0f) {
                float3 tangent, bitangent;
                orthonormalBasis(normal_G, tangent, bitangent);
                float3x3 TBN_G = float3x3(tangent, bitangent, normal_G);

                BSDFMaterial material = pushConst.sceneInfoAddress.materials[hitPayload.materialIndex];
                float3 bsdf = getBSDF(material, -sampleDir, normal_G, outgoing, TBN_G, true);
                float d = distance(samplePos_E, hitPayload.hitPos);
                float cosine = dot(normal_G, -sampleDir);
                float cosine_Light = dot(nonCanonicalSampleData.normal, sampleDir);
                f += nonCanonicalSampleData.radiance * bsdf * cosine * cosine_Light / max(d * d, 0.01f);
            }

            //if (pushConst.frameIndex == 23 && indivisibleNodeIndex == 49 && outgoingIndex == 0) {
            //    pc_hat = dot(f, 1.0f);
            //    printf("candidateIndex: %d\n wSum: %f, pc_hat: %f hit: %f\n E: normal: %f %f %f, pos: %f %f %f\n G: normal: %f %f %f, pos: %f %f %f\n E AABB: %f %f %f %f %f %f\n G AABB: %f %f %f %f %f %f\n sampleDir: %f %f %f\n", groupThreadIndex, 
            //        yData_i.w, pc_hat, hitPayload.hit,
            //        nonCanonicalSampleData.normal.x, nonCanonicalSampleData.normal.y, nonCanonicalSampleData.normal.z,
            //        samplePos_E.x, samplePos_E.y, samplePos_E.z,
            //        normal_G.x, normal_G.y, normal_G.z,
            //        samplePos_G.x, samplePos_G.y, samplePos_G.z,
            //        nonCanonicalSampleData.aabb.minimum.x, nonCanonicalSampleData.aabb.minimum.y, nonCanonicalSampleData.aabb.minimum.z,
            //        nonCanonicalSampleData.aabb.maximum.x, nonCanonicalSampleData.aabb.maximum.y, nonCanonicalSampleData.aabb.maximum.z,
            //        curClusterData.aabb.minimum.x, curClusterData.aabb.minimum.y, curClusterData.aabb.minimum.z,
            //        curClusterData.aabb.maximum.x, curClusterData.aabb.maximum.y, curClusterData.aabb.maximum.z,
            //        sampleDir.x, sampleDir.y, sampleDir.z
            //    );
            //}
        }

        f /= sampleCount;
        pc_hat = dot(f, 1.0f);
        mi = (validCandidateCount - 1) / validCandidateCount * yData_i.p_hat / (pc_hat + (validCandidateCount - 1) * yData_i.p_hat);
    }
    //-------------------------------------------
    float wi = mi * pc_hat / yData_i.p_hat * yData_i.w;
    if (!(wi >= 0.0f) || isinf(wi)) wi = 0.0f;

    CandidateSampleData_FzbPG ycData = { wi, pc_hat };
    candidateSampleData2Buffer[candidateGlobalIndex] = ycData;
    candidateSampleAABBBuffer[candidateGlobalIndex] = nonCanonicalSampleData.aabb;
#else
    printf("computeMain_SpatialReuse忘记写Candidate_Samples_Count != 32的情况了");
#endif
 
}
#else
[numthreads(Candidate_Samples_Count, 1, 1)]
[shader("compute")]
void computeMain_SpatialReuse(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int indivisibleNodeCount = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = indivisibleNodeCount;

    int indivisibleNodeIndex = threadGroupIndex / OUTGOING_COUNT_FZBPG;
    int candidateLocalIndex = groupThreadIndex;
    int outgoingIndex = threadGroupIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);

    uint adjClusterIndex = AdjClusterInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + candidateLocalIndex];
    OctreeNodeData_G_FzbPG adjClusterData;

    CandidateSampleData_FzbPG yData_i; CandidateLeafNodeData yNodeData_i; float3 normal_E;
    {
        yData_i.w = 0.0f;
        yData_i.p_hat = 1.0f;
        if (adjClusterIndex < indivisibleNodeCount) {
            uint2 adjClusterInfo = IndivisibleNodeInfoBuffer_G[adjClusterIndex];
            adjClusterData = OctreeNodeDataBuffer_G[adjClusterInfo.x][adjClusterInfo.y];

            int adjClusterCandidateLocalIndex = candidateSampleIndexBuffer[adjClusterIndex * OUTGOING_COUNT_FZBPG + outgoingIndex];
            int adjClusterCandidateGlobalStartIndex = outgoingIndex * indivisibleNodeCount * Candidate_Samples_Count + adjClusterIndex * Candidate_Samples_Count;
            yData_i = candidateSampleDataBuffer[adjClusterCandidateGlobalStartIndex + adjClusterCandidateLocalIndex];
            yData_i.w = candidateSamplePrefixWSumBuffer[adjClusterCandidateGlobalStartIndex + Candidate_Samples_Count - 1]; // wSum
            yData_i.p_hat = max(yData_i.p_hat, 1e-6f);

            yNodeData_i = candidateLeafNodeDataBuffer[adjClusterIndex * Candidate_Samples_Count + adjClusterCandidateLocalIndex];

            int subTreeNodeIndex_layer = candidateSubTreeInfoBuffer[adjClusterIndex * Candidate_Samples_Count + adjClusterCandidateLocalIndex].subTreeNodeIndex_layer;
            int layerIndex = subTreeNodeIndex_layer & 15;
            int nodeIndex = subTreeNodeIndex_layer >> 4;
            int layerNodeCount = (1 << (3 * layerIndex));
            int normalIndex = nodeIndex / layerNodeCount;
            float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
            normal_E.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
            normal_E.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
            normal_E.z = ((normalIndex & 4u) >> 2) * plusMinusSign;
        }
    }

    float poi_hat = 0.0f, poj_hat_sum = 0.0f, pci_hat = 0.0f;
    {
        HitPayload_FzbWeight hitPayload; RayDesc ray;
        [unroll]
        for (int i = 0; i < Candidate_Samples_Count; ++i) {
            AABB aabb_G; float3 normal_Gj;
            {
                aabb_G.maximum = WaveReadLaneAt(adjClusterData.aabb.maximum, i);
                aabb_G.minimum = WaveReadLaneAt(adjClusterData.aabb.minimum, i);
                normal_Gj = WaveReadLaneAt(adjClusterData.meanNormal.xyz, i);
            }
            if (WaveReadLaneAt(adjClusterIndex, i) >= indivisibleNodeCount) continue;
            if (adjClusterIndex >= indivisibleNodeCount) continue;

            float3 samplePos_G = (aabb_G.maximum + aabb_G.minimum) * 0.5f;
            float3 normal_G = SnapToDominantAxis(normal_Gj);

            float3 samplePos_E = (yNodeData_i.aabb.maximum + yNodeData_i.aabb.minimum) * 0.5f;
            float3 sampleDir = normalize(samplePos_G - samplePos_E);
            samplePos_E = getAABBPoint(yNodeData_i.aabb, samplePos_E, sampleDir);

            hitPayload.hit = -1.0f;
            hitPayload.sampleDir = sampleDir;

            bool hitTestAble = dot(normal_E, sampleDir) > 0.0f;
            hitTestAble &= dot(-sampleDir, normal_G) > 0.0f;
            if (hitTestAble) {
                ray.Direction = sampleDir;
                ray.Origin = samplePos_E + sampleDir * 0.00001f;
                ray.TMin = 0.001;
                ray.TMax = INFINITE;
                traceRay(ray, hitPayload);
                if (hitPayload.hit > 0.0f) {
                    bool check = checkOcclusion(samplePos_E, hitPayload.hitPos, aabb_G);
                    if (checkPointInAABB2(hitPayload.hitPos, aabb_G)) {
                        check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                        check &= dot(hitPayload.normal, normal_G) > 0.707f;
                    }
                    if (!check) hitPayload.hit = -2.0f;
                }
            }
            if (hitPayload.hit <= 0.0f) {
                if (i == candidateLocalIndex) break;
                continue;
            }

            float3 tangent, bitangent;
            orthonormalBasis(normal_G, tangent, bitangent);
            float3x3 TBN_G = float3x3(tangent, bitangent, normal_G);

            BSDFMaterial material = pushConst.sceneInfoAddress.materials[hitPayload.materialIndex];
            float3 bsdf = getBSDF(material, -sampleDir, normal_G, outgoing, TBN_G, true);
            float d = distance(samplePos_E, hitPayload.hitPos);
            float cosine = dot(normal_G, -sampleDir);
            float cosine_Light = dot(normal_E, sampleDir);
            float3 f = yNodeData_i.radiance * bsdf * cosine * cosine_Light / max(d * d, 0.01f);

            float poj_hat = dot(f, 1.0f);
            if (i == candidateLocalIndex) poi_hat = poj_hat;
            poj_hat_sum += poj_hat;

            if (i == 0) pci_hat = poj_hat;
        }
    }

    float mi = poi_hat / max(poj_hat_sum, 1e-6f);
    float wi = mi * pci_hat / yData_i.p_hat * yData_i.w;

    CandidateSampleData_FzbPG ycData = { wi, pci_hat };
    int candidateGlobalIndex = outgoingIndex * indivisibleNodeCount * Candidate_Samples_Count + indivisibleNodeIndex * Candidate_Samples_Count + candidateLocalIndex;
    candidateSampleData2Buffer[candidateGlobalIndex] = ycData;
    candidateSampleAABBBuffer[candidateGlobalIndex] = yNodeData_i.aabb;
    candidateSampleRadianceBuffer[candidateGlobalIndex] = yNodeData_i.radiance;
}
#endif
#endif