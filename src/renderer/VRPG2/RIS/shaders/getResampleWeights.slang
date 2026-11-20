#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

// Second half of the RIS chain: descend every cut node to a light leaf, test the visibility of
// every (geometric cluster, candidate, outgoing bin) triple and write the resampling weights plus
// their prefix sums. The math follows VRPG/Octree/shaders/GetClusterWeight.slang.
//
// Layouts (shared by both candidate-selection modes):
//   clusterPairInfoBuffer[label * M + i]                     (written by getCandidateNodes.slang)
//   candidatePoolInfoBuffer[label].candidateNodeCount
//   candidateSubTreeInfoBuffer[label * M + i]
//   candidateLeafNodeDataBuffer[label * M + i]
//   clusterPairHitInfoBuffer[(label * M + i) * S + s]
//   candidateSampleDataBuffer[(k * ClusterCount + label) * M + i]
//   candidateSamplePrefixWSumBuffer[(k * ClusterCount + label) * M + i]

[[vk::push_constant]] ConstantBuffer<RISPushConstant_VRPG2, ScalarDataLayout> pushConst;

[[vk::binding(RISBindingPoints_VRPG2::eClusterData_Octree)]] StructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_SVO)]] StructuredBuffer<OctreeNodeInfo_VRPG2> ClusterInfo_SVO;
[[vk::binding(RISBindingPoints_VRPG2::eLightNodeData_Octree)]] StructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo_VRPG2, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eClusterPairHitInfo)]] RWStructuredBuffer<ClusterPairHitInfo_VRPG2, ScalarDataLayout> clusterPairHitInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidatePoolInfo)]] StructuredBuffer<CandidatePoolInfo_VRPG2> candidatePoolInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG2, ScalarDataLayout> candidateSubTreeInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData_VRPG2, ScalarDataLayout> candidateLeafNodeDataBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_VRPG2, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;

// topLevelAS comes from pathTracingCommon.slang (dynamic set 1, binding eTlas_PT)

#define CANDIDATE_SAMPLES Candidate_Samples_Count_VRPG2
#define CLUSTER_MAX_COUNT ClusterMaxCount_VRPG
#define HITTEST_COUNT HITTEST_COUNT_VRPG2
#define OUTGOING_COUNT OUTGOING_COUNT_VRPG2

uint candidateSlotBase(uint outgoingIndex, uint label) {
    // the stride has to match the reader (guiding.slang): the live cluster count, not the capacity
    return (outgoingIndex * GlobalInfoBuffer[0].ClusterCount + label) * CANDIDATE_SAMPLES;
}

//-----------------------------------------------------------------------------------------------
// descend one cut node to a light leaf, following the children by radiance importance
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_GET_CANDIDATE_LEAF_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getCandidateLeafNodes(uint threadIndex: SV_DispatchThreadID) {
    uint label = threadIndex / CANDIDATE_SAMPLES;
    uint candidate = threadIndex % CANDIDATE_SAMPLES;
    if (label >= CLUSTER_MAX_COUNT) return;

    uint candidateIndex = label * CANDIDATE_SAMPLES + candidate;
    CandidateSubTreeInfo_VRPG2 emptySubTree;
    emptySubTree.ESum = 0.0f;
    emptySubTree.subTreeNodeIndex_layer = -1;

    if (candidate >= uint(candidatePoolInfoBuffer[label].candidateNodeCount)) {
        // keep the terminator the reader (getPgPdf2) expects
        candidateSubTreeInfoBuffer[candidateIndex] = emptySubTree;
        return;
    }

    ClusterPairInfo_VRPG2 pair = clusterPairInfoBuffer[candidateIndex];
    if (pair.layerIndex_E < 0) {
        // the cut never left the root: there is no light subtree for this cluster
        pair.leafNodeIndex_E = -1;
        clusterPairInfoBuffer[candidateIndex] = pair;
        candidateSubTreeInfoBuffer[candidateIndex] = emptySubTree;
        return;
    }

    // ESum is the radiance sum of the *cut* subtree: the probability of reaching the sampled leaf
    // is radiance(leaf) / ESum, which is exactly leafNodePdf.
    LightNodeData_Octree_VRPG cutNode = LightNodeData_Octree[pair.layerIndex_E][pair.nodeIndex_E];
    emptySubTree.ESum = dot(cutNode.radiance, float3(1.0f));
    emptySubTree.subTreeNodeIndex_layer = (pair.nodeIndex_E << 4) | pair.layerIndex_E;
    candidateSubTreeInfoBuffer[candidateIndex] = emptySubTree;

    int layer = pair.layerIndex_E;
    int node = pair.nodeIndex_E;
    float pdf = 1.0f;
    uint seed = xxhash32(uint3(candidateIndex, 0u, (uint)pushConst.randomFrameIndex));

    int lightMaxLayer = int(pushConst.lightOctreeMaxLayer);
    bool valid = emptySubTree.ESum > 0.0f;
    while (valid && layer < lightMaxLayer) {
        int childLayer = layer + 1;
        int childStart = node * 8;

        float weights[8];
        float weightSum = 0.0f;
        int childCount = 0;
        for (int c = 0; c < 8; ++c) {
            if (childLayer == 0 && c >= 6) break;
            int child = childStart + c;
            float importance = dot(LightNodeData_Octree[childLayer][child].radiance, float3(1.0f));
            weights[c] = importance;
            weightSum += importance;
            ++childCount;
        }
        if (weightSum <= 0.0f) {
            valid = false;
            break;
        }

        float r = rand(seed) * weightSum;
        int selected = 0;
        float accumulated = 0.0f;
        for (int c = 0; c < childCount; ++c) {
            selected = c;
            accumulated += weights[c];
            if (accumulated >= r) break;
        }
        pdf *= weights[selected] / weightSum;
        layer = childLayer;
        node = childStart + selected;
    }

    if (!valid) {
        pair.leafNodeIndex_E = -1;
        pair.leafNodePdf = 1.0f;
        clusterPairInfoBuffer[candidateIndex] = pair;
        return;
    }

    pair.leafNodeIndex_E = node;
    pair.leafNodePdf = max(pdf, 1e-6f);
    clusterPairInfoBuffer[candidateIndex] = pair;

    LightNodeData_Octree_VRPG leaf = LightNodeData_Octree[layer][node];
    CandidateLeafNodeData_VRPG2 leafData;
    leafData.aabb = leaf.aabb;
    leafData.radiance = leaf.radiance;
    candidateLeafNodeDataBuffer[candidateIndex] = leafData;
}

//-----------------------------------------------------------------------------------------------
// S visibility samples per candidate: light leaf <-> geometric cluster
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_CLUSTER_PAIR_HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_clusterPairHitTest(uint threadIndex: SV_DispatchThreadID) {
    uint label = threadIndex / CANDIDATE_SAMPLES;
    uint candidate = threadIndex % CANDIDATE_SAMPLES;
    if (label >= CLUSTER_MAX_COUNT) return;

    uint pairIndex = threadIndex;
    uint hitStart = pairIndex * HITTEST_COUNT;
    if (candidate >= uint(candidatePoolInfoBuffer[label].candidateNodeCount)) {
        for (int s = 0; s < HITTEST_COUNT; ++s) clusterPairHitInfoBuffer[hitStart + s].isHit = 0.0f;
        return;
    }

    ClusterPairInfo_VRPG2 pair = clusterPairInfoBuffer[label * CANDIDATE_SAMPLES + candidate];
    if (pair.leafNodeIndex_E < 0) {
        for (int s = 0; s < HITTEST_COUNT; ++s) clusterPairHitInfoBuffer[hitStart + s].isHit = 0.0f;
        return;
    }

    OctreeNodeInfo_VRPG2 clusterNode = ClusterInfo_SVO[label];
    ClusterData_Octree_VRPG nodeG = ClusterData_Octree[clusterNode.layerIndex][clusterNode.nodeIndex];
    LightNodeData_Octree_VRPG leaf = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][pair.leafNodeIndex_E];

    // the normal partition the geometric cluster belongs to
    uint octreeLayerNode = 1u << (3 * pair.layerIndex_G);
    uint normalIndex = uint(pair.nodeIndex_G) / octreeLayerNode;
    float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
    float3 normalG;
    normalG.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
    normalG.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
    normalG.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

    for (int s = 0; s < HITTEST_COUNT; ++s) {
        uint hitSeed = xxhash32(uint3(pairIndex, (uint)s, (uint)pushConst.randomFrameIndex));

        float3 randXYZ = float3(rand(hitSeed), rand(hitSeed), rand(hitSeed));
        float3 samplePoint_G = randXYZ * (nodeG.aabb.maximum - nodeG.aabb.minimum) + nodeG.aabb.minimum;

        randXYZ = float3(rand(hitSeed), rand(hitSeed), rand(hitSeed));
        float3 samplePoint_E = randXYZ * (leaf.aabb.maximum - leaf.aabb.minimum) + leaf.aabb.minimum;
        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        // move the light sample onto the light AABB surface first, otherwise the emitter blocks itself
        samplePoint_E = getAABBPoint(leaf.aabb, samplePoint_E, sampleDir);

        ClusterPairHitInfo_VRPG2 info;
        info.pos_G = samplePoint_G;
        info.normal_G = normalG;
        info.materialIndex = -1;
        info.sampleDir = sampleDir;
        info.d = distance(samplePoint_E, samplePoint_G);
        info.cosine = 0.0f;
        info.cosine_Light = 0.0f;
        info.isHit = -1.0f;

        bool hitTestAble = dot(leaf.meanNormal.xyz, sampleDir) > 0.0f;
        hitTestAble = hitTestAble && dot(-sampleDir, normalG) > 0.0f;
        if (hitTestAble) {
            RayDesc ray;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.Direction = sampleDir;
            ray.TMin = 0.001f;
            ray.TMax = INFINITE;

            RayQuery<RAY_FLAG_NONE> query;
            query.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
            while (query.Proceed()) {
                if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
                    query.CommitNonOpaqueTriangleHit();
            }

            if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
                float2 bary2 = query.CommittedTriangleBarycentrics();
                float4x3 objectToWorld = query.CommittedObjectToWorld4x3();
                float4x3 worldToObject = query.CommittedWorldToObject4x3();
                int instanceIndex = query.CommittedInstanceIndex();

                SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
                Instance instance = sceneInfo.instances[instanceIndex];

                float3 triPos[3] = query.CommittedGetIntersectionTriangleVertexPositions();
                float3 localPosition = triPos[0] * (1.0f - bary2.x - bary2.y) + triPos[1] * bary2.x + triPos[2] * bary2.y;
                float3 hitPos = mul(float4(localPosition, 1.0), objectToWorld).xyz;
                float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
                float3 hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
                if (dot(hitNormal, sampleDir) > 0.0f) hitNormal = -hitNormal;

                bool check = checkPointInAABB2(hitPos, nodeG.aabb);
                check = check && dot(hitNormal, -sampleDir) > 0.0f;
                check = check && all(GetPrincipalAxis(hitNormal) == GetPrincipalAxis(normalG));

                info.pos_G = hitPos;
                info.normal_G = hitNormal;
                info.materialIndex = instance.materialIndex;
                info.isHit = check ? 1.0f : -2.0f;
                info.cosine = max(dot(-sampleDir, hitNormal), 0.0f);
                info.cosine_Light = max(dot(sampleDir, leaf.meanNormal.xyz), 0.0f);
            }
        }

        clusterPairHitInfoBuffer[hitStart + s] = info;
    }
}

//-----------------------------------------------------------------------------------------------
// target density p_hat and resampling weight w = p_hat / pdf for every (outgoing, candidate)
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_GET_CANDIDATE_NODE_WEIGHTS_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodeWeight(uint threadIndex: SV_DispatchThreadID) {
    uint clusterCount = GlobalInfoBuffer[0].ClusterCount;
    if (clusterCount == 0) return;
    uint pairIndex = threadIndex / OUTGOING_COUNT;
    if (pairIndex >= CLUSTER_MAX_COUNT * CANDIDATE_SAMPLES) return;

    uint label = pairIndex / CANDIDATE_SAMPLES;
    uint candidate = pairIndex % CANDIDATE_SAMPLES;
    uint outgoingIndex = threadIndex % OUTGOING_COUNT;
    if (label >= clusterCount) return;

    uint slot = candidateSlotBase(outgoingIndex, label) + candidate;
    if (candidate >= uint(candidatePoolInfoBuffer[label].candidateNodeCount)) {
        candidateSampleDataBuffer[slot].w = 0.0f;
        candidateSampleDataBuffer[slot].p_hat = 0.0f;
        return;
    }

    ClusterPairInfo_VRPG2 pair = clusterPairInfoBuffer[label * CANDIDATE_SAMPLES + candidate];
    if (pair.leafNodeIndex_E < 0) {
        candidateSampleDataBuffer[slot].w = 0.0f;
        candidateSampleDataBuffer[slot].p_hat = 0.0f;
        return;
    }

    CandidateLeafNodeData_VRPG2 leaf = candidateLeafNodeDataBuffer[label * CANDIDATE_SAMPLES + candidate];

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 outgoing = mul(getOutgoing_VRPG2(outgoingIndex, OUTGOING_COUNT), pushConst.randomRotateMatrix);

    float3 f = float3(0.0f);
    uint hitStart = pairIndex * HITTEST_COUNT;
    for (int s = 0; s < HITTEST_COUNT; ++s) {
        ClusterPairHitInfo_VRPG2 info = clusterPairHitInfoBuffer[hitStart + s];
        if (info.isHit <= 0.0f) continue;
        if (info.materialIndex < 0) continue;

        BSDFMaterial material = sceneInfo.materials[info.materialIndex];
        float3 tangent, bitangent;
        orthonormalBasis(info.normal_G, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, info.normal_G);

        float3 bsdf = getBSDF(material, -info.sampleDir, info.normal_G, outgoing, TBN_G, true);
        float d = info.d;
        f += leaf.radiance * bsdf * info.cosine * info.cosine_Light / max(d * d, 1e-3f);
    }
    f /= float(HITTEST_COUNT);

    float p_hat = dot(f, float3(1.0f));
    candidateSampleDataBuffer[slot].p_hat = p_hat;
    candidateSampleDataBuffer[slot].w = p_hat / max(pair.leafNodePdf, 1e-6f);
}

//-----------------------------------------------------------------------------------------------
// prefix sum of w over the M candidates of every (outgoing, cluster)
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_PREFIX_WSUM_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createCandidatePrefixWSum(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= OUTGOING_COUNT * CLUSTER_MAX_COUNT) return;
    uint clusterCount = GlobalInfoBuffer[0].ClusterCount;
    uint outgoingIndex = threadIndex / CLUSTER_MAX_COUNT;
    uint label = threadIndex % CLUSTER_MAX_COUNT;
    if (label >= clusterCount) return;

    uint slotBase = candidateSlotBase(outgoingIndex, label);
    int candidateCount = candidatePoolInfoBuffer[label].candidateNodeCount;
    float accumulated = 0.0f;
    for (int i = 0; i < CANDIDATE_SAMPLES; ++i) {
        if (i < candidateCount) accumulated += candidateSampleDataBuffer[slotBase + i].w;
        candidateSamplePrefixWSumBuffer[slotBase + i] = accumulated;
    }
}














