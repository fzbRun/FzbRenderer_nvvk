#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

// Candidate selection over the light tree: builds, for every indivisible geometric cluster, a cut
// of at most Candidate_Samples_Count_VRPG2 light nodes. Two strategies share the same output
// layout (clusterPairInfoBuffer[label * M + i], candidatePoolInfoBuffer[label].candidateNodeCount):
//
//   VRPG_Candidate_Samples_SampleMode == 1 : bounded pool light cut. The pool is expanded for a
//       fixed number of passes; when the pool of a cluster would exceed M, the expansion with the
//       smallest upper bound is rolled back (the parent node stays in the cut).
//   VRPG_Candidate_Samples_SampleMode == 0 : greedy light cut. The node with the largest error is
//       expanded until the relative error drops below LIGHTCUT_RELATIVE_ERROR_VRPG2.
//
// No label information is needed for the light tree: the cut only reads the light node data
// (radiance / mean normal / aabb) and stops at the leaf layer or when the budget is exhausted.

[[vk::push_constant]] ConstantBuffer<RISPushConstant_VRPG2, ScalarDataLayout> pushConst;

[[vk::binding(RISBindingPoints_VRPG2::eClusterData_Octree)]] StructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_SVO)]] StructuredBuffer<OctreeNodeInfo_VRPG2> ClusterInfo_SVO;
[[vk::binding(RISBindingPoints_VRPG2::eLightNodeData_Octree)]] StructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo_VRPG2, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG2, ScalarDataLayout> candidateSubTreeInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eCandidatePoolInfo)]] RWStructuredBuffer<CandidatePoolInfo_VRPG2> candidatePoolInfoBuffer;
#if VRPG_Candidate_Samples_SampleMode == 1
[[vk::binding(RISBindingPoints_VRPG2::eCandidatePoolNodeInfo)]] RWStructuredBuffer<CandidatePoolNodeInfo_VRPG2> candidatePoolNodeInfoBuffer;
#endif

#define CANDIDATE_SAMPLES Candidate_Samples_Count_VRPG2
#define CLUSTER_MAX_COUNT ClusterMaxCount_VRPG

// Conservative upper bound of the contribution of a light node seen from a geometric cluster.
float lightNodeUpperBound(float3 posG, float3 normalG, LightNodeData_Octree_VRPG node) {
    float intensity = dot(node.radiance, float3(1.0f));
    if (intensity <= 0.0f) return 0.0f;

    float3 center = (node.aabb.maximum + node.aabb.minimum) * 0.5f;
    // delta points from the light node to the geometric cluster; every cosine below uses the
    // direction from the cluster towards the light (-delta), exactly like MaxCosToAABB in VRPG.
    float3 delta = posG - center;
    float diagonal = length(node.aabb.maximum - node.aabb.minimum);
    float distanceSquared = max(dot(delta, delta), diagonal * diagonal + 1e-6f);

    float cosG = max(0.0f, dot(normalize(-delta + float3(1e-9f)), normalize(normalG + float3(1e-9f))));
    // cos at the light side: the node is only visible from the front of its own normal
    float cosL = 1.0f;
    if (length(node.meanNormal.xyz) > 1e-6f) {
        // light-side cosine: the node has to emit towards the cluster, i.e. along +delta
        cosL = max(0.0f, dot(normalize(node.meanNormal.xyz), normalize(delta + float3(1e-9f))));
        if (cosL <= 0.0f) return 0.0f;
    }
    return intensity * cosG * cosL / distanceSquared;
}

void writeClusterPair(uint label, int candidateIndex, int layerE, int nodeE) {
    ClusterPairInfo_VRPG2 pair;
    pair.layerIndex_G = ClusterInfo_SVO[label].layerIndex;
    pair.nodeIndex_G = ClusterInfo_SVO[label].nodeIndex;
    pair.indivisibleNodeLabel = label;
    pair.layerIndex_E = layerE;
    pair.nodeIndex_E = nodeE;
    pair.candidateNodeIndex = candidateIndex;
    pair.leafNodeIndex_E = -1;
    pair.leafNodePdf = 1.0f;
    clusterPairInfoBuffer[label * CANDIDATE_SAMPLES + candidateIndex] = pair;
}

#if VRPG_Candidate_Samples_SampleMode == 1
//-----------------------------------------------------------------------------------------------
// Mode 1: bounded pool light cut
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initCandidatePool(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= CLUSTER_MAX_COUNT) return;

    candidatePoolInfoBuffer[threadIndex].candidateNodeCount = 0;
    if (threadIndex >= GlobalInfoBuffer[0].ClusterCount) return;

    for (int i = 0; i < CANDIDATE_SAMPLES; ++i) {
        clusterPairInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].leafNodeIndex_E = -1;
        // -1 is the "empty slot" terminator used by getPgPdf2 in guiding.slang
        candidateSubTreeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].ESum = 0.0f;
        candidateSubTreeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].subTreeNodeIndex_layer = -1;
    }

    candidatePoolInfoBuffer[threadIndex].candidateNodeCount = 1;
    CandidatePoolNodeInfo_VRPG2 root;
    root.noExpand = 0;
    root.layerIndex_E = -1;
    root.nodeIndex_E = 0;
    candidatePoolNodeInfoBuffer[threadIndex * CANDIDATE_SAMPLES] = root;
}

// One expansion pass. A node is expanded into its children unless the cluster budget would be
// exceeded; rolled back nodes stay in the cut (noExpand = 1).
[numthreads(RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_expandCandidatePool(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= CLUSTER_MAX_COUNT) return;
    int poolCount = candidatePoolInfoBuffer[threadIndex].candidateNodeCount;
    if (poolCount <= 0) return;

    ClusterData_Octree_VRPG nodeG = ClusterData_Octree[ClusterInfo_SVO[threadIndex].layerIndex][ClusterInfo_SVO[threadIndex].nodeIndex];
    float3 posG = (nodeG.aabb.maximum + nodeG.aabb.minimum) * 0.5f;

    CandidatePoolNodeInfo_VRPG2 newPool[CANDIDATE_SAMPLES];
    int newCount = 0;

    for (int i = 0; i < poolCount; ++i) {
        CandidatePoolNodeInfo_VRPG2 item = candidatePoolNodeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i];
        int childLayer = item.layerIndex_E + 1;

        if (item.noExpand || childLayer > int(pushConst.lightOctreeMaxLayer)) {
            if (newCount < CANDIDATE_SAMPLES) newPool[newCount++] = item;
            continue;
        }

        // gather the valid children together with their upper bound
        int childIndex[8];
        float childBound[8];
        int validChildCount = 0;
        float boundSum = 0.0f;
        for (int c = 0; c < 8; ++c) {
            if (childLayer == 0 && c >= 6) break;
            int child = item.nodeIndex_E * 8 + c;
            LightNodeData_Octree_VRPG childData = LightNodeData_Octree[childLayer][child];
            float bound = lightNodeUpperBound(posG, nodeG.meanNormal.xyz, childData);
            if (bound <= 0.0f) continue;
            childIndex[validChildCount] = child;
            childBound[validChildCount] = bound;
            boundSum += bound;
            ++validChildCount;
        }

        if (validChildCount == 0) {
            if (newCount < CANDIDATE_SAMPLES) newPool[newCount++] = item;
            continue;
        }
        if (newCount + validChildCount > CANDIDATE_SAMPLES) {
            item.noExpand = 1;
            if (newCount < CANDIDATE_SAMPLES) newPool[newCount++] = item;
            continue;
        }
        for (int c = 0; c < validChildCount; ++c) {
            CandidatePoolNodeInfo_VRPG2 child;
            child.noExpand = 0;
            child.layerIndex_E = childLayer;
            child.nodeIndex_E = childIndex[c];
            newPool[newCount++] = child;
        }
    }

    candidatePoolInfoBuffer[threadIndex].candidateNodeCount = newCount;
    for (int i = 0; i < newCount; ++i) candidatePoolNodeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i] = newPool[i];
}

// Store the final pool as cluster pairs.
[numthreads(RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_emitClusterPairs(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= CLUSTER_MAX_COUNT) return;
    int poolCount = candidatePoolInfoBuffer[threadIndex].candidateNodeCount;
    for (int i = 0; i < poolCount; ++i) {
        CandidatePoolNodeInfo_VRPG2 item = candidatePoolNodeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i];
        writeClusterPair(threadIndex, i, item.layerIndex_E, item.nodeIndex_E);
    }
}

#else
//-----------------------------------------------------------------------------------------------
// Mode 0: greedy light cut driven by a relative error threshold
//-----------------------------------------------------------------------------------------------
[numthreads(RIS_INIT_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initCandidatePool(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= CLUSTER_MAX_COUNT) return;
    candidatePoolInfoBuffer[threadIndex].candidateNodeCount = 0;
    if (threadIndex >= GlobalInfoBuffer[0].ClusterCount) return;

    for (int i = 0; i < CANDIDATE_SAMPLES; ++i) {
        clusterPairInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].leafNodeIndex_E = -1;
        candidateSubTreeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].ESum = 0.0f;
        candidateSubTreeInfoBuffer[threadIndex * CANDIDATE_SAMPLES + i].subTreeNodeIndex_layer = -1;
    }
}

[numthreads(RIS_GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_lightCut(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex >= GlobalInfoBuffer[0].ClusterCount) return;

    ClusterData_Octree_VRPG nodeG = ClusterData_Octree[ClusterInfo_SVO[threadIndex].layerIndex][ClusterInfo_SVO[threadIndex].nodeIndex];
    float3 posG = (nodeG.aabb.maximum + nodeG.aabb.minimum) * 0.5f;

    LightCutNode_VRPG2 cutNodes[CANDIDATE_SAMPLES];
    int cutCount = 1;
    cutNodes[0].layerIndex = -1;
    cutNodes[0].nodeIndex = 0;
    cutNodes[0].error = 0.0f;
    cutNodes[0].I = 0.0f;

    float intensitySum = 0.0f;
    int iteration = LIGHTCUT_MAX_ITERATION_VRPG2;
    while (iteration-- > 0) {
        int maxErrorIndex = -1;
        float maxError = -3.4e38f;
        for (int i = 0; i < cutCount; ++i) {
            if (cutNodes[i].layerIndex >= int(pushConst.lightOctreeMaxLayer)) continue;
            if (cutNodes[i].error > maxError) { maxError = cutNodes[i].error; maxErrorIndex = i; }
        }
        // the root has no error yet: force the first expansion
        if (maxErrorIndex < 0 && cutCount == 1 && cutNodes[0].layerIndex < 0) maxErrorIndex = 0;
        if (maxErrorIndex < 0) break;

        int childLayer = cutNodes[maxErrorIndex].layerIndex + 1;
        int childStart = cutNodes[maxErrorIndex].nodeIndex * 8;

        int childIndex[8];
        float childError[8];
        int validChildCount = 0;
        for (int c = 0; c < 8; ++c) {
            if (childLayer == 0 && c >= 6) break;
            int child = childStart + c;
            LightNodeData_Octree_VRPG childData = LightNodeData_Octree[childLayer][child];
            float bound = lightNodeUpperBound(posG, nodeG.meanNormal.xyz, childData);
            if (bound <= 0.0f) continue;
            childIndex[validChildCount] = child;
            childError[validChildCount] = bound;
            ++validChildCount;
        }
        if (validChildCount == 0) { cutNodes[maxErrorIndex].error = 0.0f; continue; }
        if (childLayer == 0) {
            for (int c = 0; c < validChildCount; ++c) intensitySum += dot(LightNodeData_Octree[childLayer][childIndex[c]].radiance, float3(1.0f));
        }

        // replace the expanded node by its children (first child reuses the slot)
        int first = maxErrorIndex;
        if (cutCount - 1 + validChildCount > CANDIDATE_SAMPLES) break;
        for (int c = 0; c < validChildCount; ++c) {
            int slot = (c == 0) ? first : cutCount++;
            cutNodes[slot].layerIndex = childLayer;
            cutNodes[slot].nodeIndex = childIndex[c];
            cutNodes[slot].error = childError[c];
            cutNodes[slot].I = dot(LightNodeData_Octree[childLayer][childIndex[c]].radiance, float3(1.0f));
        }

        float errorThreshold = LIGHTCUT_RELATIVE_ERROR_VRPG2 * max(intensitySum, 1e-6f);
        if (maxError <= errorThreshold && maxErrorIndex != 0) break;
    }

    candidatePoolInfoBuffer[threadIndex].candidateNodeCount = cutCount;
    for (int i = 0; i < cutCount; ++i) writeClusterPair(threadIndex, i, cutNodes[i].layerIndex, cutNodes[i].nodeIndex);
}
#endif
