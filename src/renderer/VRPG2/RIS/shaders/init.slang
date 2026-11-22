#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<RISPushConstant_VRPG2> pushConst;
[[vk::binding(RISBindingPoints_VRPG2::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];
[[vk::binding(RISBindingPoints_VRPG2::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];

[[vk::binding(RISBindingPoints_VRPG2::eClusterInfo_SVO)]] RWStructuredBuffer<OctreeNodeInfo_VRPG2> ClusterInfo_SVO;

[[vk::binding(RISBindingPoints_VRPG2::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eRISGlobalInfo)]] RWStructuredBuffer<RISGlobalInfo_VRPG2, ScalarDataLayout> risGlobalInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eClusterPairInfo)]] RWStructuredBuffer<ClusterPairInfo_VRPG2, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eClusterPairHitInfo)]] RWStructuredBuffer<ClusterPairHitInfo_VRPG2, ScalarDataLayout> clusterPairHitTestInfoBuffer;

[[vk::binding(RISBindingPoints_VRPG2::eCandidatePoolInfo)]] RWStructuredBuffer<CandidatePoolInfo_VRPG2> candidatePoolInfoBuffer;
#if VRPG_Candidate_Samples_SampleMode == 1
[[vk::binding(RISBindingPoints_VRPG2::eCandidatePoolNodeInfo)]] RWStructuredBuffer<CandidatePoolNodeInfo_VRPG2, ScalarDataLayout> candidatePoolNodeInfoBuffer;
#endif

[[vk::binding(RISBindingPoints_VRPG2::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG2, ScalarDataLayout> candidateSubTreeInfoBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData_VRPG2, ScalarDataLayout> candidateLeafNodeDataBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_VRPG2, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(RISBindingPoints_VRPG2::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;
//----------------------------------------------------------------------------------------------------------------------------------------------------
groupshared uint groupClusterCount;
//----------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_init(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
    GroupMemoryBarrierWithGroupSync();

    int clusterCount = groupClusterCount;
    if (clusterCount == 0) {
        if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = 0;
        return;
    }
    int threadTotalCount = OUTGOING_COUNT_VRPG2 * clusterCount * Candidate_Samples_Count;
    if (threadIndex >= threadTotalCount) return;

    CandidateSampleData_VRPG2 candidateData;
    candidateData.w = 0.0f;
    candidateData.p_hat = 0.0f;
    candidateSampleDataBuffer[threadIndex] = candidateData;
    if (threadIndex < clusterCount * Candidate_Samples_Count) candidateSubTreeInfoBuffer[threadIndex].subTreeNodeIndex_layer = -1;

    if (threadIndex < clusterCount) {
        candidatePoolInfoBuffer[threadIndex].candidateNodeCount = 1;

        int candidateNodeIndex = threadIndex * Candidate_Samples_Count + 0;
        CandidatePoolNodeInfo_VRPG2 nodeInfo;
        nodeInfo.valid = true;
        nodeInfo.layer_E = -1;
        nodeInfo.nodeIndex_E = 0;
        candidatePoolNodeInfoBuffer[candidateNodeIndex] = nodeInfo;
    }

    if (threadIndex == 0) {
        risGlobalInfoBuffer[0].clusterPairCount = 0;
        DispatchIndirectCommand cmd;
        #if VRPG_Candidate_Samples_SampleMode == 1
        cmd.x = clusterCount;
        #else
        cmd.x = (clusterCount + GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE - 1) / GET_CANDIDATE_NODES_CS_THREADGROUP_SIZE;
        #endif
        cmd.y = 1; cmd.z = 1;
        GlobalInfoBuffer[0].cmd = cmd;
        bool hasLightNode = false;
        for (int i = 0; i < 6; ++i) {
            if (any(LightNodeData_Octree[0][i].radiance > 0.0f)) {
                hasLightNode = true;
                break;
            }
        }
        if (!hasLightNode) GlobalInfoBuffer[0].cmd.x = 0;
    }

    #ifndef NDEBUG
    if (threadIndex < MAX_OCTREE_LAYER_VRPG2) pushConst.globalInfoAddress.candidateNodeCounts[threadIndex] = 0;
    #endif
}