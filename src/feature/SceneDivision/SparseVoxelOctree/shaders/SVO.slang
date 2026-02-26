#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOPushConstant> pushConst;
[[vk::binding(BindingPoints_SVO::eOctreeArray_G_SVO)]] StructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_SVO::eOctreeArray_E_SVO)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];
[[vk::binding(BindingPoints_SVO::eSVOArray_G_SVO)]] RWStructuredBuffer<SVONodeData_G, ScalarDataLayout> SVOArray_G[];
[[vk::binding(BindingPoints_SVO::eSVOArray_E_SVO)]] RWStructuredBuffer<SVONodeData_E, ScalarDataLayout> SVOArray_E[];
[[vk::binding(BindingPoints_SVO::eSVOLayerInfos_G_SVO)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_G;
[[vk::binding(BindingPoints_SVO::eSVOLayerInfos_E_SVO)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_E;
[[vk::binding(BindingPoints_SVO::eSVODivisibleNodeIndices_G_SVO)]] RWStructuredBuffer<uint> SVODivisibleNodeIndices_G;
[[vk::binding(BindingPoints_SVO::eSVODivisibleNodeIndices_E_SVO)]] RWStructuredBuffer<uint> SVODivisibleNodeIndices_E;
[[vk::binding(BindingPoints_SVO::eSVOThreadGroupInfos_SVO)]] RWStructuredBuffer<SVOThreadGroupInfo> SVOThreadGroupInfos;

//----------------------------------------------initSVOArray--------------------------------------------
[numthreads(512, 1, 1)]
[shader("compute")]
void computeMain_initSVOArray(uint threadIndex: SV_DispatchThreadID) {
    uint maxNodeCount = pushConst.sizes[pushConst.maxDepth];
    if (threadIndex >= maxNodeCount) return;

    if (threadIndex == 0) {
        SVOLayerInfos_G[0].divsibleNodeCount = 1;
        SVOLayerInfos_G[0].indivisibleNodeCount = 0;
        SVOLayerInfos_E[0].divsibleNodeCount = 1;
        SVOLayerInfos_E[0].indivisibleNodeCount = 0;
    }
    if (threadIndex > 0 && threadIndex <= pushConst.maxDepth) {
        SVOLayerInfos_G[threadIndex].divsibleNodeCount = 0;
        SVOLayerInfos_G[threadIndex].indivisibleNodeCount = 0;
        SVOLayerInfos_E[threadIndex].divsibleNodeCount = 0;
        SVOLayerInfos_E[threadIndex].indivisibleNodeCount = 0;
    }

    SVODivisibleNodeIndices_G[threadIndex] = 0;
    SVODivisibleNodeIndices_E[threadIndex] = 0;

    if (threadIndex < maxNodeCount / THREADGROUP_SIZE) {
        SVOThreadGroupInfo threadGroupInfo;
        threadGroupInfo.threadGroupDivisibleNodeCount_G = 0;
        threadGroupInfo.threadGroupDivisibleNodeCount_E = 0;
        threadGroupInfo.threadGroupIndivisibleNodeCount_G = 0;
        threadGroupInfo.threadGroupIndivisibleNodeCount_E = 0;
        SVOThreadGroupInfos[threadIndex] = threadGroupInfo;
    }

    SVONodeData_G nodeData_G;
    nodeData_G.meanNormal = float4(0.0f);
    nodeData_G.aabb.minimum.x = 3.402823466e+38F;
    nodeData_G.aabb.minimum.y = 3.402823466e+38F;
    nodeData_G.aabb.minimum.z = 3.402823466e+38F;
    nodeData_G.aabb.maximum.x = -3.402823466e+38F;
    nodeData_G.aabb.maximum.y = -3.402823466e+38F;
    nodeData_G.aabb.maximum.z = -3.402823466e+38F;
    nodeData_G.indivisible = 1;
    nodeData_G.label = 0;
    nodeData_G.materialType = 0;

    SVONodeData_E nodeData_E;
    nodeData_E.irradiance = float3(0.0f);
    nodeData_E.notIgnoreRatio = 1.0f;
    nodeData_E.meanNormal = float4(0.0f);
    nodeData_E.aabb.minimum.x = 3.402823466e+38F;
    nodeData_E.aabb.minimum.y = 3.402823466e+38F;
    nodeData_E.aabb.minimum.z = 3.402823466e+38F;
    nodeData_E.aabb.maximum.x = -3.402823466e+38F;
    nodeData_E.aabb.maximum.y = -3.402823466e+38F;
    nodeData_E.aabb.maximum.z = -3.402823466e+38F;
    nodeData_E.indivisible = 1;
    nodeData_E.label = 0;

    for (int depth = pushConst.maxDepth; depth >= 0; --depth) {
        if (threadIndex >= pushConst.sizes[depth]) return;
        SVOArray_G[depth][threadIndex] = nodeData_G;
        SVOArray_E[depth][threadIndex] = nodeData_E;
    }
}
//-----------------------------------------------createSVOArray-------------------------------------------
groupshared uint warpDivisibleNodeCountSums[MAX_WARP_COUNT];
groupshared uint warpIndivisibleNodeCountSums[MAX_WARP_COUNT];
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    // Each thread processes one node. eight thread processes a block of nodes whose father nodes are the same.    
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    uint fatherLayerDivisibleNodeCount = SVOLayerInfos_G[pushConst.currentDepth - 1].divsibleNodeCount;
    if (fatherLayerDivisibleNodeCount == 0) return;
    uint totalNodeCount = fatherLayerDivisibleNodeCount * 8;
    if (threadIndex >= totalNodeCount) return;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) {
        if(threadIndex == 0)
            printf("Error: totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,   maxNodeCount: %u\n", totalNodeCount, pushConst.sizes[pushConst.currentDepth]);
        return;
    }

    uint fatherNodeIndex = threadIndex / 8;
    uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
    SVONodeData_G nodeData = OctreeArray_G[pushConst.currentDepth][nodeIndex];

    bool hasData = nodeData.meanNormal.w > 0.0f;
    bool divisible = hasData && nodeData.indivisible == 0;
    bool indivisible = hasData && nodeData.indivisible == 1;

    // Count the number of divisible and indivisible nodes in the warp
    uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
    uint divisibleLabelInWarp = WavePrefixSum(divisible);
    uint x = WavePrefixSum(divisible);

    uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
    uint indivisibleLayerInWarp = WavePrefixSum(indivisible);

    uint threadGroupNodeCount = 
                totalNodeCount - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : totalNodeCount % THREADGROUP_SIZE;
    uint warpCount = (threadGroupNodeCount + 31) / 32;

    // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
    if (fatherLayerDivisibleNodeCount <= 4) {
        if (divisible) nodeData.label = divisibleLabelInWarp + 1;
        if (indivisible) nodeData.label = indivisibleLayerInWarp + 1;
    } else {
        if (warpLane == 0 && warpIndex < MAX_WARP_COUNT - 1) {
            uint warpDivisibleNodeCount = countbits(warpDivisibleNodeMask);
            warpDivisibleNodeCountSums[warpIndex + 1] = warpDivisibleNodeCount;

            uint warpIndivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            warpIndivisibleNodeCountSums[warpIndex + 1] = warpIndivisibleNodeCount;
        }
        GroupMemoryBarrierWithGroupSync();

        if (groupThreadIndex == 0) {
            for (int i = 2; i < warpCount; ++i) {
                warpDivisibleNodeCountSums[i] += warpDivisibleNodeCountSums[i - 1];
                warpIndivisibleNodeCountSums[i] += warpIndivisibleNodeCountSums[i - 1];
            }
            warpDivisibleNodeCountSums[0] = 0;
            warpIndivisibleNodeCountSums[0] = 0;
        }
        GroupMemoryBarrierWithGroupSync();

        uint divisibleLabelInThreadGroup = warpDivisibleNodeCountSums[warpIndex] + divisibleLabelInWarp;
        uint inDivisibleLabelInThreadGroup = warpIndivisibleNodeCountSums[warpIndex] + indivisibleLayerInWarp;

        if (divisible) nodeData.label = divisibleLabelInThreadGroup + 1;
        if (indivisible) nodeData.label = inDivisibleLabelInThreadGroup + 1;
    }

    if (warpIndex == warpCount - 1 && warpLane == 0) {
        if (totalNodeCount <= WARP_SIZE) {
            SVOLayerInfos_G[pushConst.currentDepth].divsibleNodeCount = countbits(warpDivisibleNodeMask);
            SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
        } else if (totalNodeCount <= THREADGROUP_SIZE) {
            SVOLayerInfos_G[pushConst.currentDepth].divsibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
            SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
        }
        else {
            uint threadGroupDivisibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
            uint threadGroupInDivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);

            InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].divsibleNodeCount, threadGroupDivisibleNodeCount);
            InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

            SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_G = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
            SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_G = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
        }
    }

    if (divisible && totalNodeCount <= THREADGROUP_SIZE) SVODivisibleNodeIndices_G[nodeData.label - 1] = nodeIndex;
    if (hasData) SVOArray_G[pushConst.currentDepth][threadIndex] = nodeData;
}

groupshared uint threadGroupDivisibleNodeCountSum_G;
groupshared uint threadGroupIndivisibleNodeCountSum_G;
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_offsetLabelMultiBlock(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint fatherLayerDivisibleNodeCount = SVOLayerInfos_G[pushConst.currentDepth - 1].divsibleNodeCount;
    uint totalNodeCount = fatherLayerDivisibleNodeCount * 8;
    if (totalNodeCount < THREADGROUP_SIZE || threadIndex >= totalNodeCount) return;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) return;

    if (groupThreadIndex == 0) {
        threadGroupDivisibleNodeCountSum_G = 0;
        threadGroupIndivisibleNodeCountSum_G = 0;
        for (int i = 0; i < threadGroupIndex; ++i) {
            SVOThreadGroupInfo threadGroupInfo = SVOThreadGroupInfos[i];
            threadGroupDivisibleNodeCountSum_G += threadGroupInfo.threadGroupDivisibleNodeCount_G;
            threadGroupIndivisibleNodeCountSum_G += threadGroupInfo.threadGroupIndivisibleNodeCount_G;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    SVONodeData_G nodeData = SVOArray_G[pushConst.currentDepth][threadIndex];
    bool hasData = nodeData.meanNormal.w > 0.0f;
    bool divisible = hasData && nodeData.indivisible == 0;
    bool indivisible = hasData && nodeData.indivisible == 1;
    if (divisible) {
        uint label = SVOArray_G[pushConst.currentDepth][threadIndex].label + threadGroupDivisibleNodeCountSum_G;
        SVOArray_G[pushConst.currentDepth][threadIndex].label = label;

        uint fatherNodeIndex = threadIndex / 8;
        uint blockLane = groupThreadIndex & 7;
        uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
        SVODivisibleNodeIndices_G[label - 1] = nodeIndex;
    }
    else if (indivisible) SVOArray_G[pushConst.currentDepth][threadIndex].label += threadGroupIndivisibleNodeCountSum_G;
}