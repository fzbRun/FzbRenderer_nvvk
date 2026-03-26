#include "common/Shader/shaderStructType.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOPushConstant_SVOPG> pushConst;
[[vk::binding(BindingPoints_SVOPG::eOctreeArray_G_SVOPG)]] StructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_SVOPG::eOctreeArray_E_SVOPG)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];
[[vk::binding(BindingPoints_SVOPG::eSVO_G_SVOPG)]] RWStructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(BindingPoints_SVOPG::eSVO_E_SVOPG)]] RWStructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E;
[[vk::binding(BindingPoints_SVOPG::eSVOGlobalInfo_SVOPG)]] RWStructuredBuffer<SVOGlobalInfo_SVOPG> SVOGlobalInfo;
[[vk::binding(BindingPoints_SVOPG::eSVODivisibleNodeIndices_G_SVOPG)]] RWStructuredBuffer<uint2> SVODivisibleNodeInfos_G;
[[vk::binding(BindingPoints_SVOPG::eSVODivisibleNodeIndices_E_SVOPG)]] RWStructuredBuffer<uint> SVODivisibleNodeInfos_E;
[[vk::binding(BindingPoints_SVOPG::eSVOThreadGroupInfos_SVOPG)]] RWStructuredBuffer<SVOThreadGroupInfo> SVOThreadGroupInfos;

//----------------------------------------------initSVOArray--------------------------------------------
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initSVOArray(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) {
        SVOGlobalInfo_SVOPG globalInfo;

        globalInfo.layerInfos_G[0].divisibleNodeCount = 6;
        globalInfo.layerInfos_G[0].indivisibleNodeCount = 0;
        globalInfo.layerInfos_E[0].divisibleNodeCount = 1;
        globalInfo.layerInfos_E[0].indivisibleNodeCount = 0;

        for (int i = 1; i <= pushConst.maxDepth_Octree; ++i) {
            globalInfo.layerInfos_G[i].divisibleNodeCount = 0;
            globalInfo.layerInfos_G[i].indivisibleNodeCount = 0;
            globalInfo.layerInfos_E[i].divisibleNodeCount = 0;
            globalInfo.layerInfos_E[i].indivisibleNodeCount = 0;
        }

        globalInfo.cmd = { (48 + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE, 1, 1 };
        SVOGlobalInfo[0] = globalInfo;

        SVODivisibleNodeInfos_E[threadIndex] = 0;
    }
    if (threadIndex < 6) SVODivisibleNodeInfos_G[threadIndex] = uint2(threadIndex * (pushConst.maxDepth_Octree + 1), 0);

    if (threadIndex < SVOSize / THREADGROUP_SIZE) {
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
    nodeData_G.materialIndex = 0;

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

    for (int i = threadIndex; i < SVOSize_G; i += THREADGROUP_SIZE) SVO_G[i] = nodeData_G;
    for (int i = threadIndex; i < SVOSize_E; i += THREADGROUP_SIZE) SVO_E[i] = nodeData_E; 
}
//-----------------------------------------------createSVOArray-------------------------------------------
#define MAX_WARP_COUNT (THREADGROUP_SIZE / WARP_SIZE)

groupshared uint groupFatherLayerDivisibleNodeCount_G;
groupshared uint groupFatherLayerDivisibleNodeCount_E;
groupshared uint groupSVONodeTotalCount_G;
groupshared uint groupSVONodeTotalCount_E;
groupshared uint groupWarpDivisibleNodeCountSums_G[MAX_WARP_COUNT];
groupshared uint groupWarpIndivisibleNodeCountSums_G[MAX_WARP_COUNT];
groupshared uint groupWarpDivisibleNodeCountSums_E[MAX_WARP_COUNT];
groupshared uint groupWarpIndivisibleNodeCountSums_E[MAX_WARP_COUNT];

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];
        groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentDepth_SVO - 1].divisibleNodeCount;
        groupFatherLayerDivisibleNodeCount_E = globalInfo.layerInfos_E[pushConst.currentDepth_SVO - 1].divisibleNodeCount;

        uint svoNodeTotalCount_G = 0, svoNodeTotalCount_E = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentDepth_SVO - 1; ++layerIndex) {
            svoNodeTotalCount_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
            svoNodeTotalCount_E += globalInfo.layerInfos_E[layerIndex].divisibleNodeCount * 8;
        }
        groupSVONodeTotalCount_G = svoNodeTotalCount_G;
        groupSVONodeTotalCount_E = svoNodeTotalCount_E;
    }
    GroupMemoryBarrierWithGroupSync();

    //------------------------------------------------G---------------------------------------------------------
    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    uint lastLayerNodeCountSum_G = groupSVONodeTotalCount_G;
    if (layerNodeCount_G + lastLayerNodeCountSum_G > SVOSize_G) {
        if (threadIndex == 0) {
            SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount = 0;
            SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount = 0;
            printf("Error: G totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,  maxNodeCount: %u\n",
                   layerNodeCount_G + lastLayerNodeCountSum_G, SVOSize_G);
        }
        return;
    }
    if (threadIndex < ((layerNodeCount_G + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)){
        uint fatherNodeIndex = threadIndex / 8;
        
        uint octreeIndex = 0;
        uint nodeIndex = 0;
        SVONodeData_G nodeData;
        bool hasData = false;
        if (threadIndex < layerNodeCount_G) {
            uint2 fatherNodeInfo = SVODivisibleNodeInfos_G[fatherNodeIndex];
            octreeIndex = fatherNodeInfo.x + 1;
            nodeIndex = fatherNodeInfo.y * 8 + blockLane;

            nodeData = OctreeArray_G[octreeIndex][nodeIndex];
            hasData = nodeData.meanNormal.w > 0.0f;
        }

        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount = 
                layerNodeCount_G - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : layerNodeCount_G % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        if (layerNodeCount_G <= WARP_SIZE) {
            if (divisible) nodeData.label = divisibleLabelInWarp + 1;
            if (indivisible) nodeData.label = indivisibleLabelInWarp + 1;
        } else {
            if (warpLane == 0) {
                uint warpDivisibleNodeCount = countbits(warpDivisibleNodeMask);
                groupWarpDivisibleNodeCountSums_G[warpIndex] = warpDivisibleNodeCount;

                uint warpIndivisibleNodeCount = countbits(warpIndivisibleNodeMask);
                groupWarpIndivisibleNodeCountSums_G[warpIndex] = warpIndivisibleNodeCount;
            }
            GroupMemoryBarrierWithGroupSync(); // The post-votal architecture can accommodate branching

            if (warpLane < warpIndex) { // WARP_SIZE >= 32, so warpIndex <= 32 <= WARP_SIZE
                threadGroupWarpDivisibleNodeLabel = groupWarpDivisibleNodeCountSums_G[warpLane];
                threadGroupWarpIndivisibleNodeLabel = groupWarpIndivisibleNodeCountSums_G[warpLane];
            }
            threadGroupWarpDivisibleNodeLabel = WaveActiveSum(threadGroupWarpDivisibleNodeLabel);
            threadGroupWarpIndivisibleNodeLabel = WaveActiveSum(threadGroupWarpIndivisibleNodeLabel);

            uint divisibleLabelInThreadGroup = threadGroupWarpDivisibleNodeLabel + divisibleLabelInWarp;
            uint indivisibleLabelInThreadGroup = threadGroupWarpIndivisibleNodeLabel + indivisibleLabelInWarp;

            if (divisible) nodeData.label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) nodeData.label = indivisibleLabelInThreadGroup + 1;
        }

        if (divisible && layerNodeCount_G <= THREADGROUP_SIZE) SVODivisibleNodeInfos_G[nodeData.label - 1] = uint2(octreeIndex, nodeIndex);
        if (hasData) SVO_G[lastLayerNodeCountSum_G + threadIndex] = nodeData;

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (layerNodeCount_G <= WARP_SIZE) {
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (layerNodeCount_G <= THREADGROUP_SIZE) {
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_G = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_G = threadGroupInDivisibleNodeCount;
            }
        }
    }

    uint layerNodeCount_E = groupFatherLayerDivisibleNodeCount_E * 8;
    uint lastLayerNodeCountSum_E = groupSVONodeTotalCount_E;
    if (layerNodeCount_E + lastLayerNodeCountSum_E > SVOSize_E) {
        if (threadIndex == 0) {
            SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount = 0;
            SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount = 0;
            printf("Error: E totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,  maxNodeCount: %u\n",
                   layerNodeCount_E + lastLayerNodeCountSum_E, SVOSize_E);
        }
        return;
    }
    if (threadIndex < ((layerNodeCount_E + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)) {
        uint nodeIndex = 0;
        SVONodeData_E nodeData;
        bool hasData = false;
        if (threadIndex < layerNodeCount_E) {
            uint fatherNodeIndex = threadIndex / 8;
            nodeIndex = SVODivisibleNodeInfos_E[fatherNodeIndex] * 8 + blockLane;

            nodeData = OctreeArray_E[pushConst.currentDepth_SVO][nodeIndex];
            hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount = 
                layerNodeCount_E - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : layerNodeCount_E % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        if (layerNodeCount_E <= WARP_SIZE) {
            if (divisible) nodeData.label = divisibleLabelInWarp + 1;
            if (indivisible) nodeData.label = indivisibleLabelInWarp + 1;
        } else {
            if (warpLane == 0) {
                uint warpDivisibleNodeCount = countbits(warpDivisibleNodeMask);
                groupWarpDivisibleNodeCountSums_E[warpIndex] = warpDivisibleNodeCount;

                uint warpIndivisibleNodeCount = countbits(warpIndivisibleNodeMask);
                groupWarpIndivisibleNodeCountSums_E[warpIndex] = warpIndivisibleNodeCount;
            }
            GroupMemoryBarrierWithGroupSync();

            if (warpLane < warpIndex) {
                threadGroupWarpDivisibleNodeLabel = groupWarpDivisibleNodeCountSums_E[warpLane];
                threadGroupWarpIndivisibleNodeLabel = groupWarpIndivisibleNodeCountSums_E[warpLane];
            }
            threadGroupWarpDivisibleNodeLabel = WaveActiveSum(threadGroupWarpDivisibleNodeLabel);
            threadGroupWarpIndivisibleNodeLabel = WaveActiveSum(threadGroupWarpIndivisibleNodeLabel);

            uint divisibleLabelInThreadGroup = threadGroupWarpDivisibleNodeLabel + divisibleLabelInWarp;
            uint indivisibleLabelInThreadGroup = threadGroupWarpIndivisibleNodeLabel + indivisibleLabelInWarp;

            if (divisible) nodeData.label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) nodeData.label = indivisibleLabelInThreadGroup + 1;
        }

        if (divisible && layerNodeCount_E <= THREADGROUP_SIZE) SVODivisibleNodeInfos_E[nodeData.label - 1] = nodeIndex;
        if (hasData) SVO_E[lastLayerNodeCountSum_E + threadIndex] = nodeData;

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (layerNodeCount_E <= WARP_SIZE) {
                SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (layerNodeCount_E <= THREADGROUP_SIZE) {
                SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_E = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_E = threadGroupInDivisibleNodeCount;
            }
        }
    }

    if (threadIndex == 0 && max(layerNodeCount_G, layerNodeCount_E) <= THREADGROUP_SIZE) SVOGlobalInfo[0].cmd.x = 1;
}

groupshared uint groupDivisibleNodeCountSum_G;
groupshared uint groupIndivisibleNodeCountSum_G;
groupshared uint groupDivisibleNodeCountSum_E;
groupshared uint groupIndivisibleNodeCountSum_E;
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_offsetLabelMultiBlock(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (warpIndex == 0) { // this assume groupThreadCount < WARP_SIZE
        if (groupThreadIndex == 0) {
            SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];
            groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentDepth_SVO - 1].divisibleNodeCount;
            groupFatherLayerDivisibleNodeCount_E = globalInfo.layerInfos_E[pushConst.currentDepth_SVO - 1].divisibleNodeCount;

            uint svoNodeTotalCount_G = 0, svoNodeTotalCount_E = 0;
            for (int layerIndex = 0; layerIndex < pushConst.currentDepth_SVO - 1; ++layerIndex) {
                svoNodeTotalCount_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
                svoNodeTotalCount_E += globalInfo.layerInfos_E[layerIndex].divisibleNodeCount * 8;
            }
            groupSVONodeTotalCount_G = svoNodeTotalCount_G;
            groupSVONodeTotalCount_E = svoNodeTotalCount_E;
        }

        uint threadGroupDivisibleNodeCount = 0;
        uint threadGroupIndivisibleNodeCount = 0;
        if (groupThreadIndex < threadGroupIndex) {
            SVOThreadGroupInfo threadGroupInfo = SVOThreadGroupInfos[groupThreadIndex];
            threadGroupDivisibleNodeCount += threadGroupInfo.threadGroupDivisibleNodeCount_G;
            threadGroupIndivisibleNodeCount += threadGroupInfo.threadGroupIndivisibleNodeCount_G;
        }
        threadGroupDivisibleNodeCount = WaveActiveSum(threadGroupDivisibleNodeCount);
        threadGroupIndivisibleNodeCount = WaveActiveSum(threadGroupIndivisibleNodeCount);
        if (warpLane == 0) {
            groupDivisibleNodeCountSum_G = threadGroupDivisibleNodeCount;
            groupIndivisibleNodeCountSum_G = threadGroupIndivisibleNodeCount;
        }

        threadGroupDivisibleNodeCount = 0;
        threadGroupIndivisibleNodeCount = 0;
        if (groupThreadIndex < threadGroupIndex) {
            SVOThreadGroupInfo threadGroupInfo = SVOThreadGroupInfos[groupThreadIndex];
            threadGroupDivisibleNodeCount += threadGroupInfo.threadGroupDivisibleNodeCount_E;
            threadGroupIndivisibleNodeCount += threadGroupInfo.threadGroupIndivisibleNodeCount_E;
        }
        threadGroupDivisibleNodeCount = WaveActiveSum(threadGroupDivisibleNodeCount);
        threadGroupIndivisibleNodeCount = WaveActiveSum(threadGroupIndivisibleNodeCount);
        if (warpLane == 0) {
            groupDivisibleNodeCountSum_E = threadGroupDivisibleNodeCount;
            groupIndivisibleNodeCountSum_E = threadGroupIndivisibleNodeCount;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    //------------------------------------------------G---------------------------------------------------------
    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    uint lastLayerNodeCountSum_G = groupSVONodeTotalCount_G;
    if (layerNodeCount_G + lastLayerNodeCountSum_G > SVOSize_G) return;
    if (layerNodeCount_G > THREADGROUP_SIZE && threadIndex < layerNodeCount_G) {
        SVONodeData_G nodeData = SVO_G[lastLayerNodeCountSum_G + threadIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = nodeData.label + groupDivisibleNodeCountSum_G;
            SVO_G[lastLayerNodeCountSum_G + threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;

            uint2 fatherNodeInfo = SVODivisibleNodeInfos_G[fatherNodeIndex];
            uint octreeIndex = fatherNodeInfo.x + 1;
            uint nodeIndex = fatherNodeInfo.y * 8 + blockLane;

            SVODivisibleNodeInfos_G[label - 1] = uint2(octreeIndex, nodeIndex);
        }
        else if (indivisible) SVO_G[lastLayerNodeCountSum_G + threadIndex].label += groupIndivisibleNodeCountSum_G;
    }
    //------------------------------------------------E---------------------------------------------------------
    uint layerNodeCount_E = groupFatherLayerDivisibleNodeCount_E * 8;
    uint lastLayerNodeCountSum_E = groupSVONodeTotalCount_E;
    if (layerNodeCount_E + lastLayerNodeCountSum_E > SVOSize_E) return;
    if (layerNodeCount_E > THREADGROUP_SIZE && threadIndex < layerNodeCount_E) {
        SVONodeData_E nodeData = SVO_E[lastLayerNodeCountSum_E + threadIndex];
        bool hasData = any(nodeData.irradiance > 0.0f);
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = nodeData.label + groupDivisibleNodeCountSum_E;
            SVO_E[lastLayerNodeCountSum_E + threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;
            uint nodeIndex = SVODivisibleNodeInfos_E[fatherNodeIndex] * 8 + blockLane;
            SVODivisibleNodeInfos_E[label - 1] = nodeIndex;
        }
        else if (indivisible) SVO_E[lastLayerNodeCountSum_E + threadIndex].label += groupIndivisibleNodeCountSum_E;
    }

    if (threadIndex == 0) {
        uint nextLayerNodeCount =
                max(SVOGlobalInfo[0].layerInfos_G[pushConst.currentDepth_SVO].divisibleNodeCount, SVOGlobalInfo[0].layerInfos_E[pushConst.currentDepth_SVO].divisibleNodeCount) * 8;
        uint cmdX = (nextLayerNodeCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        SVOGlobalInfo[0].cmd.x = cmdX;

        
        if (cmdX == 0) {
            uint SVOTotalNodeCount_G = 0;
            uint SVOTotalNodeCount_E = 0;
            for (int i = 0; i < pushConst.maxDepth_Octree; ++i) {
                SVOTotalNodeCount_G += SVOGlobalInfo[0].layerInfos_G[i].divisibleNodeCount * 8;
                SVOTotalNodeCount_E += SVOGlobalInfo[0].layerInfos_E[i].divisibleNodeCount * 8;
            }
            SVOGlobalInfo[0].totalNodeCount_G = SVOTotalNodeCount_G;
            SVOGlobalInfo[0].totalNodeCount_E = SVOTotalNodeCount_E;

            #ifndef NDEBUG
            SVOGlobalInfo[0].drawCmd.instanceCount = SVOTotalNodeCount_G + SVOTotalNodeCount_E;
            if (pushConst.frameIndex == 0) {
                printf("SVOTotalNodeCount_G: %d, SVOTotalNodeCount_E: %d\n", SVOTotalNodeCount_G, SVOTotalNodeCount_E);
            }
            #endif
        }
        
    }
}
#ifndef NDEBUG
//----------------------------------------debugDrawWireframe----------------------------------------
static const float4 Colors[8] = {
    float4(0.0f, 0.0f, 0.0f, 1.0f),
    float4(1.0f, 1.0f, 1.0f, 1.0f),
    float4(1.0f, 0.0f, 0.0f, 1.0f),
    float4(0.0f, 1.0f, 0.0f, 1.0f),
    float4(0.0f, 0.0f, 1.0f, 1.0f),
    float4(1.0f, 1.0f, 0.0f, 1.0f),
    float4(1.0f, 0.0f, 1.0f, 1.0f),
    float4(0.0f, 1.0f, 1.0f, 1.0f),
};

struct VSout_Wireframe {
    float4 sv_position : SV_Position;
    float4 color;
    bool isG;
};
struct PSout_Wireframe
{
    float4 color0 : SV_Target0;
    float4 color1 : SV_Target1;
};
[shader("vertex")]
VSout_Wireframe vertexMain_Wireframe(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout_Wireframe output;

    SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];

    bool isG = true;
    uint nodeIndex = instanceIndex;
    if (nodeIndex >= globalInfo.totalNodeCount_G) {
        nodeIndex -= globalInfo.totalNodeCount_G;
        isG = false;
    }
    output.isG = isG;

    uint layerIndex = 1;
    uint nodeIndex_temp = nodeIndex;
    for (; layerIndex <= pushConst.maxDepth_Octree; ++layerIndex) {
        uint layerNodeCount =
            (isG ? globalInfo.layerInfos_G[layerIndex - 1].divisibleNodeCount : globalInfo.layerInfos_E[layerIndex - 1].divisibleNodeCount) * 8;
        if (nodeIndex_temp < layerNodeCount) break;
        nodeIndex_temp -= layerNodeCount;
    }
    output.color = Colors[layerIndex];

    AABB nodeAABB; bool show = true;
    if (isG) {
        SVONodeData_G nodeData = SVO_G[nodeIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        show = hasData && nodeData.indivisible == 1;
        nodeAABB = nodeData.aabb;
    } else {
        SVONodeData_E nodeData = SVO_E[nodeIndex];
        bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        show = hasData && nodeData.indivisible == 1;
        nodeAABB = nodeData.aabb;
    }

    if (show) {
        float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += nodeAABB.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    } else output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    return output;
}
[shader("pixel")]
PSout_Wireframe fragmentMain_Wireframe(VSout_Wireframe input) {
    PSout_Wireframe output;
    if (input.isG) {
        output.color0 = input.color;
        output.color1 = float4(0.0f);
    }
    else {
        output.color0 = float4(0.0f);
        output.color1 = input.color;
    }
    return output;
}
#endif