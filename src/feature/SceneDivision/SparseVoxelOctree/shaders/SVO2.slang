#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOPushConstant> pushConst;
[[vk::binding(BindingPoints_SVO::eOctreeArray_G_SVO)]] StructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_SVO::eOctreeArray_E_SVO)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];
[[vk::binding(BindingPoints_SVO::eSVOArray_G_SVO)]] RWStructuredBuffer<SVONodeData_G, ScalarDataLayout> SVOArray_G[];
[[vk::binding(BindingPoints_SVO::eSVOArray_E_SVO)]] RWStructuredBuffer<SVONodeData_E, ScalarDataLayout> SVOArray_E[];
[[vk::binding(BindingPoints_SVO::eSVOLayerInfos_G_SVO)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_G;
[[vk::binding(BindingPoints_SVO::eSVOLayerInfos_E_SVO)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_E;
[[vk::binding(BindingPoints_SVO::eSVOGlobalInfo_SVO)]] RWStructuredBuffer<SVOGloablInfo_SVO> SVOGloablInfo;
[[vk::binding(BindingPoints_SVO::eSVODivisibleNodeIndices_G_SVO)]] RWStructuredBuffer<uint> SVODivisibleNodeIndices_G;
[[vk::binding(BindingPoints_SVO::eSVODivisibleNodeIndices_E_SVO)]] RWStructuredBuffer<uint> SVODivisibleNodeIndices_E;
[[vk::binding(BindingPoints_SVO::eSVOThreadGroupInfos_SVO)]] RWStructuredBuffer<SVOThreadGroupInfo> SVOThreadGroupInfos;

//----------------------------------------------initSVOArray--------------------------------------------
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initSVOArray(uint threadIndex: SV_DispatchThreadID) {
    uint maxNodeCount_G = pushConst.sizes_G[pushConst.maxDepth];
    uint maxNodeCount_E = pushConst.sizes_E[pushConst.maxDepth];
    uint maxNodeCount = max(maxNodeCount_G, maxNodeCount_E);
    if (threadIndex >= maxNodeCount) return;

    if (threadIndex == 0) {
        SVOLayerInfos_G[0].divisibleNodeCount = 1;
        SVOLayerInfos_G[0].indivisibleNodeCount = 0;
        SVOLayerInfos_E[0].divisibleNodeCount = 1;
        SVOLayerInfos_E[0].indivisibleNodeCount = 0;

        SVOGloablInfo_SVO globalInfo;
        globalInfo.cmd = { (max(SVONodeCount_G_Layer1, SVONodeCount_E_Layer1) + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE, 1, 1 };
        SVOGloablInfo[0] = globalInfo;
    }

    if (threadIndex > 0 && threadIndex <= pushConst.maxDepth) {
        SVOLayerInfos_G[threadIndex].divisibleNodeCount = 0;
        SVOLayerInfos_G[threadIndex].indivisibleNodeCount = 0;
        SVOLayerInfos_E[threadIndex].divisibleNodeCount = 0;
        SVOLayerInfos_E[threadIndex].indivisibleNodeCount = 0;
    }

    if (threadIndex < maxNodeCount_G) SVODivisibleNodeIndices_G[threadIndex] = 0;
    if (threadIndex < maxNodeCount_E) SVODivisibleNodeIndices_E[threadIndex] = 0;

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
    nodeData_G.materialIndex = 0;

    SVONodeData_E nodeData_E;
    nodeData_E.irradiance = float3(0.0f);
    nodeData_E.pdf = 1.0f;
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
        if (threadIndex < pushConst.sizes_G[depth]) SVOArray_G[depth][threadIndex] = nodeData_G;
        if (threadIndex < pushConst.sizes_E[depth]) SVOArray_E[depth][threadIndex] = nodeData_E;
    }
}
//-----------------------------------------------createSVOArray-------------------------------------------
#define MAX_WARP_COUNT (THREADGROUP_SIZE / WARP_SIZE)

groupshared uint groupFatherLayerDivisibleNodeCount_G;
groupshared uint groupFatherLayerDivisibleNodeCount_E;
groupshared uint groupWarpDivisibleNodeCountSums_G[MAX_WARP_COUNT];
groupshared uint groupWarpIndivisibleNodeCountSums_G[MAX_WARP_COUNT];
groupshared uint groupWarpDivisibleNodeCountSums_E[MAX_WARP_COUNT];
groupshared uint groupWarpIndivisibleNodeCountSums_E[MAX_WARP_COUNT];
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    // Each thread processes one node. eight thread processes a block of nodes whose father nodes are the same.
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        groupFatherLayerDivisibleNodeCount_G = SVOLayerInfos_G[pushConst.currentDepth - 1].divisibleNodeCount;
        groupFatherLayerDivisibleNodeCount_E = SVOLayerInfos_E[pushConst.currentDepth - 1].divisibleNodeCount;
    }
    GroupMemoryBarrierWithGroupSync();
    //------------------------------------------------G---------------------------------------------------------
    uint totalNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    if (totalNodeCount_G > pushConst.sizes_G[pushConst.currentDepth]) {
        if (threadIndex == 0) {
            SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = 0;
            SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = 0;
            printf("Error: G totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u, maxNodeCount: %u\n",
                totalNodeCount_G, pushConst.sizes_G[pushConst.currentDepth]);
        }
        return;
    }
    if (threadIndex < totalNodeCount_G) {
        uint fatherNodeIndex = threadIndex / 8;
        uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
        SVONodeData_G nodeData = OctreeArray_G[pushConst.currentDepth][nodeIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        //if (threadIndex < totalNodeCount_G) {
        //    nodeData = OctreeArray_G[pushConst.currentDepth][nodeIndex];
        //    hasData = nodeData.meanNormal.w > 0.0f;
        //}
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount = 
                totalNodeCount_G - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : totalNodeCount_G % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        if (totalNodeCount_G <= WARP_SIZE) {
            if (divisible) nodeData.label = divisibleLabelInWarp + 1;
            if (indivisible) nodeData.label = indivisibleLabelInWarp + 1;
        } else {
            if (warpLane == 0) {
                uint warpDivisibleNodeCount = countbits(warpDivisibleNodeMask);
                groupWarpDivisibleNodeCountSums_G[warpIndex] = warpDivisibleNodeCount;

                uint warpIndivisibleNodeCount = countbits(warpIndivisibleNodeMask);
                groupWarpIndivisibleNodeCountSums_G[warpIndex] = warpIndivisibleNodeCount;
            }
            GroupMemoryBarrierWithGroupSync();      //The post-votal architecture can accommodate branching

            if (warpLane < warpIndex) {     //WARP_SIZE >= 32, so warpIndex <= 32 <= WARP_SIZE
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

        if (divisible && totalNodeCount_G <= THREADGROUP_SIZE) SVODivisibleNodeIndices_G[nodeData.label - 1] = nodeIndex;
        if (hasData) SVOArray_G[pushConst.currentDepth][threadIndex] = nodeData;

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (totalNodeCount_G <= WARP_SIZE) {
                SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (totalNodeCount_G <= THREADGROUP_SIZE) {
                SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_G = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_G = threadGroupInDivisibleNodeCount;
            }
        }
    }

    uint totalNodeCount_E = groupFatherLayerDivisibleNodeCount_E * 8;
    if (totalNodeCount_E > pushConst.sizes_E[pushConst.currentDepth]) {
        if (threadIndex == 0) {
            SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = 0;
            SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = 0;
            printf("Error: E totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,   maxNodeCount: %u\n",
                   totalNodeCount_E, pushConst.sizes_E[pushConst.currentDepth]);
        }
        return;
    }
    if (threadIndex < totalNodeCount_E){
        uint fatherNodeIndex = threadIndex / 8;
        uint nodeIndex = SVODivisibleNodeIndices_E[fatherNodeIndex] * 8 + blockLane;
        SVONodeData_E nodeData = OctreeArray_E[pushConst.currentDepth][nodeIndex];
        bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        //if (threadIndex < totalNodeCount_E) {
        //    nodeData = OctreeArray_E[pushConst.currentDepth][nodeIndex];
        //    hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        //}
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount = 
                totalNodeCount_E - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : totalNodeCount_E % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        if (totalNodeCount_E <= WARP_SIZE) {
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

        if (divisible && totalNodeCount_E <= THREADGROUP_SIZE) SVODivisibleNodeIndices_E[nodeData.label - 1] = nodeIndex;
        if (hasData) SVOArray_E[pushConst.currentDepth][threadIndex] = nodeData;

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (totalNodeCount_E <= WARP_SIZE) {
                SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (totalNodeCount_E <= THREADGROUP_SIZE) {
                SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_E = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_E = threadGroupInDivisibleNodeCount;
            }
        }
    }

    if (threadIndex == 0 && max(totalNodeCount_G, totalNodeCount_E) <= THREADGROUP_SIZE) SVOGloablInfo[0].cmd.x = 1;
}

/*
In our program, the maximum nodeCount of SVOLayer will not exceed 2048, 4096, etc., which is much smaller than THREADGROUP_SIZE * THREADGROUP_SIZE.
Therefore, computeMain_offsetLabelMultiBlock generally only has one thread group
Generally, 4 or 8 threads are sufficient to obtain all the thread group information for computeMain_createSVOArray
*/
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
            groupFatherLayerDivisibleNodeCount_G = SVOLayerInfos_G[pushConst.currentDepth - 1].divisibleNodeCount;
            groupFatherLayerDivisibleNodeCount_E = SVOLayerInfos_E[pushConst.currentDepth - 1].divisibleNodeCount;
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
    uint totalNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    if (totalNodeCount_G > pushConst.sizes_G[pushConst.currentDepth]) return;
    if (totalNodeCount_G > THREADGROUP_SIZE) {
        SVONodeData_G nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount_G) {
            nodeData = SVOArray_G[pushConst.currentDepth][threadIndex];
            hasData = nodeData.meanNormal.w > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = SVOArray_G[pushConst.currentDepth][threadIndex].label + groupDivisibleNodeCountSum_G;
            SVOArray_G[pushConst.currentDepth][threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;
            uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
            SVODivisibleNodeIndices_G[label - 1] = nodeIndex;
        }
        else if (indivisible) SVOArray_G[pushConst.currentDepth][threadIndex].label += groupIndivisibleNodeCountSum_G;
    }
    //------------------------------------------------E---------------------------------------------------------
    uint totalNodeCount_E = groupFatherLayerDivisibleNodeCount_E * 8;
    if (totalNodeCount_E > pushConst.sizes_E[pushConst.currentDepth]) return;
    if (totalNodeCount_E > THREADGROUP_SIZE) {
        SVONodeData_E nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount_E) {
            nodeData = SVOArray_E[pushConst.currentDepth][threadIndex];
            hasData = nodeData.meanNormal.w > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = SVOArray_E[pushConst.currentDepth][threadIndex].label + groupDivisibleNodeCountSum_E;
            SVOArray_E[pushConst.currentDepth][threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;
            uint nodeIndex = SVODivisibleNodeIndices_E[fatherNodeIndex] * 8 + blockLane;
            SVODivisibleNodeIndices_E[label - 1] = nodeIndex;
        }
        else if (indivisible) SVOArray_E[pushConst.currentDepth][threadIndex].label += groupIndivisibleNodeCountSum_E;
    }

    if (threadIndex == 0) {
        uint nextLayerNodeCount =
                max(SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount, SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount) * 8;
        uint cmdX = (nextLayerNodeCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        SVOGloablInfo[0].cmd.x = cmdX;

        #ifndef NDEBUG
        if (cmdX == 0) {
            uint SVOTotalNodeCount_G = 0;
            uint SVOTotalNodeCount_E = 0;
            for (int i = 0; i < pushConst.maxDepth; ++i) {
                SVOTotalNodeCount_G += SVOLayerInfos_G[i].divisibleNodeCount * 8;
                SVOTotalNodeCount_E += SVOLayerInfos_E[i].divisibleNodeCount * 8;
            }
            SVOGloablInfo[0].drawCmd.instanceCount = SVOTotalNodeCount_G + SVOTotalNodeCount_E;
            SVOGloablInfo[0].totalNodeCount_G = SVOTotalNodeCount_G;
            SVOGloablInfo[0].totalNodeCount_E = SVOTotalNodeCount_E;
        }
        #endif
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

    SVOGloablInfo_SVO globalInfo = SVOGloablInfo[0];

    bool isG = true;
    uint layerIndex = 1;
    uint nodeIndex = instanceIndex;
    if (nodeIndex >= globalInfo.totalNodeCount_G) {
        nodeIndex -= globalInfo.totalNodeCount_G;
        isG = false;
    }
    output.isG = isG;

    for (; layerIndex <= pushConst.maxDepth; ++layerIndex) {
        uint layerNodeCount = (isG ? SVOLayerInfos_G[layerIndex - 1].divisibleNodeCount : SVOLayerInfos_E[layerIndex - 1].divisibleNodeCount) * 8;
        if (nodeIndex >= layerNodeCount) nodeIndex -= layerNodeCount;
        else break;
    }
    output.color = Colors[layerIndex];

    AABB nodeAABB; bool show = true;
    if (isG) {
        SVONodeData_G nodeData = SVOArray_G[layerIndex][nodeIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        show = hasData && nodeData.indivisible == 1;
        nodeAABB = nodeData.aabb;
    } else {
        SVONodeData_E nodeData = SVOArray_E[layerIndex][nodeIndex];
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