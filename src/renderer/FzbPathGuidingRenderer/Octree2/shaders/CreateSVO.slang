#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_FzbPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> DivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eThreadGroupInfos)]] RWStructuredBuffer<OctreeThreadGroupInfo_FzbPG> OctreeThreadGroupInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
//------------------------------------------------getLabel---------------------------------------------------------
groupshared uint groupDivisibleNodeIndices_layer0[6];
groupshared uint groupDivisibleNodeCount_layer0;
groupshared uint groupWarpDivisibleNodeCount_layer1[2];

[numthreads(GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getOctreeLabel1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint divisibleNodeCount_layer0 = 0;
    uint divisibleNodeCount_layer1 = 0;

    if (threadIndex < 6) {
        OctreeNodeInfo_G_FzbPG nodeData = OctreeNodeInfoBuffer_G[0][threadIndex];
        bool divisible = nodeData.label_indivisible == 2;

        divisibleNodeCount_layer0 = WaveActiveCountBits(divisible);
        uint divisibleLabel = WavePrefixCountBits(divisible);

        int label = 0;
        if (divisible) {
            label = divisibleLabel + 1;
            groupDivisibleNodeIndices_layer0[divisibleLabel] = threadIndex;
        }
        if (divisible) OctreeNodeInfoBuffer_G[0][threadIndex].label_indivisible = (label << 1);

        if (threadIndex == 0) groupDivisibleNodeCount_layer0 = divisibleNodeCount_layer0;
    }
    GroupMemoryBarrierWithGroupSync();

    int nodeCount_layer2 = groupDivisibleNodeCount_layer0 * 8; // max 48
    if (nodeCount_layer2 <= 32) {
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            uint nodeIndex = fatherNodeIndex * 8 + threadIndex & 7;

            OctreeNodeInfo_G_FzbPG nodeData = OctreeNodeInfoBuffer_G[1][nodeIndex];
            bool divisible = nodeData.label_indivisible == 2;
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);
            uint divisibleLabel = WavePrefixCountBits(divisible);

            int label = 0;
            if (divisible) {
                label = divisibleLabel + 1;
                DivisibleNodeInfoBuffer_G[divisibleLabel] = uint2(nodeIndex);
            }
            OctreeNodeInfoBuffer_G[1][nodeIndex].label_indivisible = (label << 1);
        }
    } else {
        uint warpIndex = threadIndex / 32;
        uint warpLane = threadIndex & 31;
        uint blockLane = threadIndex & 7;

        OctreeNodeInfo_G_FzbPG nodeData; uint nodeIndex = 0;
        uint divisibleLabel; bool divisible = false;
        int label = 0;
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            nodeIndex = fatherNodeIndex * 8 + blockLane;

            nodeData = OctreeNodeInfoBuffer_G[1][nodeIndex];
            divisible = nodeData.label_indivisible == 2;
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);
            divisibleLabel = WavePrefixCountBits(divisible);

            if (divisible) label = divisibleLabel + 1;

            if (warpLane == 0) groupWarpDivisibleNodeCount_layer1[warpIndex] = divisibleNodeCount_layer1;
        }
        GroupMemoryBarrierWithGroupSync();
        if (threadIndex < nodeCount_layer2) {
            if (warpIndex == 1) {
                if (divisible) label += groupWarpDivisibleNodeCount_layer1[0];
            }
            if (divisible) DivisibleNodeInfoBuffer_G[label - 1] = uint2(nodeIndex);
            if (divisible) OctreeNodeInfoBuffer_G[1][nodeIndex].label_indivisible = (label << 1);
        }
    }

    if (threadIndex == 0) {
        OctreeGlobalInfo_FzbPG globalInfo;

        globalInfo.layerInfos_G[0].divisibleNodeCount = divisibleNodeCount_layer0;
        globalInfo.layerInfos_G[0].indivisibleNodeCount = 0;

        if (nodeCount_layer2 <= 32) globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1;
        else globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1 + groupWarpDivisibleNodeCount_layer1[1];
        globalInfo.layerInfos_G[1].indivisibleNodeCount = 0;

        for (int i = 2; i < MAX_OCTREE_LAYER_FZBPG; ++i) {
            globalInfo.layerInfos_G[i].divisibleNodeCount = 0;
            globalInfo.layerInfos_G[i].indivisibleNodeCount = 0;
        }

        globalInfo.cmd = { (globalInfo.layerInfos_G[1].divisibleNodeCount * 8 + GETOCTREELABEL_CS_THREADGROUP_SIZE - 1) / GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1 };
        GlobalInfoBuffer[0] = globalInfo;
    }
}

#define DivisibleNodeInfoCount_G_FZBPG ((IndivisibleNodeCount_G_FZBPG + 7) / 8)
groupshared uint groupFatherLayerDivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCountSum_G;
groupshared uint groupWarpDivisibleNodeCountSums_G[GETOCTREELABEL_CS_THREADGROUP_SIZE / 32];
groupshared uint groupWarpIndivisibleNodeCountSums_G[GETOCTREELABEL_CS_THREADGROUP_SIZE / 32];
[numthreads(GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getOctreeLabel2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
        groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

        uint indivisibleNodeCountSum_G = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentLayer; ++layerIndex)
            indivisibleNodeCountSum_G += globalInfo.layerInfos_G[layerIndex].indivisibleNodeCount;
        groupIndivisibleNodeCountSum_G = indivisibleNodeCountSum_G;
    }
    GroupMemoryBarrierWithGroupSync();

    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    if (threadIndex < ((layerNodeCount_G + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)) {
        bool oddLayer = (pushConst.currentLayer & 1) == 1;

        uint nodeIndex = 0;
        OctreeNodeInfo_G_FzbPG nodeData; nodeData.label_indivisible = 1;
        if (threadIndex < layerNodeCount_G) {
            uint fatherNodeLabel = threadIndex / 8;
            uint fatherNodeIndex;
            if (oddLayer) fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].x;
            else fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].y;
            nodeIndex = fatherNodeIndex * 8 + blockLane;
            nodeData = OctreeNodeInfoBuffer_G[pushConst.currentLayer][nodeIndex];

            //if (pushConst.frameIndex == 1 && pushConst.currentLayer == 3 && blockLane == 0 && fatherNodeLabel == 66)
            //    printf("oddLayer: %d, fatherNodeLabel: %d, fatherNodeIndex: %d\n", oddLayer, fatherNodeLabel, DivisibleNodeInfoBuffer_G[fatherNodeLabel].x);
        }
        bool indivisible = nodeData.label_indivisible == 3;
        bool divisible = nodeData.label_indivisible == 2;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount =
            layerNodeCount_G - threadGroupIndex * GETOCTREELABEL_CS_THREADGROUP_SIZE >= GETOCTREELABEL_CS_THREADGROUP_SIZE ? GETOCTREELABEL_CS_THREADGROUP_SIZE : layerNodeCount_G % GETOCTREELABEL_CS_THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        int label = 0;
        if (layerNodeCount_G <= WARP_SIZE) {
            if (divisible) label = divisibleLabelInWarp + 1;
            if (indivisible) label = indivisibleLabelInWarp + 1;
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

            if (divisible) label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) label = indivisibleLabelInThreadGroup + 1;
        }

        if (divisible && layerNodeCount_G <= GETOCTREELABEL_CS_THREADGROUP_SIZE) {
            if (oddLayer) DivisibleNodeInfoBuffer_G[label - 1].y = nodeIndex;
            else DivisibleNodeInfoBuffer_G[label - 1].x = nodeIndex;
        }
        if (indivisible && layerNodeCount_G <= GETOCTREELABEL_CS_THREADGROUP_SIZE) {
            IndivisibleNodeInfoBuffer_G[groupIndivisibleNodeCountSum_G + label - 1] = uint2(pushConst.currentLayer, nodeIndex);
        }

        if (indivisible || divisible) OctreeNodeInfoBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + uint(indivisible);

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (layerNodeCount_G <= WARP_SIZE) {
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (layerNodeCount_G <= GETOCTREELABEL_CS_THREADGROUP_SIZE) {
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                OctreeThreadGroupInfoBuffer[threadGroupIndex].threadGroupDivisibleNodeCount_G = threadGroupDivisibleNodeCount;
                OctreeThreadGroupInfoBuffer[threadGroupIndex].threadGroupIndivisibleNodeCount_G = threadGroupInDivisibleNodeCount;
            }
        }
    }

    if (threadIndex == 0 && layerNodeCount_G <= GETOCTREELABEL_CS_THREADGROUP_SIZE) GlobalInfoBuffer[0].cmd.x = 1;
}

groupshared uint groupThreadGroupDivisibleNodeCountSum_G;
groupshared uint groupThreadGroupIndivisibleNodeCountSum_G;
[numthreads(GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getOctreeLabel3(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (warpIndex == 0) { // this assume groupThreadCount < 32
        OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
        if (groupThreadIndex == 0)
            groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

        uint indivisibleNodeCountSum_G = 0;
        if (groupThreadIndex < pushConst.currentLayer) indivisibleNodeCountSum_G = globalInfo.layerInfos_G[groupThreadIndex].indivisibleNodeCount;
        groupIndivisibleNodeCountSum_G = WaveActiveSum(indivisibleNodeCountSum_G);

        uint threadGroupDivisibleNodeCount = 0;
        uint threadGroupIndivisibleNodeCount = 0;
        if (groupThreadIndex < threadGroupIndex) {
            OctreeThreadGroupInfo_FzbPG threadGroupInfo = OctreeThreadGroupInfoBuffer[groupThreadIndex];
            threadGroupDivisibleNodeCount += threadGroupInfo.threadGroupDivisibleNodeCount_G;
            threadGroupIndivisibleNodeCount += threadGroupInfo.threadGroupIndivisibleNodeCount_G;
        }
        threadGroupDivisibleNodeCount = WaveActiveSum(threadGroupDivisibleNodeCount);
        threadGroupIndivisibleNodeCount = WaveActiveSum(threadGroupIndivisibleNodeCount);
        if (warpLane == 0) {
            groupThreadGroupDivisibleNodeCountSum_G = threadGroupDivisibleNodeCount;
            groupThreadGroupIndivisibleNodeCountSum_G = threadGroupIndivisibleNodeCount;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    bool oddLayer = (pushConst.currentLayer & 1) == 1;
    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;

    if (layerNodeCount_G > GETOCTREELABEL_CS_THREADGROUP_SIZE && threadIndex < layerNodeCount_G) {
        uint blockLane = threadIndex & 7;
        uint fatherNodeLabel = threadIndex / 8;
        uint fatherNodeIndex;
        if (oddLayer) fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].x;
        else fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].y;
        uint nodeIndex = fatherNodeIndex * 8 + blockLane;
        OctreeNodeInfo_G_FzbPG nodeData = OctreeNodeInfoBuffer_G[pushConst.currentLayer][nodeIndex];

        bool indivisible = (nodeData.label_indivisible & 1) == 1 && nodeData.label_indivisible > 0;
        bool divisible = (nodeData.label_indivisible & 1) == 0 && nodeData.label_indivisible > 0;
        if (divisible) {
            uint label = (nodeData.label_indivisible >> 1) + groupThreadGroupDivisibleNodeCountSum_G;
            OctreeNodeInfoBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = label << 1;

            uint fatherNodeLabel = threadIndex / 8;
            if (oddLayer) {
                uint fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].x;
                DivisibleNodeInfoBuffer_G[label - 1].y = fatherNodeIndex * 8 + blockLane;
            } else {
                uint fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].y;
                DivisibleNodeInfoBuffer_G[label - 1].x = fatherNodeIndex * 8 + blockLane;
            }
        }
        else if (indivisible) {
            uint label = (nodeData.label_indivisible >> 1) + groupThreadGroupIndivisibleNodeCountSum_G;
            OctreeNodeInfoBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + 1;
            IndivisibleNodeInfoBuffer_G[groupIndivisibleNodeCountSum_G + label - 1] = uint2(pushConst.currentLayer, nodeIndex);
        }
    }

    if (threadIndex == 0) {
        OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
        uint nextLayerNodeCount = globalInfo.layerInfos_G[pushConst.currentLayer].divisibleNodeCount * 8;
        uint cmdX = (nextLayerNodeCount + GETOCTREELABEL_CS_THREADGROUP_SIZE - 1) / GETOCTREELABEL_CS_THREADGROUP_SIZE;
        globalInfo.cmd.x = cmdX;
        GlobalInfoBuffer[0] = globalInfo;
    }
}

groupshared uint groupIndivisibleNodeCount_G;
[numthreads(GETOCTREELABEL4_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getOctreeLabel4(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < WARP_SIZE) {
        OctreeGlobalInfo_FzbPG octreeGlobalInfo = GlobalInfoBuffer[0];

        uint indivisibleNodeCount_G = 0;
        if (groupThreadIndex < MAX_OCTREE_LAYER_FZBPG) indivisibleNodeCount_G = octreeGlobalInfo.layerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);

        if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = indivisibleNodeCount_G;
    }
    GroupMemoryBarrierWithGroupSync();

    uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    if (indivisibleNodeCount_G > IndivisibleNodeCount_G_FZBPG) {
        if (threadIndex == 0) {
            printf("Error: G indivisibleNodeCount_G exceeds the maximum node count. maxNodeCount: %u,   currentNodeCount: %u\n",
                   IndivisibleNodeCount_G_FZBPG, groupIndivisibleNodeCount_G);
            GlobalInfoBuffer[0].indivisibleNodeCount_G = 0;
        }
        return;
    }

    if (threadIndex == 0) {
        OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
        globalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
        GlobalInfoBuffer[0] = globalInfo;

        #ifndef NDEBUG
        if (pushConst.frameIndex == 1) {
            printf("indivisibleNodeCount_G: %d\n", indivisibleNodeCount_G);
            for (int i = 0; i <= pushConst.octreeMaxLayer; ++i) {
                printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                       globalInfo.layerInfos_G[i].divisibleNodeCount, globalInfo.layerInfos_G[i].indivisibleNodeCount
                );
            }
        }
        #endif
    }
}