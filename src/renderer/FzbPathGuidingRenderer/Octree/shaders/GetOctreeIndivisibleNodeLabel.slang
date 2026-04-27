#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eClusterLayerData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> ClusterlayerDataBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> DivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eThreadGroupInfos)]] RWStructuredBuffer<OctreeThreadGroupInfo_FzbPG> OctreeThreadGroupInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> IndivisibleNodeInfoBuffer_E;
//------------------------------------------------getLabel---------------------------------------------------------
groupshared uint groupDivisibleNodeIndices_layer0[6];
groupshared uint groupDivisibleNodeCount_layer0;
groupshared uint groupIndivisibleNodeCount_layer0;
groupshared uint groupWarpIndivisibleNodeCount_layer1[2];
groupshared uint groupWarpDivisibleNodeCount_layer1[2];

[numthreads(GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getOctreeLabel1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint indivisibleNodeCount_layer0 = 0;
    uint divisibleNodeCount_layer0 = 0;
    uint indivisibleNodeCount_layer1 = 0;
    uint divisibleNodeCount_layer1 = 0;

    if (threadIndex < 6) {
        OctreeNodeData_G_FzbPG nodeData = OctreeDataBuffer_G[0][threadIndex];
        bool indivisible = nodeData.label_indivisible == 3;
        bool divisible = nodeData.label_indivisible == 2;

        indivisibleNodeCount_layer0 = WaveActiveCountBits(indivisible);
        divisibleNodeCount_layer0 = WaveActiveCountBits(divisible);

        uint indivisibleLabel = WavePrefixCountBits(indivisible);
        uint divisibleLabel = WavePrefixCountBits(divisible);

        int label = 0;
        if (indivisible) {
            label = indivisibleLabel + 1;
            IndivisibleNodeInfoBuffer_G[indivisibleLabel] = uint2(0, threadIndex);
        }
        if (divisible) {
            label = divisibleLabel + 1;
            groupDivisibleNodeIndices_layer0[divisibleLabel] = threadIndex;
        }
        if (indivisible || divisible) OctreeDataBuffer_G[0][threadIndex].label_indivisible = (label << 1) + uint(indivisible);

        if (threadIndex == 0) {
            groupDivisibleNodeCount_layer0 = divisibleNodeCount_layer0;
            groupIndivisibleNodeCount_layer0 = indivisibleNodeCount_layer0;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int nodeCount_layer2 = groupDivisibleNodeCount_layer0 * 8; // max 48
    if (nodeCount_layer2 <= 32) {
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            uint nodeIndex = fatherNodeIndex * 8 + threadIndex & 7;

            OctreeNodeData_G_FzbPG nodeData = OctreeDataBuffer_G[1][nodeIndex];
            bool indivisible = nodeData.label_indivisible == 3;
            bool divisible = nodeData.label_indivisible == 2;

            indivisibleNodeCount_layer1 = WaveActiveCountBits(indivisible);
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);

            uint indivisibleLabel = WavePrefixCountBits(indivisible);
            uint divisibleLabel = WavePrefixCountBits(divisible);

            int label = 0;
            if (indivisible) {
                label = indivisibleLabel + 1;
                IndivisibleNodeInfoBuffer_G[groupIndivisibleNodeCount_layer0 + indivisibleLabel] = uint2(1, nodeIndex);
            }
            if (divisible) {
                label = divisibleLabel + 1;
                DivisibleNodeInfoBuffer_G[divisibleLabel] = nodeIndex;
            }
            OctreeDataBuffer_G[1][threadIndex].label_indivisible = (label << 1) | uint(indivisible);
        }
    } else {
        uint warpIndex = threadIndex / 32;
        uint warpLane = threadIndex & 31;
        uint blockLane = threadIndex & 7;

        OctreeNodeData_G_FzbPG nodeData; uint nodeIndex = 0;
        uint indivisibleLabel; uint divisibleLabel;
        bool indivisible = false; bool divisible = false;
        int label = 0;
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            nodeIndex = fatherNodeIndex * 8 + blockLane;

            nodeData = OctreeDataBuffer_G[1][nodeIndex];
            indivisible = nodeData.label_indivisible == 3;
            divisible = nodeData.label_indivisible == 2;

            indivisibleNodeCount_layer1 = WaveActiveCountBits(indivisible);
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);

            indivisibleLabel = WavePrefixCountBits(indivisible);
            divisibleLabel = WavePrefixCountBits(divisible);

            if (indivisible) label = indivisibleLabel + 1;
            if (divisible) label = divisibleLabel + 1;

            if (warpLane == 0) {
                groupWarpIndivisibleNodeCount_layer1[warpIndex] = indivisibleNodeCount_layer1;
                groupWarpDivisibleNodeCount_layer1[warpIndex] = divisibleNodeCount_layer1;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (threadIndex < nodeCount_layer2) {
            if (warpIndex == 1) {
                if (indivisible) label += groupWarpIndivisibleNodeCount_layer1[0];
                if (divisible) label += groupWarpDivisibleNodeCount_layer1[0];
            }
            if (divisible) DivisibleNodeInfoBuffer_G[label - 1] = nodeIndex;
            else IndivisibleNodeInfoBuffer_G[groupIndivisibleNodeCount_layer0 + label - 1] = uint2(1, nodeIndex);
            if (indivisible || divisible) OctreeDataBuffer_G[1][threadIndex].label_indivisible = (label << 1) + uint(indivisible);
        }
    }

    if (threadIndex == 0) {
        OctreeGlobalInfo_FzbPG globalInfo;

        globalInfo.layerInfos_G[0].divisibleNodeCount = divisibleNodeCount_layer0;
        globalInfo.layerInfos_G[0].indivisibleNodeCount = 6 - divisibleNodeCount_layer0;

        if (nodeCount_layer2 <= 32) {
            globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1;
            globalInfo.layerInfos_G[1].indivisibleNodeCount = indivisibleNodeCount_layer1;
        } else {
            globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1 + groupWarpDivisibleNodeCount_layer1[1];
            globalInfo.layerInfos_G[1].indivisibleNodeCount = indivisibleNodeCount_layer1 + groupWarpIndivisibleNodeCount_layer1[1];
        }

        for (int i = 2; i < MAX_OCTREE_LAYER_FZBPG; ++i) {
            globalInfo.layerInfos_G[i].divisibleNodeCount = 0;
            globalInfo.layerInfos_G[i].indivisibleNodeCount = 0;
        }

        globalInfo.cmd = { (globalInfo.layerInfos_G[1].divisibleNodeCount * 8 + GETOCTREELABEL_CS_THREADGROUP_SIZE - 1) / GETOCTREELABEL_CS_THREADGROUP_SIZE, 1, 1 };
        GlobalInfoBuffer[0] = globalInfo;
    }
}
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
        OctreeNodeData_G_FzbPG nodeData; nodeData.label_indivisible = 1;
        if (threadIndex < layerNodeCount_G) {
            uint fatherNodeLabel = threadIndex / 8;
            uint fatherNodeIndex;
            if (oddLayer) fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].x;
            else fatherNodeIndex = DivisibleNodeInfoBuffer_G[fatherNodeLabel].y;
            nodeIndex = fatherNodeIndex * 8 + blockLane;
            nodeData = OctreeDataBuffer_G[pushConst.currentLayer][nodeIndex];
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
        if (indivisible && layerNodeCount_G <= GETOCTREELABEL_CS_THREADGROUP_SIZE)
            IndivisibleNodeInfoBuffer_G[groupIndivisibleNodeCountSum_G + label - 1] = uint2(pushConst.currentLayer, nodeIndex);
        if (indivisible || divisible) OctreeDataBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + uint(indivisible);

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
        OctreeNodeData_G_FzbPG nodeData = OctreeDataBuffer_G[pushConst.currentLayer][nodeIndex];

        bool indivisible = nodeData.label_indivisible == 3;
        bool divisible = nodeData.label_indivisible == 2;
        if (divisible) {
            uint label = (nodeData.label_indivisible >> 1) + groupThreadGroupDivisibleNodeCountSum_G;
            OctreeDataBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = label << 1;

            uint fatherNodeLabel = threadIndex / 8;

            uint nodeIndex;
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
            OctreeDataBuffer_G[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + 1;
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
groupshared uint groupWarpIndivisibleNodeCount_E[(OCTREE_NODECOUNT_E_FZBPG + 31) / 32];
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
            GlobalInfoBuffer[0].indivisibleNodeCount_E = 0;
        }
        return;
    }

    if (threadGroupIndex == 0) {
        uint warpIndex = threadIndex / 32;
        uint warpLane = threadIndex & 31;

        uint warpCount = (CLUSTER_LAYER_NODECOUNT_E_FZBPG + 31) / 32; //(384 + 31) / 32;
        bool indivisible = false;
        uint warpIndivisibleNodeCount;
        if (threadIndex < warpCount * 32) {
            if (threadIndex < CLUSTER_LAYER_NODECOUNT_E_FZBPG) {
                OctreeNodeData_E_FzbPG nodeData = ClusterlayerDataBuffer_E[threadIndex];
                indivisible = nodeData.label > 0;
            }
            warpIndivisibleNodeCount = WaveActiveCountBits(indivisible);
            if (warpLane == 0) groupWarpIndivisibleNodeCount_E[warpIndex] = warpIndivisibleNodeCount;
        }
        GroupMemoryBarrierWithGroupSync();
        if (threadIndex < warpCount * 32) {
            uint nodeWarpLabel = WavePrefixCountBits(indivisible);

            uint nodeGroupLabel = 0;
            if (warpLane < warpIndex) nodeGroupLabel = groupWarpIndivisibleNodeCount_E[warpLane];
            nodeGroupLabel = WaveActiveSum(nodeGroupLabel);

            if (indivisible) {
                uint label = nodeWarpLabel + nodeGroupLabel + 1;
                IndivisibleNodeInfoBuffer_E[label - 1] = threadIndex;
                ClusterlayerDataBuffer_E[threadIndex].label = label;
            }

            if (warpIndex == warpCount - 1 && warpLane == 0) {
                OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
                globalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
                globalInfo.indivisibleNodeCount_E = nodeGroupLabel + warpIndivisibleNodeCount;
                GlobalInfoBuffer[0] = globalInfo;

                #ifndef NDEBUG
                if (pushConst.frameIndex == 1) {
                    printf("indivisibleNodeCount: G: %d  E: %d\n",
                        indivisibleNodeCount_G, globalInfo.indivisibleNodeCount_E
                    );
                    for (int i = 0; i <= pushConst.octreeMaxLayer; ++i) {
                        printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               globalInfo.layerInfos_G[i].divisibleNodeCount, globalInfo.layerInfos_G[i].indivisibleNodeCount
                        );
                    }
                }
                #endif
            }
        }
    }
}
//------------------------------------------------debug---------------------------------------------------------
#ifndef NDEBUG
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_G)]] RWStructuredBuffer<OctreeNodeClusterData_G_FzbPG, ScalarDataLayout> OctreeClusterDataBuffer_G[];

static const float4 Colors[8] = {
    float4(1.0f, 1.0f, 1.0f, 1.0f),
    float4(1.0f, 0.0f, 0.0f, 1.0f),
    float4(0.0f, 1.0f, 0.0f, 1.0f),
    float4(0.0f, 0.0f, 1.0f, 1.0f),
    float4(1.0f, 1.0f, 0.0f, 1.0f),
    float4(1.0f, 0.0f, 1.0f, 1.0f),
    float4(0.0f, 1.0f, 1.0f, 1.0f),
    float4(0.0f, 0.0f, 0.0f, 1.0f),
};
struct VSout_OctreeIndivisibleNodes {
    float4 sv_position : SV_Position;
    float4 color;
};
struct PSout_OctreeIndivisibleNodes
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout_OctreeIndivisibleNodes vertexMain_OctreeIndivisibleNodes(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout_OctreeIndivisibleNodes output;
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];

    if (instanceIndex >= globalInfo.indivisibleNodeCount_G) return output;

    uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[instanceIndex];
    uint layerIndex = nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG;
    uint nodeIndex = nodeInfo.y;
    output.color = Colors[layerIndex];

    OctreeNodeClusterData_G_FzbPG nodeData = OctreeClusterDataBuffer_G[layerIndex][nodeIndex];
    bool hasData = nodeData.meanNormal.w > 0.0f;
    bool show = hasData && nodeData.indivisible == 1;
    AABB nodeAABB = nodeData.aabb;

    if (show) {
        float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += nodeAABB.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    }

    return output;
}
[shader("pixel")]
PSout_OctreeIndivisibleNodes fragmentMain_OctreeIndivisibleNodes(VSout_OctreeIndivisibleNodes input) {
    PSout_OctreeIndivisibleNodes output;
    output.color0 = input.color;
    return output;
}
#endif