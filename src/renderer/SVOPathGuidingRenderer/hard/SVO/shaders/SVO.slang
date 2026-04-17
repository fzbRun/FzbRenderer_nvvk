#include "common/Shader/shaderStructType.h"
#include "renderer/SVOPathGuidingRenderer/hard/SVO/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOPushConstant_SVOPG> pushConst;
[[vk::binding(BindingPoints_SVOPG::eOctreeArray_G_SVOPG)]] StructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_SVOPG::eSVO_G_SVOPG)]] RWStructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(BindingPoints_SVOPG::eSVOGlobalInfo_SVOPG)]] RWStructuredBuffer<SVOGlobalInfo_SVOPG> SVOGlobalInfo;
[[vk::binding(BindingPoints_SVOPG::eSVODivisibleNodeIndices_G_SVOPG)]] RWStructuredBuffer<uint2> SVODivisibleNodeInfos_G;
[[vk::binding(BindingPoints_SVOPG::eSVOThreadGroupInfos_SVOPG)]] RWStructuredBuffer<SVOThreadGroupInfo> SVOThreadGroupInfos;

groupshared uint groupDivisibleNodeCount_layer1;
groupshared uint groupDivisibleNodeIndices_layer1[6];
groupshared uint groupWarpIndivisibleNodeCount_layer2[2];
groupshared uint groupWarpDivisibleNodeCount_layer2[2];

groupshared uint groupFatherLayerDivisibleNodeCount_G;
groupshared uint groupSVONodeTotalCount_G;
groupshared uint groupWarpDivisibleNodeCountSums_G[CREATESVO_CS_THREADGROUP_SIZE / 32];
groupshared uint groupWarpIndivisibleNodeCountSums_G[CREATESVO_CS_THREADGROUP_SIZE / 32];

groupshared uint groupDivisibleNodeCountSum_G;
groupshared uint groupIndivisibleNodeCountSum_G;
//----------------------------------------------initSVOArray--------------------------------------------
[numthreads(INITSVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    SVONodeData_G nodeData_init;
    {
        nodeData_init.meanNormal = float4(0.0f);
        nodeData_init.aabb.minimum.x = 3.402823466e+38F;
        nodeData_init.aabb.minimum.y = 3.402823466e+38F;
        nodeData_init.aabb.minimum.z = 3.402823466e+38F;
        nodeData_init.aabb.maximum.x = -3.402823466e+38F;
        nodeData_init.aabb.maximum.y = -3.402823466e+38F;
        nodeData_init.aabb.maximum.z = -3.402823466e+38F;
        nodeData_init.indivisible = 1;
        nodeData_init.label = 0;

        #ifdef CLUSTER_WITH_MATERIAL
        nodeData_init.materialIndex = 0;
        #endif
        //[unroll]
        //for (int i = 0; i < NEARBY_NODE_COUNT; ++i)
        //    nodeData_init.nearbyNodeIndices[i] = -1;
    }
  
    if (threadGroupIndex == 0) {
        uint indivisibleNodeCount_layer1 = 0;
        uint divisibleNodeCount_layer1 = 0;
        uint indivisibleNodeCount_layer2 = 0;
        uint divisibleNodeCount_layer2 = 0;

        if (threadIndex < 6) {
            OctreeNodeData_G nodeData = OctreeArray_G[0][threadIndex];
            bool hasData = nodeData.meanNormal.w > 0.0f;
            bool indivisible = hasData && nodeData.indivisible == 1;
            bool divisible = hasData && nodeData.indivisible == 0;

            indivisibleNodeCount_layer1 = WaveActiveCountBits(indivisible);
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);

            uint indivisibleLabel = WavePrefixCountBits(indivisible);
            uint divisibleLabel = WavePrefixCountBits(divisible);

            if (indivisible) nodeData.label = indivisibleLabel + 1;
            if (divisible) {
                nodeData.label = divisibleLabel + 1;
                groupDivisibleNodeIndices_layer1[divisibleLabel] = threadIndex;
            }
            #ifdef CLUSTER_WITH_MATERIAL
            uint materialIndex = 0; uint maxMaterialCount = 0;
            for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                if (nodeData.materialCounts[i] > maxMaterialCount) materialIndex = i;
            }
            SVO_G[threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label, materialIndex };
            #else
            SVO_G[threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label };
            #endif

            if (threadIndex == 0) groupDivisibleNodeCount_layer1 = divisibleNodeCount_layer1;
        }
        GroupMemoryBarrierWithGroupSync();

        int nodeCount_layer2 = groupDivisibleNodeCount_layer1 * 8; // max 48
        if (nodeCount_layer2 <= 32) {
            if (threadIndex < nodeCount_layer2) {
                uint fatherNodeIndex = groupDivisibleNodeIndices_layer1[threadIndex / 8];
                uint nodeIndex = fatherNodeIndex * 8;

                OctreeNodeData_G nodeData = OctreeArray_G[1][nodeIndex];
                bool hasData = nodeData.meanNormal.w > 0.0f;
                bool indivisible = hasData && nodeData.indivisible == 1;
                bool divisible = hasData && nodeData.indivisible == 0;

                indivisibleNodeCount_layer2 = WaveActiveCountBits(indivisible);
                divisibleNodeCount_layer2 = WaveActiveCountBits(divisible);

                uint indivisibleLabel = WavePrefixCountBits(indivisible);
                uint divisibleLabel = WavePrefixCountBits(divisible);

                if (indivisible) nodeData.label = indivisibleLabel + 1;
                if (divisible) {
                    nodeData.label = divisibleLabel + 1;
                    SVODivisibleNodeInfos_G[divisibleLabel] = nodeIndex;
                }

                #ifdef CLUSTER_WITH_MATERIAL
                uint materialIndex = 0; uint maxMaterialCount = 0;
                for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                    if (nodeData.materialCounts[i] > maxMaterialCount) materialIndex = i;
                }
                SVO_G[8 + threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label, materialIndex };
                #else
                SVO_G[8 + threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label };
                #endif
            }
        } else {
            uint warpIndex = threadIndex / 32;
            uint warpLane = threadIndex & 31;
            uint blockLane = threadIndex & 7;

            OctreeNodeData_G nodeData; uint nodeIndex = 0;
            uint indivisibleLabel; uint divisibleLabel;
            bool indivisible; bool divisible;
            if (threadIndex < nodeCount_layer2) {
                uint fatherNodeIndex = groupDivisibleNodeIndices_layer1[threadIndex / 8];
                nodeIndex = fatherNodeIndex * 8 + blockLane;

                nodeData = OctreeArray_G[1][nodeIndex];
                bool hasData = nodeData.meanNormal.w > 0.0f;
                indivisible = hasData && nodeData.indivisible == 1;
                divisible = hasData && nodeData.indivisible == 0;

                indivisibleNodeCount_layer2 = WaveActiveCountBits(indivisible);
                divisibleNodeCount_layer2 = WaveActiveCountBits(divisible);

                indivisibleLabel = WavePrefixCountBits(indivisible);
                divisibleLabel = WavePrefixCountBits(divisible);

                if (indivisible) nodeData.label = indivisibleLabel + 1;
                if (divisible) nodeData.label = divisibleLabel + 1;

                if (warpLane == 0) {
                    groupWarpIndivisibleNodeCount_layer2[warpIndex] = indivisibleNodeCount_layer2;
                    groupWarpDivisibleNodeCount_layer2[warpIndex] = divisibleNodeCount_layer2;
                }
            }
            GroupMemoryBarrierWithGroupSync();
            if (threadIndex < nodeCount_layer2) {
                if (warpIndex == 1) {
                    if (indivisible) nodeData.label += groupWarpIndivisibleNodeCount_layer2[0];
                    if (divisible) nodeData.label += groupWarpDivisibleNodeCount_layer2[0];
                }
                if (divisible) SVODivisibleNodeInfos_G[nodeData.label - 1] = nodeIndex;

                #ifdef CLUSTER_WITH_MATERIAL
                uint materialIndex = 0; uint maxMaterialCount = 0;
                for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                    if (nodeData.materialCounts[i] > maxMaterialCount) materialIndex = i;
                }
                SVO_G[8 + threadIndex] = SVO_G[threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label, materialIndex };
                #else
                SVO_G[8 + threadIndex] = SVO_G[threadIndex] = { nodeData.meanNormal, nodeData.aabb, nodeData.indivisible, nodeData.label };
                #endif
            }
        }

        if (threadIndex < 2) SVO_G[threadIndex + 6] = nodeData_init;
        uint initNodeCount = 56 - 8 - nodeCount_layer2;
        if (threadIndex < initNodeCount) SVO_G[8 + nodeCount_layer2 + threadIndex] = nodeData_init;

        if (threadIndex == 0) {
            SVOGlobalInfo_SVOPG globalInfo;

            globalInfo.layerInfos_G[0].divisibleNodeCount = 1;
            globalInfo.layerInfos_G[0].indivisibleNodeCount = 0;

            globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1;
            globalInfo.layerInfos_G[1].indivisibleNodeCount = indivisibleNodeCount_layer1;

            if (nodeCount_layer2 <= 32) {
                globalInfo.layerInfos_G[2].divisibleNodeCount = divisibleNodeCount_layer2;
                globalInfo.layerInfos_G[2].indivisibleNodeCount = indivisibleNodeCount_layer2;
            } else {
                globalInfo.layerInfos_G[2].divisibleNodeCount = divisibleNodeCount_layer2 + groupWarpDivisibleNodeCount_layer2[1];
                globalInfo.layerInfos_G[2].indivisibleNodeCount = indivisibleNodeCount_layer2 + groupWarpIndivisibleNodeCount_layer2[1];
            }

            for (int i = 3; i < MAX_SVO_LAYER; ++i) {
                globalInfo.layerInfos_G[i].divisibleNodeCount = 0;
                globalInfo.layerInfos_G[i].indivisibleNodeCount = 0;
            }

            globalInfo.cmd = { (globalInfo.layerInfos_G[2].divisibleNodeCount * 8 + CREATESVO_CS_THREADGROUP_SIZE - 1) / CREATESVO_CS_THREADGROUP_SIZE, 1, 1 };
            SVOGlobalInfo[0] = globalInfo;
        }
    }
    {
        if (threadIndex < SVOSize / CREATESVO_CS_THREADGROUP_SIZE) {
            SVOThreadGroupInfo threadGroupInfo;
            threadGroupInfo.threadGroupDivisibleNodeCount_G = 0;
            threadGroupInfo.threadGroupDivisibleNodeCount_E = 0;
            threadGroupInfo.threadGroupIndivisibleNodeCount_G = 0;
            threadGroupInfo.threadGroupIndivisibleNodeCount_E = 0;
            SVOThreadGroupInfos[threadIndex] = threadGroupInfo;
        }

        if (threadGroupIndex > 0) {
            uint startIndex = 56 + threadIndex - INITSVO_CS_THREADGROUP_SIZE;
            for (int i = startIndex; i < SVOSize_G; i += INITSVO_CS_THREADGROUP_SIZE) SVO_G[i] = nodeData_init;
        }
    }
}
//-----------------------------------------------createSVOArray-------------------------------------------
[numthreads(CREATESVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];
        groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

        uint svoNodeTotalCount_G = 0, svoNodeTotalCount_E = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentLayer - 1; ++layerIndex)
            svoNodeTotalCount_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
        groupSVONodeTotalCount_G = svoNodeTotalCount_G;
    }
    GroupMemoryBarrierWithGroupSync();

    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    uint lastLayerNodeCountSum_G = groupSVONodeTotalCount_G;
    if (layerNodeCount_G + lastLayerNodeCountSum_G > SVOSize_G) {
        if (threadIndex == 0) {
            SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = 0;
            printf("Error: G totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,   maxNodeCount: %u\n",
                   layerNodeCount_G + lastLayerNodeCountSum_G, SVOSize_G);
        }
        return;
    }
    if (threadIndex < ((layerNodeCount_G + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)) {
        bool oddLayer = (pushConst.currentLayer & 1) == 1;

        uint nodeIndex = 0;
        SVONodeData_G nodeData;
        bool hasData = false;
        if (threadIndex < layerNodeCount_G) {
            uint fatherNodeLabel = threadIndex / 8;
            uint fatherNodeIndex;
            if (oddLayer) fatherNodeIndex = SVODivisibleNodeInfos_G[fatherNodeLabel].x;
            else fatherNodeIndex = SVODivisibleNodeInfos_G[fatherNodeLabel].y;
            nodeIndex = fatherNodeIndex * 8 + blockLane;
            OctreeNodeData_G octreeNodeData = OctreeArray_G[pushConst.currentLayer - 1][nodeIndex];

            #ifdef CLUSTER_WITH_MATERIAL
            uint materialIndex = 0; uint maxMaterialCount = 0;
            for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                if (octreeNodeData.materialCounts[i] > maxMaterialCount) materialIndex = i;
            }
            nodeData = { octreeNodeData.meanNormal, octreeNodeData.aabb, octreeNodeData.indivisible, octreeNodeData.label, materialIndex };
            #else
            nodeData = { octreeNodeData.meanNormal, octreeNodeData.aabb, octreeNodeData.indivisible, octreeNodeData.label };
            #endif
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
            layerNodeCount_G - threadGroupIndex * CREATESVO_CS_THREADGROUP_SIZE >= CREATESVO_CS_THREADGROUP_SIZE ? CREATESVO_CS_THREADGROUP_SIZE : layerNodeCount_G % CREATESVO_CS_THREADGROUP_SIZE;
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

        if (divisible && layerNodeCount_G <= CREATESVO_CS_THREADGROUP_SIZE) {
            if (oddLayer) SVODivisibleNodeInfos_G[nodeData.label - 1].y = nodeIndex;
            else SVODivisibleNodeInfos_G[nodeData.label - 1].x = nodeIndex;
        }
        if (hasData) SVO_G[lastLayerNodeCountSum_G + threadIndex] = nodeData;

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (layerNodeCount_G <= WARP_SIZE) {
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (layerNodeCount_G <= CREATESVO_CS_THREADGROUP_SIZE) {
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_G = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_G = threadGroupInDivisibleNodeCount;
            }
        }
    }

    if (threadIndex == 0 && layerNodeCount_G <= CREATESVO_CS_THREADGROUP_SIZE) SVOGlobalInfo[0].cmd.x = 1;
}

[numthreads(CREATESVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_offsetLabelMultiBlock(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (warpIndex == 0) { // this assume groupThreadCount < 32
        if (groupThreadIndex == 0) {
            SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];
            groupFatherLayerDivisibleNodeCount_G = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

            uint svoNodeTotalCount_G = 0, svoNodeTotalCount_E = 0;
            for (int layerIndex = 0; layerIndex < pushConst.currentLayer - 1; ++layerIndex)
                svoNodeTotalCount_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
            groupSVONodeTotalCount_G = svoNodeTotalCount_G;
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
    }
    GroupMemoryBarrierWithGroupSync();

    bool oddLayer = (pushConst.currentLayer & 1) == 1;
    uint layerNodeCount_G = groupFatherLayerDivisibleNodeCount_G * 8;
    uint lastLayerNodeCountSum_G = groupSVONodeTotalCount_G;
    if (layerNodeCount_G + lastLayerNodeCountSum_G > SVOSize_G) return;
    if (layerNodeCount_G > CREATESVO_CS_THREADGROUP_SIZE && threadIndex < layerNodeCount_G) {
        SVONodeData_G nodeData = SVO_G[lastLayerNodeCountSum_G + threadIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = nodeData.label + groupDivisibleNodeCountSum_G;
            SVO_G[lastLayerNodeCountSum_G + threadIndex].label = label;

            uint fatherNodeLabel = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;

            uint nodeIndex;
            if (oddLayer) {
                uint fatherNodeIndex = SVODivisibleNodeInfos_G[fatherNodeLabel].x;
                SVODivisibleNodeInfos_G[label - 1].y = fatherNodeIndex * 8 + blockLane;
            } else {
                uint fatherNodeIndex = SVODivisibleNodeInfos_G[fatherNodeLabel].y;
                SVODivisibleNodeInfos_G[label - 1].x = fatherNodeIndex * 8 + blockLane;
            }
        }
        else if (indivisible) SVO_G[lastLayerNodeCountSum_G + threadIndex].label += groupIndivisibleNodeCountSum_G;
    }

    if (threadIndex == 0) {
        uint nextLayerNodeCount = SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount * 8;
        uint cmdX = (nextLayerNodeCount + CREATESVO_CS_THREADGROUP_SIZE - 1) / CREATESVO_CS_THREADGROUP_SIZE;
        SVOGlobalInfo[0].cmd.x = cmdX;

        if (pushConst.frameIndex == 0) {
            uint divisibleNodeCount = SVOGlobalInfo[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount;
            printf("layer: %d, layerNodeCount: %d, divisibleNodeCount: %d\n", pushConst.currentLayer, layerNodeCount_G, divisibleNodeCount);
        }

        if (cmdX == 0) {
            uint SVOTotalNodeCount_G = 0;
            for (int i = 0; i <= pushConst.currentLayer; ++i)
                SVOTotalNodeCount_G += SVOGlobalInfo[0].layerInfos_G[i].divisibleNodeCount * 8;
            SVOGlobalInfo[0].totalNodeCount_G = SVOTotalNodeCount_G;

            #ifndef NDEBUG
            SVOGlobalInfo[0].drawCmd.instanceCount = SVOTotalNodeCount_G;
            if (pushConst.frameIndex == 0) printf("SVOTotalNodeCount_G: %d\n", SVOTotalNodeCount_G);
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
};
struct PSout_Wireframe
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout_Wireframe vertexMain_Wireframe(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout_Wireframe output;

    SVOGlobalInfo_SVOPG globalInfo = SVOGlobalInfo[0];

    uint nodeIndex = instanceIndex;
    uint layerIndex = 1;
    for (; layerIndex <= pushConst.svoMaxLayer; ++layerIndex) {
        uint layerNodeCount = globalInfo.layerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
        if (nodeIndex < layerNodeCount) break;
        nodeIndex -= layerNodeCount;
    }
    output.color = Colors[layerIndex];

    SVONodeData_G nodeData = SVO_G[instanceIndex];
    bool hasData = nodeData.meanNormal.w > 0.0f;
    bool show = hasData && nodeData.indivisible == 1;
    AABB nodeAABB = nodeData.aabb;

    //if (layerIndex == 6) show = false;
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
    output.color0 = input.color;
    return output;
}
#endif