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
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initSVOArray(uint threadIndex: SV_DispatchThreadID) {
    uint maxNodeCount = pushConst.sizes[pushConst.maxDepth];
    if (threadIndex >= maxNodeCount) return;

    if (threadIndex == 0) {
        SVOLayerInfos_G[0].divisibleNodeCount = 1;
        SVOLayerInfos_G[0].indivisibleNodeCount = 0;
        SVOLayerInfos_E[0].divisibleNodeCount = 1;
        SVOLayerInfos_E[0].indivisibleNodeCount = 0;
    }
    if (threadIndex > 0 && threadIndex <= pushConst.maxDepth) {
        SVOLayerInfos_G[threadIndex].divisibleNodeCount = 0;
        SVOLayerInfos_G[threadIndex].indivisibleNodeCount = 0;
        SVOLayerInfos_E[threadIndex].divisibleNodeCount = 0;
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

    for (int depth = pushConst.maxDepth; depth >= 0; --depth) {
        if (threadIndex >= pushConst.sizes[depth]) return;
        SVOArray_G[depth][threadIndex] = nodeData_G;
        SVOArray_E[depth][threadIndex] = nodeData_E;
    }
}
//-----------------------------------------------createSVOArray-------------------------------------------
groupshared uint warpDivisibleNodeCountSums[MAX_WARP_COUNT];
groupshared uint warpIndivisibleNodeCountSums[MAX_WARP_COUNT];
groupshared uint fatherLayerDivisibleNodeCount_G;
groupshared uint fatherLayerDivisibleNodeCount_E;
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVOArray(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    // Each thread processes one node. eight thread processes a block of nodes whose father nodes are the same.
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        fatherLayerDivisibleNodeCount_G = SVOLayerInfos_G[pushConst.currentDepth - 1].divisibleNodeCount;
        fatherLayerDivisibleNodeCount_E = SVOLayerInfos_E[pushConst.currentDepth - 1].divisibleNodeCount;
    }
    GroupMemoryBarrierWithGroupSync();
    //------------------------------------------------G---------------------------------------------------------
    uint totalNodeCount = fatherLayerDivisibleNodeCount_G * 8;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) {
        if (threadIndex == 0) {
            SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = 0;
            SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = 0;
            printf("Error: totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,          maxNodeCount: %u\n", totalNodeCount, pushConst.sizes[pushConst.currentDepth]);
        }
        return;
    }
    if (fatherLayerDivisibleNodeCount_G > 0) {
        uint fatherNodeIndex = threadIndex / 8;
        uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
        SVONodeData_G nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount) {
            nodeData = OctreeArray_G[pushConst.currentDepth][nodeIndex];
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
                totalNodeCount - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : totalNodeCount % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        if (fatherLayerDivisibleNodeCount_G <= 4) {
            if (divisible) nodeData.label = divisibleLabelInWarp + 1;
            if (indivisible) nodeData.label = indivisibleLabelInWarp + 1;
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
            uint indivisibleLabelInThreadGroup = warpIndivisibleNodeCountSums[warpIndex] + indivisibleLabelInWarp;

            if (divisible) nodeData.label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) nodeData.label = indivisibleLabelInThreadGroup + 1;
        }

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (totalNodeCount <= WARP_SIZE) {
                SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (totalNodeCount <= THREADGROUP_SIZE) {
                SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOLayerInfos_G[pushConst.currentDepth].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_G = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_G = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
            }
        }

        if (divisible && totalNodeCount <= THREADGROUP_SIZE) SVODivisibleNodeIndices_G[nodeData.label - 1] = nodeIndex;
        if (hasData) SVOArray_G[pushConst.currentDepth][threadIndex] = nodeData;
    }
    //------------------------------------------------E---------------------------------------------------------
    totalNodeCount = fatherLayerDivisibleNodeCount_E * 8;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) {
        if (threadIndex == 0) {
            SVOLayerInfos_G[pushConst.currentDepth].divisibleNodeCount = 0;
            SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = 0;
            printf("Error: totalNodeCount exceeds the maximum node count of the current layer. totalNodeCount: %u,         maxNodeCount: %u\n", totalNodeCount, pushConst.sizes[pushConst.currentDepth]);
        }
        return;
    }
    if (fatherLayerDivisibleNodeCount_E > 0) {
        uint fatherNodeIndex = threadIndex / 8;
        uint nodeIndex = SVODivisibleNodeIndices_E[fatherNodeIndex] * 8 + blockLane;
        SVONodeData_E nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount) {
            nodeData = OctreeArray_E[pushConst.currentDepth][nodeIndex];
            hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLableInWarp = WavePrefixSum(indivisible);

        uint threadGroupNodeCount = 
                totalNodeCount - threadGroupIndex * THREADGROUP_SIZE >= THREADGROUP_SIZE ? THREADGROUP_SIZE : totalNodeCount % THREADGROUP_SIZE;
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        if (fatherLayerDivisibleNodeCount_E <= 4) {
            if (divisible) nodeData.label = divisibleLabelInWarp + 1;
            if (indivisible) nodeData.label = indivisibleLableInWarp + 1;
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
            uint indivisibleLabelInThreadGroup = warpIndivisibleNodeCountSums[warpIndex] + indivisibleLableInWarp;

            if (divisible) nodeData.label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) nodeData.label = indivisibleLabelInThreadGroup + 1;
        }

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (totalNodeCount <= WARP_SIZE) {
                SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (totalNodeCount <= THREADGROUP_SIZE) {
                SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(SVOLayerInfos_E[pushConst.currentDepth].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(SVOLayerInfos_E[pushConst.currentDepth].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount_E = warpDivisibleNodeCountSums[warpIndex] + countbits(warpDivisibleNodeMask);
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount_E = warpIndivisibleNodeCountSums[warpIndex] + countbits(warpIndivisibleNodeMask);
            }
        }

        if (divisible && totalNodeCount <= THREADGROUP_SIZE) SVODivisibleNodeIndices_E[nodeData.label - 1] = nodeIndex;
        if (hasData) SVOArray_E[pushConst.currentDepth][threadIndex] = nodeData;
    }
}

groupshared uint threadGroupDivisibleNodeCountSum;
groupshared uint threadGroupIndivisibleNodeCountSum;
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_offsetLabelMultiBlock(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        fatherLayerDivisibleNodeCount_G = SVOLayerInfos_G[pushConst.currentDepth - 1].divisibleNodeCount;
        fatherLayerDivisibleNodeCount_E = SVOLayerInfos_E[pushConst.currentDepth - 1].divisibleNodeCount;
    }
    GroupMemoryBarrierWithGroupSync();
    //------------------------------------------------G---------------------------------------------------------
    uint totalNodeCount = fatherLayerDivisibleNodeCount_G * 8;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) return;
    if (totalNodeCount > THREADGROUP_SIZE) {
        if (groupThreadIndex == 0) {
            threadGroupDivisibleNodeCountSum = 0;
            threadGroupIndivisibleNodeCountSum = 0;
            for (int i = 0; i < threadGroupIndex; ++i) {
                SVOThreadGroupInfo threadGroupInfo = SVOThreadGroupInfos[i];
                threadGroupDivisibleNodeCountSum += threadGroupInfo.threadGroupDivisibleNodeCount_G;
                threadGroupIndivisibleNodeCountSum += threadGroupInfo.threadGroupIndivisibleNodeCount_G;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        SVONodeData_G nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount) {
            nodeData = SVOArray_G[pushConst.currentDepth][threadIndex];
            hasData = nodeData.meanNormal.w > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = SVOArray_G[pushConst.currentDepth][threadIndex].label + threadGroupDivisibleNodeCountSum;
            SVOArray_G[pushConst.currentDepth][threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;
            uint nodeIndex = SVODivisibleNodeIndices_G[fatherNodeIndex] * 8 + blockLane;
            SVODivisibleNodeIndices_G[label - 1] = nodeIndex;
        }
        else if (indivisible) SVOArray_G[pushConst.currentDepth][threadIndex].label += threadGroupIndivisibleNodeCountSum;
    }
    //------------------------------------------------E---------------------------------------------------------
    totalNodeCount = fatherLayerDivisibleNodeCount_E * 8;
    if (totalNodeCount > pushConst.sizes[pushConst.currentDepth]) return;
    if (totalNodeCount > THREADGROUP_SIZE) {
        if (groupThreadIndex == 0) {
            threadGroupDivisibleNodeCountSum = 0;
            threadGroupIndivisibleNodeCountSum = 0;
            for (int i = 0; i < threadGroupIndex; ++i) {
                SVOThreadGroupInfo threadGroupInfo = SVOThreadGroupInfos[i];
                threadGroupDivisibleNodeCountSum += threadGroupInfo.threadGroupDivisibleNodeCount_E;
                threadGroupIndivisibleNodeCountSum += threadGroupInfo.threadGroupIndivisibleNodeCount_E;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        SVONodeData_E nodeData;
        bool hasData = false;
        if (threadIndex < totalNodeCount) {
            nodeData = SVOArray_E[pushConst.currentDepth][threadIndex];
            hasData = nodeData.meanNormal.w > 0.0f;
        }
        bool divisible = hasData && nodeData.indivisible == 0;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (divisible) {
            uint label = SVOArray_E[pushConst.currentDepth][threadIndex].label + threadGroupDivisibleNodeCountSum;
            SVOArray_E[pushConst.currentDepth][threadIndex].label = label;

            uint fatherNodeIndex = threadIndex / 8;
            uint blockLane = groupThreadIndex & 7;
            uint nodeIndex = SVODivisibleNodeIndices_E[fatherNodeIndex] * 8 + blockLane;
            SVODivisibleNodeIndices_E[label - 1] = nodeIndex;
        }
        else if (indivisible) SVOArray_E[pushConst.currentDepth][threadIndex].label += threadGroupIndivisibleNodeCountSum;
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

    bool isG = true;
    uint layerIndex = 1;
    uint nodeIndex = instanceIndex;
    if (nodeIndex >= pushConst.totalNodeCount) {
        nodeIndex -= pushConst.totalNodeCount;
        isG = false;
    }
    output.isG = isG;

    for (; layerIndex <= pushConst.maxDepth; ++layerIndex) {
        uint layerNodeCount = pushConst.sizes[layerIndex];
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