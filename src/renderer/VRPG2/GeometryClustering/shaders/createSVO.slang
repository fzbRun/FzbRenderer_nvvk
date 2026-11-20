#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<GeometryClusteringPushConstant_VRPG> pushConst;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_SVO)]] RWStructuredBuffer<ClusterInfo_SVO_VRPG, ScalarDataLayout> ClusterInfo_SVO;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eSVODivisibleNodeInfos)]] RWStructuredBuffer<SVODivisibleNodeInfo, ScalarDataLayout> SVODivisibleNodeInfos[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eSVOThreadGroupInfos)]] RWStructuredBuffer<SVOThreadGroupInfo_VRPG, ScalarDataLayout> SVOThreadGroupInfos;
//---------------------------------------------------------------------------------------------------------
bool hasValidGNodeData(ClusterData_Octree_VRPG data) {
    return data.meanNormal.w > 0.0f &&
           data.aabb.minimum.x <= data.aabb.maximum.x &&
           data.aabb.minimum.y <= data.aabb.maximum.y &&
           data.aabb.minimum.z <= data.aabb.maximum.z;
}
//---------------------------------------------------------------------------------------------------------
groupshared uint groupDivisibleNodeIndices_layer0[6];
groupshared uint groupDivisibleNodeCount_layer0;
groupshared uint groupWarpDivisibleNodeCount_layer1[2];

[numthreads(CreateSVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVO1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint divisibleNodeCount_layer0 = 0;
    uint divisibleNodeCount_layer1 = 0;

    if (threadIndex < 6) {
        ClusterInfo_Octree_VRPG nodeInfo = ClusterInfo_Octree[0][threadIndex];
        bool divisible = nodeInfo.label_indivisible == 2;

        divisibleNodeCount_layer0 = WaveActiveCountBits(divisible);
        uint divisibleLabel = WavePrefixCountBits(divisible);

        int label = 0;
        if (divisible) {
            label = divisibleLabel + 1;
            groupDivisibleNodeIndices_layer0[divisibleLabel] = threadIndex;
        }
        if (divisible) ClusterInfo_Octree[0][threadIndex].label_indivisible = (label << 1);

        if (threadIndex == 0) groupDivisibleNodeCount_layer0 = divisibleNodeCount_layer0;
    }
    GroupMemoryBarrierWithGroupSync();

    int nodeCount_layer2 = groupDivisibleNodeCount_layer0 * 8; // max 48
    if (nodeCount_layer2 <= 32) {
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            uint nodeIndex = fatherNodeIndex * 8 + threadIndex & 7;

            ClusterInfo_Octree_VRPG nodeInfo = ClusterInfo_Octree[1][nodeIndex];
            bool divisible = nodeInfo.label_indivisible == 2;
            divisibleNodeCount_layer1 = WaveActiveCountBits(divisible);
            uint divisibleLabel = WavePrefixCountBits(divisible);

            int label = 0;
            if (divisible) {
                label = divisibleLabel + 1;
                SVODivisibleNodeInfos[0][divisibleLabel].nodeIndex = nodeIndex;
            }
            ClusterInfo_Octree[1][nodeIndex].label_indivisible = (label << 1);
        }
    } else {
        uint warpIndex = threadIndex / 32;
        uint warpLane = threadIndex & 31;
        uint blockLane = threadIndex & 7;

        ClusterInfo_Octree_VRPG nodeInfo; uint nodeIndex = 0;
        uint divisibleLabel; bool divisible = false;
        int label = 0;
        if (threadIndex < nodeCount_layer2) {
            uint fatherNodeIndex = groupDivisibleNodeIndices_layer0[threadIndex / 8];
            nodeIndex = fatherNodeIndex * 8 + blockLane;

            nodeInfo = ClusterInfo_Octree[1][nodeIndex];
            divisible = nodeInfo.label_indivisible == 2;
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
            if (divisible) SVODivisibleNodeInfos[0][label - 1].nodeIndex = nodeIndex;
            if (divisible) ClusterInfo_Octree[1][nodeIndex].label_indivisible = (label << 1);
        }
    }

    if (threadIndex == 0) {
        GlobalInfo_VRPG2 globalInfo = GlobalInfoBuffer[0];

        globalInfo.layerInfos_G[0].divisibleNodeCount = divisibleNodeCount_layer0;
        globalInfo.layerInfos_G[0].indivisibleNodeCount = 0;

        if (nodeCount_layer2 <= 32) globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1;
        else globalInfo.layerInfos_G[1].divisibleNodeCount = divisibleNodeCount_layer1 + groupWarpDivisibleNodeCount_layer1[1];
        globalInfo.layerInfos_G[1].indivisibleNodeCount = 0;

        for (int i = 2; i < MAX_OCTREE_LAYER_VRPG2; ++i) {
            globalInfo.layerInfos_G[i].divisibleNodeCount = 0;
            globalInfo.layerInfos_G[i].indivisibleNodeCount = 0;
        }
        // the layer counters live in the same buffer the later createSVO passes read
        GlobalInfoBuffer[0] = globalInfo;

        GlobalInfoBuffer[0].cmd.x = (globalInfo.layerInfos_G[1].divisibleNodeCount * 8 + CreateSVO_CS_THREADGROUP_SIZE - 1) / CreateSVO_CS_THREADGROUP_SIZE;
    }
}

#define DivisibleNodeInfoCount_VRPG ((ClusterMaxCount_VRPG + 7) / 8)
groupshared uint groupFatherLayerDivisibleNodeCount;
groupshared uint groupIndivisibleNodeCountSum;
groupshared uint groupWarpDivisibleNodeCountSums[CreateSVO_CS_THREADGROUP_SIZE / 32];
groupshared uint groupWarpIndivisibleNodeCountSums[CreateSVO_CS_THREADGROUP_SIZE / 32];
[numthreads(CreateSVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVO2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint blockLane = groupThreadIndex & 7;
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (groupThreadIndex == 0) {
        GlobalInfo_VRPG2 globalInfo = GlobalInfoBuffer[0];
        groupFatherLayerDivisibleNodeCount = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

        uint indivisibleNodeCountSum = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentLayer; ++layerIndex)
            indivisibleNodeCountSum += globalInfo.layerInfos_G[layerIndex].indivisibleNodeCount;
        groupIndivisibleNodeCountSum = indivisibleNodeCountSum;
    }
    GroupMemoryBarrierWithGroupSync();

    uint layerNodeCount = groupFatherLayerDivisibleNodeCount * 8;
    if (threadIndex < ((layerNodeCount + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE)) {
        int oddLayer = (pushConst.currentLayer & 1);

        uint nodeIndex = 0;
        ClusterInfo_Octree_VRPG nodeInfo; nodeInfo.label_indivisible = 1;
        if (threadIndex < layerNodeCount) {
            uint fatherNodeLabel = threadIndex / 8;
            uint fatherNodeIndex = SVODivisibleNodeInfos[oddLayer][fatherNodeLabel].nodeIndex;
            nodeIndex = fatherNodeIndex * 8 + blockLane;
            nodeInfo = ClusterInfo_Octree[pushConst.currentLayer][nodeIndex];
        }
        bool indivisible = nodeInfo.label_indivisible == 3;
        bool divisible = nodeInfo.label_indivisible == 2;
        if (indivisible && !hasValidGNodeData(ClusterData_Octree[pushConst.currentLayer][nodeIndex])) {
            indivisible = false;
            ClusterInfo_Octree[pushConst.currentLayer][nodeIndex].label_indivisible = 1;
        }

        // Count the number of divisible and indivisible nodes in the warp
        uint warpDivisibleNodeMask = WaveActiveBallot(divisible).x;
        uint divisibleLabelInWarp = WavePrefixSum(divisible);

        uint warpIndivisibleNodeMask = WaveActiveBallot(indivisible).x;
        uint indivisibleLabelInWarp = WavePrefixSum(indivisible);

        // The last group can be full; modulo gives 0 when layerNodeCount_G is an
        // exact multiple of the group size, which drops its warp counts and
        // corrupts every label in that group.
        uint threadGroupNodeCount = min(
            layerNodeCount - min(threadGroupIndex * CreateSVO_CS_THREADGROUP_SIZE, layerNodeCount),
                CreateSVO_CS_THREADGROUP_SIZE);
        uint warpCount = (threadGroupNodeCount + 31) / 32;

        // if the number of nodes in the father layer < one warp size, we can directly calculate the label of the node.
        uint threadGroupWarpDivisibleNodeLabel = 0;
        uint threadGroupWarpIndivisibleNodeLabel = 0;
        int label = 0;
        if (layerNodeCount <= WARP_SIZE) {
            if (divisible) label = divisibleLabelInWarp + 1;
            if (indivisible) label = indivisibleLabelInWarp + 1;
        } else {
            if (warpLane == 0) {
                uint warpDivisibleNodeCount = countbits(warpDivisibleNodeMask);
                groupWarpDivisibleNodeCountSums[warpIndex] = warpDivisibleNodeCount;

                uint warpIndivisibleNodeCount = countbits(warpIndivisibleNodeMask);
                groupWarpIndivisibleNodeCountSums[warpIndex] = warpIndivisibleNodeCount;
            }
            GroupMemoryBarrierWithGroupSync(); // The post-votal architecture can accommodate branching

            if (warpLane < warpIndex) { // WARP_SIZE >= 32, so warpIndex <= 32 <= WARP_SIZE
                threadGroupWarpDivisibleNodeLabel = groupWarpDivisibleNodeCountSums[warpLane];
                threadGroupWarpIndivisibleNodeLabel = groupWarpIndivisibleNodeCountSums[warpLane];
            }
            threadGroupWarpDivisibleNodeLabel = WaveActiveSum(threadGroupWarpDivisibleNodeLabel);
            threadGroupWarpIndivisibleNodeLabel = WaveActiveSum(threadGroupWarpIndivisibleNodeLabel);

            uint divisibleLabelInThreadGroup = threadGroupWarpDivisibleNodeLabel + divisibleLabelInWarp;
            uint indivisibleLabelInThreadGroup = threadGroupWarpIndivisibleNodeLabel + indivisibleLabelInWarp;

            if (divisible) label = divisibleLabelInThreadGroup + 1;
            else if (indivisible) label = indivisibleLabelInThreadGroup + 1;
        }

        if (divisible && layerNodeCount <= CreateSVO_CS_THREADGROUP_SIZE) SVODivisibleNodeInfos[1 - oddLayer][label - 1].nodeIndex = nodeIndex;
        if (indivisible && layerNodeCount <= CreateSVO_CS_THREADGROUP_SIZE)
            ClusterInfo_SVO[groupIndivisibleNodeCountSum + label - 1] = { pushConst.currentLayer, nodeIndex };

        if (indivisible || divisible) ClusterInfo_Octree[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + uint(indivisible);

        if (warpIndex == warpCount - 1 && warpLane == 0) {
            if (layerNodeCount <= WARP_SIZE) {
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = countbits(warpDivisibleNodeMask);
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = countbits(warpIndivisibleNodeMask);
            } else if (layerNodeCount <= CreateSVO_CS_THREADGROUP_SIZE) {
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);
            }
            else {
                uint threadGroupDivisibleNodeCount = threadGroupWarpDivisibleNodeLabel + countbits(warpDivisibleNodeMask);
                uint threadGroupInDivisibleNodeCount = threadGroupWarpIndivisibleNodeLabel + countbits(warpIndivisibleNodeMask);

                InterlockedAdd<uint>(GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount, threadGroupDivisibleNodeCount);
                InterlockedAdd<uint>(GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].indivisibleNodeCount, threadGroupInDivisibleNodeCount);

                SVOThreadGroupInfos[threadGroupIndex].threadGroupDivisibleNodeCount = threadGroupDivisibleNodeCount;
                SVOThreadGroupInfos[threadGroupIndex].threadGroupIndivisibleNodeCount = threadGroupInDivisibleNodeCount;
            }
        }
    }

    if (threadIndex == 0 && layerNodeCount <= CreateSVO_CS_THREADGROUP_SIZE) GlobalInfoBuffer[0].cmd.x = 1;
}

groupshared uint groupThreadGroupDivisibleNodeCountSum;
groupshared uint groupThreadGroupIndivisibleNodeCountSum;
[numthreads(CreateSVO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createSVO3(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint warpLane = groupThreadIndex & (WARP_SIZE - 1);
    uint warpIndex = groupThreadIndex / WARP_SIZE;

    if (warpIndex == 0) { // this assume groupThreadCount < 32
        GlobalInfo_VRPG2 globalInfo = GlobalInfoBuffer[0];
        if (groupThreadIndex == 0)
            groupFatherLayerDivisibleNodeCount = globalInfo.layerInfos_G[pushConst.currentLayer - 1].divisibleNodeCount;

        uint indivisibleNodeCountSum = 0;
        if (groupThreadIndex < pushConst.currentLayer) indivisibleNodeCountSum = globalInfo.layerInfos_G[groupThreadIndex].indivisibleNodeCount;
        groupIndivisibleNodeCountSum = WaveActiveSum(indivisibleNodeCountSum);

        uint threadGroupDivisibleNodeCount = 0;
        uint threadGroupIndivisibleNodeCount = 0;
        if (groupThreadIndex < threadGroupIndex) {
            SVOThreadGroupInfo_VRPG threadGroupInfo = SVOThreadGroupInfos[groupThreadIndex];
            threadGroupDivisibleNodeCount += threadGroupInfo.threadGroupDivisibleNodeCount;
            threadGroupIndivisibleNodeCount += threadGroupInfo.threadGroupIndivisibleNodeCount;
        }
        threadGroupDivisibleNodeCount = WaveActiveSum(threadGroupDivisibleNodeCount);
        threadGroupIndivisibleNodeCount = WaveActiveSum(threadGroupIndivisibleNodeCount);
        if (warpLane == 0) {
            groupThreadGroupDivisibleNodeCountSum = threadGroupDivisibleNodeCount;
            groupThreadGroupIndivisibleNodeCountSum = threadGroupIndivisibleNodeCount;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int oddLayer = int(pushConst.currentLayer & 1u);
    uint layerNodeCount = groupFatherLayerDivisibleNodeCount * 8;

    if (layerNodeCount > CreateSVO_CS_THREADGROUP_SIZE && threadIndex < layerNodeCount) {
        uint blockLane = threadIndex & 7;
        uint fatherNodeLabel = threadIndex / 8;
        uint fatherNodeIndex = SVODivisibleNodeInfos[oddLayer][fatherNodeLabel].nodeIndex;
        uint nodeIndex = fatherNodeIndex * 8 + blockLane;
        ClusterInfo_Octree_VRPG nodeInfo = ClusterInfo_Octree[pushConst.currentLayer][nodeIndex];

        bool indivisible = (nodeInfo.label_indivisible & 1) == 1 && nodeInfo.label_indivisible > 0;
        bool divisible = (nodeInfo.label_indivisible & 1) == 0 && nodeInfo.label_indivisible > 0;
        if (indivisible && !hasValidGNodeData(ClusterData_Octree[pushConst.currentLayer][nodeIndex])) {
            indivisible = false;
            ClusterInfo_Octree[pushConst.currentLayer][nodeIndex].label_indivisible = 1;
        }
        if (divisible) {
            uint label = (nodeInfo.label_indivisible >> 1) + groupThreadGroupDivisibleNodeCountSum;
            ClusterInfo_Octree[pushConst.currentLayer][nodeIndex].label_indivisible = label << 1;

            SVODivisibleNodeInfos[1 - oddLayer][label - 1].nodeIndex = fatherNodeIndex * 8 + blockLane;
        }
        else if (indivisible) {
            uint label = (nodeInfo.label_indivisible >> 1) + groupThreadGroupIndivisibleNodeCountSum;
            ClusterInfo_Octree[pushConst.currentLayer][nodeIndex].label_indivisible = (label << 1) + 1;
            ClusterInfo_SVO[groupIndivisibleNodeCountSum + label - 1] = { pushConst.currentLayer, nodeIndex };
        }
    }

    if (threadIndex == 0) {
        uint nextLayerNodeCount = GlobalInfoBuffer[0].layerInfos_G[pushConst.currentLayer].divisibleNodeCount * 8;
        GlobalInfoBuffer[0].cmd.x = (nextLayerNodeCount + CreateSVO_CS_THREADGROUP_SIZE - 1) / CreateSVO_CS_THREADGROUP_SIZE;
    }
}

[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain_createSVO4(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint indivisibleNodeCount = 0;
    if (groupThreadIndex < MAX_OCTREE_LAYER_VRPG2) indivisibleNodeCount = GlobalInfoBuffer[0].layerInfos_G[groupThreadIndex].indivisibleNodeCount;
    indivisibleNodeCount = WaveActiveSum(indivisibleNodeCount);

    if (indivisibleNodeCount > ClusterMaxCount_VRPG) {
        if (threadIndex == 0) {
            printf( "Error: G ClusterCount exceeds the maximum node count. maxNodeCount: %u,  currentNodeCount: %u\n",
                ClusterMaxCount_VRPG, indivisibleNodeCount);
            GlobalInfoBuffer[0].ClusterCount = 0;
        }
        return;
    }

    if (threadIndex == 0) {
        GlobalInfoBuffer[0].ClusterCount = indivisibleNodeCount;

        #ifndef NDEBUG
        if (pushConst.frameIndex == 1) {
            printf("ClusterCount: %d\n", indivisibleNodeCount);
            for (int i = 0; i <= pushConst.octreeMaxLayer; ++i) {
                printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                       GlobalInfoBuffer[0].layerInfos_G[i].divisibleNodeCount, GlobalInfoBuffer[0].layerInfos_G[i].indivisibleNodeCount
                );
            }
        }
        #endif

        // if (threadIndex == 0) {
        //     OctreeNodeInfo_G_VRPG info = OctreeNodeInfoBuffer_G[3][1537];
        //     printf("layer: 3, index: 1537\n label: %d, indivisible: %d\n", info.label_indivisible >> 1, info.label_indivisible & 1);
        //
        //     int2 info2 = IndivisibleNodeInfoBuffer_G[219];
        //     printf("layer: %d, index: %d\n label: %d, indivisible: %d\n", info2.x, info2.y, info.label_indivisible >> 1, info.label_indivisible & 1);
        //
        //     info = OctreeNodeInfoBuffer_G[4][12301];
        //     printf("layer: 4, index: 12301\n label: %d, indivisible: %d\n", info.label_indivisible >> 1, info.label_indivisible & 1);
        //
        //     info2 = IndivisibleNodeInfoBuffer_G[672];
        //     printf("layer: %d, index: %d\n label: %d, indivisible: %d\n", info2.x, info2.y, info.label_indivisible >> 1, info.label_indivisible & 1);
        // }
    }
}