#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/GeometryInjection/GeometryInjectionShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<GeometryClusteringPushConstant_VRPG> pushConst;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eVGB)]] RWStructuredBuffer<VoxelData_VRPG, ScalarDataLayout> VGBs[];

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterBlockInfos)]] RWStructuredBuffer<ClusterBlockInfo> ClusterBlockInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eHasDataClusterBlockIndices)]] RWStructuredBuffer<ClusterBlockIndex> HasDataClusterBlockIndexBuffer;
//----------------------------------------------init--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_init(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    AABB aabb; {
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    ClusterData_Octree_VRPG clusterData; {
        clusterData.indivisible = 1;
        clusterData.aabb = aabb;
        clusterData.fillRate = 1.0f;
        clusterData.meanNormal = float4(0.0f);
    }

    if (threadIndex < pushConst.voxelTotalCount * 6) { // must be 1024 Integer multiple, so warp and threadGroup can synchronization
        uint VGBIndex = threadIndex / pushConst.voxelTotalCount;
        uint voxelIndex = threadIndex - VGBIndex * pushConst.voxelTotalCount;
        VoxelData_VRPG voxelData = VGBs[VGBIndex][voxelIndex];

        bool hasData = voxelData.sumNormal_G.w > 0.0f;
        AABB nodeAabb;
        if (hasData) {
            nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
            nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
            nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
            nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
            nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
            nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

            clusterData.aabb = nodeAabb;
            clusterData.meanNormal = voxelData.sumNormal_G / voxelData.sumNormal_G.w;
        }
        //------------------------------------------getHasDataInfo------------------------------------
        {
            uint warpLane = groupThreadIndex & 31;
            uint blockIndex = threadIndex / 8;
            uint blockLane = threadIndex & 7;
            uint blockFirstWarpLane = warpLane / 8 * 8;
            uint blockNodeMask = 0xff << blockFirstWarpLane;
            //-----------------------------------------------------------
            uint warpHasDataMask = WaveActiveBallot(hasData).x;
            uint blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
            bool blockHasData = blockHasDataNodeCount > 0;
            if (blockHasData) {
                ClusterData_Octree[pushConst.octreeMaxLayer][threadIndex] = clusterData;
                ClusterInfo_Octree[pushConst.octreeMaxLayer][threadIndex].label_indivisible = (uint(hasData) << 1) + 1; // 0x11, label > 0 mean hasData
            }
            if (blockLane == 0) ClusterBlockInfoBuffer[blockIndex].hasData = (uint)blockHasData;
        }
    }
    else {
        uint nodeIndex = threadIndex - pushConst.voxelTotalCount * 6;
        uint layerIndex = 0, layerNodeCount = 6;
        for (; layerIndex < pushConst.octreeMaxLayer; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        ClusterInfo_Octree[layerIndex][nodeIndex].label_indivisible = 1;
        ClusterData_Octree[layerIndex][nodeIndex] = clusterData;
    }
}