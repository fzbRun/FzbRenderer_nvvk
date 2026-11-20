#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<GeometryClusteringPushConstant_VRPG> pushConst;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterBlockInfos)]] RWStructuredBuffer<ClusterBlockInfo> ClusterBlockInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eHasDataClusterBlockIndices)]] RWStructuredBuffer<ClusterBlockIndex> HasDataClusterBlockIndexBuffer;
//-------------------------------------------------CreateOctree---------------------------------------------
groupshared uint groupWarpHasDataCount[32];
groupshared uint groupHasDataGlobalLabel;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_GetHasDataClusterBlockInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerBlockCount) return;
    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    bool hasData = ClusterBlockInfoBuffer[threadIndex].hasData == 1;
    uint warpHasDataBlockCount = WaveActiveCountBits(hasData);

    if (warpLane == 0) groupWarpHasDataCount[warpIndex] = warpHasDataBlockCount;
    GroupMemoryBarrierWithGroupSync();

    uint hasDataGroupLabel = 0;
    if (warpLane < warpIndex) hasDataGroupLabel = groupWarpHasDataCount[warpLane];
    hasDataGroupLabel = WaveActiveSum(hasDataGroupLabel);

    uint threadAllowance = pushConst.currentLayerBlockCount - threadGroupIndex * 1024;
    uint threadGroupWarpCount = threadAllowance >= 1024 ? 32 : (threadAllowance + 31) / 32;

    if (warpIndex == threadGroupWarpCount - 1 && warpLane == 0) {
        uint groupHasDataNodeCount = hasDataGroupLabel + warpHasDataBlockCount;
        InterlockedAdd<uint>(ClusteringGlobalInfoBuffer[0].hasDataClusterBlockCount, groupHasDataNodeCount, groupHasDataGlobalLabel);
    }
    GroupMemoryBarrierWithGroupSync();

    uint hasDataWarpLabel = WavePrefixCountBits(hasData);
    if (hasData) HasDataClusterBlockIndexBuffer[groupHasDataGlobalLabel + hasDataGroupLabel + hasDataWarpLabel].blockIndex = threadIndex;

    if (threadIndex < pushConst.currentLayerBlockCount / 8) ClusterBlockInfoBuffer[threadIndex].hasData = 0;
}
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_DispatchClustering(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) {
        int hasDataNodeCounts = ClusteringGlobalInfoBuffer[0].hasDataClusterBlockCount;
        DispatchIndirectCommand cmd;
        uint hasDataNodeCount = hasDataNodeCounts * 8;
        cmd.x = (hasDataNodeCount + Clustering_CS_THREADGROUP_SIZE - 1) / Clustering_CS_THREADGROUP_SIZE;
        cmd.y = 1; cmd.z = 1;
        GlobalInfoBuffer[0].cmd = cmd;
    }
}

void atomicMergeAABB(inout AABB aabb) {
    uint laneIndex = WaveGetLaneIndex();

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.x, targetLane);
        aabb.minimum.x = min(aabb.minimum.x, other_val);
    }
    // aabb.minimum.x = WaveReadLaneAt(aabb.minimum.x, blockFirstWarpLane);

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.y, targetLane);
        aabb.minimum.y = min(aabb.minimum.y, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.z, targetLane);
        aabb.minimum.z = min(aabb.minimum.z, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.x, targetLane);
        aabb.maximum.x = max(aabb.maximum.x, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.y, targetLane);
        aabb.maximum.y = max(aabb.maximum.y, other_val);
    }

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.maximum.z, targetLane);
        aabb.maximum.z = max(aabb.maximum.z, other_val);
    }
}
[numthreads(Clustering_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_Clustering1(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint blockLane = threadIndex & 7;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;
    if (threadIndex >= ClusteringGlobalInfoBuffer[0].hasDataClusterBlockCount * 8) return;
    //-----------------------------------------------------------------------------------
    uint currentBlockLabel = threadIndex / 8;
    uint nodeIndex = HasDataClusterBlockIndexBuffer[currentBlockLabel].blockIndex * 8 + blockLane;
    uint fatherNodeIndex = nodeIndex / 8;
    ClusterData_Octree_VRPG nodeData = ClusterData_Octree[pushConst.currentLayer][nodeIndex];

    bool hasData = nodeData.meanNormal.w > 0.0f;
    uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
    uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

    if (blockHasDataNodeCount == 1) {
        if (hasData) {
            ClusterData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
            ClusterInfo_Octree[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3; // 0x11
        }
    }
    //-----------------------------------------------------------------------------------
    else {
        AABB mergeAABB = nodeData.aabb;
        atomicMergeAABB(mergeAABB);

        // Normal consistency + surface coverage.
        float3 mergeNormal = nodeData.meanNormal.xyz;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
            mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
            mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
        }
        mergeNormal = normalize(mergeNormal / max(float(blockHasDataNodeCount), 1.0f) + float3(1e-8f));
        mergeNormal.x = WaveReadLaneAt(mergeNormal.x, blockFirstWarpLane);
        mergeNormal.y = WaveReadLaneAt(mergeNormal.y, blockFirstWarpLane);
        mergeNormal.z = WaveReadLaneAt(mergeNormal.z, blockFirstWarpLane);

        float3 nodeNormal = normalize(nodeData.meanNormal.xyz + float3(1e-8f));
        float minNormalCos = hasData ? dot(nodeNormal, mergeNormal) : 1.0f;
        for (int offset = 4; offset > 0; offset /= 2)
            minNormalCos = min(minNormalCos, WaveReadLaneAt(minNormalCos, warpLane + offset));

        float3 childExtent = max(nodeData.aabb.maximum - nodeData.aabb.minimum, 0.0f);
        float childSurfaceArea = hasData
                    ? 2.0f * (childExtent.x * childExtent.y +
                              childExtent.x * childExtent.z +
                                                           childExtent.y * childExtent.z)
                    : 0.0f;
        // fillRate is the child's surface coverage, so an empty child
        // contributes less than its full AABB surface area.
        float childContribution = childSurfaceArea * clamp(nodeData.fillRate, 0.0f, 1.0f);
        float surfaceAreaSum = childContribution;
        for (int offset = 4; offset > 0; offset /= 2)
            surfaceAreaSum += WaveReadLaneAt(surfaceAreaSum, warpLane + offset);

        bool indivisible = true;
        if (warpLane == blockFirstWarpLane) {
            float3 mergeExtent = max(mergeAABB.maximum - mergeAABB.minimum, float3(1e-5f));
            float mergeSurfaceArea = 2.0f * (mergeExtent.x * mergeExtent.y +
                                                     mergeExtent.x * mergeExtent.z +
                                                     mergeExtent.y * mergeExtent.z);
            float surfaceFill = clamp(surfaceAreaSum / max(mergeSurfaceArea, 1e-8f), 0.0f, 1.0f);

            float score = clamp(minNormalCos, 0.0f, 1.0f) * surfaceFill;
            // Keep fillRate as pure surface coverage for the next layer.
            nodeData.fillRate = surfaceFill;
            indivisible = score >= pushConst.clusteringThreshold;

            nodeData.indivisible = indivisible;
            nodeData.aabb = mergeAABB;
            nodeData.meanNormal = float4(mergeNormal, 1.0f);
            ClusterData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
            ClusterInfo_Octree[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2 + uint(indivisible);
        }
    }

    if (blockLane == 0 && blockHasDataNodeCount > 0) {
        uint fatherBlockIndex = fatherNodeIndex / 8;
        ClusterBlockInfoBuffer[fatherBlockIndex].hasData = 1; // have Data competition, but input same value
    }
}
[numthreads(Clustering_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_Clustering2(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerNodeCount) return;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint fatherNodeIndex = threadIndex / 8;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;

    ClusterInfo_Octree_VRPG nodeInfo = ClusterInfo_Octree[pushConst.currentLayer][threadIndex];
    bool hasData = (nodeInfo.label_indivisible >> 1) > 0;
    uint warpHasDataMask = WaveActiveBallot(hasData).x;
    uint blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
    if (blockHasDataNodeCount > 0 && warpLane == blockFirstWarpLane) ClusterInfo_Octree[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2;
}