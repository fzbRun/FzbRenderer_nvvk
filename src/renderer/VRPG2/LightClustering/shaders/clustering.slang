#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<LightClusteringPushConstant_VRPG> pushConst;

[[vk::binding(LightClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(LightClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(LightClusteringBindingPoints_VRPG::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];

[[vk::binding(LightClusteringBindingPoints_VRPG::eLightNodeBlockInfos)]] RWStructuredBuffer<LightNodeBlockInfo> LightNodeBlockInfoBuffer;
[[vk::binding(LightClusteringBindingPoints_VRPG::eHasDataLightNodeBlockIndices)]] RWStructuredBuffer<LightNodeBlockIndex> HasDataLightNodeBlockIndexBuffer;
//-------------------------------------------------CreateOctree---------------------------------------------
groupshared uint groupWarpHasDataCount[32];
groupshared uint groupHasDataGlobalLabel;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_GetHasDataLightNodeBlockInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerBlockCount) return;
    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    bool hasData = LightNodeBlockInfoBuffer[threadIndex].hasData == 1;
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
    if (hasData) HasDataLightNodeBlockIndexBuffer[groupHasDataGlobalLabel + hasDataGroupLabel + hasDataWarpLabel].blockIndex = threadIndex;

    if (threadIndex < pushConst.currentLayerBlockCount / 8) LightNodeBlockInfoBuffer[threadIndex].hasData = 0;
}
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_DispatchClustering(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) {
        int hasDataNodeCounts = ClusteringGlobalInfoBuffer[0].hasDataClusterBlockCount;
        DispatchIndirectCommand cmd;
        uint hasDataNodeCount = hasDataNodeCounts * 8;
        cmd.x = (hasDataNodeCount + LightClustering_CS_THREADGROUP_SIZE - 1) / LightClustering_CS_THREADGROUP_SIZE;
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
[numthreads(LightClustering_CS_THREADGROUP_SIZE, 1, 1)]
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
    uint nodeIndex = HasDataLightNodeBlockIndexBuffer[currentBlockLabel].blockIndex * 8 + blockLane;
    uint fatherNodeIndex = nodeIndex / 8;
    LightNodeData_Octree_VRPG nodeData = LightNodeData_Octree[pushConst.currentLayer][nodeIndex];

    bool hasData = nodeData.meanNormal.w > 0.0f;
    uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
    uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

    if (blockHasDataNodeCount == 1) {
        if (hasData) LightNodeData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
    }
    //-----------------------------------------------------------------------------------
    else {
        float3 mergeR = nodeData.radiance;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeR += WaveReadLaneAt(mergeR, targetLane);
        }
        mergeR = WaveReadLaneAt(mergeR, blockFirstWarpLane);

        AABB mergeAABB = nodeData.aabb;
        atomicMergeAABB(mergeAABB);

        float weight = hasData ? length(nodeData.radiance) : 0.0f;
        float3 mergeNormal = nodeData.meanNormal.xyz * weight;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeNormal += WaveReadLaneAt(mergeNormal, targetLane);
            weight += WaveReadLaneAt(weight, targetLane);
        }
        mergeNormal = WaveReadLaneAt(mergeNormal, blockFirstWarpLane);
        weight = WaveReadLaneAt(weight, blockFirstWarpLane);

        float halfAngle = M_PI;
        if (any(abs(mergeNormal) > 1e-3f)) {
            mergeNormal = normalize(mergeNormal / weight);
            float cosine = dot(nodeData.meanNormal.xyz, mergeNormal);
            float maxCosine = cosine;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                maxCosine = max(WaveReadLaneAt(maxCosine, targetLane), maxCosine);
            }
            maxCosine = WaveReadLaneAt(maxCosine, blockFirstWarpLane);
            halfAngle = acos(maxCosine);
        } else {
            uint octreeLayerNode = 1 << (3 * pushConst.currentLayer);
            uint normalIndex = nodeIndex / octreeLayerNode;
            float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
            mergeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
            mergeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
            mergeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;
        }

        if (warpLane == blockFirstWarpLane) {
            nodeData.radiance = mergeR;
            nodeData.halfAngle = halfAngle;
            nodeData.meanNormal = float4(mergeNormal, 1.0f);
            nodeData.aabb = mergeAABB;
            LightNodeData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
        }
    }

    if (blockLane == 0 && blockHasDataNodeCount > 0) {
        uint fatherBlockIndex = fatherNodeIndex / 8;
        LightNodeBlockInfoBuffer[fatherBlockIndex].hasData = 1;
    }
}

[numthreads(384, 1, 1)]
[shader("compute")]
void computeMain_Clustering2(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerNodeCount) return;

    uint warpLane = groupThreadIndex & 31;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;

    uint fatherNodeIndex = threadIndex / 8;
    LightNodeData_Octree_VRPG nodeData = LightNodeData_Octree[pushConst.currentLayer][threadIndex];

    bool hasData = nodeData.meanNormal.w > 0.0f || any(nodeData.radiance > 0.0f);
    uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
    uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

    if (blockHasDataNodeCount == 1) {
        if (hasData) LightNodeData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
        return;
    }

    float3 mergeR = nodeData.radiance;
    for (int offset = 4; offset > 0; offset /= 2) {
        mergeR += WaveReadLaneAt(mergeR, warpLane + offset);
    }
    mergeR = WaveReadLaneAt(mergeR, blockFirstWarpLane);

    AABB mergeAABB = nodeData.aabb;
    atomicMergeAABB(mergeAABB);

    float weight = hasData ? length(nodeData.radiance) : 0.0f;
    float3 mergeNormal = nodeData.meanNormal.xyz * weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        mergeNormal += WaveReadLaneAt(mergeNormal, warpLane + offset);
        weight += WaveReadLaneAt(weight, warpLane + offset);
    }
    mergeNormal = WaveReadLaneAt(mergeNormal, blockFirstWarpLane);
    weight = WaveReadLaneAt(weight, blockFirstWarpLane);

    float halfAngle = M_PI;
    if (any(abs(mergeNormal) > 1e-3f)) {
        mergeNormal = normalize(mergeNormal / max(weight, 1e-6f));
        float cosine = dot(nodeData.meanNormal.xyz, mergeNormal);
        float maxCosine = cosine;
        for (int offset = 4; offset > 0; offset /= 2) {
            maxCosine = max(WaveReadLaneAt(maxCosine, warpLane + offset), maxCosine);
        }
        maxCosine = WaveReadLaneAt(maxCosine, blockFirstWarpLane);
        halfAngle = acos(maxCosine);
    } else {
        uint octreeLayerNode = 1 << (3 * pushConst.currentLayer);
        uint normalIndex = threadIndex / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        mergeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        mergeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        mergeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;
    }

    if (warpLane == blockFirstWarpLane) {
        nodeData.radiance = mergeR;
        nodeData.halfAngle = halfAngle;
        nodeData.meanNormal = float4(mergeNormal, 1.0f);
        nodeData.aabb = mergeAABB;
        LightNodeData_Octree[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData;
    }
}
