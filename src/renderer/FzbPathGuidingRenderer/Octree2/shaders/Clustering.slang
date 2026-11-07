#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/RasterVoxelization/RasterVoxelizationShaderio_FzbPG.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eVGB)]] StructuredBuffer<VGBVoxelData_FzbPG, ScalarDataLayout> VGBs[];

[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_FzbPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_G)]] RWStructuredBuffer<uint> BlockInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_E)]] RWStructuredBuffer<uint> BlockInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_G)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_E)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockCount)]] RWStructuredBuffer<HasDataOctreeBlockCount_FzbPG> HasDataBlockCountBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    AABB aabb; {
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    OctreeNodeData_G_FzbPG nodeClusterData_G; {
        nodeClusterData_G.indivisible = 1;
        nodeClusterData_G.aabb = aabb;
        nodeClusterData_G.fillRate = 1.0f;
        nodeClusterData_G.meanNormal = float4(0.0f);
    }
    OctreeNodeData_E_FzbPG nodeClusterData_E; {
        nodeClusterData_E.radiance = 0.0f;
        nodeClusterData_E.halfAngle = M_PI;
        nodeClusterData_E.meanNormal = float4(0.0f);
        nodeClusterData_E.aabb = aabb;
    }

    if (threadIndex < pushConst.VGBVoxelTotalCount * 6) { // must be 1024 Integer multiple, so warp and threadGroup can synchronization
        uint VGBIndex = threadIndex / pushConst.VGBVoxelTotalCount;
        uint voxelIndex = threadIndex - VGBIndex * pushConst.VGBVoxelTotalCount;
        VGBVoxelData_FzbPG voxelData = VGBs[VGBIndex][voxelIndex];

        bool hasData_G = voxelData.sumNormal_G.w > 0.0f;
        AABB nodeAabb;
        if (hasData_G) {
            nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
            nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
            nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
            nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
            nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
            nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

            nodeClusterData_G.aabb = nodeAabb;
            nodeClusterData_G.meanNormal = voxelData.sumNormal_G / voxelData.sumNormal_G.w;
        }
        //-----------------------------------------------E--------------------------------------------
        bool hasData_E = hasData_G && voxelData.irradiance.w > 0.0f;
        if (hasData_E) {
            float3 lengthAxis = nodeAabb.maximum - nodeAabb.minimum;
            int3 axisSign = sign(lengthAxis);
            lengthAxis += float3(int3(1) - axisSign) * 0.0001f; // if x == 0.0, then x = 1.0 else x = x
            if (all(axisSign == 0)) lengthAxis = float3(0.0f);
            float aabbVolume = lengthAxis.x * lengthAxis.y + lengthAxis.x * lengthAxis.z + lengthAxis.y * lengthAxis.z;

            nodeClusterData_E.radiance = aabbVolume * voxelData.irradiance.xyz / voxelData.irradiance.w;
            if (any(abs(nodeClusterData_G.meanNormal) > 0.0f)) nodeClusterData_E.meanNormal.xyz = normalize(nodeClusterData_G.meanNormal.xyz);
            else {
                uint normalIndex = threadIndex / pushConst.VGBVoxelTotalCount;
                float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
                nodeClusterData_E.meanNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
                nodeClusterData_E.meanNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
                nodeClusterData_E.meanNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;
            }
            nodeClusterData_E.aabb = nodeAabb;
        }
        //------------------------------------------getHasDataInfo------------------------------------
        {
            uint warpLane = groupThreadIndex & 31;
            uint blockIndex = threadIndex / 8;
            uint blockLane = threadIndex & 7;
            uint blockFirstWarpLane = warpLane / 8 * 8;
            uint blockNodeMask = 0xff << blockFirstWarpLane;
            //-----------------------------------------------------------
            uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
            uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
            bool blockHasData_G = blockHasDataNodeCount_G > 0;
            if (blockHasData_G) {
                OctreeNodeDataBuffer_G[pushConst.octreeMaxLayer][threadIndex] = nodeClusterData_G;
                OctreeNodeInfoBuffer_G[pushConst.octreeMaxLayer][threadIndex].label_indivisible = (uint(hasData_G) << 1) + 1; // 0x11, label > 0 mean hasData
            }
            if (blockLane == 0) BlockInfoBuffer_G[blockIndex] = (uint)blockHasData_G;

            uint warpHasDataMask_E = WaveActiveBallot(hasData_E).x;
            uint blockHasDataNodeCount_E = countbits(warpHasDataMask_E & blockNodeMask);
            bool blockHasData_E = blockHasDataNodeCount_E > 0;
            if (blockHasData_E) OctreeNodeDataBuffer_E[pushConst.octreeMaxLayer][threadIndex] = nodeClusterData_E; // cmdFillData 0 for maxLayerNode, before use aabb, check E > 0
            if (blockLane == 0) BlockInfoBuffer_E[blockIndex] = (uint)blockHasData_E;
        }
    }
    else {
        uint nodeIndex = threadIndex - pushConst.VGBVoxelTotalCount * 6;
        uint layerIndex = 0, layerNodeCount = 6;
        for (; layerIndex < pushConst.octreeMaxLayer; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        OctreeNodeInfoBuffer_G[layerIndex][nodeIndex].label_indivisible = 1;
        OctreeNodeDataBuffer_G[layerIndex][nodeIndex] = nodeClusterData_G;
        OctreeNodeDataBuffer_E[layerIndex][nodeIndex] = nodeClusterData_E;
    }
}
//-------------------------------------------------CreateOctree---------------------------------------------
groupshared uint groupWarpHasDataCount_G[32];
groupshared uint groupWarpHasDataCount_E[32];
groupshared uint groupHasDataGlobalLabel_G;
groupshared uint groupHasDataGlobalLabel_E;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initHasDataBlockInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerBlockCount) return;
    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    bool hasData_G = BlockInfoBuffer_G[threadIndex] == 1;
    uint warpHasDataBlockCount_G = WaveActiveCountBits(hasData_G);

    bool hasData_E = BlockInfoBuffer_E[threadIndex] == 1;
    uint warpHasDataBlockCount_E = WaveActiveCountBits(hasData_E);

    if (warpLane == 0) {
        groupWarpHasDataCount_G[warpIndex] = warpHasDataBlockCount_G;
        groupWarpHasDataCount_E[warpIndex] = warpHasDataBlockCount_E;
    }
    GroupMemoryBarrierWithGroupSync();

    uint hasDataGroupLabel_G = 0;
    uint hasDataGroupLabel_E = 0;
    if (warpLane < warpIndex) {
        hasDataGroupLabel_G = groupWarpHasDataCount_G[warpLane];
        hasDataGroupLabel_E = groupWarpHasDataCount_E[warpLane];
    }
    hasDataGroupLabel_G = WaveActiveSum(hasDataGroupLabel_G);
    hasDataGroupLabel_E = WaveActiveSum(hasDataGroupLabel_E);

    uint threadAllowance = pushConst.currentLayerBlockCount - threadGroupIndex * 1024;
    uint threadGroupWarpCount = threadAllowance >= 1024 ? 32 : (threadAllowance + 31) / 32;

    if (warpIndex == threadGroupWarpCount - 1 && warpLane == 0) {
        uint groupHasDataNodeCount_G = hasDataGroupLabel_G + warpHasDataBlockCount_G;
        InterlockedAdd<uint>(HasDataBlockCountBuffer[0].count_G, groupHasDataNodeCount_G, groupHasDataGlobalLabel_G);

        uint groupHasDataNodeCount_E = hasDataGroupLabel_E + warpHasDataBlockCount_E;
        InterlockedAdd<uint>(HasDataBlockCountBuffer[0].count_E, groupHasDataNodeCount_E, groupHasDataGlobalLabel_E);
    }
    GroupMemoryBarrierWithGroupSync();

    uint hasDataWarpLabel_G = WavePrefixCountBits(hasData_G);
    if (hasData_G) HasDataBlockIndicsBuffer_G[groupHasDataGlobalLabel_G + hasDataGroupLabel_G + hasDataWarpLabel_G] = threadIndex;

    uint hasDataWarpLabel_E = WavePrefixCountBits(hasData_E);
    if (hasData_E) HasDataBlockIndicsBuffer_E[groupHasDataGlobalLabel_E + hasDataGroupLabel_E + hasDataWarpLabel_E] = threadIndex;

    if (threadIndex < pushConst.currentLayerBlockCount / 8) {
        BlockInfoBuffer_G[threadIndex] = false;
        BlockInfoBuffer_E[threadIndex] = false;
    }
}
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getGlobalInfo(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) {
        HasDataOctreeBlockCount_FzbPG hasDataNodeCounts = HasDataBlockCountBuffer[0];
        OctreeGlobalInfo_FzbPG globalInfo;
        uint hasDataNodeCount = max(hasDataNodeCounts.count_G, hasDataNodeCounts.count_E) * 8;
        globalInfo.cmd.x = (hasDataNodeCount + CREATEOCTREE_CS_THREADGROUP_SIZE - 1) / CREATEOCTREE_CS_THREADGROUP_SIZE;
        globalInfo.cmd.y = 1; globalInfo.cmd.z = 1;
        GlobalInfoBuffer[0] = globalInfo;
    }
}

groupshared uint groupHasDataNodeCount_G;
groupshared uint groupHasDataNodeCount_E;

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
[numthreads(CREATEOCTREE_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        HasDataOctreeBlockCount_FzbPG hasDataBlockCounts = HasDataBlockCountBuffer[0];
        groupHasDataNodeCount_G = hasDataBlockCounts.count_G * 8;
        groupHasDataNodeCount_E = hasDataBlockCounts.count_E * 8;
    }
    GroupMemoryBarrierWithGroupSync();

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint blockLane = threadIndex & 7;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;
    //----------------------------------------getOctree_G-------------------------------------------
    if (threadIndex < groupHasDataNodeCount_G) {
        uint currentBlockLabel = threadIndex / 8;
        uint nodeIndex = HasDataBlockIndicsBuffer_G[currentBlockLabel] * 8 + blockLane;
        uint fatherNodeIndex = nodeIndex / 8;
        OctreeNodeData_G_FzbPG nodeData_G = OctreeNodeDataBuffer_G[pushConst.currentLayer][nodeIndex];

        bool hasData = nodeData_G.meanNormal.w > 0.0f;
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

        if (blockHasDataNodeCount == 1) {
            if (hasData) {
                OctreeNodeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
                OctreeNodeInfoBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3; // 0x11
            }
        } else {
            AABB mergeAABB = nodeData_G.aabb;
            atomicMergeAABB(mergeAABB);

            float3 mergeNormal = nodeData_G.meanNormal.xyz; // get mergeNormal
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
                mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
                mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
            }
            mergeNormal /= blockHasDataNodeCount;

            float mergeAABBVolume = 0.0f; // get mergeAABB's surfaceArea
            if (warpLane == blockFirstWarpLane) {
                float3 length = mergeAABB.maximum - mergeAABB.minimum;
                length += float3(int3(1) - sign(length)) * 0.0001f; // if x == 0.0, then x = 1.0 else x = x
                mergeAABBVolume = length.x * length.y * length.z;
            }

            float aabbVolume = 0.0f;
            if (hasData) {
                // The volume is needed to calculate the scaling factor only when there is lighting.
                // Although it is said that where there is lighting, there must be geometry, rasterization may not be injected
                float3 length = nodeData_G.aabb.maximum - nodeData_G.aabb.minimum;
                int3 axisSign = sign(length);
                length += float3(int3(1) - axisSign) * 0.0001f; // if x == 0.0, then x = 1.0 else x = x
                if (all(axisSign == 0)) length = float3(0.0f);
                aabbVolume = length.x * length.y * length.z;
                aabbVolume *= nodeData_G.fillRate;
            }

            float aabbVolumeSum = aabbVolume; // get merge node's aabb volume
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                aabbVolumeSum += WaveReadLaneAt(aabbVolumeSum, targetLane);
            }

            bool indivisible = true;
            if (warpLane == blockFirstWarpLane) {
                nodeData_G.fillRate = aabbVolumeSum / mergeAABBVolume;
                indivisible &= length(mergeNormal) > 0.707f &&
                    (nodeData_G.fillRate >= 0.75 ||
                    (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG) * 1.5f);

                nodeData_G.indivisible = indivisible;
                nodeData_G.aabb = mergeAABB;
                nodeData_G.meanNormal = float4(mergeNormal, 1.0f);
                OctreeNodeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
                OctreeNodeInfoBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2 + uint(indivisible);
            }
        }

        if (blockLane == 0 && blockHasDataNodeCount > 0 && pushConst.currentLayer > OCTREE_CLUSTER_LAYER_FZBPG + 1) {
            uint fatherBlockIndex = fatherNodeIndex / 8;
            BlockInfoBuffer_G[fatherBlockIndex] = true; // have Data competition, but input same value
        }
    }
    //----------------------------------------getOctree_E-------------------------------------------
    if (threadIndex < groupHasDataNodeCount_E) {
        uint currentBlockLabel = threadIndex / 8;
        uint nodeIndex = HasDataBlockIndicsBuffer_E[currentBlockLabel] * 8 + blockLane;
        uint fatherNodeIndex = nodeIndex / 8;
        OctreeNodeData_E_FzbPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.currentLayer][nodeIndex];

        bool hasData = any(nodeData_E.radiance > 0.0f);
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
        if (blockHasDataNodeCount == 1) {
            if (hasData) OctreeNodeDataBuffer_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
        } else {
            float3 mergeR = nodeData_E.radiance;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeR += WaveReadLaneAt(mergeR, targetLane);
            }
            mergeR = WaveReadLaneAt(mergeR, blockFirstWarpLane);

            AABB mergeAABB = nodeData_E.aabb;
            atomicMergeAABB(mergeAABB);

            float weight = hasData ? length(nodeData_E.radiance) : 0.0f;
            float3 mergeNormal = nodeData_E.meanNormal.xyz * weight;
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
                float cosine = dot(nodeData_E.meanNormal.xyz, mergeNormal);
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
                nodeData_E.radiance = mergeR;
                nodeData_E.halfAngle = halfAngle;
                nodeData_E.meanNormal = float4(mergeNormal, 1.0f);
                nodeData_E.aabb = mergeAABB;
                OctreeNodeDataBuffer_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
            }
        }

        if (blockLane == 0 && blockHasDataNodeCount > 0 && pushConst.currentLayer > OCTREE_CLUSTER_LAYER_FZBPG + 1) {
            uint fatherBlockIndex = fatherNodeIndex / 8;
            BlockInfoBuffer_E[fatherBlockIndex] = true;
        }
    }
}
[numthreads(CREATEOCTREE_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray2(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.currentLayerNodeCount) return;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint fatherNodeIndex = threadIndex / 8;
    uint blockFirstWarpLane = (warpLane / 8) * 8;
    uint blockNodeMask = 0xff << blockFirstWarpLane;

    OctreeNodeInfo_G_FzbPG nodeInfo_G = OctreeNodeInfoBuffer_G[pushConst.currentLayer][threadIndex];
    bool hasData_G = (nodeInfo_G.label_indivisible >> 1) > 0;
    uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
    uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
    if (blockHasDataNodeCount_G == 0) {}
    else OctreeNodeInfoBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2;

    OctreeNodeData_E_FzbPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.currentLayer][threadIndex];
    bool hasData = any(nodeData_E.radiance > 0.0f);
    uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
    uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
    if (blockHasDataNodeCount == 1) {
        if (hasData) OctreeNodeDataBuffer_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
    } else {
        float3 mergeR = nodeData_E.radiance;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeR += WaveReadLaneAt(mergeR, targetLane);
        }
        mergeR = WaveReadLaneAt(mergeR, blockFirstWarpLane);

        AABB mergeAABB = nodeData_E.aabb;
        atomicMergeAABB(mergeAABB);

        float weight = hasData ? length(nodeData_E.radiance) : 0.0f;
        float3 mergeNormal = nodeData_E.meanNormal.xyz * weight;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeNormal += WaveReadLaneAt(mergeNormal, targetLane);
            weight += WaveReadLaneAt(weight, targetLane);
        }
        mergeNormal = WaveReadLaneAt(mergeNormal, blockFirstWarpLane);
        weight = WaveReadLaneAt(weight, blockFirstWarpLane);

        float halfAngle = M_PI;
        if (any(abs(mergeNormal) > 1e-3f)) {
            mergeNormal = normalize(mergeNormal / max(weight, 1e-6f));
            float cosine = dot(nodeData_E.meanNormal.xyz, mergeNormal);
            float maxCosine = cosine;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                maxCosine = max(WaveReadLaneAt(maxCosine, targetLane), maxCosine);
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
            nodeData_E.radiance = mergeR;
            nodeData_E.meanNormal = float4(mergeNormal, 1.0f);
            nodeData_E.aabb = mergeAABB;
            nodeData_E.halfAngle = halfAngle;
            OctreeNodeDataBuffer_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
        }
    }
}