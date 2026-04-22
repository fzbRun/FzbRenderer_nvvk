#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eVGB)]] StructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_G)]] RWStructuredBuffer<OctreeNodeClusterData_G_FzbPG, ScalarDataLayout> OctreeClusterDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_E)]] RWStructuredBuffer<OctreeNodeClusterData_E_FzbPG, ScalarDataLayout> OctreeClusterDataBuffer_E[];
[[vk::binding(BindingPoints_Octree_FzbPG::eClusterLayerData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> ClusterlayerDataBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_G)]] RWStructuredBuffer<uint> BlockInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_E)]] RWStructuredBuffer<uint> BlockInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_G)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_E)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockCount)]] RWStructuredBuffer<HasDataOctreeBlockCount_FzbPG> HasDataBlockCountBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> DivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eThreadGroupInfos)]] RWStructuredBuffer<OctreeThreadGroupInfo_FzbPG> OctreeThreadGroupInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> IndivisibleNodeInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairVisibleData)]] RWStructuredBuffer<OctreeNodePairVisibleData_FzbPG, ScalarDataLayout> OctreeNodePairVisibleDataBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairWeight)]] RWStructuredBuffer<float> OctreeNodePairWeightBuffer;

//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    if (threadIndex < CLUSTER_LAYER_NODECOUNT_E_FZBPG) ClusterlayerDataBuffer_E[threadIndex].label = 0;

    AABB aabb;{
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    OctreeNodeClusterData_G_FzbPG nodeClusterData_G; {
        nodeClusterData_G.indivisible = 1;
        nodeClusterData_G.aabb = aabb;
        nodeClusterData_G.fillRate = 1.0f;
        nodeClusterData_G.meanNormal = float4(0.0f);}
    OctreeNodeClusterData_E_FzbPG nodeClusterData_E; {
        nodeClusterData_E.E = 0.0f;
        nodeClusterData_E.meanNormal = float4(0.0f);
        nodeClusterData_E.aabb = aabb;}

    if (threadIndex < pushConst.VGBVoxelTotalCount * 6) { // must be 1024 Integer multiple, so warp and threadGroup can synchronization
        uint VGBIndex = threadIndex / pushConst.VGBVoxelTotalCount;
        uint voxelIndex = threadIndex - VGBIndex * pushConst.VGBVoxelTotalCount;
        VGBVoxelData_SVOPG voxelData = VGBs[VGBIndex][voxelIndex];

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
        bool hasData_E = hasData_G && voxelData.sumNormal_E.w > 0.0f;
        if (hasData_E) {
            nodeClusterData_E.E = length(voxelData.irradiance.xyz / voxelData.irradiance.w);
            nodeClusterData_E.meanNormal = voxelData.sumNormal_E / voxelData.sumNormal_E.w;
            nodeClusterData_E.aabb = nodeAabb;
        }
        //------------------------------------------getHasDataInfo------------------------------------
        {
            uint warpLane = groupThreadIndex & 31;
            uint blockIndex = threadIndex / 8;
            uint blockLane = threadIndex & 7;
            uint blockFirstWarpLane = warpLane / 8 * 8;
            uint blockNodeMask = 0xff << blockFirstWarpLane;

            uint layerIndex = pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG;
            //-----------------------------------------------------------
            uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
            uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
            bool blockHasData_G = blockHasDataNodeCount_G > 0;
            if (blockHasData_G) {
                OctreeClusterDataBuffer_G[layerIndex][threadIndex] = nodeClusterData_G;
                OctreeDataBuffer_G[pushConst.octreeMaxLayer][threadIndex].label_indivisible = (uint(hasData_G) << 1) + 1;    //0x11, label > 0 mean hasData
            }
            if (blockLane == 0) BlockInfoBuffer_G[blockIndex] = (uint)blockHasData_G;

            uint warpHasDataMask_E = WaveActiveBallot(hasData_E).x;
            uint blockHasDataNodeCount_E = countbits(warpHasDataMask_E & blockNodeMask);
            bool blockHasData_E = blockHasDataNodeCount_E > 0;
            if (blockHasData_E) OctreeClusterDataBuffer_E[layerIndex][threadIndex] = nodeClusterData_E;
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
        OctreeDataBuffer_G[layerIndex][nodeIndex].label_indivisible = 1;
        if (layerIndex >= OCTREE_CLUSTER_LAYER_FZBPG) {
            layerIndex -= OCTREE_CLUSTER_LAYER_FZBPG;
            OctreeClusterDataBuffer_G[layerIndex][nodeIndex] = nodeClusterData_G;
            OctreeClusterDataBuffer_E[layerIndex][nodeIndex] = nodeClusterData_E;
        } 
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

    uint layerIndex = pushConst.currentLayer - OCTREE_CLUSTER_LAYER_FZBPG;
    //----------------------------------------getOctree_G-------------------------------------------
    if (threadIndex < groupHasDataNodeCount_G) {
        uint currentBlockLabel = threadIndex / 8;
        uint nodeIndex = HasDataBlockIndicsBuffer_G[currentBlockLabel] * 8 + blockLane;
        uint fatherNodeIndex = nodeIndex / 8;
        OctreeNodeClusterData_G_FzbPG nodeData_G = OctreeClusterDataBuffer_G[layerIndex][nodeIndex];

        bool hasData = nodeData_G.meanNormal.w > 0.0f;
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

        if (blockHasDataNodeCount == 1) {
            if (hasData) {
                OctreeClusterDataBuffer_G[layerIndex - 1][fatherNodeIndex] = nodeData_G;
                OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3;  //0x11
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
                                (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG) * 2);

                nodeData_G.indivisible = indivisible;
                nodeData_G.aabb = mergeAABB;
                nodeData_G.meanNormal = float4(mergeNormal, 1.0f);
                OctreeClusterDataBuffer_G[layerIndex - 1][fatherNodeIndex] = nodeData_G;
                OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2 + uint(indivisible);
            }
        }

        if (blockLane == 0 && blockHasDataNodeCount > 0 && layerIndex > 1) {
            uint fatherBlockIndex = fatherNodeIndex / 8;
            BlockInfoBuffer_G[fatherBlockIndex] = true; // have Data competition, but input same value
        }
    }
    //----------------------------------------getOctree_E-------------------------------------------
    if (threadIndex < groupHasDataNodeCount_E) {
        uint currentBlockLabel = threadIndex / 8;
        uint nodeIndex = HasDataBlockIndicsBuffer_E[currentBlockLabel] * 8 + blockLane;
        uint fatherNodeIndex = nodeIndex / 8;
        OctreeNodeClusterData_E_FzbPG nodeData_E = OctreeClusterDataBuffer_E[layerIndex][nodeIndex];

        bool hasData = nodeData_E.E > 0.0f;
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
        if (blockHasDataNodeCount == 1) {
            if (hasData) OctreeClusterDataBuffer_E[layerIndex - 1][fatherNodeIndex] = nodeData_E;
        } else {
            float mergeE = nodeData_E.E;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeE += WaveReadLaneAt(mergeE, targetLane);
            }
            mergeE = WaveReadLaneAt(mergeE, blockFirstWarpLane);

            AABB mergeAABB = nodeData_E.aabb;
            atomicMergeAABB(mergeAABB);

            float3 mergeNormal = nodeData_E.meanNormal.xyz;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
                mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
                mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
            }
            mergeNormal /= blockHasDataNodeCount;

            if (warpLane == blockFirstWarpLane) {
                nodeData_E.E = mergeE;
                nodeData_E.aabb = mergeAABB;
                nodeData_E.meanNormal = float4(mergeNormal, 1.0f);
                OctreeClusterDataBuffer_E[layerIndex - 1][fatherNodeIndex] = nodeData_E;
            }
        }

        if (blockLane == 0 && blockHasDataNodeCount > 0 && layerIndex > 1) {
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

    OctreeNodeData_G_FzbPG nodeData_G = OctreeDataBuffer_G[pushConst.currentLayer][threadIndex];
    bool hasData_G = (nodeData_G.label_indivisible >> 1) > 0;
    uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
    uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
    if (blockHasDataNodeCount_G == 0) {}
    else if (blockHasDataNodeCount_G == 1 && hasData_G)
        OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3;
    else OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2;

    if (pushConst.currentLayer == OCTREE_CLUSTER_LAYER_FZBPG) {
        if (OctreeClusterDataBuffer_E[0][threadIndex].E > 0.0f)
            ClusterlayerDataBuffer_E[threadIndex].label = 1;
    }
}
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
            printf("Error: G indivisibleNodeCount_G exceeds the maximum node count. maxNodeCount: %u,  currentNodeCount: %u\n",
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
                if (pushConst.frameIndex == 0) {
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
//----------------------------------------------getNodePairData-------------------------------------------------
[numthreads(INITWEIGHT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int threadTotalCount = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG;
    if (threadIndex >= threadTotalCount) return;
    OctreeNodePairWeightBuffer[threadIndex] = 0.0f;

    if (threadIndex == 0) {

        GlobalInfoBuffer[0].cmd.x = (GlobalInfoBuffer[0].indivisibleNodeCount_E * indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE;
    }
}

struct IndivisibleNodeData_E {
    float3 normal;
    float3x3 TBN;
    AABB aabb;
    uint nodeIndex;
};
struct IndivisibleNodeData_G {
    float3 normal;
    float3x3 TBN;
    AABB aabb;
};
struct HitPayload_SVOWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
};

groupshared uint groupIndivisibleNodeCount_E;
#define MAX_INDIVISIBLE_NODE_E_COUNT 2 // if indivisibleNodeCount_G > (THREADGROUP_SIZE / OUTGOING_COUNT), indivisibleNodeCount_E <= 2
groupshared IndivisibleNodeData_E groupNodeDatas_E[MAX_INDIVISIBLE_NODE_E_COUNT];
#define MAX_INDIVISIBLE_NODE_G_COUNT HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG
groupshared IndivisibleNodeData_G groupNodeDatas_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_SVOWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT_FZBPG];
groupshared AABB groupNodePairVisibleAabb[MAX_INDIVISIBLE_NODE_G_COUNT];

void traceRay(RayDesc ray, inout HitPayload_SVOWeight payload, AABB nodeAABB_G)
{
    // Initialize inline ray query with no special flags
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    // Process all potential intersections along the ray
    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept all intersections as solid
    }

    // Check if ray hit any geometry
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        // Extract intersection data from the committed hit
        float2 barycentricCoords = q.CommittedTriangleBarycentrics(); // (u,v) coordinates on triangle
        float4x3 worldToObject = q.CommittedWorldToObject4x3();       // Transform matrix
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix
        int instanceIndex = q.CommittedInstanceIndex();               // Instance index in scene

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex];         // Instance data
        payload.material = sceneInfo.materials[instance.materialIndex]; // Material properties

        // Convert built-in barycentric coordinates to full barycentric coordinates
        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        // Process intersection data into world-space shading information
        // HitState hitState = getHitStateWithPositionFetch(barycentrics);
        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        if (!checkPointInAABB(payload.hitPos, nodeAABB_G)) {
            payload.hit = -1.0f;
            return;
        }

        payload.hit = 1.0f;

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        // payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        payload.normal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    }
    else payload.hit = 0.0f;
}
AABB atomicMergeAABB(float3 hitPos, bool hitAble) {
    #if HITTEST_COUNT_FZBPG == 32
    AABB aabb;
    if (!hitAble) hitPos = float3(3.402823466e+38F);
    aabb.minimum = WaveActiveMin(hitPos);
    if (!hitAble) hitPos = float3(-3.402823466e+38F);
    aabb.maximum = WaveActiveMax(hitPos);
    #else
    uint laneIndex = WaveGetLaneIndex();
    AABB aabb; {
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}

    if (!hitAble) hitPos = float3(3.402823466e+38F);
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.x, targetLane);
        aabb.minimum.x = min(aabb.minimum.x, other_val);
    }
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.y, targetLane);
        aabb.minimum.y = min(aabb.minimum.y, other_val);
    }
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.z, targetLane);
        aabb.minimum.z = min(aabb.minimum.z, other_val);
    }

    if (!hitAble) hitPos = float3(-3.402823466e+38F);
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.x, targetLane);
        aabb.maximum.x = max(aabb.maximum.x, other_val);
    }
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.y, targetLane);
        aabb.maximum.y = max(aabb.maximum.y, other_val);
    }
    for (int offset = HITTEST_COUNT_FZBPG / 2; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(hitPos.z, targetLane);
        aabb.maximum.z = max(aabb.maximum.z, other_val);
    }
    #endif

    return aabb;
}
[numthreads(HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeStartLabel_E = threadGroupIndex * HITTEST_CS_THREADGROUP_SIZE / threadCount_perNodeE;
    uint currentGroupNodeEndLabel_E = ((threadGroupIndex + 1) * HITTEST_CS_THREADGROUP_SIZE - 1) / threadCount_perNodeE;
    uint currentGroupNodeECount = currentGroupNodeEndLabel_E - currentGroupNodeStartLabel_E + 1;
    if (groupThreadIndex < currentGroupNodeECount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_E;
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[nodeLabel];
        OctreeNodeClusterData_E_FzbPG nodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        IndivisibleNodeData_E groupNodeData_E;
        groupNodeData_E.normal = normalize(nodeData.meanNormal.xyz);
        groupNodeData_E.nodeIndex = nodeIndex;
        groupNodeData_E.aabb = nodeData.aabb;

        float3 tangent, bitangent;
        orthonormalBasis(groupNodeData_E.normal, tangent, bitangent);
        groupNodeData_E.TBN = float3x3(tangent, bitangent, groupNodeData_E.normal);

        groupNodeDatas_E[groupThreadIndex] = groupNodeData_E;
    }

    uint currentGroupNodeStartLabel_G = threadGroupIndex * HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeEndLabel_G = ((threadGroupIndex + 1) * HITTEST_CS_THREADGROUP_SIZE - 1) / OUTGOING_COUNT_FZBPG;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = (groupThreadIndex + currentGroupNodeStartLabel_G) % indivisibleNodeCount_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG nodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G groupNodeData_G;
        groupNodeData_G.normal = normalize(nodeData.meanNormal.xyz);
        groupNodeData_G.aabb = nodeData.aabb;

        float3 tangent, bitangent;
        orthonormalBasis(groupNodeData_G.normal, tangent, bitangent);
        groupNodeData_G.TBN = float3x3(tangent, bitangent, groupNodeData_G.normal);

        groupNodeDatas_G[groupThreadIndex] = groupNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    for (int i = groupThreadIndex; i < currentGoupNodeGCount * HITTEST_COUNT_FZBPG; i += HITTEST_CS_THREADGROUP_SIZE) {
        uint nodeLabel_G_group = i / HITTEST_COUNT_FZBPG;
        uint nodeLabel_G = nodeLabel_G_group + currentGroupNodeStartLabel_G;

        uint nodeLabel_E = nodeLabel_G / indivisibleNodeCount_G;
        uint nodeLabel_E_group = nodeLabel_E - currentGroupNodeStartLabel_E;

        IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        IndivisibleNodeData_G nodeData_G = groupNodeDatas_G[nodeLabel_G_group];

        uint hitTestIndex = nodeLabel_G_group * HITTEST_COUNT_FZBPG + i % HITTEST_COUNT_FZBPG;
        uint seed = xxhash32(uint3(threadIndex, hitTestIndex, pushConst.frameIndex));

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) + nodeData_E.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);

        HitPayload_SVOWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool normalCheck = dot(sampleDir, nodeData_E.normal) > 0.0f && dot(-sampleDir, nodeData_G.normal) > 0.0f;
        if (normalCheck) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;

            traceRay(ray, hitPayload, nodeData_G.aabb);
        }
        AABB visibleAabb = atomicMergeAABB(hitPayload.hitPos, hitPayload.hit > 0.0f);
        if (i % HITTEST_COUNT_FZBPG == 0) groupNodePairVisibleAabb[nodeLabel_G_group] = visibleAabb;

        groupHitPayloads[hitTestIndex] = hitPayload;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupIndivisibleNodeCount_E * indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);
    uint nodeLabel_E_group = threadIndex / (indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG) - currentGroupNodeStartLabel_E; // groupIndex
    uint nodeLabel_G_group = threadIndex / OUTGOING_COUNT_FZBPG - currentGroupNodeStartLabel_G; // groupIndex
    float weight = 0.0f; uint hitCount = 0;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT_FZBPG; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT_FZBPG + hitTestIndex;
        HitPayload_SVOWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        if (cosine <= 0.0f) continue;
        bool isExt = true;

        float3 tangent, bitangent;
        orthonormalBasis(hitPayload.normal, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

        float3 bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        float3 weight3 = bsdf * cosine;
        weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);
        ++hitCount;
    }
    // Before, it was possible to hit or miss, so it was uneven. Let v be a part of the weight; Now it's 100 percent, so we're averaging
    if (hitCount > 0) weight /= hitCount;

    if (weight > 0.0f) {
        uint nodeLabel_G = (threadIndex / OUTGOING_COUNT_FZBPG) % indivisibleNodeCount_G;
        uint nodeIndex_E = groupNodeDatas_E[nodeLabel_E_group].nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                           nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG + (nodeIndex_E + 56);
        OctreeNodePairWeightBuffer[weightIndex] = weight;
    }

    if (threadIndex == 0) {
        uint threadTotalCount = OUTGOING_COUNT * indivisibleNodeCount_G * NODECOUNT_E;
        GlobalInfo[0].cmd.x = (threadTotalCount + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
    }
}
//------------------------------------------------debug---------------------------------------------------------
#ifndef NDEBUG
struct VSout_OctreeLayer {
    float4 sv_position : SV_Position;
    uint renderTargetIndex;
    bool debug_color;
    float4 color;
};
struct PSout_OctreeLayer
{
    float4 color0 : SV_Target0;
    #if OctreeLayerMapCount > 1
        float4 color1 : SV_Target1;
    #endif
    #if OctreeLayerMapCount > 2
        float4 color2 : SV_Target2;
    #endif
    #if OctreeLayerMapCount > 3
        float4 color3 : SV_Target3;
    #endif
    #if OctreeLayerMapCount > 4
        float4 color4 : SV_Target4;
    #endif
    #if OctreeLayerMapCount > 5
        float4 color5 : SV_Target5;
    #endif
    #if OctreeLayerMapCount > 6
        float4 color6 : SV_Target6; // max VGBSize = 128x128x128
    #endif
    #if OctreeLayerMapCount > 7
        float4 color7 : SV_Target7;
    #endif
    #if OctreeLayerMapCount > 8
        float4 color8 : SV_Target8;
    #endif
    #if OctreeLayerMapCount > 9
        float4 color9 : SV_Target9;
    #endif
    #if OctreeLayerMapCount > 10
        float4 color10 : SV_Target10;
    #endif
    #if OctreeLayerMapCount > 11
        float4 color11 : SV_Target11;
    #endif
    #if OctreeLayerMapCount > 12
        float4 color12 : SV_Target12;
    #endif
    #if OctreeLayerMapCount > 13
        float4 color13 : SV_Target13; // max VGBSize = 128x128x128
    #endif
};
[shader("vertex")]
VSout_OctreeLayer vertexMain_OctreeLayer(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VSout_OctreeLayer output;
    output.renderTargetIndex = 0;

    bool isG = true;
    uint nodeIndex = instanceIndex;
    if (nodeIndex >= pushConst.showOctreeNodeTotalCount * 6) {
        nodeIndex -= pushConst.showOctreeNodeTotalCount * 6;
        isG = false;
        output.renderTargetIndex += pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG;
    }
    uint normalIndex = nodeIndex / pushConst.showOctreeNodeTotalCount;
    nodeIndex = nodeIndex % pushConst.showOctreeNodeTotalCount;

    uint layerIndex = OCTREE_CLUSTER_LAYER_FZBPG;
    uint layerNodeCount = uint(pow(8, layerIndex));
    for (; layerIndex <= pushConst.octreeMaxLayer; ++layerIndex) {
        if (nodeIndex < layerNodeCount) break;
        nodeIndex -= layerNodeCount;
        layerNodeCount *= 8;
    }
    nodeIndex += normalIndex * uint(pow(8, layerIndex));
    layerIndex -= OCTREE_CLUSTER_LAYER_FZBPG;

    AABB nodeAABB; uint invisible = 1;
    if (isG) {
        OctreeNodeClusterData_G_FzbPG octreeData_G = OctreeClusterDataBuffer_G[layerIndex][nodeIndex];
        nodeAABB = octreeData_G.aabb;
        invisible = octreeData_G.meanNormal.w > 0.0f && octreeData_G.indivisible == 1;
    } else {
        OctreeNodeClusterData_E_FzbPG octreeData_E = OctreeClusterDataBuffer_E[layerIndex][nodeIndex];
        nodeAABB = octreeData_E.aabb;
        invisible = octreeData_E.E > 0.0f;
    }

    if (normalIndex != pushConst.normalIndex) invisible = 0;

    if (nodeAABB.minimum.x > 10000.0f || invisible == 0) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += nodeAABB.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    }
    output.renderTargetIndex += layerIndex;

    return output;
}
[shader("pixel")]
PSout_OctreeLayer fragmentMain_OctreeLayer(VSout_OctreeLayer input) {
    PSout_OctreeLayer output;
    float4 colors[OctreeLayerMapCount];
    for (int i = 0; i < OctreeLayerMapCount; ++i) colors[i] = float4(0.0f);
    colors[input.renderTargetIndex] = float4(1.0f, 0.0f, 0.0f, 1.0f);
    if (input.debug_color) colors[input.renderTargetIndex] = input.color;

    #if OctreeLayerMapCount > 0
        output.color0 = colors[0];
    #endif
    #if OctreeLayerMapCount > 1
        output.color1 = colors[1];
    #endif
    #if OctreeLayerMapCount > 2
        output.color2 = colors[2];
    #endif
    #if OctreeLayerMapCount > 3
        output.color3 = colors[3];
    #endif
    #if OctreeLayerMapCount > 4
        output.color4 = colors[4];
    #endif
    #if OctreeLayerMapCount > 5
        output.color5 = colors[5];
    #endif
    #if OctreeLayerMapCount > 6
        output.color6 = colors[6];
    #endif
    #if OctreeLayerMapCount > 7
        output.color7 = colors[7];
    #endif
    #if OctreeLayerMapCount > 8
        output.color8 = colors[81];
    #endif
    #if OctreeLayerMapCount > 9
        output.color9 = colors[9];
    #endif
    #if OctreeLayerMapCount > 10
        output.color10 = colors[10];
    #endif
    #if OctreeLayerMapCount > 11
        output.color11 = colors[11];
    #endif
    #if OctreeLayerMapCount > 12
        output.color12 = colors[12];
    #endif
    #if OctreeLayerMapCount > 13
        output.color13 = colors[13];
    #endif

    return output;
}

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