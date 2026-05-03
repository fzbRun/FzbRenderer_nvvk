#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/RasterVoxelization/RasterVoxelizationShaderio_FzbPG.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eVGB)]] StructuredBuffer<VGBVoxelData_FzbPG, ScalarDataLayout> VGBs[];
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
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairWeight)]] RWStructuredBuffer<float> OctreeNodePairWeightBuffer;
//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    if (threadIndex < CLUSTER_LAYER_NODECOUNT_E_FZBPG) ClusterlayerDataBuffer_E[threadIndex].label = 0;

    AABB aabb; {
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
        nodeClusterData_G.meanNormal = float4(0.0f);
        #ifdef GEOMETRY_CLUSTER_WITH_E
        nodeClusterData_G.E = 0.0f;
        #endif
    }
    OctreeNodeClusterData_E_FzbPG nodeClusterData_E; {
        nodeClusterData_E.pdf = 1.0f;
        nodeClusterData_E.E = 0.0f;
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
            nodeClusterData_E.E = length(voxelData.irradiance.xyz / voxelData.irradiance.w);
            nodeClusterData_E.meanNormal = nodeClusterData_G.meanNormal;    // voxelData.sumNormal_E / voxelData.sumNormal_E.w;
            nodeClusterData_E.aabb = nodeAabb;

            #ifdef GEOMETRY_CLUSTER_WITH_E
            nodeClusterData_G.E = nodeClusterData_E.E;
            #endif
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
                OctreeDataBuffer_G[pushConst.octreeMaxLayer][threadIndex].label_indivisible = (uint(hasData_G) << 1) + 1; // 0x11, label > 0 mean hasData
            }
            if (blockLane == 0) BlockInfoBuffer_G[blockIndex] = (uint)blockHasData_G;

            uint warpHasDataMask_E = WaveActiveBallot(hasData_E).x;
            uint blockHasDataNodeCount_E = countbits(warpHasDataMask_E & blockNodeMask);
            bool blockHasData_E = blockHasDataNodeCount_E > 0;
            if (blockHasData_E) OctreeClusterDataBuffer_E[layerIndex][threadIndex] = nodeClusterData_E; // cmdFillData 0 for maxLayerNode, before use aabb, check E > 0
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
                OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3; // 0x11
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
            #ifdef GEOMETRY_CLUSTER_WITH_E
            {
                float mergeE = nodeData_G.E;
                for (int offset = 4; offset > 0; offset /= 2) {
                    uint targetLane = warpLane + offset;
                    mergeE += WaveReadLaneAt(mergeE, targetLane);
                }
                mergeE = WaveReadLaneAt(mergeE, blockFirstWarpLane);

                bool indivisibleE = true;
                if (mergeE > 0.0f) {
                    float ratioE = hasData ? nodeData_G.E / mergeE : 1.0f;
                    indivisibleE = ratioE >= (1.0f / blockHasDataNodeCount);
                    for (int offset = 4; offset > 0; offset /= 2) {
                        uint targetLane = warpLane + offset;
                        indivisibleE &= WaveReadLaneAt(indivisibleE, targetLane);
                    }
                }
                indivisible &= indivisibleE;

                nodeData_G.E = mergeE;
            }
            #endif
            
            if (warpLane == blockFirstWarpLane) {
                nodeData_G.fillRate = aabbVolumeSum / mergeAABBVolume;
                indivisible &= length(mergeNormal) > 0.707f &&
                               (nodeData_G.fillRate >= 0.75 ||
                               (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG) * 1.5f);

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

            float probability = nodeData_E.E / mergeE;
            uint32_t sampleWarpLane = blockFirstWarpLane;
            float sampleProbability = 1.0f;
            if (warpLane == blockFirstWarpLane) {
                uint randomSeed = xxhash32(uint3(layerIndex, nodeIndex, (uint)pushConst.frameIndex));
                sampleProbability = rand(randomSeed);
            }
            sampleProbability = WaveReadLaneAt(sampleProbability, blockFirstWarpLane);
            float testPDF = sampleProbability;
            for (int offset = 0; offset < 8; ++offset) {
                sampleProbability -= WaveReadLaneAt(probability, blockFirstWarpLane + offset);
                if (sampleProbability <= 0.0f) {
                    sampleWarpLane += offset;
                    break;
                }
            }

            if (warpLane == sampleWarpLane) {
                nodeData_E.E = mergeE;
                nodeData_E.pdf *= probability;
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
    // hitTest need indivisibleNode's aabb, but we haven't 0 - clusterlayer - 1 layer's node data, so can't  set these layer's node to be indivisible
    // else if (blockHasDataNodeCount_G == 1 && hasData_G)  
    //     OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 3;
    else OctreeDataBuffer_G[pushConst.currentLayer - 1][fatherNodeIndex].label_indivisible = 2;


    if (pushConst.currentLayer == OCTREE_CLUSTER_LAYER_FZBPG) {
        if (OctreeClusterDataBuffer_E[0][threadIndex].E > 0.0f) {
            OctreeNodeClusterData_E_FzbPG clusterNodeData_E = OctreeClusterDataBuffer_E[0][threadIndex];
            OctreeNodeData_E_FzbPG nodeData_E;
            #ifndef ADAPTIVE_IMPORTANCE_SAMPLING
            nodeData_E.aabb = clusterNodeData_E.aabb;
            nodeData_E.pdf = clusterNodeData_E.pdf;
            #endif
            nodeData_E.label = 1;
            ClusterlayerDataBuffer_E[threadIndex] = nodeData_E;
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
        GlobalInfoBuffer[0].cmd.x = GlobalInfoBuffer[0].indivisibleNodeCount_E * ((indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE);
        #ifdef NEARBYNODE_JITTER_FZBPG
        DispatchIndirectCommand cmd2;
        cmd2.x = indivisibleNodeCount_G * ((indivisibleNodeCount_G + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE);
        cmd2.y = 1; cmd2.z = 1;
        GlobalInfoBuffer[0].cmd2 = cmd2;
        #endif
    }
}

struct IndivisibleNodeData_E {
    AABB aabb;
    float3 normal;
    uint nodeIndex;
    float E;
};
struct IndivisibleNodeData_G {
    AABB aabb;
    float3 nodeNormal; // isn't node's normal, is one of six normal
    uint layerIndex;
    uint nodeIndex;
};
struct HitPayload_FzbWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
};

groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared IndivisibleNodeData_E groupNodeData_E;

void traceRay(RayDesc ray, inout HitPayload_FzbWeight payload)
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
        payload.hit = 1.0f;

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        // payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        payload.normal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    }
    else payload.hit = 0.0f;
}

#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairData)]] RWStructuredBuffer<OctreeNodePairData_FzbPG, ScalarDataLayout> OctreeNodePairDataBuffer;
//[[vk::binding(BindingPoints_Octree_FzbPG::ePartialHitNodePairCount)]] RWStructuredBuffer<uint> PartialHitNodePairCountBuffer;
//[[vk::binding(BindingPoints_Octree_FzbPG::ePartialHitNodePairTempData)]] RWStructuredBuffer<OctreePartialHiNodePairTempData_FzbPG, ScalarDataLayout> PartialHitNodePairTempDataBuffer[];
//[[vk::binding(BindingPoints_Octree_FzbPG::eHitTestNodePairCount)]] RWStructuredBuffer<uint> HitTestNodePairCountBuffer;
//[[vk::binding(BindingPoints_Octree_FzbPG::eHitTestNodePairInfo)]] RWStructuredBuffer<OctreeHitTestNodePairInfo_FzbPG, ScalarDataLayout> HitTestNodePairInfoBuffer[];

struct ChildNodeInfo_E {
    AABB aabb;
    float3 normal;
    float E;
    float pdf;
}

#define MAX_INDIVISIBLE_NODE_G_COUNT HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG
groupshared IndivisibleNodeData_G groupNodeData_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_FzbWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * 8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG];

groupshared ChildNodeInfo_E groupChildNodeInfos_E[8];
groupshared uint groupHasDataChildNodeCount;

[numthreads(HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int indivisibleNodeCount_E = groupIndivisibleNodeCount_E;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG;
    uint threadGroupCount_perNodeE = (threadCount_perNodeE + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE;
    uint nodeLabel_E = threadGroupIndex / threadGroupCount_perNodeE;
    if (groupThreadIndex < 8) {
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[nodeLabel_E];
        OctreeNodeClusterData_E_FzbPG clusterNodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        if (groupThreadIndex == 0) {
            IndivisibleNodeData_E indivisibleNodeData_E;
            indivisibleNodeData_E.nodeIndex = nodeIndex;
            indivisibleNodeData_E.E = clusterNodeData.E;
            indivisibleNodeData_E.normal = normalize(clusterNodeData.meanNormal.xyz);
            indivisibleNodeData_E.aabb = clusterNodeData.aabb;

            groupNodeData_E = indivisibleNodeData_E;
        }
        {
            //float3 voxelSize = pushConst.VGBVoxelSize.xyz * (1 << (3 * (pushConst.octreeMaxLayer - pushConst.currentLayer)));
            //float3 childVoxelStartPos =
            //    int3((clusterNodeData.aabb.minimum - pushConst.VGBStartPos_Size.xyz) / voxelSize) * voxelSize + pushConst.VGBStartPos_Size.xyz;
            //float3 childVoxelSize = voxelSize / 8;
            //uint3 nodeOffset = uint3(groupThreadIndex & 1u, (groupThreadIndex & 2u) >> 1, (groupThreadIndex & 4u) >> 2);
            //AABB childNodeAabb;
            //childNodeAabb.minimum = childVoxelStartPos + nodeOffset * childVoxelSize;
            //childNodeAabb.maximum = childNodeAabb.minimum + childVoxelSize;
            //childNodeAabb.minimum = max(childNodeAabb.minimum, clusterNodeData.aabb.minimum);
            //childNodeAabb.maximum = min(childNodeAabb.maximum, clusterNodeData.aabb.maximum);
        }
        OctreeNodeClusterData_E_FzbPG childNodeData = OctreeClusterDataBuffer_E[1][nodeIndex * 8 + groupThreadIndex];
        bool hasData = childNodeData.E > 0.0f;

        uint warpHasDataChildNodeCount = WaveActiveCountBits(hasData);
        uint warpHasDataChildNodeLabel = WavePrefixCountBits(hasData);
        if (warpLane == 0) groupHasDataChildNodeCount = warpHasDataChildNodeCount;

        if (hasData) {
            ChildNodeInfo_E childNodeInfo;
            childNodeInfo.aabb = childNodeData.aabb; // this aabb is importantSample result
            childNodeInfo.normal = normalize(childNodeData.meanNormal.xyz);
            childNodeInfo.E = childNodeData.E;
            childNodeInfo.pdf = childNodeData.pdf;
            groupChildNodeInfos_E[warpHasDataChildNodeLabel] = childNodeInfo;
        }
    }

    uint threadGroupIndex_inNodeE = threadGroupIndex % threadGroupCount_perNodeE;
    uint remainingThreadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG - threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE;
    uint threadCount_currentThreadGroup = remainingThreadCount_perNodeE >= HITTEST_CS_THREADGROUP_SIZE ? HITTEST_CS_THREADGROUP_SIZE : remainingThreadCount_perNodeE;

    uint currentGroupNodeStartLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE) / OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeEndLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE + threadCount_currentThreadGroup - 1) / OUTGOING_COUNT_FZBPG;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG clusterNodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G IndivisibleNodeData_G;
        IndivisibleNodeData_G.aabb = clusterNodeData.aabb;
        IndivisibleNodeData_G.layerIndex = nodeInfo.x;
        IndivisibleNodeData_G.nodeIndex = nodeInfo.y;

        uint octreeLayerNode = 1 << (3 * nodeInfo.x);
        uint normalIndex = nodeInfo.y / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        IndivisibleNodeData_G.nodeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

        groupNodeData_G[groupThreadIndex] = IndivisibleNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    IndivisibleNodeData_E nodeData_E = groupNodeData_E;
    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));

    uint hasDataChildNodeCount_E = groupHasDataChildNodeCount;
    uint hitTestCount_perNodeG = hasDataChildNodeCount_E * HITTEST_COUNT_PER_CHILDNODE_FZBPG;
    for (int hitTestIndex = groupThreadIndex; hitTestIndex < currentGoupNodeGCount * hitTestCount_perNodeG; hitTestIndex += HITTEST_CS_THREADGROUP_SIZE) {
        uint nodeLocalLabel_G = hitTestIndex / hitTestCount_perNodeG;
        uint nodeLabel_G = nodeLocalLabel_G + currentGroupNodeStartLabel_G;
        IndivisibleNodeData_G nodeData_G = groupNodeData_G[nodeLocalLabel_G];

        uint childNodeLabel = (hitTestIndex % hitTestCount_perNodeG) / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        ChildNodeInfo_E childNodeInfo_E = groupChildNodeInfos_E[childNodeLabel];

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (childNodeInfo_E.aabb.maximum - childNodeInfo_E.aabb.minimum) + childNodeInfo_E.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        samplePoint_E = getAABBPoint(childNodeInfo_E.aabb, samplePoint_E, sampleDir); // make samplePoint_E move to the aabb surface, prevent self shielding

        HitPayload_FzbWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
        hitTestAble &= dot(-sampleDir, nodeData_G.nodeNormal) > 0.0f;
        if (hitTestAble) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;
            traceRay(ray, hitPayload);
            if (hitPayload.hit > 0.0f) {
                bool check = checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb);
                check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                check &= dot(hitPayload.normal, nodeData_G.nodeNormal) > 0.707f;
                if (!check) hitPayload.hit = -1.0f;
            }
        }
        groupHitPayloads[hitTestIndex] = hitPayload;

        bool printfAble = pushConst.frameIndex == 2;
        printfAble &= nodeLabel_G == 48;
        printfAble &= nodeLabel_E == 15;
        if (printfAble) {
            printf("childLabel: %d, hitTestIndex: %d, hit: %f\nsamplePoint_E: %f %f %f  samplePoint_G: %f %f %f\nchildNodeAabb: %f %f %f %f %f %f\n\n",
                   childNodeLabel, hitTestIndex % HITTEST_COUNT_PER_CHILDNODE_FZBPG, hitPayload.hit,
                   samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
                   samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
                   childNodeInfo_E.aabb.minimum.x, childNodeInfo_E.aabb.minimum.y, childNodeInfo_E.aabb.minimum.z,
                   childNodeInfo_E.aabb.maximum.x, childNodeInfo_E.aabb.maximum.y, childNodeInfo_E.aabb.maximum.z,
            );
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex >= threadCount_currentThreadGroup) return;

    if (warpIndex < currentGoupNodeGCount && warpLane < hitTestCount_perNodeG) {
        uint childNodeLabel = warpLane / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        float childNodeE = groupChildNodeInfos_E[childNodeLabel].E;

        uint childNodeFirstWarpLane = warpLane / HITTEST_COUNT_PER_CHILDNODE_FZBPG * HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        uint hitTestIndex_inChildNode = childNodeFirstWarpLane + warpLane % HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        bool hit = groupHitPayloads[warpIndex * hitTestCount_perNodeG + hitTestIndex_inChildNode].hit == 1.0f;
        uint hitCount = uint(hit);
        for (int offset = HITTEST_COUNT_PER_CHILDNODE_FZBPG / 2; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            hitCount += WaveReadLaneAt(hitCount, targetLane);
        }
        if (warpLane == childNodeFirstWarpLane) {
            childNodeE *= (float)hitCount / HITTEST_COUNT_PER_CHILDNODE_FZBPG;

            float mergeE = WaveActiveSum(childNodeE);
            float probability = childNodeE / max(mergeE, 1e-16f);

            float selectProbablity = rand(seed);
            selectProbablity = WaveReadLaneFirst(selectProbablity);
            uint sampleChildLabel = 0;
            for (int childNodeTargetLabel = 0; childNodeTargetLabel < hasDataChildNodeCount_E; ++childNodeTargetLabel) {
                float other_probablity = WaveReadLaneAt(probability, childNodeTargetLabel * HITTEST_COUNT_PER_CHILDNODE_FZBPG);
                if (selectProbablity <= other_probablity) {
                    sampleChildLabel = childNodeTargetLabel;
                    break;
                }
                selectProbablity -= other_probablity;
            }

            if (childNodeLabel == sampleChildLabel) {
                ChildNodeInfo_E childNodeInfo = groupChildNodeInfos_E[childNodeLabel];
                probability *= childNodeInfo.pdf;
                OctreeNodePairData_FzbPG nodePairData = { childNodeInfo.aabb, probability };

                uint nodeLabel_G = currentGroupNodeStartLabel_G + warpIndex;
                uint nodePairIndex = nodeLabel_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG + nodeData_E.nodeIndex;
                OctreeNodePairDataBuffer[nodePairIndex] = nodePairData;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint nodeLocalIndex_G = groupThreadIndex / OUTGOING_COUNT_FZBPG;
    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);
    float weightSum = 0.0f;
    for (int childNodeLabel = 0; childNodeLabel < hasDataChildNodeCount_E; ++childNodeLabel) {
        float weight = 0.0f;
        for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT_PER_CHILDNODE_FZBPG; ++hitTestIndex) {
            uint hitPayloadIndex = nodeLocalIndex_G * hitTestCount_perNodeG + childNodeLabel * HITTEST_COUNT_PER_CHILDNODE_FZBPG +  hitTestIndex;
            HitPayload_FzbWeight hitPayload = groupHitPayloads[hitPayloadIndex];
            if (hitPayload.hit <= 0.0f) continue;

            float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
            if (cosine <= 0.0f) continue;

            float3 bsdf = float3(0.0f);
            if (hitPayload.material.type == MaterialType::Diffuse) {
                bsdf = getBSDF_Diffuse(hitPayload.material.albedo, -hitPayload.sampleDir, hitPayload.normal);
            } else {
                float3 tangent, bitangent;
                orthonormalBasis(hitPayload.normal, tangent, bitangent);
                float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

                bool isExt = true;
                bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
            }

            float3 weight3 = bsdf * cosine;
            weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);
        }
        weight *= groupChildNodeInfos_E[childNodeLabel].E / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        weightSum += weight;
    }
    if (weightSum > 0.0f) {
        uint nodeLabel_G = currentGroupNodeStartLabel_G + nodeLocalIndex_G;
        uint nodeIndex_E = nodeData_E.nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                           nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG + (nodeIndex_E + 56);
        OctreeNodePairWeightBuffer[weightIndex] = weightSum;
    }

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G;
}

/*
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getOctreeNodeHitTestCmd(uint threadIndex: SV_DispatchThreadID) {
    GlobalInfoBuffer[0].cmd.x = (HitTestNodePairCountBuffer[0] * 8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG + HITTEST_CS_THREADGROUP_SIZE2 - 1) / HITTEST_CS_THREADGROUP_SIZE2;
}

groupshared uint groupHitTestNodePairCount;

[numthreads(HITTEST_CS_THREADGROUP_SIZE2, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHitTestNodePairCount = HitTestNodePairCountBuffer[0];
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHitTestNodePairCount * 8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG) return;

    OctreeHitTestNodePairInfo_FzbPG nodePairInfo = HitTestNodePairInfoBuffer[pushConst.currentLayer - 1][threadIndex];
    uint layerIndex_G = nodePairInfo.layerIndex_G;
    uint nodeIndex_G = nodePairInfo.nodeIndex_G;
    OctreeNodeClusterData_G_FzbPG nodeData_G = OctreeClusterDataBuffer_G[layerIndex_G - OCTREE_CLUSTER_LAYER_FZBPG][nodeIndex_G];
    float3 normal_G = normalize(nodeData_G.meanNormal.xyz);

    uint nodeIndex_E = nodePairInfo.nodeIndex_E * 8 + ;
    OctreeNodeClusterData_E_FzbPG nodeData_E = OctreeClusterDataBuffer_E[pushConst.currentLayer][nodeIndex_E];
    float3 normal_E = normalize(nodeData_E.meanNormal.xyz);

    uint childNodeIndex = (groupThreadIndex & (8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG - 1)) / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
    uint hitTestIndex = groupThreadIndex % HITTEST_COUNT_PER_CHILDNODE_FZBPG;
    OctreeNodeClusterData_E_FzbPG childNodeData = OctreeClusterDataBuffer_E[pushConst.currentLayer + 1][nodeIndex_E * 8 + childNodeIndex];
    bool hasData_E = childNodeData.E > 0.0f;

    uint seed = xxhash32(uint3(threadIndex, hitTestIndex, (uint)pushConst.frameIndex));

    float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

    randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 samplePoint_E = randXYZ * (childNodeData.aabb.maximum - childNodeData.aabb.minimum) + childNodeData.aabb.minimum;

    float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
    samplePoint_E = getAABBPoint(childNodeData.aabb, samplePoint_E, sampleDir); // make samplePoint_E move to the aabb surface, prevent self shielding

    HitPayload_FzbWeight hitPayload;
    hitPayload.hit = -1.0f;
    hitPayload.sampleDir = sampleDir;

    bool hitTestAble = dot(normal_E, sampleDir) > 0.0f;
    hitTestAble &= dot(sampleDir, normal_G) > 0.0f;
    if (dot(normal_E, sampleDir) > 0.0f) {
        RayDesc ray;
        ray.Direction = sampleDir;
        ray.Origin = samplePoint_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        traceRay(ray, hitPayload);
        if (hitPayload.hit > 0.0f) {
            bool check = checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb);
            check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
            check &= dot(hitPayload.normal, normal_G) > 0.707f;
            if (!check) hitPayload.hit = -1.0f;
        }
    }
    bool hit = hitPayload.hit == 1.0f;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint nodePairHitTestCount = 8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG;
    uint nodePairIndex = groupThreadIndex / nodePairHitTestCount;
    uint nodePairFirstWarpLane = warpLane / nodePairHitTestCount * nodePairHitTestCount;
    bool allHit = hit, allNotHit = hit;
    for (int offset = nodePairHitTestCount / 2; offset >= 0; offset /= 2) {
        uint targetLane = warpLane + offset;
        allHit &= WaveReadLaneAt(allHit, targetLane);
        allNotHit |= WaveReadLaneAt(allNotHit, targetLane);
    }
    allHit = WaveReadLaneAt(allHit, nodePairFirstWarpLane);
    allNotHit = WaveReadLaneAt(allNotHit, nodePairFirstWarpLane);

    if (allHit && warpLane == nodePairFirstWarpLane) {
        uint allHitLabel;
        InterlockedAdd<uint>(groupAllHitNodeCount_G, 1, allHitLabel);   //should use warp shuffle
        groupAllHitNodeLocalIndex_G[allHitLabel] = warpIndex;
    }

    bool partialHit = !allHit && !allNotHit;
    if (partialHit && warpLane == nodePairFirstWarpLane) {
        uint nodePairLabel = 0;
        InterlockedAdd<uint>(HitTestNodePairCountBuffer[0], 1, nodePairLabel);
        OctreeHitTestNodePairInfo_FzbPG nodePairInfo;
        nodePairInfo.nodeIndex_E = nodeData_E.nodeIndex;
        nodePairInfo.layerIndex_G = groupNodeData_G[nodeLocalIndex_G].layerIndex;
        nodePairInfo.nodeIndex_G = groupNodeData_G[nodeLocalIndex_G].nodeIndex;
        HitTestNodePairInfoBuffer[0][nodePairLabel] = nodePairInfo;
    }
}
*/
#else
#define MAX_INDIVISIBLE_NODE_G_COUNT HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG
groupshared IndivisibleNodeData_G groupNodeData_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_FzbWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT_FZBPG];

[numthreads(HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int indivisibleNodeCount_E = groupIndivisibleNodeCount_E;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG;
    uint threadGroupCount_perNodeE = (threadCount_perNodeE + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE;
    uint nodeLabel_E = threadGroupIndex / threadGroupCount_perNodeE;
    if (groupThreadIndex == 0) {
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[nodeLabel_E];
        OctreeNodeClusterData_E_FzbPG clusterNodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        IndivisibleNodeData_E indivisibleNodeData_E;
        indivisibleNodeData_E.nodeIndex = nodeIndex;
        indivisibleNodeData_E.E = clusterNodeData.E;
        indivisibleNodeData_E.normal = normalize(clusterNodeData.meanNormal.xyz);
        indivisibleNodeData_E.aabb = clusterNodeData.aabb;

        groupNodeData_E = indivisibleNodeData_E;
    }

    uint threadGroupIndex_inNodeE = threadGroupIndex % threadGroupCount_perNodeE;
    uint remainingThreadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG - threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE;
    uint threadCount_currentThreadGroup = remainingThreadCount_perNodeE >= HITTEST_CS_THREADGROUP_SIZE ? HITTEST_CS_THREADGROUP_SIZE : remainingThreadCount_perNodeE;

    uint currentGroupNodeStartLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE) / OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeEndLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE + threadCount_currentThreadGroup - 1) / OUTGOING_COUNT_FZBPG;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG clusterNodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G IndivisibleNodeData_G;
        IndivisibleNodeData_G.aabb = clusterNodeData.aabb;

        uint octreeLayerNode = 1 << (3 * nodeInfo.x);
        uint normalIndex = nodeInfo.y / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        IndivisibleNodeData_G.nodeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

        groupNodeData_G[groupThreadIndex] = IndivisibleNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    IndivisibleNodeData_E nodeData_E = groupNodeData_E;

    for (int hitTestIndex = groupThreadIndex; hitTestIndex < currentGoupNodeGCount * HITTEST_COUNT_FZBPG; hitTestIndex += HITTEST_CS_THREADGROUP_SIZE) {
        uint nodeLabel_G_group = hitTestIndex / HITTEST_COUNT_FZBPG;
        IndivisibleNodeData_G nodeData_G = groupNodeData_G[nodeLabel_G_group];

        uint seed = xxhash32(uint3(threadIndex, hitTestIndex, (uint)pushConst.frameIndex));

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) + nodeData_E.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        samplePoint_E = getAABBPoint(nodeData_E.aabb, samplePoint_E, sampleDir); // make samplePoint_E move to the aabb surface, prevent self shielding

        HitPayload_FzbWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
        hitTestAble &= dot(-sampleDir, nodeData_G.nodeNormal) > 0.0f;
        if (hitTestAble) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;
            traceRay(ray, hitPayload);
            if (hitPayload.hit > 0.0f) {
                bool check = checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb);
                check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                check &= dot(hitPayload.normal, nodeData_G.nodeNormal) > 0.707f;
                if (!check) hitPayload.hit = -1.0f;
            }
        }
        groupHitPayloads[hitTestIndex] = hitPayload;

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= nodeLabel_G % indivisibleNodeCount_G == 87;
        // printfAble &= nodeLabel_E == 30;
        ////printfAble &= i % HITTEST_COUNT_FZBPG == 0;
        // if (printfAble) {
        //     printf("nodeLabel_E: %d hit: %f\nhitPos: %f %f %f, hitNormal: %f %f %f\nsampleDir: %f %f %f\naabb_E: %f %f %f %f %f %f\naabb_G: %f %f %f %f %f %f\nnodeNormal_G: %f %f %f\n\n",
        //            nodeLabel_E, hitPayload.hit,
        //            hitPayload.hitPos.x, hitPayload.hitPos.y, hitPayload.hitPos.z,
        //            hitPayload.normal.x, hitPayload.normal.y, hitPayload.normal.z,
        //            sampleDir.x, sampleDir.y, sampleDir.z,
        //            nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
        //            nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
        //            nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
        //            nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
        //            nodeData_G.nodeNormal.x, nodeData_G.nodeNormal.y, nodeData_G.nodeNormal.z,
        //     );
        // }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex >= threadCount_currentThreadGroup) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);
    uint nodeLabel_G_group = groupThreadIndex / OUTGOING_COUNT_FZBPG;
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT_FZBPG; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT_FZBPG + hitTestIndex;
        HitPayload_FzbWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        if (cosine <= 0.0f) continue;

        float3 bsdf = float3(0.0f);
        if (hitPayload.material.type == MaterialType::Diffuse) {
            bsdf = getBSDF_Diffuse(hitPayload.material.albedo, -hitPayload.sampleDir, hitPayload.normal);
        } else {
            float3 tangent, bitangent;
            orthonormalBasis(hitPayload.normal, tangent, bitangent);
            float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

            bool isExt = true;
            bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        }

        float3 weight3 = bsdf * cosine;
        weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);
    }
    weight = weight * nodeData_E.E / HITTEST_COUNT_FZBPG;

    if (weight > 0.0f) {
        uint nodeLabel_G = currentGroupNodeStartLabel_G + nodeLabel_G_group;
        uint nodeIndex_E = nodeData_E.nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                           nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG + (nodeIndex_E + 56);
        OctreeNodePairWeightBuffer[weightIndex] = weight;
    }

    if (threadIndex == 0) {
        // uint threadTotalCount = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG;
        // GlobalInfoBuffer[0].cmd.x = (threadTotalCount + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
        GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G;
    }
}
#endif


groupshared float groupWeights[CLUSTER_LAYER_NODECOUNT_E_FZBPG / 8];
[numthreads(CLUSTER_LAYER_NODECOUNT_E_FZBPG, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    uint layerIndex = pushConst.currentLayer;
    uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;

    uint outgoingIndex = threadGroupIndex / indivisibleNodeCount_G; // outgoing Index
    uint nodeLabel_G = threadGroupIndex % indivisibleNodeCount_G;

    uint weightStartIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                       nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG;
    uint weightIndex = weightStartIndex + OctreeLayerStartIndex_FzbPG[OCTREE_CLUSTER_LAYER_FZBPG] + groupThreadIndex;
    float weight = OctreeNodePairWeightBuffer[weightIndex];

    uint warplane = groupThreadIndex & 31;
    uint blockFirstWarplane = warplane / 8 * 8;
    float weightSum = weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = warplane + offset;
        weightSum += WaveReadLaneAt(weightSum, targetLane);
    }
    weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
    float probability = weight / max(weightSum, 1e-20f);
    if (probability > 0.0f) OctreeNodePairWeightBuffer[weightIndex] = probability;
    if (warplane == blockFirstWarplane) groupWeights[groupThreadIndex / 8] = weightSum;
    GroupMemoryBarrierWithGroupSync();

    //bool printfAble = pushConst.frameIndex == 1;
    //printfAble &= outgoingIndex == 57;
    //printfAble &= nodeLabel_G == 0;
    //if (printfAble) {
    //    printf("layerIndex: 2, nodeIndex_E: %d\nweight:%f, probability: %f\n\n", groupThreadIndex, weight, probability);
    //}

    for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG - 1; layerIndex >= 0; --layerIndex) {
        uint layerNodeCount = OctreeLayerNodeCount_FzbPG[layerIndex];
        if(groupThreadIndex >= layerNodeCount) return;      //after votla

        weight = groupWeights[groupThreadIndex];
        weightSum = weight;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warplane + offset;
            weightSum += WaveReadLaneAt(weightSum, targetLane);
        }
        weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
        probability = weight / max(weightSum, 1e-20f);

        //if (printfAble) {
        //    printf("layerIndex: %d, nodeIndex_E: %d\nweight:%f, probability: %f\n\n", layerIndex, groupThreadIndex, weight, probability);
        //}

        if (probability > 0.0f) {
            weightIndex = weightStartIndex + OctreeLayerStartIndex_FzbPG[layerIndex] + groupThreadIndex;
            OctreeNodePairWeightBuffer[weightIndex] = probability;
        }

        if (layerIndex > 0) {
            if (warplane == blockFirstWarplane) groupWeights[groupThreadIndex / 8] = weightSum;
            GroupMemoryBarrierWithGroupSync();
        }
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
    output.debug_color = false;

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

        uint nodeIndex_sample = IndivisibleNodeInfoBuffer_E[pushConst.sampleNodeLabel_E];
        if (layerIndex == 0 && nodeIndex == nodeIndex_sample) {
            output.debug_color = true;
            output.color = float4(0.0f, 1.0f, 0.0f, 1.0f);
        }
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

struct VSout_OctreeNodePairHitTestResult {
    float4 sv_position : SV_Position;
    float4 color;
};
struct PSout_OctreeNodePairHitTestResult
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout_OctreeNodePairHitTestResult vertexMain_OctreeNodePairHitTestResult(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout_OctreeNodePairHitTestResult output;
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    uint indivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    uint indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (instanceIndex >= indivisibleNodeCount_E + 1 ||
        pushConst.sampleNodeLabel_G <= 0 ||
        pushConst.sampleNodeLabel_G >= indivisibleNodeCount_G + 1) return output;

    uint nodeLabel_G = pushConst.sampleNodeLabel_G - 1;
    uint2 nodeInfo_G = IndivisibleNodeInfoBuffer_G[nodeLabel_G];
    OctreeNodeClusterData_G_FzbPG nodeData_G = OctreeClusterDataBuffer_G[nodeInfo_G.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo_G.y];

    float3 nodeCenter = (nodeData_G.aabb.minimum + nodeData_G.aabb.maximum) * 0.5f;
    float3 viewDir = normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter);
    float3 outgoing = mul(pushConst.randomRotateMatrix, viewDir);
    uint outgoingIndex = inverseOutgoing_FzbPG(outgoing, OUTGOING_COUNT_FZBPG);
    uint weightStartIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG + nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG;

    AABB nodeAABB;
    if (instanceIndex == indivisibleNodeCount_E) {
        nodeAABB = nodeData_G.aabb;
        output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);

        if (pushConst.frameIndex == 1 && vertexIndex == 0) printf("outgongIndex: %d\n\n", outgoingIndex);
    } else {
        uint nodeLabel_E = instanceIndex;
        uint nodeIndex_E = IndivisibleNodeInfoBuffer_E[instanceIndex];
        float nodePdf = 1.0f;
        #ifdef ADAPTIVE_IMPORTANCE_SAMPLING
        uint nodePairIndex = nodeLabel_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG + nodeIndex_E;
        OctreeNodePairData_FzbPG nodePairData = OctreeNodePairDataBuffer[nodePairIndex];
        nodeAABB = nodePairData.aabb;
        nodePdf = nodePairData.pdf;
        #else
        OctreeNodeData_E_FzbPG nodeData_E = ClusterlayerDataBuffer_E[nodeIndex_E];
        nodeAABB = nodeData_E.aabb;
        nodePdf = nodeData_E.pdf;
        #endif

        float layerPdf = 1.0f;
        for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG; layerIndex >= 0; --layerIndex) {
            uint weightIndex = weightStartIndex + nodeIndex_E + OctreeLayerStartIndex_FzbPG[layerIndex];
            layerPdf *= OctreeNodePairWeightBuffer[weightIndex];
            nodeIndex_E /= 8;
        }

        //if (pushConst.frameIndex == 1 && vertexIndex == 0){
        //    printf("nodeIndex_E: %d\naabb: %f %f %f %f %f %f\nnodePdf: %f, layerPdf: %f\n\n",
        //           IndivisibleNodeInfoBuffer_E[instanceIndex],
        //           nodeAABB.minimum.x, nodeAABB.minimum.y, nodeAABB.minimum.z,
        //           nodeAABB.maximum.x, nodeAABB.maximum.y, nodeAABB.maximum.z,
        //           nodePdf, layerPdf
        //    );
        //}

        float pdf = layerPdf * nodePdf;
        output.color = float4(0.0f, pdf * 100.0f, 0.0f, 1.0f);
    }
    if (nodeAABB.minimum.x > 1e10f) return output;

    float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

    output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
    output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
    output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
    output.sv_position.xyz += nodeAABB.minimum;

    output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);

    return output;
}
[shader("pixel")]
PSout_OctreeNodePairHitTestResult fragmentMain_OctreeNodePairHitTestResult(VSout_OctreeNodePairHitTestResult input) {
    return { input.color };
}
#endif