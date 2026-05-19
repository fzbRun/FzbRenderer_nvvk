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
            float3 lengthAxis = nodeAabb.maximum - nodeAabb.minimum;
            int3 axisSign = sign(lengthAxis);
            lengthAxis += float3(int3(1) - axisSign) * 0.0001f; // if x == 0.0, then x = 1.0 else x = x
            if (all(axisSign == 0)) lengthAxis = float3(0.0f);
            float aabbVolume = lengthAxis.x * lengthAxis.y + lengthAxis.x * lengthAxis.z + lengthAxis.y * lengthAxis.z;

            nodeClusterData_E.E = aabbVolume * length(voxelData.irradiance.xyz / voxelData.irradiance.w);
            nodeClusterData_E.meanNormal = nodeClusterData_G.meanNormal;    // voxelData.sumNormal_E / voxelData.sumNormal_E.w;
            nodeClusterData_E.aabb = nodeAabb;
            #ifndef NDEBUG
            nodeClusterData_E.irradiance = voxelData.irradiance.xyz / voxelData.irradiance.w;
            #endif

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
            //{
            //    bool printfAble = pushConst.frameIndex == 1;
            //    bool nodeCheck = false;
            //    nodeCheck |= pushConst.currentLayer == 4 && fatherNodeIndex == 201;
            //    nodeCheck |= pushConst.currentLayer == 3 && fatherNodeIndex == 25;
            //    nodeCheck |= pushConst.currentLayer == 2 && fatherNodeIndex == 3;
            //    printfAble &= nodeCheck;
            //    if (printfAble) {
            //        printf("nodeIndex: %d\naabb: %f %f %f %f %f %f\n\n",
            //            nodeIndex,
            //            nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
            //            nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
            //        );
            //    }
            //}

            if (warpLane == blockFirstWarpLane) {
                nodeData_G.fillRate = aabbVolumeSum / mergeAABBVolume;
                indivisible &= length(mergeNormal) > 0.707f &&
                    (nodeData_G.fillRate >= 0.75 ||
                     (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG) * 1.5f);

                //{
                //    bool printfAble = pushConst.frameIndex == 1;
                //    bool nodeCheck = false;
                //    nodeCheck |= pushConst.currentLayer == 4 && fatherNodeIndex == 201;
                //    nodeCheck |= pushConst.currentLayer == 3 && fatherNodeIndex == 344;
                //    nodeCheck |= pushConst.currentLayer == 2 && fatherNodeIndex == 43;
                //    printfAble &= nodeCheck;
                //    if (printfAble) {
                //        printf("indivisible: %d, normalLenght: %f, fillRate: %f\n\n", indivisible, length(mergeNormal), nodeData_G.fillRate);
                //    }
                //}

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

        // uint nodeIndex_sample = IndivisibleNodeInfoBuffer_E[pushConst.sampleNodeLabel_E];
        // if (layerIndex == 0 && nodeIndex == nodeIndex_sample) {
        //     output.debug_color = true;
        //     output.color = float4(0.0f, 1.0f, 0.0f, 1.0f);
        // }

        output.debug_color = true;
        output.color = float4(octreeData_E.irradiance, 1.0f);
    }

    if (normalIndex != pushConst.normalIndex) invisible = 0;

    //{
    //    bool printfAble = pushConst.frameIndex == 1;
    //    printfAble &= isG;
    //    printfAble &= invisible == 1;
    //    printfAble &= vertexIndex == 0;
    //    printfAble &= nodeAABB.minimum.x - 0.7 < 0.0f && nodeAABB.maximum.x - 0.7 > 0.0f;
    //    printfAble &= nodeAABB.minimum.y - 4.118 < 0.0f && nodeAABB.maximum.y - 4.118 > 0.0f;
    //    // printfAble &= abs(nodeAABB.minimum.x - 2.495325) < 0.01f;
    //    if (printfAble) {
    //        printf("layerIndex: %d, nodeIndex: %d, instanceIndex: %d\nnodeaabb: %f %f %f %f %f %f\n\n",
    //               layerIndex + OCTREE_CLUSTER_LAYER_FZBPG, nodeIndex, instanceIndex,
    //               nodeAABB.minimum.x, nodeAABB.minimum.y, nodeAABB.minimum.z,
    //               nodeAABB.maximum.x, nodeAABB.maximum.y, nodeAABB.maximum.z,
    //        );
    //    }
    //    bool debugColor = layerIndex == 0 && nodeIndex == 344;
    //    debugColor |= layerIndex == 1 && nodeIndex == 2753;
    //    if (debugColor) {
    //        output.debug_color = true;
    //        output.color = float4(0.0f, 10.0f, 0.0f, 1.0f);
    //    }
    //}

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
#endif