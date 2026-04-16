#include "renderer/SVOPathGuidingRenderer/hard/Octree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_SVOPG> pushConst;
[[vk::binding(BindingPoints_Octree_SVOPG::eVGB)]] StructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(BindingPoints_Octree_SVOPG::eVGBMaterialInfos)]] StructuredBuffer<VGBMaterialInfo_SVOPG, ScalarDataLayout> VGBMaterialInfos_Octree[];
[[vk::binding(BindingPoints_Octree_SVOPG::eOctreeArray_G)]] RWStructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_Octree_SVOPG::eOctreeArray_E)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];
[[vk::binding(BindingPoints_Octree_SVOPG::eNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> NodeData_E;
[[vk::binding(BindingPoints_Octree_SVOPG::eBlockInfos_G)]] RWStructuredBuffer<uint> BlockInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_SVOPG::eBlockInfos_E)]] RWStructuredBuffer<uint> BlockInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_SVOPG::eHasDataBlockIndices_G)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_G;
[[vk::binding(BindingPoints_Octree_SVOPG::eHasDataBlockIndices_E)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_E;
[[vk::binding(BindingPoints_Octree_SVOPG::eHasDataBlockCount)]] RWStructuredBuffer<HasDataOctreeBlockCount> HasDataBlockCountBuffer;
[[vk::binding(BindingPoints_Octree_SVOPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo, ScalarDataLayout> GlobalInfoBuffer;

groupshared uint groupWarpHasDataCount_G[32];
groupshared uint groupWarpHasDataCount_E[32];
groupshared uint groupHasDataGlobalLabel_G;
groupshared uint groupHasDataGlobalLabel_E;

groupshared uint groupHasDataNodeCount_G;
groupshared uint groupHasDataNodeCount_E;

int3 getVoxelIndexXYZ(int voxelIndex) {
    int voxelCount = int(pushConst.VGBStartPos_Size.w);
    int voxelGroupTotalCount = voxelCount * voxelCount * voxelCount;
    int3 voxelIndexXYZ = int3(0, 0, 0);
    while (voxelIndex > 0) {
        voxelGroupTotalCount /= 8;
        voxelCount /= 2;
        int groupIndex = voxelIndex / voxelGroupTotalCount;
        int3 groupIndexXYZ = int3(groupIndex & 1, (groupIndex & 2) >> 1, (groupIndex & 4) >> 2);
        voxelIndexXYZ += groupIndexXYZ * voxelCount;

        voxelIndex -= groupIndexXYZ.x * voxelGroupTotalCount + 
                      groupIndexXYZ.y * voxelGroupTotalCount * 2 +
                      groupIndexXYZ.z * voxelGroupTotalCount * 4;
    }
    return voxelIndexXYZ;
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
//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    AABB aabb;
    aabb.minimum.x = 3.402823466e+38F;
    aabb.minimum.y = 3.402823466e+38F;
    aabb.minimum.z = 3.402823466e+38F;
    aabb.maximum.x = -3.402823466e+38F;
    aabb.maximum.y = -3.402823466e+38F;
    aabb.maximum.z = -3.402823466e+38F;

    OctreeNodeData_G nodeData_G;
    nodeData_G.indivisible = 1;
    nodeData_G.label = 0;
    nodeData_G.aabb = aabb;
    nodeData_G.fillRate = 1.0f;
    nodeData_G.meanNormal = float4(0.0f);
    nodeData_G.materialCountSum = 0;
    for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) nodeData_G.materialCounts[i] = 0;

    OctreeNodeData_E nodeData_E;
    nodeData_E.E = 0.0f;
    nodeData_E.pdf = 1.0f;
    nodeData_E.meanNormal = float4(0.0f);
    nodeData_E.indivisible = 1;
    nodeData_E.aabb = aabb;

    if (threadIndex < NODECOUNT_E) NodeData_E[threadIndex] = nodeData_E;

    if (threadIndex < pushConst.VGBVoxelTotalCount * 6) { // must be 1024 Integer multiple, so warp and threadGroup can synchronization
        uint VGBIndex = threadIndex / pushConst.VGBVoxelTotalCount;
        uint voxelIndex = threadIndex - VGBIndex * pushConst.VGBVoxelTotalCount;
        VGBVoxelData_SVOPG voxelData = VGBs[VGBIndex][voxelIndex];

        bool hasData_G = voxelData.sumNormal_G.w > 0.0f;
        AABB nodeAabb;
        if (hasData_G) {
            VGBMaterialInfo_SVOPG voxelMaterialInfos = VGBMaterialInfos_Octree[VGBIndex][voxelIndex];
            for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) nodeData_G.materialCountSum += voxelMaterialInfos.materialCount[i];
            nodeData_G.materialCounts = voxelMaterialInfos.materialCount;

            nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
            nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
            nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
            nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
            nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
            nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

            //int3 voxelIndexXYZ = int3((nodeAabb.minimum - pushConst.VGBStartPos_Size.xyz) / pushConst.VGBVoxelSize.xyz);
            //float3 voxelStartPos = (float3)voxelIndexXYZ * pushConst.VGBVoxelSize.xyz + pushConst.VGBStartPos_Size.xyz;
            //float3 voxelEndPos = voxelStartPos + pushConst.VGBVoxelSize.xyz;
            //// nodeAabb.minimum = nodeAabb.minimum - (nodeAabb.minimum - voxelStartPos) * 0.1f;
            //// nodeAabb.maximum = nodeAabb.maximum + (voxelEndPos - nodeAabb.maximum) * 0.1f;
            //int3 axisSign = sign(nodeAabb.maximum - nodeAabb.minimum);
            //nodeAabb.minimum = nodeAabb.minimum - (nodeAabb.minimum - voxelStartPos) * axisSign;
            //nodeAabb.maximum = nodeAabb.maximum + (voxelEndPos - nodeAabb.maximum) * axisSign;
            //nodeAabb.minimum = max(nodeAabb.minimum, voxelStartPos);
            //nodeAabb.maximum = min(nodeAabb.maximum, voxelEndPos);
            nodeData_G.aabb = nodeAabb;

            nodeData_G.meanNormal = voxelData.sumNormal_G / voxelData.sumNormal_G.w;
        }
        //-----------------------------------------------E--------------------------------------------
        bool hasData_E = hasData_G && voxelData.sumNormal_E.w > 0.0f;
        if (hasData_E) {
            nodeData_E.E = length(voxelData.irradiance.xyz / voxelData.irradiance.w);
            nodeData_E.meanNormal = voxelData.sumNormal_E / voxelData.sumNormal_E.w;
            nodeData_E.aabb = nodeAabb;
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
            if (blockHasData_G) OctreeArray_G[pushConst.octreeMaxLayer][threadIndex] = nodeData_G;
            if (blockLane == 0) BlockInfoBuffer_G[blockIndex] = (uint)blockHasData_G;

            uint warpHasDataMask_E = WaveActiveBallot(hasData_E).x;
            uint blockHasDataNodeCount_E = countbits(warpHasDataMask_E & blockNodeMask);
            bool blockHasData_E = blockHasDataNodeCount_E > 0;
            if (blockHasData_E) OctreeArray_E[pushConst.octreeMaxLayer][threadIndex] = nodeData_E;
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
        OctreeArray_G[layerIndex][nodeIndex] = nodeData_G;
        OctreeArray_E[layerIndex][nodeIndex] = nodeData_E;
    }
}
//-------------------------------------------------CreateOctree---------------------------------------------
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
        HasDataOctreeBlockCount hasDataNodeCounts = HasDataBlockCountBuffer[0];
        OctreeGlobalInfo globalInfo;
        uint hasDataNodeCount = max(hasDataNodeCounts.count_G, hasDataNodeCounts.count_E) * 8;
        globalInfo.cmd.x = (hasDataNodeCount + CREATEOCTREE_CS_THREADGROUP_SIZE - 1) / CREATEOCTREE_CS_THREADGROUP_SIZE;
        globalInfo.cmd.y = 1; globalInfo.cmd.z = 1;
        GlobalInfoBuffer[0] = globalInfo;

        //#ifndef NDEBUG
        //if (pushConst.frameIndex == 0) {
        //    printf("Layer: %d, hasDataBlockCount G: %d,  E: %d\n", pushConst.currentLayer, hasDataNodeCounts.count_G, hasDataNodeCounts.count_E);
        //}
        //#endif
    }
}

[numthreads(CREATEOCTREE_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        HasDataOctreeBlockCount hasDataBlockCounts = HasDataBlockCountBuffer[0];
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
        OctreeNodeData_G nodeData_G = OctreeArray_G[pushConst.currentLayer][nodeIndex];

        bool hasData = nodeData_G.meanNormal.w > 0.0f;
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);

        if (blockHasDataNodeCount == 1) {
            if (hasData) OctreeArray_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
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

            for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                for (int offset = 4; offset > 0; offset /= 2) {
                    uint targetLane = warpLane + offset;
                    nodeData_G.materialCounts[i] += WaveReadLaneAt(nodeData_G.materialCounts[i], targetLane);
                }
            }
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                nodeData_G.materialCountSum += WaveReadLaneAt(nodeData_G.materialCountSum, targetLane);
            }
            float materialEntropy = 0.0f;
            if (warpLane == blockFirstWarpLane) {
                for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
                    float pdf = (float)nodeData_G.materialCounts[i] / nodeData_G.materialCountSum;
                    if(pdf > 0.0f) materialEntropy -= pdf * log2(pdf);
                }
            }
            /*
            if (pushConst.frameIndex == 0 && pushConst.currentLayer == 3) {
                bool forward = abs(mergeNormal.y) > abs(mergeNormal.x) && abs(mergeNormal.y) > abs(mergeNormal.z);
                aabbVolumeSum = WaveReadLaneAt(aabbVolumeSum, blockFirstWarpLane);
                mergeAABBVolume = WaveReadLaneAt(mergeAABBVolume, blockFirstWarpLane);
                materialEntropy = WaveReadLaneAt(materialEntropy, blockFirstWarpLane);
                if (forward && fatherNodeIndex == 158) {
                    printf("\nfatherNodeIndex: %d\naabb: %f %f %f %f %f %f\nfillRate: %f,aabbVolume: %f\n",
                           fatherNodeIndex,
                           nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
                           nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
                           nodeData_G.fillRate, aabbVolume,
                           
                    );
                    if (warpLane == blockFirstWarpLane) {
                        printf("\ncluster:\nmergeAABB: %f %f %f %f %f %f\nfillRate: %f, aabbVolumeSum: %f, mergeAABBVolume: %f\nmaterailEntropy: %f\n",
                               mergeAABB.minimum.x, mergeAABB.minimum.y, mergeAABB.minimum.z,
                               mergeAABB.maximum.x, mergeAABB.maximum.y, mergeAABB.maximum.z,
                               aabbVolumeSum / mergeAABBVolume, aabbVolumeSum, mergeAABBVolume,
                               materialEntropy
                        );
                    }
                }
            }
            */

            bool indivisible = true;
            if (warpLane == blockFirstWarpLane) {
                nodeData_G.fillRate = aabbVolumeSum / mergeAABBVolume;
                indivisible = //materialEntropy < 0.2f && 
                              length(mergeNormal) > 0.707f &&
                              (nodeData_G.fillRate >= 0.7 ||
                              (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER) * 4);
                /*
                if (pushConst.frameIndex == 0 && pushConst.currentLayer == 3) {
                    bool forward = abs(mergeNormal.y) > abs(mergeNormal.x) && abs(mergeNormal.y) > abs(mergeNormal.z);
                    if (forward && fatherNodeIndex == 158) {
                        printf("\ncluster:%d\nmergeAABB: %f %f %f %f %f %f\nfillRate: %f, aabbVolumeSum: %f,   mergeAABBVolume: %f\nmaterailEntropy: %f\n",
                               fatherNodeIndex,       
                               mergeAABB.minimum.x, mergeAABB.minimum.y, mergeAABB.minimum.z,
                               mergeAABB.maximum.x, mergeAABB.maximum.y, mergeAABB.maximum.z,
                               nodeData_G.fillRate, aabbVolumeSum, mergeAABBVolume,
                               materialEntropy
                        );
                    }
                }
                */

                nodeData_G.indivisible = indivisible;
                nodeData_G.label = 0;
                nodeData_G.aabb = mergeAABB;
                nodeData_G.meanNormal = float4(mergeNormal, 1.0f);
                OctreeArray_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
            }
        }

        uint fatherBlockIndex = fatherNodeIndex / 8;
        if (blockLane == 0) BlockInfoBuffer_G[fatherBlockIndex] = true;     // have Data competition, but input same value
    }
    //----------------------------------------getOctree_E-------------------------------------------
    if (threadIndex < groupHasDataNodeCount_E) {
        uint currentBlockLabel = threadIndex / 8;
        uint nodeIndex = HasDataBlockIndicsBuffer_E[currentBlockLabel] * 8 + blockLane;
        uint fatherNodeIndex = nodeIndex / 8;
        OctreeNodeData_E nodeData_E = OctreeArray_E[pushConst.currentLayer][nodeIndex];

        bool hasData = nodeData_E.E > 0.0f;
        uint32_t warpHasDataMask = WaveActiveBallot(hasData).x;
        uint32_t blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
        if (blockHasDataNodeCount == 1) {
            if (hasData) OctreeArray_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
        } else {
            float mergeE = nodeData_E.E;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeE += WaveReadLaneAt(mergeE, targetLane);
            }
            mergeE = WaveReadLaneAt(mergeE, blockFirstWarpLane);

            //if (pushConst.frameIndex == 0 && mergeE > 1000000.0f) {
            //    printf("layerIndex: %d\nnodeIndex: %d, mergeE: %f E: %f\n",
            //        pushConst.currentLayer,
            //        nodeIndex, mergeE, nodeData_E.E
            //    );
            //}

            if (pushConst.currentLayer > 10) {
                AABB mergeAABB = nodeData_E.aabb;
                atomicMergeAABB(mergeAABB);

                float3 mergeNormal = nodeData_E.meanNormal.xyz; // get mergeNormal
                for (int offset = 4; offset > 0; offset /= 2) {
                    uint targetLane = warpLane + offset;
                    mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
                    mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
                    mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
                }
                mergeNormal /= blockHasDataNodeCount;

                if (warpLane == blockFirstWarpLane) {
                    nodeData_E.E = mergeE;
                    nodeData_E.pdf = 1.0f;
                    nodeData_E.meanNormal = float4(mergeNormal, 1.0f);
                    nodeData_E.aabb = mergeAABB;
                    nodeData_E.indivisible = 1;
                    OctreeArray_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
                }
            } else {
                float probability = nodeData_E.E / mergeE;

                uint32_t sampleWarpLane = blockFirstWarpLane;
                float sampleProbability = 1.0f;
                if (warpLane == blockFirstWarpLane) {
                    uint randomSeed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));
                    sampleProbability = rand(randomSeed);
                }
                sampleProbability = WaveReadLaneAt(sampleProbability, blockFirstWarpLane);
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
                    nodeData_E.indivisible = 1;
                    OctreeArray_E[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_E;
                }
            }
        }

        uint fatherBlockIndex = fatherNodeIndex / 8;
        if (blockLane == 0) BlockInfoBuffer_E[fatherBlockIndex] = true;
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
    //----------------------------------------getOctree_G-------------------------------------------
    OctreeNodeData_G nodeData_G = OctreeArray_G[pushConst.currentLayer][threadIndex];
    bool hasData_G = nodeData_G.meanNormal.w > 0.0f;
    uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
    uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
    if (blockHasDataNodeCount_G == 0) {}
    else if (blockHasDataNodeCount_G == 1) {
        if (hasData_G) OctreeArray_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
    } else {
        AABB mergeAABB = nodeData_G.aabb;
        atomicMergeAABB(mergeAABB);
        if (warpLane == blockFirstWarpLane) {
            nodeData_G.indivisible = 0;
            nodeData_G.aabb = mergeAABB;
            nodeData_G.meanNormal = float4(0.0f, 0.0f, 0.0f, 1.0f);
            OctreeArray_G[pushConst.currentLayer - 1][fatherNodeIndex] = nodeData_G;
        }
    }
    //----------------------------------------getNodeData_E-------------------------------------------
    if (pushConst.currentLayer == OCTREE_CLUSTER_LAYER)
        NodeData_E[threadIndex] = OctreeArray_E[pushConst.currentLayer][threadIndex];
}

#ifndef NDEBUG
//------------------------------------------------debug---------------------------------------------------------
struct VSout_Wireframe {
    float4 sv_position : SV_Position;
    uint renderTargetIndex;
    bool debug_color;
    float4 color;
};
struct PSout_Wireframe
{
    float4 color0 : SV_Target0;
    #if WireframeMapCount > 1
        float4 color1 : SV_Target1;
    #endif
    #if WireframeMapCount > 2
        float4 color2 : SV_Target2;
    #endif
    #if WireframeMapCount > 3
        float4 color3 : SV_Target3;
    #endif
    #if WireframeMapCount > 4
        float4 color4 : SV_Target4;
    #endif
    #if WireframeMapCount > 5
        float4 color5 : SV_Target5;
    #endif
    #if WireframeMapCount > 6
        float4 color6 : SV_Target6; // max VGBSize = 128x128x128
    #endif
    #if WireframeMapCount > 7
        float4 color7 : SV_Target7;
    #endif
    #if WireframeMapCount > 8
        float4 color8 : SV_Target8;
    #endif
    #if WireframeMapCount > 9
        float4 color9 : SV_Target9;
    #endif
    #if WireframeMapCount > 10
        float4 color10 : SV_Target10;
    #endif
    #if WireframeMapCount > 11
        float4 color11 : SV_Target11;
    #endif
    #if WireframeMapCount > 12
        float4 color12 : SV_Target12;
    #endif
    #if WireframeMapCount > 13
        float4 color13 : SV_Target13; // max VGBSize = 128x128x128
    #endif
};

[shader("vertex")]
VSout_Wireframe vertexMain_Wireframe(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VSout_Wireframe output;
    output.renderTargetIndex = 0;

    bool isG = true;
    uint nodeIndex = instanceIndex;
    if (nodeIndex >= pushConst.showOctreeNodeTotalCount * 6) {
        nodeIndex -= pushConst.showOctreeNodeTotalCount * 6;
        isG = false;
        output.renderTargetIndex += pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER;
    }
    uint normalIndex = nodeIndex / pushConst.showOctreeNodeTotalCount;
    nodeIndex = nodeIndex % pushConst.showOctreeNodeTotalCount;

    uint layerIndex = OCTREE_CLUSTER_LAYER;
    uint layerNodeCount = uint(pow(8, layerIndex));
    for (; layerIndex <= pushConst.octreeMaxLayer; ++layerIndex) {
        if (nodeIndex < layerNodeCount) break;
        nodeIndex -= layerNodeCount;
        layerNodeCount *= 8;
    }
    nodeIndex += normalIndex * uint(pow(8, layerIndex));

    AABB nodeAABB; uint invisible = 1;
    if (isG) {
        OctreeNodeData_G octreeData_G = OctreeArray_G[layerIndex][nodeIndex];
        nodeAABB = octreeData_G.aabb;
        invisible = octreeData_G.meanNormal.w > 0.0f && octreeData_G.indivisible == 1;
    } else {
        OctreeNodeData_E octreeData_E = OctreeArray_E[layerIndex][nodeIndex];
        nodeAABB = octreeData_E.aabb;
        invisible = (octreeData_E.E > 0.0f) && octreeData_E.indivisible == 1;
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
    output.renderTargetIndex += layerIndex - OCTREE_CLUSTER_LAYER;

    if (layerIndex == 2 && nodeIndex == 158) {
        output.debug_color = true;
        output.color = float4(1.0f);
    } else output.debug_color = false;

    return output;
}
[shader("pixel")]
PSout_Wireframe fragmentMain_Wireframe(VSout_Wireframe input) {
    PSout_Wireframe output;
    float4 colors[WireframeMapCount];
    for (int i = 0; i < WireframeMapCount; ++i) colors[i] = float4(0.0f);
    colors[input.renderTargetIndex] = float4(1.0f, 0.0f, 0.0f, 1.0f);
    if (input.debug_color) colors[input.renderTargetIndex] = input.color;

    #if WireframeMapCount > 0
        output.color0 = colors[0];
    #endif
    #if WireframeMapCount > 1
        output.color1 = colors[1];
    #endif
    #if WireframeMapCount > 2
        output.color2 = colors[2];
    #endif
    #if WireframeMapCount > 3
        output.color3 = colors[3];
    #endif
    #if WireframeMapCount > 4
        output.color4 = colors[4];
    #endif
    #if WireframeMapCount > 5
        output.color5 = colors[5];
    #endif
    #if WireframeMapCount > 6
        output.color6 = colors[6];
    #endif
    #if WireframeMapCount > 7
        output.color7 = colors[7];
    #endif
    #if WireframeMapCount > 8
        output.color8 = colors[81];
    #endif
    #if WireframeMapCount > 9
        output.color9 = colors[9];
    #endif
    #if WireframeMapCount > 10
        output.color10 = colors[10];
    #endif
    #if WireframeMapCount > 11
        output.color11 = colors[11];
    #endif
    #if WireframeMapCount > 12
        output.color12 = colors[12];
    #endif
    #if WireframeMapCount > 13
        output.color13 = colors[13];
    #endif

    return output;
}

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_MergeResult(uint3 dispatchId: SV_DispatchThreadID) {
    /*
    uint2 wireframeMapSize;
    wireframeMap[0].GetDimensions(wireframeMapSize.x, wireframeMapSize.y);
    if (dispatchId.x >= wireframeMapSize.x || dispatchId.y >= wireframeMapSize.y) return;

    uint2 baseMapSize;
    baseMap.GetDimensions(baseMapSize.x, baseMapSize.y);
    float2 texCoords = (float2)dispatchId.xy / wireframeMapSize;
    uint2 baseMapTexCoords = uint2(texCoords * baseMapSize);
    float4 baseColor = baseMap[baseMapTexCoords];

    for (int i = 0; i < WireframeMapCount; ++i) {
        float4 result = float4(1.0f);
        float4 wireframeColor = wireframeMap[i][dispatchId.xy];
        if (wireframeColor.w == 0.0f) result = baseColor;
        else result = wireframeColor;
        wireframeMap[i][dispatchId.xy] = result;
    }
    */
}
#endif