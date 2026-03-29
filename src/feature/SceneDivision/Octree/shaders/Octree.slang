#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/Octree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant> pushConst;
[[vk::binding(BindingPoints_Octree::eVGB_Octree)]] StructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(BindingPoints_Octree::eVGBMaterialInfos_Octree)]] StructuredBuffer<VGBMaterialInfo_SVOPG, ScalarDataLayout> eVGBMaterialInfos_Octree;
[[vk::binding(BindingPoints_Octree::eOctreeArray_G_Octree)]] RWStructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_Octree::eOctreeArray_E_Octree)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];
#ifndef NDEBUG
[[vk::binding(BindingPoints_Octree::eWireframeMap_Octree)]] RWTexture2D<float4> wireframeMap[];
[[vk::binding(BindingPoints_Octree::eBaseMap_Octree)]] Texture2D<float4> baseMap;
#endif

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

//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID) {
    uint maxLayerNodeTotalCount = uint(pow(8, pushConst.maxDepth));     //one thread per one node
    if (threadIndex >= maxLayerNodeTotalCount) return;

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

    VGBMaterialInfo_SVOPG voxelMaterialInfos = eVGBMaterialInfos_Octree[threadIndex];
    uint materialIndex = 0, materialCount = 0;
    for (int i = 0; i < MAX_MATERIAL_COUNT; ++i) {
        if (voxelMaterialInfos.materialCount[i] > materialCount) {
            materialCount = voxelMaterialInfos.materialCount[i];
            materialIndex = i;
        }
    }

    uint irradianceCount = 0;
    OctreeNodeData_E nodeData_E;
    nodeData_E.irradiance = float3(0.0f);
    nodeData_E.pdf = 1.0f;
    nodeData_E.meanNormal = float4(0.0f);
    nodeData_E.aabb = aabb;
    nodeData_E.indivisible = 1;
    nodeData_E.label = 0;
    for (int normalIndex = 0; normalIndex < 6; ++normalIndex) {
        VGBVoxelData_SVOPG voxelData = VGBs[normalIndex][threadIndex];

        AABB nodeAabb;
        nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
        nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
        nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
        nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
        nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
        nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

        bool hasNormal = voxelData.sumNormal_G.w > 0.0f;
        nodeData_G.meanNormal = hasNormal ? voxelData.sumNormal_G / voxelData.sumNormal_G.w : float4(0.0f);
        nodeData_G.aabb = nodeAabb;
        nodeData_G.materialIndex = materialIndex;

        uint32_t octreeIndex = normalIndex * (pushConst.maxDepth + 1) + pushConst.maxDepth;
        OctreeArray_G[octreeIndex][threadIndex] = nodeData_G;

        hasNormal = voxelData.sumNormal_E.w > 0.0f;
        if (hasNormal && nodeAabb.minimum.x >= 3.402823466e+36F) {
            hasNormal = false;
            voxelData.irradiance = float4(0.0f);
            voxelData.sumNormal_E = float4(0.0f);
        }
        nodeData_E.irradiance += voxelData.irradiance.xyz;
        irradianceCount += (uint)voxelData.irradiance.w;

        nodeData_E.meanNormal += voxelData.sumNormal_E;
        nodeData_E.aabb.minimum.x = min(nodeData_E.aabb.minimum.x, nodeAabb.minimum.x);
        nodeData_E.aabb.minimum.y = min(nodeData_E.aabb.minimum.y, nodeAabb.minimum.y);
        nodeData_E.aabb.minimum.z = min(nodeData_E.aabb.minimum.z, nodeAabb.minimum.z);
        nodeData_E.aabb.maximum.x = max(nodeData_E.aabb.maximum.x, nodeAabb.maximum.x);
        nodeData_E.aabb.maximum.y = max(nodeData_E.aabb.maximum.y, nodeAabb.maximum.y);
        nodeData_E.aabb.maximum.z = max(nodeData_E.aabb.maximum.z, nodeAabb.maximum.z);
    }
    if (irradianceCount > 0) {
        nodeData_E.irradiance /= irradianceCount;
        nodeData_E.meanNormal /= nodeData_E.meanNormal.w == 0.0f ? 1.0f : nodeData_E.meanNormal.w;
    }
    OctreeArray_E[pushConst.maxDepth][threadIndex] = nodeData_E;
    
    nodeData_G.meanNormal = float4(0.0f);
    nodeData_G.aabb = aabb;
    nodeData_G.materialIndex = 0;

    nodeData_E.irradiance = float3(0.0f);
    nodeData_E.meanNormal = float4(0.0f);
    nodeData_E.aabb = aabb;

    uint nodeTotalCount = maxLayerNodeTotalCount;
    for (int i = pushConst.maxDepth - 1; i >= 0; --i) {
        nodeTotalCount /= 8;
        if (threadIndex >= nodeTotalCount) continue;
        OctreeArray_E[i][threadIndex] = nodeData_E;
    }

    uint nodeTotalCount_oneOctree_G = pushConst.OctreeNodeTotalCount - maxLayerNodeTotalCount;     //except max layer
    uint nodeTotalCount_allOctree_G = nodeTotalCount_oneOctree_G * 6;
    for (int i = threadIndex; i < nodeTotalCount_allOctree_G; i += maxLayerNodeTotalCount) {
        uint normalIndex = i / nodeTotalCount_oneOctree_G;
        uint nodeIndex = i - normalIndex * nodeTotalCount_oneOctree_G;
        int layerIndex = 0;
        uint layerNodeCount = (uint)pow(8, layerIndex);
        for (; layerIndex < pushConst.maxDepth; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        uint octreeIndex = normalIndex * (pushConst.maxDepth + 1) + layerIndex;
        OctreeArray_G[octreeIndex][nodeIndex] = nodeData_G;
    }
}

// only blockFirstWarpLane has right mergeAABB
void atomicMergeAABB(inout AABB aabb) {
    uint laneIndex = WaveGetLaneIndex();

    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = laneIndex + offset;
        float other_val = WaveReadLaneAt(aabb.minimum.x, targetLane);
        aabb.minimum.x = min(aabb.minimum.x, other_val);
    }
    //aabb.minimum.x = WaveReadLaneAt(aabb.minimum.x, blockFirstWarpLane);

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
[numthreads(THREADGROUP_SIZE2, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint nodeTotalCount = pushConst.currentLayerNodeCount;
    if (threadIndex >= nodeTotalCount * 6) return;

    uint32_t warpIndex = groupThreadIndex / 32;
    uint32_t warpLane = groupThreadIndex & 31;

    uint32_t indexInBlock = threadIndex & 7;          // index of eight brothers
    uint32_t blockFirstWarpLane = (warpLane / 8) * 8; // first thread of block in warp
    uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
    uint32_t blockIndexInGroup = groupThreadIndex / 8;
    //----------------------------------------getOctree_G-------------------------------------------
    uint normalIndex = threadIndex / nodeTotalCount;
    uint octreeIndex = normalIndex * (pushConst.maxDepth + 1) + pushConst.currentDepth;
    uint nodeIndex = threadIndex % nodeTotalCount;
    uint32_t blockIndex = nodeIndex / 8; // blockIndex of global = fatherNodeIndex

    OctreeNodeData_G nodeData_G = OctreeArray_G[octreeIndex][nodeIndex];
    bool hasGeometry = nodeData_G.meanNormal.w > 0.0f;
    uint32_t warpHasGeometryMask = WaveActiveBallot(hasGeometry).x;
    uint32_t blockHasGeometryNodeCount = countbits(warpHasGeometryMask & blockNodeMask);
    if (blockHasGeometryNodeCount == 0) {}
    else if (blockHasGeometryNodeCount == 1) {
        if (hasGeometry) OctreeArray_G[octreeIndex - 1][blockIndex] = nodeData_G;
    } else {
        //----------------------------------------calc node data-------------------------------------
        //get mergeAABB
        AABB mergeAABB = nodeData_G.aabb;
        atomicMergeAABB(mergeAABB);
        //get mergeNormal
        float3 mergeNormal = nodeData_G.meanNormal.xyz;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeNormal.x += WaveReadLaneAt(mergeNormal.x, targetLane);
            mergeNormal.y += WaveReadLaneAt(mergeNormal.y, targetLane);
            mergeNormal.z += WaveReadLaneAt(mergeNormal.z, targetLane);
        }
        mergeNormal /= blockHasGeometryNodeCount;

        // get mergeAABB's surfaceArea
        float mergeAABBVolume = 0.0f;
        if (warpLane == blockFirstWarpLane) {
            float3 length = mergeAABB.maximum - mergeAABB.minimum;
            length += float3(int3(1) - sign(length)); // if x == 0.0, then x = 1.0 else x = x
            mergeAABBVolume = length.x * length.y * length.z;
        }

        float aabbVolume = 0.0f;
        // The volume is needed to calculate the scaling factor only when there is lighting.
        // Although it is said that where there is lighting, there must be geometry, rasterization may not be injected
        if (hasGeometry) {
            float3 length = nodeData_G.aabb.maximum - nodeData_G.aabb.minimum;
            length += float3(int3(1) - sign(length)); // if x == 0.0, then x = 1.0 else x = x
            aabbVolume = length.x * length.y * length.z;
        }

        // get merge node's aabb volume
        float aabbVolumeSum = aabbVolume;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            aabbVolumeSum += WaveReadLaneAt(aabbVolumeSum, targetLane);
        }

        uint blockMaterialTypeMask = hasGeometry ? 1u << nodeData_G.materialIndex : 0u;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            blockMaterialTypeMask |= WaveReadLaneAt(blockMaterialTypeMask, targetLane);
        }
        uint materialTypeCount = countbits(blockMaterialTypeMask);
        //--------------------------------------------judge divisible--------------------------------------------
        //bool indivisible = true;   // = nodeData_G.indivisible == 1;
        //for (int offset = 4; offset > 0; offset /= 2) {
        //    uint targetLane = warpLane + offset;
        //    indivisible &= WaveReadLaneAt(indivisible, targetLane);
        //}
        //indivisible &= hasGeometry ? materialTypeCount == 1 : true;

        bool indivisible = true;
        if (warpLane == blockFirstWarpLane) {
            indivisible = materialTypeCount == 1 &&
                          length(mergeNormal) > pushConst.entropyThreshold.x &&
                          (mergeAABBVolume - aabbVolumeSum) < pushConst.voxelVolume * (pushConst.maxDepth - pushConst.currentDepth + 1) * 4;

            nodeData_G.indivisible = indivisible;
            nodeData_G.label = 0;
            nodeData_G.aabb = mergeAABB;
            nodeData_G.meanNormal = float4(mergeNormal, 1.0f);
            nodeData_G.materialIndex = firstbitlow(blockMaterialTypeMask);
            OctreeArray_G[octreeIndex - 1][blockIndex] = nodeData_G;
        }

        // octreeIndex: 23, nodeIndex: 12047, aabb: -0.381061 1.671665 -0.171104 -0.359361 1.678067 -0.166743, normal: 0.221960 0.774146 0.587211
        if (pushConst.frameIndex == 0) {
            //if (octreeIndex == 23 && nodeIndex >= 12040 && nodeIndex <= 12047) { // && mergeAABB.maximum.x < -0.99f && mergeAABB.maximum.y > 1.9f
            //    printf("\nnodeIndex: %d, indivisible: %d, nodeData_G.meanNormal: %f %f %f\n",
            //            nodeIndex, indivisible,
            //            nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z
            //        );
            //}
            //if (octreeIndex == 22 && nodeIndex >= 1504 && nodeIndex <= 1511) { // && mergeAABB.maximum.x < -0.99f && mergeAABB.maximum.y > 1.9f
            //    printf("\nnodeIndex: %d, indivisible: %d, nodeData_G.meanNormal: %f %f %f\n",
            //            nodeIndex, indivisible,
            //            nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z
            //        );
            //}
            //if(nodeData_G.materialIndex == 6 && normalIndex == 3 && hasGeometry){
            //    printf("aabb: %f %f %f %f %f %f\nnormal: %f %f %f\noctreeIndex: %d, nodeIndex: %d\n\n",
            //        nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
            //        nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
            //        nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z,
            //        octreeIndex, nodeIndex
            //    );
            //}
        }
    }

    //----------------------------------------getOctree_E-------------------------------------------
    if (threadIndex >= nodeTotalCount) return;

    blockIndex = threadIndex / 8;

    OctreeNodeData_E nodeData_E = OctreeArray_E[pushConst.currentDepth][threadIndex];
    bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;
    uint32_t warpHasIrradianceMask = WaveActiveBallot(hasIrradiance).x;
    uint32_t blockHasIrradianceNodeCount = countbits(warpHasIrradianceMask & blockNodeMask);

    if (blockHasIrradianceNodeCount == 0) return;
    if (blockHasIrradianceNodeCount == 1) {
        if (hasIrradiance) OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        return;
    } else {
        float irradianceValue = length(nodeData_E.irradiance);
        float irradianceSum = irradianceValue;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            irradianceSum += WaveReadLaneAt(irradianceSum, targetLane);
        }
        irradianceSum = WaveReadLaneAt(irradianceSum, blockFirstWarpLane);
        float pdf = irradianceValue / irradianceSum;

        uint32_t sampleWarpLane = blockFirstWarpLane;
        float sampleProbability = 1.0f;
        if (warpLane == blockFirstWarpLane) {
            uint randomSeed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));
            sampleProbability = rand(randomSeed) - pdf;
        }
        sampleProbability = WaveReadLaneAt(sampleProbability, blockFirstWarpLane);
        for (int offset = 1; offset < 8; ++offset) {
            uint targetLane = blockFirstWarpLane + offset;
            sampleProbability -= WaveReadLaneAt(pdf, targetLane);
            if (sampleProbability <= 0.0f) {
                sampleWarpLane += offset;
                break;
            }
        }
        //--------------------------------calc mergeNode data------------------------------------------
        float3 mergeIrradiance = nodeData_E.irradiance.xyz;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeIrradiance.x += WaveReadLaneAt(mergeIrradiance.x, targetLane);
            mergeIrradiance.y += WaveReadLaneAt(mergeIrradiance.y, targetLane);
            mergeIrradiance.z += WaveReadLaneAt(mergeIrradiance.z, targetLane);
        }
        //--------------------------------------assignment------------------------------------------------
        if (warpLane == sampleWarpLane) {
            nodeData_E.irradiance = mergeIrradiance;
            nodeData_E.pdf *= pdf;
            //nodeData_E.meanNormal = float4(meanNormal, 1.0f);
            //nodeData_E.aabb = mergeAABB;
            nodeData_E.indivisible = 1;
            nodeData_E.label = 0;
            OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        }
    }
}
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray2(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint nodeTotalCount = pushConst.currentLayerNodeCount;
    if (threadIndex >= nodeTotalCount * 6) return;

    uint32_t warpIndex = groupThreadIndex / 32;
    uint32_t warpLane = groupThreadIndex & 31;

    uint32_t indexInBlock = threadIndex & 7;          // index of eight brothers
    uint32_t blockFirstWarpLane = (warpLane / 8) * 8; // first thread of block in warp
    uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
    uint32_t blockIndexInGroup = groupThreadIndex / 8;
    //----------------------------------------getOctree_G-------------------------------------------
    uint normalIndex = threadIndex / nodeTotalCount;
    uint octreeIndex = normalIndex * (pushConst.maxDepth + 1) + pushConst.currentDepth;
    uint nodeIndex = threadIndex % nodeTotalCount;
    uint32_t blockIndex = nodeIndex / 8; // blockIndex of global = fatherNodeIndex

    OctreeNodeData_G nodeData_G = OctreeArray_G[octreeIndex][nodeIndex];
    bool hasGeometry = nodeData_G.meanNormal.w > 0.0f;
    uint32_t warpHasGeometryMask = WaveActiveBallot(hasGeometry).x;
    uint32_t blockHasGeometryNodeCount = countbits(warpHasGeometryMask & blockNodeMask);
    // Geometry injection is not 100% effective. It may not be injected completely.
    // However, if there is light in this area, we cannot directly return. We also need to handle E
    if (blockHasGeometryNodeCount == 0) {}
    else if (blockHasGeometryNodeCount == 1) {
        if (hasGeometry) OctreeArray_G[octreeIndex - 1][blockIndex] = nodeData_G;
    } else {
        AABB mergeAABB = nodeData_G.aabb;
        atomicMergeAABB(mergeAABB);
        if (warpLane == blockFirstWarpLane) {
            nodeData_G.indivisible = 0;
            nodeData_G.aabb = mergeAABB;
            nodeData_G.meanNormal = float4(0.0f, 0.0f, 0.0f, 1.0f);
            OctreeArray_G[octreeIndex - 1][blockIndex] = nodeData_G;
        }
    }
    //----------------------------------------getOctree_E-------------------------------------------
    if (threadIndex >= nodeTotalCount) return;

    blockIndex = threadIndex / 8;

    OctreeNodeData_E nodeData_E = OctreeArray_E[pushConst.currentDepth][threadIndex];
    bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;
    uint32_t warpHasIrradianceMask = WaveActiveBallot(hasIrradiance).x;
    uint32_t blockHasIrradianceNodeCount = countbits(warpHasIrradianceMask & blockNodeMask);

    if (blockHasIrradianceNodeCount == 0) {}
    else if (blockHasIrradianceNodeCount == 1) {
        if (hasIrradiance) OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
    } else {
        AABB mergeAABB = nodeData_E.aabb;
        atomicMergeAABB(mergeAABB);
        if (warpLane == blockFirstWarpLane) {
            nodeData_E.indivisible = 0;
            nodeData_E.aabb = mergeAABB;
            nodeData_E.irradiance = float3(1.0f, 0.0f, 0.0f);
            nodeData_E.meanNormal = float4(0.0f, 0.0f, 0.0f, 1.0f);
            OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        }
    }
}

#ifndef NDEBUG
//------------------------------------------------debug---------------------------------------------------------
struct VSout_Wireframe {
    float4 sv_position : SV_Position;
    uint renderTargetIndex;
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
        float4 color6 : SV_Target6;     //max VGBSize = 128x128x128
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
        float4 color13 : SV_Target13;     //max VGBSize = 128x128x128
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
        output.renderTargetIndex += pushConst.maxDepth - pushConst.clusteringLevel;
    } else nodeIndex = nodeIndex % pushConst.showOctreeNodeTotalCount;

    uint layerIndex = pushConst.clusteringLevel;
    uint layerNodeCount = uint(pow(8, layerIndex));
    for (; layerIndex <= pushConst.maxDepth; ++layerIndex) {
        if (nodeIndex < layerNodeCount) break;
        nodeIndex -= layerNodeCount;
        layerNodeCount *= 8;
    }
    
    AABB nodeAABB; uint invisible = 1;
    if (isG) {
        uint normalIndex = instanceIndex / pushConst.showOctreeNodeTotalCount;
        uint octreeIndex = normalIndex * (pushConst.maxDepth + 1) + layerIndex;
        OctreeNodeData_G octreeData_G = OctreeArray_G[octreeIndex][nodeIndex];
        nodeAABB = octreeData_G.aabb;
        invisible = octreeData_G.meanNormal.w > 0.0f && octreeData_G.indivisible == 1;
    } else {
        OctreeNodeData_E octreeData_E = OctreeArray_E[layerIndex][nodeIndex];
        nodeAABB = octreeData_E.aabb;
        invisible = (octreeData_E.irradiance.x + octreeData_E.irradiance.y + octreeData_E.irradiance.z > 0.0f) && octreeData_E.indivisible == 1;
    }

    if (isG && instanceIndex / pushConst.showOctreeNodeTotalCount != pushConst.normalIndex) invisible = 0;

    if (nodeAABB.minimum.x > 10000.0f || invisible == 0) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += nodeAABB.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    }
    output.renderTargetIndex += layerIndex - pushConst.clusteringLevel;

    return output;
}
[shader("pixel")]
PSout_Wireframe fragmentMain_Wireframe(VSout_Wireframe input) {
    PSout_Wireframe output;
    float4 colors[WireframeMapCount];
    for (int i = 0; i < WireframeMapCount; ++i) colors[i] = float4(0.0f);
    colors[input.renderTargetIndex] = float4(1.0f, 0.0f, 0.0f, 1.0f);

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
}
#endif