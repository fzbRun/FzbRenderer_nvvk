#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/Octree/shaderio.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant> pushConst;
[[vk::binding(BindingPoints_Octree::eVGB_Octree)]] RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;
[[vk::binding(BindingPoints_Octree::eOctreeArray_G_Octree)]] RWStructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(BindingPoints_Octree::eOctreeArray_E_Octree)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeArray_E[];

//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(512, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint3 dispatchId: SV_DispatchThreadID) {
    uint nodeTotalCount = uint(pow(2, pushConst.maxDepth));
    if (dispatchId.x >= nodeTotalCount) return;
    VGBVoxelData voxelData = VGB[dispatchId.x];

    AABB aabb;
    aabb.minimum.x = asfloat(voxelData.aabbU.minimum.x);
    aabb.minimum.y = asfloat(voxelData.aabbU.minimum.y);
    aabb.minimum.z = asfloat(voxelData.aabbU.minimum.z);
    aabb.maximum.x = asfloat(voxelData.aabbU.maximum.x);
    aabb.maximum.y = asfloat(voxelData.aabbU.maximum.y);
    aabb.maximum.z = asfloat(voxelData.aabbU.maximum.z);

    OctreeNodeData_G nodeData_G;
    nodeData_G.meanNormal = voxelData.sumNormal_G.w > 0.0f ? float4(voxelData.sumNormal_G.xyz / voxelData.sumNormal_G.w, 1.0f) : float4(0.0f);
    nodeData_G.aabb = aabb;
    nodeData_G.indivisible = 1;
    nodeData_G.label = 0;
    OctreeArray_G[pushConst.maxDepth][dispatchId.x] = nodeData_G;

    OctreeNodeData_E nodeData_E;
    nodeData_E.irradiance = voxelData.irradiance.w > 0.0f ? voxelData.irradiance.xyz / voxelData.irradiance.w : float3(0.0f);
    nodeData_E.notIgnoreRatio = 1.0f;
    nodeData_E.meanNormal = voxelData.sumNormal_E.w > 0.0f ? float4(normalize(voxelData.sumNormal_E.xyz), 1.0f) : float4(0.0f);
    nodeData_E.aabb = aabb;
    nodeData_E.indivisible = 1;
    nodeData_E.label = 0;
    OctreeArray_E[pushConst.maxDepth][dispatchId.x] = nodeData_E;

    aabb.minimum.x = 3.402823466e+38F;
    aabb.minimum.y = 3.402823466e+38F;
    aabb.minimum.z = 3.402823466e+38F;
    aabb.maximum.x = -3.402823466e+38F;
    aabb.maximum.y = -3.402823466e+38F;
    aabb.maximum.z = -3.402823466e+38F;

    nodeData_G.meanNormal = float4(0.0f);
    nodeData_G.aabb = aabb;

    nodeData_E.meanNormal = float4(0.0f);
    nodeData_E.aabb = aabb;
    for (int i = pushConst.maxDepth - 1; i > 0; --i) {
        nodeTotalCount /= 8;
        if (dispatchId.x >= nodeTotalCount) return;

        OctreeArray_G[i][dispatchId.x] = nodeData_G;
        OctreeArray_E[i][dispatchId.x] = nodeData_E;
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
[numthreads(256, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint nodeTotalCount = uint(pow(2, pushConst.currentDepth));
    if (threadIndex >= nodeTotalCount) return;

    uint warpSize = WaveGetLaneCount();

    uint32_t warpIndex = groupThreadIndex / warpSize;
    uint32_t warpLane = groupThreadIndex & (warpSize - 1);

    uint32_t indexInBlock = threadIndex & 7;            // index of eight brothers
    uint32_t blockIndex = threadIndex / 8;              // blockIndex of global = fatherNodeIndex
    uint32_t blockFirstWarpLane = (blockIndex & 3) * 8; // first thread of block in warp
    uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
    uint32_t blockIndexInGroup = groupThreadIndex / 8;
    //----------------------------------------getOctree_G-------------------------------------------
    OctreeNodeData_G nodeData_G = OctreeArray_G[pushConst.currentDepth][threadIndex];
    bool hasGeometry = nodeData_G.aabb.minimum.x > 10000.0f;
    uint32_t warpHasGeometryMask = WaveActiveBallot(hasGeometry).x;
    uint32_t blockHasGeometryNodeCount = countbits(warpHasGeometryMask & blockNodeMask);
    //Geometry injection is not 100% effective. It may not be injected completely. 
    //However, if there is light in this area, we cannot directly return. We also need to handle E
    if (blockHasGeometryNodeCount == 0) {}
    else if (blockHasGeometryNodeCount == 1) {
        if (hasGeometry) OctreeArray_G[pushConst.currentDepth - 1][blockIndex] = nodeData_G;
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
        float mergeAABBSurfaceArea = 0.0f;
        if (warpLane == blockFirstWarpLane) {
            float lengthX = mergeAABB.maximum.x - mergeAABB.minimum.x;
            float lengthY = mergeAABB.maximum.y - mergeAABB.minimum.y;
            float lengthZ = mergeAABB.maximum.z - mergeAABB.minimum.z;
            mergeAABBSurfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
        }

        //get node aabb's surfaceArea
        float surfaceArea = 0.0f;
        // The surfaceArea is needed to calculate the scaling factor only when there is lighting.
        // Although it is said that where there is lighting, there must be geometry, rasterization may not be injected
        if (hasGeometry) {
            float lengthX = nodeData_G.aabb.maximum.x - nodeData_G.aabb.minimum.x;
            float lengthY = nodeData_G.aabb.maximum.y - nodeData_G.aabb.minimum.y;
            float lengthZ = nodeData_G.aabb.maximum.z - nodeData_G.aabb.minimum.z;
            surfaceArea = (lengthX * lengthY + lengthX * lengthZ + lengthY * lengthZ) * 2;
        }

        // get merge node's aabb surfaceArea
        float surfaceAreaSum = surfaceArea;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            surfaceAreaSum += WaveReadLaneAt(surfaceAreaSum, targetLane);
        }

        //--------------------------------------------judge divisible--------------------------------------------
        bool indivisible = true;
        // calc mergeNode's entropy
        float entropy = 0.0f;
        if (warpLane == blockFirstWarpLane) {
            float normalEntropy = max(1.0f - length(mergeNormal), 0.0f);
            float surfaceAreaEntropy = max(1.0f - surfaceAreaSum / mergeAABBSurfaceArea, 0.0f);
            entropy = (normalEntropy + surfaceAreaEntropy) * 0.5f;
            //if node is very tidy or very chaos, we assume it can be clustering
            indivisible = entropy < pushConst.entropyThreshold.x || entropy > pushConst.entropyThreshold.y;
        }
        //---------------------------------------------assignment------------------------------------------------
        if (warpLane == blockFirstWarpLane) {
            nodeData_G.indivisible = indivisible;
            nodeData_G.label = 0;
            nodeData_G.aabb = mergeAABB;
            nodeData_G.meanNormal = float4(mergeNormal, 1.0f);
            OctreeArray_G[pushConst.currentDepth - 1][blockIndex] = nodeData_G;
        }
    }

    //----------------------------------------getOctree_E-------------------------------------------
    OctreeNodeData_E nodeData_E = OctreeArray_E[pushConst.currentDepth][threadIndex];
    bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;
    uint32_t warpHasIrradianceMask = WaveActiveBallot(hasIrradiance).x;
    uint32_t blockHasIrradianceNodeCount = countbits(warpHasIrradianceMask & blockNodeMask);

    if (blockHasIrradianceNodeCount == 0) return;
    if (blockHasIrradianceNodeCount == 1) {
        if (hasIrradiance) OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        return;
    } else {
        //--------------------------------get max rradiance node------------------------------------------
        float irradianceValue = length(nodeData_E.irradiance);
        float maxEIrradianceValue = irradianceValue;
        uint32_t maxEWarpLane = warpLane;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            float other_maxEIrradianceValue = WaveReadLaneAt(maxEIrradianceValue, targetLane);
            uint32_t other_maxEWarpLane = WaveReadLaneAt(maxEWarpLane, targetLane);
            if (other_maxEIrradianceValue > maxEIrradianceValue) {
                maxEIrradianceValue = other_maxEIrradianceValue;
                maxEWarpLane = other_maxEWarpLane;
            }
        }
        maxEIrradianceValue = WaveReadLaneAt(maxEIrradianceValue, blockFirstWarpLane);
        //--------------------------------judge ignore node---------------------------------------------------
        uint32_t ignore = 0;
        float relIrradianceRatio = irradianceValue / maxEIrradianceValue;
        if (relIrradianceRatio <= pushConst.irradianceRelRatioThreshold) ignore = 1;

        uint32_t blockIgnoreMask = ignore << indexInBlock;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            blockIgnoreMask |= WaveReadLaneAt(blockIgnoreMask, targetLane);
        }
        blockIgnoreMask = WaveReadLaneAt(blockIgnoreMask, blockFirstWarpLane);
        uint32_t blockNotIgnoreCount = 8 - countbits(blockIgnoreMask);

        // if only one node has irradiance, directly assign
        if (blockNotIgnoreCount == 1) {
            if (hasIrradiance) OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
            return;
        }
        //--------------------------------calc mergeNode data------------------------------------------
        // calc not ignore node's mergeNormal
        float3 meanNormal = ignore ? float3(0.0f) : nodeData_E.meanNormal.xyz;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            meanNormal.x += WaveReadLaneAt(meanNormal.x, targetLane);
            meanNormal.y += WaveReadLaneAt(meanNormal.y, targetLane);
            meanNormal.z += WaveReadLaneAt(meanNormal.z, targetLane);
        }
        meanNormal /= blockNotIgnoreCount;
        // calc not ignore node's mergeAABB
        AABB mergeAABB = nodeData_E.aabb;
        if (ignore == 1) {
            mergeAABB.minimum.x = 3.402823466e+38F;
            mergeAABB.minimum.y = 3.402823466e+38F;
            mergeAABB.minimum.z = 3.402823466e+38F;
            mergeAABB.maximum.x = -3.402823466e+38F;
            mergeAABB.maximum.y = -3.402823466e+38F;
            mergeAABB.maximum.z = -3.402823466e+38F;
        }
        atomicMergeAABB(mergeAABB);

        // calc not ignore node's mergeIrradiance
        float3 mergeIrradiance = nodeData_E.irradiance.xyz;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            mergeIrradiance.x += WaveReadLaneAt(mergeIrradiance.x, targetLane);
            mergeIrradiance.y += WaveReadLaneAt(mergeIrradiance.y, targetLane);
            mergeIrradiance.z += WaveReadLaneAt(mergeIrradiance.z, targetLane);
        }
        float notIgnoreRatio = ignore ? 0.0f : irradianceValue;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            notIgnoreRatio += WaveReadLaneAt(notIgnoreRatio, targetLane);
        }
        notIgnoreRatio = min(notIgnoreRatio / length(mergeIrradiance), 1.0f);
        //---------------------------------------------assignment------------------------------------------------
        if (warpLane == blockFirstWarpLane) {
            nodeData_E.irradiance = mergeIrradiance;
            nodeData_E.notIgnoreRatio = notIgnoreRatio;
            nodeData_E.meanNormal = float4(meanNormal, 1.0f);
            nodeData_E.aabb = mergeAABB;
            nodeData_E.indivisible = 1;
            nodeData_E.label = 0;
            OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        }
    }
}
[numthreads(512, 1, 1)]
[shader("compute")]
void computeMain_createOctreeArray2(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint nodeTotalCount = uint(pow(2, pushConst.currentDepth));
    if (threadIndex >= nodeTotalCount) return;

    uint warpSize = WaveGetLaneCount();

    uint32_t warpIndex = groupThreadIndex / warpSize;
    uint32_t warpLane = groupThreadIndex & (warpSize - 1);

    uint32_t indexInBlock = threadIndex & 7;            // index of eight brothers
    uint32_t blockIndex = threadIndex / 8;              // blockIndex of global = fatherNodeIndex
    uint32_t blockFirstWarpLane = (blockIndex & 3) * 8; // first thread of block in warp
    uint32_t blockNodeMask = 0xff << blockFirstWarpLane;
    uint32_t blockIndexInGroup = groupThreadIndex / 8;
    //----------------------------------------getOctree_G-------------------------------------------
    OctreeNodeData_G nodeData_G = OctreeArray_G[pushConst.currentDepth][threadIndex];
    bool hasGeometry = nodeData_G.aabb.minimum.x > 10000.0f;
    uint32_t warpHasGeometryMask = WaveActiveBallot(hasGeometry).x;
    uint32_t blockHasGeometryNodeCount = countbits(warpHasGeometryMask & blockNodeMask);
    // Geometry injection is not 100% effective. It may not be injected completely.
    // However, if there is light in this area, we cannot directly return. We also need to handle E
    if (blockHasGeometryNodeCount == 0) {}
    else if (blockHasGeometryNodeCount == 1) {
        if (hasGeometry) OctreeArray_G[pushConst.currentDepth - 1][blockIndex] = nodeData_G;
    } else {
        AABB mergeAABB = nodeData_G.aabb;
        atomicMergeAABB(mergeAABB);
        if (warpLane == blockFirstWarpLane) {
            nodeData_G.indivisible = 0;
            nodeData_G.aabb = mergeAABB;
            OctreeArray_G[pushConst.currentDepth - 1][blockIndex] = nodeData_G;
        }
    }
    //----------------------------------------getOctree_E-------------------------------------------
    OctreeNodeData_E nodeData_E = OctreeArray_E[pushConst.currentDepth][threadIndex];
    bool hasIrradiance = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;
    uint32_t warpHasIrradianceMask = WaveActiveBallot(hasIrradiance).x;
    uint32_t blockHasIrradianceNodeCount = countbits(warpHasIrradianceMask & blockNodeMask);

    if (blockHasIrradianceNodeCount == 0) return;
    if (blockHasIrradianceNodeCount == 1) {
        if (hasIrradiance) OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        return;
    } else {
        AABB mergeAABB = nodeData_E.aabb;
        atomicMergeAABB(mergeAABB);
        if (warpLane == blockFirstWarpLane) {
            nodeData_E.indivisible = 0;
            nodeData_E.aabb = mergeAABB;
            OctreeArray_E[pushConst.currentDepth - 1][blockIndex] = nodeData_E;
        }
    }
}