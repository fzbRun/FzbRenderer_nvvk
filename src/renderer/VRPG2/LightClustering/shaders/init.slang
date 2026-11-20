#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"
#include "renderer/VRPG2/GeometryInjection/GeometryInjectionShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<LightClusteringPushConstant_VRPG> pushConst;

[[vk::binding(LightClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(LightClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(LightClusteringBindingPoints_VRPG::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];

[[vk::binding(LightClusteringBindingPoints_VRPG::eLightNodeBlockInfos)]] RWStructuredBuffer<LightNodeBlockInfo> LightNodeBlockInfoBuffer;
[[vk::binding(LightClusteringBindingPoints_VRPG::eHasDataLightNodeBlockIndices)]] RWStructuredBuffer<LightNodeBlockIndex> HasDataLightNodeBlockIndexBuffer;
//----------------------------------------------init--------------------------------------------
// The leaf layer of the light tree is a coarse grid (same idea as VRPG's light VGB): every light
// voxel aggregates the geometry voxels inside its cell, exactly like VRPG's
// computeMain_clusterGeometryVGB clips each geometry AABB to the light cell before merging it.
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_init(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    uint lightVoxelCount = pushConst.lightVoxelCount;
    uint lightVoxelTotalCount = lightVoxelCount * lightVoxelCount * lightVoxelCount;
    if (threadIndex >= lightVoxelTotalCount * 6) return;

    AABB aabb; {
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    LightNodeData_Octree_VRPG lightNodeData; {
        lightNodeData.radiance = 0.0f;
        lightNodeData.halfAngle = M_PI;
        lightNodeData.meanNormal = float4(0.0f);
        lightNodeData.aabb = aabb;
    }

    uint normalIndex = threadIndex / lightVoxelTotalCount;
    uint lightVoxelIndex = threadIndex - normalIndex * lightVoxelTotalCount;

    // decode the light voxel index into cell coordinates (the index is the hierarchical one)
    uint rest = lightVoxelIndex;
    uint x = 0, y = 0, z = 0;
    {
        // inverse of getVoxelIndex: the forward loop adds offset.z*4*total + offset.y*2*total + offset.x*total
        uint count = lightVoxelCount;
        uint total = lightVoxelTotalCount;
        while (total > 1) {
            count /= 2;
            total /= 8;
            uint oz = rest / (total * 4);
            rest -= oz * total * 4;
            z = z * 2 + oz;
            uint oy = rest / (total * 2);
            rest -= oy * total * 2;
            y = y * 2 + oy;
            uint ox = rest / total;
            rest -= ox * total;
            x = x * 2 + ox;
        }
    }

    float3 lightStart = pushConst.VGBStartPos_Size.xyz;
    float3 lightSize = pushConst.voxelSize.xyz;
    float3 cellMin = lightStart + float3(x, y, z) * lightSize;
    float3 cellMax = cellMin + lightSize;
    float3 geoSize = lightSize * (1.0f / float(pushConst.geoVoxelCount / max(lightVoxelCount, 1u)));

    float3 irradiance = float3(0.0f);
    float irradianceWeight = 0.0f;
    float3 normalSum = float3(0.0f);
    float normalWeight = 0.0f;

    uint geoStep = pushConst.geoVoxelCount / max(lightVoxelCount, 1u);
    for (uint gz = 0; gz < geoStep; ++gz) {
        for (uint gy = 0; gy < geoStep; ++gy) {
            for (uint gx = 0; gx < geoStep; ++gx) {
                float3 p = cellMin + (float3(gx, gy, gz) + 0.5f) * geoSize;
                int geoIndex = getVoxelIndex(p, lightStart, geoSize, int(pushConst.geoVoxelCount));
                VoxelData_VRPG voxelData = VGBs[normalIndex][geoIndex];
                if (voxelData.sumNormal_G.w <= 0.0f) continue;

                AABB nodeAabb;
                nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
                nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
                nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
                nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
                nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
                nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

                // clip the geometry AABB to this light cell (VRPG does the same)
                AABB cellAabb;
                cellAabb.minimum = max(nodeAabb.minimum, cellMin);
                cellAabb.maximum = min(nodeAabb.maximum, cellMax);
                if (any(cellAabb.minimum > cellAabb.maximum + 1e-6f)) continue;

                aabb.minimum = min(aabb.minimum, cellAabb.minimum);
                aabb.maximum = max(aabb.maximum, cellAabb.maximum);
                irradiance += voxelData.irradiance.xyz;
                irradianceWeight += voxelData.irradiance.w;
                normalSum += voxelData.sumNormal_G.xyz;
                normalWeight += voxelData.sumNormal_G.w;
            }
        }
    }

    bool hasData = normalWeight > 0.0f && irradianceWeight > 0.0f;
    if (hasData) {
        lightNodeData.aabb = aabb;
        lightNodeData.meanNormal.xyz = normalize(normalSum);
        lightNodeData.meanNormal.w = 1.0f;

        float3 lengthAxis = aabb.maximum - aabb.minimum;
        int3 axisSign = sign(lengthAxis);
        lengthAxis += float3(int3(1) - axisSign) * 0.0001f; // if x == 0.0, then x = 1.0 else x = x
        if (all(axisSign == 0)) lengthAxis = float3(0.0f);
        float aabbVolume = lengthAxis.x * lengthAxis.y + lengthAxis.x * lengthAxis.z + lengthAxis.y * lengthAxis.z;

        lightNodeData.radiance = aabbVolume * irradiance / irradianceWeight;
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
        if (blockHasData) LightNodeData_Octree[pushConst.geoOctreeMaxLayer][threadIndex] = lightNodeData;
        if (blockLane == 0) LightNodeBlockInfoBuffer[blockIndex].hasData = (uint)blockHasData;
    }
}

