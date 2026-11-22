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
//----------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_init(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.lightOctreeNodeTotalCount) return;

    AABB aabb; {
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    LightNodeData_Octree_VRPG lightNodeData; 
    if (threadIndex < pushConst.lightLeafNodeTotalCount) {
        lightNodeData = LightNodeData_Octree[pushConst.lightOctreeMaxLayer][threadIndex];
        float3 radiance = lightNodeData.radiance;
        bool hasData = any(radiance > 0.0f);
        if (!hasData) {
            lightNodeData.radiance = 0.0f;
            lightNodeData.halfAngle = M_PI;
            lightNodeData.meanNormal = float4(0.0f);
            lightNodeData.aabb = aabb;

            LightNodeData_Octree[pushConst.lightOctreeMaxLayer][threadIndex] = lightNodeData;
        }

        uint warpLane = groupThreadIndex & 31;
        uint blockIndex = threadIndex / 8;
        uint blockLane = threadIndex & 7;
        uint blockFirstWarpLane = warpLane / 8 * 8;
        uint blockNodeMask = 0xff << blockFirstWarpLane;

        uint warpHasDataMask = WaveActiveBallot(hasData).x;
        uint blockHasDataNodeCount = countbits(warpHasDataMask & blockNodeMask);
        bool blockHasData = blockHasDataNodeCount > 0;
        if (blockLane == 0) LightNodeBlockInfoBuffer[blockIndex].hasData = (uint)blockHasData;
    } else {
        lightNodeData.radiance = 0.0f;
        lightNodeData.halfAngle = M_PI;
        lightNodeData.meanNormal = float4(0.0f);
        lightNodeData.aabb = aabb;

        uint nodeIndex = threadIndex - pushConst.lightLeafNodeTotalCount;
        uint layerIndex = 0, layerNodeCount = 6;
        for (; layerIndex < pushConst.lightOctreeMaxLayer; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        LightNodeData_Octree[layerIndex][nodeIndex] = lightNodeData;
    }
}

