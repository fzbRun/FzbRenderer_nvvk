#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eVGB)]] StructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_G)]] RWStructuredBuffer<OctreeNodeClusterData_G, ScalarDataLayout> OctreeClusterDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeData_G)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> OctreeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_E)]] RWStructuredBuffer<OctreeNodeClusterData_E, ScalarDataLayout> OctreeClusterDataBuffer_E[];
[[vk::binding(BindingPoints_Octree_FzbPG::eClusterLayerData_E)]] RWStructuredBuffer<OctreeNodeData_E, ScalarDataLayout> ClusterlayerDataBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_G)]] RWStructuredBuffer<uint> BlockInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eBlockInfos_E)]] RWStructuredBuffer<uint> BlockInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_G)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockIndices_E)]] RWStructuredBuffer<uint> HasDataBlockIndicsBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eHasDataBlockCount)]] RWStructuredBuffer<HasDataOctreeBlockCount> HasDataBlockCountBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eDivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> DivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eThreadGroupInfos)]] RWStructuredBuffer<OctreeThreadGroupInfo> OctreeThreadGroupInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> IndivisibleNodeInfoBuffer_E;

//----------------------------------------------clearOctreeArray--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initOctreeArray(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex >= pushConst.octreeNodeTotalCount) return;

    {
        OctreeNodeData_E nodeData_E;
        nodeData_E.pdf = 1.0f;
        nodeData_E.label_indivisible = 0;
        if (threadIndex < NODECOUNT_E) ClusterlayerDataBuffer_E[threadIndex] = nodeData_E;
    }

    AABB aabb;{
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;}
    OctreeNodeClusterData_G nodeClusterData_G; {
        nodeClusterData_G.indivisible = 1;
        nodeClusterData_G.aabb = aabb;
        nodeClusterData_G.fillRate = 1.0f;
        nodeClusterData_G.meanNormal = float4(0.0f);}
    OctreeNodeData_G nodeData_G; nodeData_G.label_indivisible = 0;
    OctreeNodeClusterData_E nodeClusterData_E; {
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
            //-----------------------------------------------------------
            uint warpHasDataMask_G = WaveActiveBallot(hasData_G).x;
            uint blockHasDataNodeCount_G = countbits(warpHasDataMask_G & blockNodeMask);
            bool blockHasData_G = blockHasDataNodeCount_G > 0;
            if (blockHasData_G) OctreeClusterDataBuffer_G[pushConst.octreeMaxLayer][threadIndex] = nodeClusterData_G;
            if (blockLane == 0) BlockInfoBuffer_G[blockIndex] = (uint)blockHasData_G;

            uint warpHasDataMask_E = WaveActiveBallot(hasData_E).x;
            uint blockHasDataNodeCount_E = countbits(warpHasDataMask_E & blockNodeMask);
            bool blockHasData_E = blockHasDataNodeCount_E > 0;
            if (blockHasData_E) OctreeClusterDataBuffer_E[pushConst.octreeMaxLayer][threadIndex] = nodeClusterData_E;
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
