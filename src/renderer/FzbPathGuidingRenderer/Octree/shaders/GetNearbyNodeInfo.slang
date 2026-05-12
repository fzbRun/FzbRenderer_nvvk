#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_FzbPG> pushConst;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeClusterData_G)]] RWStructuredBuffer<OctreeNodeClusterData_G_FzbPG, ScalarDataLayout> OctreeClusterDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;
[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> IndivisibleNodeInfoBuffer_E;
[[vk::binding(BindingPoints_Octree_FzbPG::eNearbyNodeTempInfos)]] RWStructuredBuffer<IndivisibleNodeNearbyNodeTempInfo_FzbPG, ScalarDataLayout> NearbyNodeTempInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eNearbyNodeInfos)]] RWStructuredBuffer<OctreeNearbyNodeInfo_FzbPG, ScalarDataLayout> NearbyNodeInfoBuffer;
//------------------------------------------------getNearbyNode---------------------------------------------------------
struct IndivisibleNodeData_G {
    AABB aabb;
    uint normalIndex;
#ifdef GEOMETRY_CLUSTER_WITH_E
    float E;
#endif
};

groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared IndivisibleNodeData_G groupIndivisibleNodeData;

#ifdef COMMON_NEARBYNODE_JITTER
groupshared float groupNearbyNodeDistances[NEARBY_NODE_COUNT_FZBPG];
groupshared int2 groupNearbyNodeInfos[NEARBY_NODE_COUNT_FZBPG];
groupshared float groupWarpMinDistances[GETNEARBYNODES_CS_THREADGROUP_SIZE / 32];
groupshared uint2 groupWarpMinDistanceNodeInfos[GETNEARBYNODES_CS_THREADGROUP_SIZE / 32];
groupshared uint groupGroupMinDistanceCount;
groupshared float groupGroupMinDistance;


[numthreads(GETNEARBYNODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    if (groupThreadIndex < NEARBY_NODE_COUNT_FZBPG) {
        groupNearbyNodeDistances[groupThreadIndex] = 1e30f;
        groupNearbyNodeInfos[groupThreadIndex] = int2(-1);
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint indivisibleNodeLabel = threadGroupIndex / threadGroupCount_perIndivisibleNode;
    if (groupThreadIndex == 0) {
        uint2 nodeInfos = IndivisibleNodeInfoBuffer_G[indivisibleNodeLabel];
        OctreeNodeClusterData_G_FzbPG nodeData = OctreeClusterDataBuffer_G[nodeInfos.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfos.y];
        IndivisibleNodeData_G nodeInfo;

        // nodeInfo.normal = normalize(nodeData.meanNormal.xyz);
        uint octreeLayerNode = 1 << (3 * nodeInfos.x);
        nodeInfo.normalIndex = nodeInfos.y / octreeLayerNode;

        nodeInfo.aabb = nodeData.aabb;
        #ifdef GEOMETRY_CLUSTER_WITH_E
        nodeInfo.E = nodeData.E;
        #endif
        groupIndivisibleNodeData = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();
    uint threadGroupOffset = threadGroupIndex % threadGroupCount_perIndivisibleNode;
    uint nearbyNodeLabel = threadGroupOffset * GETNEARBYNODES_CS_THREADGROUP_SIZE + groupThreadIndex;
    if (nearbyNodeLabel >= indivisibleNodeCount) return;

    IndivisibleNodeData_G indivisibleNodeData = groupIndivisibleNodeData;
    uint2 nearbyNodeInfo = IndivisibleNodeInfoBuffer_G[nearbyNodeLabel];
    OctreeNodeClusterData_G_FzbPG nearbyNodeData = OctreeClusterDataBuffer_G[nearbyNodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nearbyNodeInfo.y];
    uint octreeLayerNode = 1 << (3 * nearbyNodeInfo.x);
    uint nearbyNormalIndex = nearbyNodeInfo.y / octreeLayerNode;

    float distance = AABBDistance2(indivisibleNodeData.aabb, nearbyNodeData.aabb);
    // float cosine = dot(indivisibleNodeData.normal, normalize(nearbyNodeData.meanNormal.xyz));
    // distance *= cosine < 0.707f ? 100000.0f : (1.1f - cosine);
    distance += nearbyNormalIndex == indivisibleNodeData.normalIndex ? 0.0f : 1e10f;
    #ifdef GEOMETRY_CLUSTER_WITH_E
    distance *= (1.1f - (min(indivisibleNodeData.E, nearbyNodeData.E) / max(indivisibleNodeData.E, nearbyNodeData.E)));
    #endif

    if (indivisibleNodeLabel == nearbyNodeLabel) distance = 1e30f;

    uint remainingThreadCount = indivisibleNodeCount - (threadGroupCount_perIndivisibleNode - 1) * GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint warpCount = (threadGroupIndex % threadGroupCount_perIndivisibleNode) == threadGroupCount_perIndivisibleNode - 1 ? (remainingThreadCount + 31) / 32 : GETNEARBYNODES_CS_THREADGROUP_SIZE / 32;
    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    for (int i = 0; i < NEARBY_NODE_COUNT_FZBPG; i += groupGroupMinDistanceCount) {
        float warpMinDistance = WaveActiveMin(distance);
        if (warpLane == 0) groupWarpMinDistances[warpIndex] = warpMinDistance;
        uint minDistanceLabel = WavePrefixCountBits(distance == warpMinDistance);
        if (distance == warpMinDistance && minDistanceLabel == 0)
            groupWarpMinDistanceNodeInfos[warpIndex] = nearbyNodeInfo;
        GroupMemoryBarrierWithGroupSync();

        if (groupThreadIndex == 0) {
            float groupMinDistance = groupWarpMinDistances[0]; uint minDistanceWarpIndex = 0;
            for (int j = 1; j < warpCount; ++j) {
                float other_warpMinDistance = groupWarpMinDistances[j];
                if (other_warpMinDistance < groupMinDistance) {
                    groupMinDistance = other_warpMinDistance;
                    minDistanceWarpIndex = j;
                }
            }
            groupGroupMinDistance = groupMinDistance;

            uint groupMinDistanceCount = 0;
            for (int j = 0; j < warpCount; ++j) {
                float other_warpMinDistance = groupWarpMinDistances[j];
                if (other_warpMinDistance == groupMinDistance) {
                    if (i + groupMinDistanceCount >= NEARBY_NODE_COUNT_FZBPG) break;
                    groupNearbyNodeInfos[i + groupMinDistanceCount] = groupWarpMinDistanceNodeInfos[j];
                    groupNearbyNodeDistances[i + groupMinDistanceCount] = groupMinDistance;
                    ++groupMinDistanceCount;
                }
            }
            groupGroupMinDistanceCount = groupMinDistanceCount;
        }
        GroupMemoryBarrierWithGroupSync();

        if (distance == groupGroupMinDistance && minDistanceLabel == 0) distance = 1e30f;
    }

    if (groupThreadIndex == 0) {
        if (threadGroupCount_perIndivisibleNode == 1) {
            NearbyNodeInfoBuffer[indivisibleNodeLabel].nearbyNodeInfos = groupNearbyNodeInfos;
        } else {
            uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + threadGroupOffset;
            NearbyNodeTempInfoBuffer[infoIndex].nodeLabel = indivisibleNodeLabel;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos = groupNearbyNodeInfos;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances = groupNearbyNodeDistances;
        }
        if (threadIndex == 0) {
            DispatchIndirectCommand cmd2;
            if (threadGroupCount_perIndivisibleNode == 1) cmd2.x = 0;
            else cmd2.x = (indivisibleNodeCount * 32 + GETNEARBYNODES2_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES2_CS_THREADGROUP_SIZE;
            cmd2.y = 1; cmd2.z = 1;

            if (threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_FZBPG > 32) {
                printf("Error: Too many threads per indivisible node for getNearbyNodes. currentThreadCount: %u, maxThreadCount: %u\n",
                       threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_FZBPG, 32);
                cmd2.x = 0; cmd2.y = 0; cmd2.z = 0;
            }

            GlobalInfoBuffer[0].cmd2 = cmd2;
        }
    }
}

[numthreads(GETNEARBYNODES2_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint indivisibleNodeLabel = threadIndex / 32;
    if (indivisibleNodeLabel >= indivisibleNodeCount) return;

    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;

    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    float distance = 1e30f; int2 nearbyNodeIndex = -1; uint nodeLabel = 0;
    if (warpLane < threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_FZBPG) {
        uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + warpLane / NEARBY_NODE_COUNT_FZBPG;
        distance = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances[warpLane % NEARBY_NODE_COUNT_FZBPG];
        nearbyNodeIndex = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos[warpLane % NEARBY_NODE_COUNT_FZBPG];

        nodeLabel = NearbyNodeTempInfoBuffer[indivisibleNodeLabel * threadGroupCount_perIndivisibleNode].nodeLabel;
    }

    for (int i = 0; i < NEARBY_NODE_COUNT_FZBPG; ++i) {
        float minDistance = WaveActiveMin(distance);

        uint minDistanceLabel = WavePrefixCountBits(distance == minDistance);
        if (distance == minDistance && minDistanceLabel == 0) {
            NearbyNodeInfoBuffer[nodeLabel].nearbyNodeInfos[i] = nearbyNodeIndex;
            distance = 1e30f;
        }
    }
}
#else

groupshared float groupDirWarpMinDistance[4][GETNEARBYNODES_CS_THREADGROUP_SIZE / 32];
groupshared float groupDirMinDistances[4];
groupshared int2 groupDirMinDistanceNodeInfos[4];

[numthreads(GETNEARBYNODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    if (groupThreadIndex < 4) {
        groupDirMinDistances[groupThreadIndex] = 1e30f;
        groupDirMinDistanceNodeInfos[groupThreadIndex] = int2(-1);
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint indivisibleNodeLabel = threadGroupIndex / threadGroupCount_perIndivisibleNode;
    if (groupThreadIndex == 0) {
        uint2 nodeInfos = IndivisibleNodeInfoBuffer_G[indivisibleNodeLabel];
        OctreeNodeClusterData_G_FzbPG nodeData = OctreeClusterDataBuffer_G[nodeInfos.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfos.y];
        IndivisibleNodeData_G nodeInfo;

        // nodeInfo.normal = normalize(nodeData.meanNormal.xyz);
        uint octreeLayerNode = 1 << (3 * nodeInfos.x);
        nodeInfo.normalIndex = nodeInfos.y / octreeLayerNode;

        nodeInfo.aabb = nodeData.aabb;
        groupIndivisibleNodeData = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();
    uint threadGroupOffset = threadGroupIndex % threadGroupCount_perIndivisibleNode;
    uint nearbyNodeLabel = threadGroupOffset * GETNEARBYNODES_CS_THREADGROUP_SIZE + groupThreadIndex;
    if (nearbyNodeLabel >= indivisibleNodeCount) return;

    uint remainingThreadCount = indivisibleNodeCount - (threadGroupCount_perIndivisibleNode - 1) * GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint warpCount = (threadGroupIndex % threadGroupCount_perIndivisibleNode) == threadGroupCount_perIndivisibleNode - 1 ? (remainingThreadCount + 31) / 32 : GETNEARBYNODES_CS_THREADGROUP_SIZE / 32;
    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    IndivisibleNodeData_G indivisibleNodeData = groupIndivisibleNodeData;
    uint2 nearbyNodeInfo = IndivisibleNodeInfoBuffer_G[nearbyNodeLabel];
    OctreeNodeClusterData_G_FzbPG nearbyNodeData = OctreeClusterDataBuffer_G[nearbyNodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nearbyNodeInfo.y];
    uint octreeLayerNode = 1 << (3 * nearbyNodeInfo.x);
    uint nearbyNormalIndex = nearbyNodeInfo.y / octreeLayerNode;

    float3 indivisibleNodeCenter = (indivisibleNodeData.aabb.minimum + indivisibleNodeData.aabb.maximum) * 0.5f;
    float3 nearbyNodeCenter = (nearbyNodeData.aabb.minimum + nearbyNodeData.aabb.maximum) * 0.5f;
    float3 centerDir = normalize(nearbyNodeCenter - indivisibleNodeCenter);
    float3 absDir = abs(centerDir);
    int dirIndex = absDir.x > absDir.y ? (absDir.x > absDir.z ? 0 : 2) : (absDir.y > absDir.z ? 1 : 2);
    dirIndex = dirIndex * 2 + (sign(centerDir[dirIndex]) + 1) / 2;

    bool isNearby = dirIndex / 2 != indivisibleNodeData.normalIndex / 2;
    if (dirIndex > indivisibleNodeData.normalIndex) dirIndex -= 2;

    float distance = AABBDistance2(indivisibleNodeData.aabb, nearbyNodeData.aabb);

    float3 length = nearbyNodeData.aabb.maximum - nearbyNodeData.aabb.minimum;
    length[indivisibleNodeData.normalIndex / 2] = 1.0f;
    float surfaceArea = length.x * length.y * length.z;
    distance += 1.0f - min(surfaceArea, 1.0f);

    distance += nearbyNormalIndex == indivisibleNodeData.normalIndex ? 0.0f : 1e32f;
    if (indivisibleNodeLabel == nearbyNodeLabel) distance = 1e30f;

    for (int i = 0; i < 4; i++) {
        float dirDistance = isNearby && i == dirIndex ? distance : 1e30f;
        float warpMinDistance = WaveActiveMin(dirDistance);
        if (warpLane == 0) groupDirWarpMinDistance[i][warpIndex] = warpMinDistance;
        uint warpMinDistanceLabel = WavePrefixCountBits(dirDistance == warpMinDistance);
        GroupMemoryBarrierWithGroupSync();

        warpMinDistance = 1e30f;
        if (warpLane < warpCount) warpMinDistance = groupDirWarpMinDistance[i][warpLane];
        float groupDirMinDistance = WaveActiveMin(warpMinDistance);
        uint groupMinDistanceLabel = WavePrefixCountBits(warpMinDistance == groupDirMinDistance);
        groupMinDistanceLabel = WaveReadLaneAt(groupMinDistanceLabel, warpIndex);

        if (dirDistance == groupDirMinDistance && warpMinDistanceLabel == 0 && groupMinDistanceLabel == 0)
            groupDirMinDistanceNodeInfos[i] = nearbyNodeInfo;

        if (groupThreadIndex == 0 && threadGroupCount_perIndivisibleNode > 1) groupDirMinDistances[i] = groupDirMinDistance;
    }

    if (groupThreadIndex == 0) {
        if (threadGroupCount_perIndivisibleNode == 1) {
            NearbyNodeInfoBuffer[indivisibleNodeLabel].nearbyNodeInfos = groupDirMinDistanceNodeInfos;
        } else {
            uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + threadGroupOffset;
            NearbyNodeTempInfoBuffer[infoIndex].nodeLabel = indivisibleNodeLabel;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos = groupDirMinDistanceNodeInfos;
            NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances = groupDirMinDistances;
        }
        if (threadIndex == 0) {
            DispatchIndirectCommand cmd2;
            if (threadGroupCount_perIndivisibleNode == 1) cmd2.x = 0;
            else cmd2.x = (indivisibleNodeCount * 32 + GETNEARBYNODES2_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES2_CS_THREADGROUP_SIZE;
            cmd2.y = 1; cmd2.z = 1;

            if (threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_FZBPG > 32) {
                printf("Error: Too many threads per indivisible node for getNearbyNodes. currentThreadCount: %u,  maxThreadCount: %u\n",
                       threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT_FZBPG, 32);
                cmd2.x = 0; cmd2.y = 0; cmd2.z = 0;
            }

            GlobalInfoBuffer[0].cmd2 = cmd2;
        }
    }
}

[numthreads(GETNEARBYNODES2_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint indivisibleNodeLabel = threadIndex / 32;
    if (indivisibleNodeLabel >= indivisibleNodeCount) return;

    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;

    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    float distance = 1e30f; int2 nearbyNodeIndex = -1; uint nodeLabel = 0;
    if (warpLane < threadGroupCount_perIndivisibleNode * 4) {
        uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + warpLane / 4;
        distance = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeDistances[warpLane % 4];
        nearbyNodeIndex = NearbyNodeTempInfoBuffer[infoIndex].nearbyNodeInfos[warpLane % 4];

        nodeLabel = NearbyNodeTempInfoBuffer[indivisibleNodeLabel * threadGroupCount_perIndivisibleNode].nodeLabel;
    }

    int dirIndex = warpLane & 3;
    for (int i = 0; i < 4; ++i) {
        float dirDistance = dirIndex == i ? distance : 1e30f;
        float minDistance = WaveActiveMin(dirDistance);

        uint minDistanceLabel = WavePrefixCountBits(dirDistance == minDistance);
        if (dirDistance == minDistance && minDistanceLabel == 0)
            NearbyNodeInfoBuffer[nodeLabel].nearbyNodeInfos[i] = nearbyNodeIndex;
    }
}
#endif
//------------------------------------------------debug---------------------------------------------------------
#ifndef NDEBUG
struct VSout_nearby {
    float4 sv_position : SV_Position;
    float4 color;
};
struct PSout_nearby
{
    float4 color : SV_Target;
};
[shader("vertex")]
VSout_nearby vertexMain_nearby(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VSout_nearby output;
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    OctreeGlobalInfo_FzbPG globalInfo = GlobalInfoBuffer[0];
    if (pushConst.sampleNodeLabel_G - 1 >= globalInfo.indivisibleNodeCount_G) return output;

    int2 nearbyNodeInfos[NEARBY_NODE_COUNT_FZBPG] = NearbyNodeInfoBuffer[pushConst.sampleNodeLabel_G - 1].nearbyNodeInfos;

    OctreeNodeClusterData_G_FzbPG nodeData;
    bool show = instanceIndex == pushConst.sampleNodeLabel_G - 1;
    if (show) {
        uint2 sampleNodeInfo = IndivisibleNodeInfoBuffer_G[instanceIndex];
        nodeData = OctreeClusterDataBuffer_G[sampleNodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][sampleNodeInfo.y];

        //if (vertexIndex == 0 && pushConst.frameIndex == 1) {
        //    printf("sampleNode layerIndex: %d, sampleNodeIndex: %d\naabb: %f %f %f %f %f %f\n\n", 
        //        sampleNodeInfo.x, sampleNodeInfo.y,
        //        nodeData.aabb.minimum.x, nodeData.aabb.minimum.y, nodeData.aabb.minimum.z,
        //        nodeData.aabb.maximum.x, nodeData.aabb.maximum.y, nodeData.aabb.maximum.z,
        //    );
        //    for (int i = 0; i < NEARBY_NODE_COUNT_FZBPG; ++i) {
        //        OctreeNodeClusterData_G_FzbPG nearbyNodeData = OctreeClusterDataBuffer_G[nearbyNodeInfos[i].x - OCTREE_CLUSTER_LAYER_FZBPG][nearbyNodeInfos[i].y];
        //
        //        printf("nearbyNode layerIndex: %d, nodeIndex: %d\naabb: %f %f %f %f %f %f\n\n",
        //            nearbyNodeInfos[i].x, nearbyNodeInfos[i].y,
        //            nearbyNodeData.aabb.minimum.x, nearbyNodeData.aabb.minimum.y, nearbyNodeData.aabb.minimum.z,
        //            nearbyNodeData.aabb.maximum.x, nearbyNodeData.aabb.maximum.y, nearbyNodeData.aabb.maximum.z,
        //        );
        //    }  
        //}

        output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    }
    else output.color = float4(0.0f, 1.0f, 0.0f, 1.0f);

    uint2 nodeInfos = IndivisibleNodeInfoBuffer_G[instanceIndex];
    for (int i = 0; i < NEARBY_NODE_COUNT_FZBPG; ++i) {
        if (all(nodeInfos == nearbyNodeInfos[i])) {
            show = true;
            nodeData = OctreeClusterDataBuffer_G[nodeInfos.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfos.y];
            break;
        }
    }
    if (show) {
        AABB nodeAABB = nodeData.aabb;
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
PSout_nearby fragmentMain_nearby(VSout_nearby input) {
    PSout_nearby output;
    output.color = input.color;
    return output;
}
#endif