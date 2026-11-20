#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<GeometryClusteringPushConstant_VRPG> pushConst;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusteringGlobalInfo)]] RWStructuredBuffer<ClusteringGlobalInfo_VRPG> ClusteringGlobalInfoBuffer;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_SVO)]] RWStructuredBuffer<ClusterInfo_SVO_VRPG, ScalarDataLayout> ClusterInfo_SVO;

[[vk::binding(GeometryClusteringBindingPoints_VRPG::eNearbyClusterTempInfos)]] RWStructuredBuffer<NearbyClusterTempInfo_VRPG, ScalarDataLayout> NearbyClusterTempInfo;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eNearbyClusterInfos)]] RWStructuredBuffer<NearbyClusterInfo_VRPG, ScalarDataLayout> NearbyClusterInfo;
//---------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_dispatchGetNearbyCluster() {
    GlobalInfoBuffer[0].cmd.x = GlobalInfoBuffer[0].ClusterCount * (GlobalInfoBuffer[0].ClusterCount + GetNearbyCluster_CS_THREADGROUP_SIZE - 1) / GetNearbyCluster_CS_THREADGROUP_SIZE;
}

struct ClusterData {
    AABB aabb;
    uint normalIndex;
    uint layer;
    float extent;
    float3 normal;
};

groupshared uint groupClusterCount;
groupshared ClusterData groupClusterData;

groupshared float groupDirWarpMinDistance[VRPG_NEARBY_CLUSTER_COUNT][GetNearbyCluster_CS_THREADGROUP_SIZE / 32];
groupshared float groupDirMinDistances[VRPG_NEARBY_CLUSTER_COUNT];
groupshared OctreeNodeInfo_VRPG2 groupDirMinDistanceClusterInfos[VRPG_NEARBY_CLUSTER_COUNT];

[numthreads(GetNearbyCluster_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyClusters1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
    if (groupThreadIndex < VRPG_NEARBY_CLUSTER_COUNT) {
        groupDirMinDistances[groupThreadIndex] = 1e30f;
        groupDirMinDistanceClusterInfos[groupThreadIndex] = { -1, -1 };
    }
    GroupMemoryBarrierWithGroupSync();
    int clusterCount = groupClusterCount;
    uint threadGroupCount_perCluster = (clusterCount + GetNearbyCluster_CS_THREADGROUP_SIZE - 1) / GetNearbyCluster_CS_THREADGROUP_SIZE;
    uint clusterLabel = threadGroupIndex / threadGroupCount_perCluster;
    if (groupThreadIndex == 0) {
        OctreeNodeInfo_VRPG2 nodeInfo = ClusterInfo_SVO[clusterLabel];
        ClusterData_Octree_VRPG nodeData_origin = ClusterData_Octree[nodeInfo.layerIndex][nodeInfo.nodeIndex];
        ClusterData nodeData;

        // nodeInfo.normal = normalize(nodeData.meanNormal.xyz);
        uint octreeLayerNode = 1 << (3 * nodeInfo.layerIndex);
        nodeData.normalIndex = nodeInfo.nodeIndex / octreeLayerNode;
        nodeData.layer = nodeInfo.layerIndex;
        float3 selectedExtent = nodeData_origin.aabb.maximum - nodeData_origin.aabb.minimum;
        nodeData.extent = max(selectedExtent.x, max(selectedExtent.y, selectedExtent.z));
        nodeData.normal = normalize(nodeData_origin.meanNormal.xyz + float3(1e-8f));

        nodeData.aabb = nodeData_origin.aabb;
        groupClusterData = nodeData;
    }
    GroupMemoryBarrierWithGroupSync();
    uint threadGroupOffset = threadGroupIndex % threadGroupCount_perCluster;
    uint nearbyNodeLabel = threadGroupOffset * GetNearbyCluster_CS_THREADGROUP_SIZE + groupThreadIndex;
    if (nearbyNodeLabel >= clusterCount) return;

    uint remainingThreadCount = clusterCount - (threadGroupCount_perCluster - 1) * GetNearbyCluster_CS_THREADGROUP_SIZE;
    uint warpCount = (threadGroupIndex % threadGroupCount_perCluster) == threadGroupCount_perCluster - 1 ? (remainingThreadCount + 31) / 32 : GetNearbyCluster_CS_THREADGROUP_SIZE / 32;
    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    ClusterData clusterData = groupClusterData;
    OctreeNodeInfo_VRPG2 nearbyClusterInfo = ClusterInfo_SVO[nearbyNodeLabel];
    ClusterData_Octree_VRPG nearbyClusterData = ClusterData_Octree[nearbyClusterInfo.layerIndex][nearbyClusterInfo.nodeIndex];
    uint octreeLayerNode = 1 << (3 * nearbyClusterInfo.layerIndex);
    uint nearbyNormalIndex = nearbyClusterInfo.nodeIndex / octreeLayerNode;

    float3 clusterCenter = (clusterData.aabb.minimum + clusterData.aabb.maximum) * 0.5f;
    float3 nearbyClusterCenter = (nearbyClusterData.aabb.minimum + nearbyClusterData.aabb.maximum) * 0.5f;

    // Use the tangent plane of the selected G normal. The two axes are the
    // world axes other than the normal axis, and each gets +/- slots.
    int normalAxis = int(clusterData.normalIndex) / 2;
    int axisA = (normalAxis + 1) % 3;
    int axisB = (normalAxis + 2) % 3;

    float3 delta = nearbyClusterCenter - clusterCenter;
    float projA = delta[axisA];
    float projB = delta[axisB];

    int dirIndex = 0;
    int extentAxis = axisA;
    if (abs(projA) >= abs(projB)) {
        dirIndex = projA >= 0.0f ? 0 : 1;
        extentAxis = axisA;
    } else {
        dirIndex = projB >= 0.0f ? 2 : 3;
        extentAxis = axisB;
    }

    // Requirement 2: only clusters with exactly the same normal partition.
    // Requirement 3: boundary AABB distance is the primary key.
    float boundaryDistance2 = AABBDistance2(clusterData.aabb, nearbyClusterData.aabb);
    float3 nearbyExtent3 = nearbyClusterData.aabb.maximum - nearbyClusterData.aabb.minimum;
    float nearbyExtent = max(nearbyExtent3.x, max(nearbyExtent3.y, nearbyExtent3.z));
    float minExtent = max(min(clusterData.extent, nearbyExtent), 1e-6f);
    float distance = boundaryDistance2 / (minExtent * minExtent);

    // A side neighbor must overlap the selected cluster on the other tangent
    // axis. Otherwise a cluster that is entirely below/above can be chosen as
    // the "left/right" neighbor just because its boundary is close.
    int otherAxis = extentAxis == axisA ? axisB : axisA;
    float overlap = min(clusterData.aabb.maximum[otherAxis],
                        nearbyClusterData.aabb.maximum[otherAxis]) -
                    max(clusterData.aabb.minimum[otherAxis],
                        nearbyClusterData.aabb.minimum[otherAxis]);
    if (overlap < 0.01f * minExtent)
        distance = 1e30f;

    // Direction alignment from the center delta is more stable than the
    // boundary vector when the AABBs overlap/touch: the boundary vector can be
    // zero there and would make every candidate look equally aligned.
    float3 directionVector = nearbyClusterCenter - clusterCenter;
    float directionLength = length(directionVector);
    float alignment = 0.0f;
    if (directionLength > 1e-6f)
        alignment = abs(directionVector[extentAxis]) / directionLength;
    distance += (1.0f - clamp(alignment, 0.0f, 1.0f)) * 1e-6f;

    // Requirement 4: when both distance and direction are equal, prefer the
    // cluster that is longer along the selected cardinal direction.
    float extentAlong = max(nearbyClusterData.aabb.maximum[extentAxis] -
                            nearbyClusterData.aabb.minimum[extentAxis], 1e-6f);
    float tieBreak = (1.0f - clamp(extentAlong / max(clusterData.extent, 1e-6f), 0.0f, 1.0f)) * 1e-9f;
    distance += tieBreak;

    #if VRPG_COSINE_NEIGHBOR
    // Cosine + coherence candidate gating instead of the hard "same normal partition" test: curved
    // (circle / sphere) and flat surfaces keep jittering across partition boundaries, while creases
    // are still refused because the average normal length drops there.
    // clusterData.normal is the normalised mean normal of the current cluster; its own
    // coherence is a per node constant and therefore does not change the ranking of the candidates.
    const float cosineCoherenceOther = length(nearbyClusterData.meanNormal.xyz);
    float cosineSim = -1.0f;
    if (cosineCoherenceOther > 1e-6f)
        cosineSim = dot(clusterData.normal, nearbyClusterData.meanNormal.xyz) / cosineCoherenceOther;
    const float cosineInRange = max(0.0f, cosineSim);
    const float cosineScore = pow(cosineInRange, VRPG_COSINE_POWER) * clamp(cosineCoherenceOther, 0.0f, 1.0f);
    if (cosineInRange < VRPG_COSINE_MIN_DOT || cosineScore <= 1e-6f)
        distance = 1e30f;
    else
        distance /= cosineScore; // the selector below minimises distance, so better score = smaller
    #else
    if (nearbyNormalIndex != clusterData.normalIndex)
        distance = 1e30f;
    #endif
    if (clusterLabel == nearbyNodeLabel) distance = 1e30f;

    for (int i = 0; i < VRPG_NEARBY_CLUSTER_COUNT; i++) {
        float dirDistance = i == dirIndex ? distance : 1e30f;
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
            groupDirMinDistanceClusterInfos[i] = nearbyClusterInfo;

        if (groupThreadIndex == 0 && threadGroupCount_perCluster > 1) groupDirMinDistances[i] = groupDirMinDistance;
    }

    if (groupThreadIndex == 0) {
        if (threadGroupCount_perCluster == 1) {
            NearbyClusterInfo[clusterLabel].nearbyClusterInfos = groupDirMinDistanceClusterInfos;
        } else {
            uint infoIndex = clusterLabel * threadGroupCount_perCluster + threadGroupOffset;
            NearbyClusterTempInfo[infoIndex].nearbyClusterInfos = groupDirMinDistanceClusterInfos;
            NearbyClusterTempInfo[infoIndex].nearbyClusterDistances = groupDirMinDistances;
        }
        if (threadIndex == 0) {
            DispatchIndirectCommand cmd;
            if (threadGroupCount_perCluster == 1) cmd.x = 0;
            else cmd.x = (clusterCount * 32 + GetNearbyCluster2_CS_THREADGROUP_SIZE - 1) / GetNearbyCluster2_CS_THREADGROUP_SIZE;
            cmd.y = 1; cmd.z = 1;

            if (threadGroupCount_perCluster * NEARBY_NODE_COUNT_VRPG > 32) {
                printf("Error: Too many threads per cluster for getNearbyNodes. currentThreadCount: %u,  maxThreadCount: %u\n",
                       threadGroupCount_perCluster * NEARBY_NODE_COUNT_VRPG, 32);
                cmd.x = 0; cmd.y = 0; cmd.z = 0;
            }

            GlobalInfoBuffer[0].cmd = cmd;
        }
    }
}

[numthreads(GetNearbyCluster2_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyClusters2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterCount = GlobalInfoBuffer[0].ClusterCount;
    GroupMemoryBarrierWithGroupSync();
    int clusterCount = groupClusterCount;
    uint clusterLabel = threadIndex / 32;
    if (clusterLabel >= clusterCount) return;

    uint threadGroupCount_perCluster = (clusterCount + GetNearbyCluster_CS_THREADGROUP_SIZE - 1) / GetNearbyCluster_CS_THREADGROUP_SIZE;

    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    float distance = 1e30f; OctreeNodeInfo_VRPG2 nearbyClusterInfo = { -1, -1 };
    if (warpLane < threadGroupCount_perCluster * VRPG_NEARBY_CLUSTER_COUNT) {
        uint infoIndex = clusterLabel * threadGroupCount_perCluster + warpLane / VRPG_NEARBY_CLUSTER_COUNT;
        distance = NearbyClusterTempInfo[infoIndex].nearbyClusterDistances[warpLane % VRPG_NEARBY_CLUSTER_COUNT];
        nearbyClusterInfo = NearbyClusterTempInfo[infoIndex].nearbyClusterInfos[warpLane % VRPG_NEARBY_CLUSTER_COUNT];
    }

    int dirIndex = warpLane & 3;
    for (int i = 0; i < VRPG_NEARBY_CLUSTER_COUNT; ++i) {
        float dirDistance = dirIndex == i ? distance : 1e30f;
        float minDistance = WaveActiveMin(dirDistance);

        uint minDistanceLabel = WavePrefixCountBits(dirDistance == minDistance);
        if (dirDistance == minDistance && minDistanceLabel == 0)
            NearbyClusterInfo[clusterLabel].nearbyClusterInfos[i] = nearbyClusterInfo;
    }
}