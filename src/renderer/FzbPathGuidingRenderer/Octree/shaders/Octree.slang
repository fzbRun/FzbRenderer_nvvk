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

[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairPartialHitNodeInfo1)]] RWStructuredBuffer<OctreeNodePairPartialHitNodeInfo1_FzbPG, ScalarDataLayout> OctreeNodePairPartialHitNodeInfo1Buffer[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOOctreeNodePairPartialHitNodeInfo2Count)]] RWStructuredBuffer<uint> OctreeNodePairPartialHitNodeInfo2CountBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairPartialHitNodeInfo2)]] RWStructuredBuffer<OctreeNodePairPartialHitNodeInfo2_FzbPG, ScalarDataLayout> OctreeNodePairPartialHitNodeInfo2Buffer[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairE)]] RWStructuredBuffer<float> OctreeNodePairEBuffer[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairData)]] RWStructuredBuffer<OctreeNodePairData_FzbPG, ScalarDataLayout> OctreeNodePairDataBuffer[];

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
        #ifdef ADAPTIVE_IMPORTANCE_SAMPLING
        #else
        nodeClusterData_E.pdf = 1.0f;
        #endif
        nodeClusterData_E.E = 0.0f;
        nodeClusterData_E.meanNormal = float4(0.0f);
        nodeClusterData_E.aabb = aabb;
        nodeClusterData_E.importantSampleNodeIndex = 0;
    }

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
        bool hasData_E = hasData_G && voxelData.irradiance.w > 0.0f;
        if (hasData_E) {
            nodeClusterData_E.E = length(voxelData.irradiance.xyz / voxelData.irradiance.w);
            nodeClusterData_E.meanNormal = nodeClusterData_G.meanNormal; // voxelData.sumNormal_E / voxelData.sumNormal_E.w;
            nodeClusterData_E.aabb = nodeAabb;
            nodeClusterData_E.importantSampleNodeIndex = threadIndex;

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

            AABB mergeAabb = nodeData_E.aabb;
            atomicMergeAABB(mergeAabb);

            if (warpLane == sampleWarpLane) {
                nodeData_E.aabb = mergeAabb;
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
        GlobalInfoBuffer[0].cmd.x =
            (GlobalInfoBuffer[0].indivisibleNodeCount_E * indivisibleNodeCount_G * 32 + 127) / 128;
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
};
struct HitPayload_FzbPG {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
};
struct NodePairHitInfo {
    uint hitType[8];
    HitPayload_FzbPG hitPayloads[8][4];
}

groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared IndivisibleNodeData_E groupNodeData_E;
groupshared uint groupHasDataChildNodeCount_E;
groupshared bool groupIsHasDataChildNode_E[8];
groupshared uint groupChildHasDataNodeIndex_E[8];
groupshared AABB groupHasDataChildNodeAABB_E[8];

groupshared float3 groupSamplePos_E[32];

groupshared IndivisibleNodeData_G groupNodeData_G[4];

groupshared NodePairHitInfo groupNodePairHitInfos[4];

groupshared uint groupNodePairAllHitCount;
groupshared uint groupNodePairAllHitNodeIndex_G[4];

void traceRay(RayDesc ray, inout HitPayload_FzbPG payload)
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

[numthreads(128, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int indivisibleNodeCount_E = groupIndivisibleNodeCount_E;

    uint threadCount_perNodeG = HITTEST_COUNT_PERCHILDNODE_FZBPG * 8;
    uint threadCount_perNodeE = indivisibleNodeCount_G * threadCount_perNodeG;
    uint groupNodeLabel_E = threadGroupIndex * HITTEST_CS_THREADGROUP_SIZE / threadCount_perNodeE;
    if (groupThreadIndex == 0) {
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[threadCount_perNodeE];
        OctreeNodeClusterData_E_FzbPG clusterNodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        IndivisibleNodeData_E nodeData_E;
        nodeData_E.nodeIndex = nodeIndex;
        nodeData_E.E = clusterNodeData.E;
        nodeData_E.normal = normalize(clusterNodeData.meanNormal.xyz);
        nodeData_E.aabb = clusterNodeData.aabb;

        groupNodeData_E = nodeData_E;
    }

    uint currentGroupNodeStartLabel_G = threadGroupIndex * HITTEST_CS_THREADGROUP_SIZE / threadCount_perNodeG;
    uint currentGroupNodeEndLabel_G = min(((threadGroupIndex + 1) * HITTEST_CS_THREADGROUP_SIZE - 1) / threadCount_perNodeG, indivisibleNodeCount_G);
    uint currentGroupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGroupNodeGCount) {
        uint nodeLabel = (groupThreadIndex + currentGroupNodeStartLabel_G) % indivisibleNodeCount_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG clusterNodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G nodeData_G;
        nodeData_G.aabb = clusterNodeData.aabb;

        uint octreeLayerNode = 1 << (3 * nodeInfo.x);
        uint normalIndex = nodeInfo.y / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        nodeData_G.nodeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        nodeData_G.nodeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        nodeData_G.nodeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

        groupNodeData_G[groupThreadIndex] = nodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    int warpIndex = groupThreadIndex / 32;
    int warpLane = groupThreadIndex & 31;

    IndivisibleNodeData_E nodeData_E = groupNodeData_E;
    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));
    if (groupThreadIndex < 32) {
        float3 childVoxelSize = pushConst.VGBVoxelSize.xyz * (1 << (3 * (pushConst.octreeMaxLayer - pushConst.currentLayer - 1)));
        uint3 childVoxelOffset = int3((nodeData_E.aabb.minimum - pushConst.VGBStartPos_Size.xyz) / childVoxelSize);
        float3 childVoxelStartPos = childVoxelOffset * childVoxelSize + pushConst.VGBStartPos_Size.xyz;

        uint childNodeIndex = groupThreadIndex / 4;
        uint sampleIndex = groupThreadIndex & 3;

        AABB childNodeAabb;
        childNodeAabb.minimum = childVoxelStartPos;
        childNodeAabb.maximum = childVoxelStartPos + float3(childNodeIndex & 1u, (childNodeIndex & 2u) >> 1, (childNodeIndex & 4u) >> 2) * childVoxelSize;
        childNodeAabb.minimum = max(childNodeAabb.minimum, nodeData_E.aabb.minimum);
        childNodeAabb.maximum = min(childNodeAabb.maximum, nodeData_E.aabb.maximum);

        bool hasData = all(childNodeAabb.maximum >= childNodeAabb.minimum);
        uint hasDataCount = WaveActiveSum(hasData) / 4;
        uint hasDataLabel = WavePrefixCountBits(hasData) / 4;
        if (sampleIndex == 0) {
            groupIsHasDataChildNode_E[childNodeIndex] = hasData;
            groupHasDataChildNodeCount_E = hasDataCount;
            if (hasData) {
                groupChildHasDataNodeIndex_E[hasDataLabel] = childNodeIndex;
                groupHasDataChildNodeAABB_E[hasDataLabel] = childNodeAabb;
            }
        }

        if (hasData) {
            float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
            float3 samplePos_E = randXYZ * (childNodeAabb.maximum - childNodeAabb.minimum) + childNodeAabb.minimum;
            groupSamplePos_E[hasDataLabel * 4 + sampleIndex] = samplePos_E;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint hasDataChildNodeCount_E = groupHasDataChildNodeCount_E;
    uint hitTestCount_perNodeG = hasDataChildNodeCount_E * 4;
    if (groupThreadIndex < currentGroupNodeGCount * hitTestCount_perNodeG) {
        uint groupNodeIndex_G = groupThreadIndex / hitTestCount_perNodeG;
        uint childNodeLabel_E = (groupThreadIndex - (groupNodeIndex_G * hitTestCount_perNodeG)) / 4;

        uint hitTestIndex = groupThreadIndex & 3;
        float3 samplePos_E = groupSamplePos_E[childNodeLabel_E * 4 + hitTestIndex];

        IndivisibleNodeData_G nodeData_G = groupNodeData_G[groupNodeIndex_G];
        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePos_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        float3 sampleDir = normalize(samplePos_G - samplePos_E);

        AABB nodeAabb_E = groupHasDataChildNodeAABB_E[childNodeLabel_E];
        samplePos_E = getAABBPoint(nodeAabb_E, samplePos_E, sampleDir);

        HitPayload_FzbPG hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
        hitTestAble &= dot(sampleDir, nodeData_G.nodeNormal) > 0.0f;
        if (dot(nodeData_E.normal, sampleDir) > 0.0f) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePos_E + sampleDir * 0.00001f;
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
        groupNodePairHitInfos[groupNodeIndex_G].hitPayloads[childNodeLabel_E][hitTestIndex] = hitPayload;

        bool hit = hitPayload.hit == 1.0f;
        bool sampleAllHit = hitPayload.hit == 1.0f, samplePartialHit = hitPayload.hit == 1.0f;
        for (int offset = HITTEST_COUNT_PERCHILDNODE_FZBPG / 2; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            bool other_hit = WaveReadLaneAt(hit, targetLane);
            sampleAllHit &= other_hit;
            samplePartialHit |= other_hit;
        }
        // 0x11 -> allHit; 0x01 -> paritialHit; 0x00 -> notHit
        if (hitTestIndex == 0) groupNodePairHitInfos[groupNodeIndex_G].hitType[childNodeLabel_E] = (sampleAllHit << 1) | samplePartialHit; 
    }
    GroupMemoryBarrierWithGroupSync();

    bool partialHit = false;
    if (groupThreadIndex < currentGroupNodeGCount * 8) {
        uint groupNodeIndex_G = groupThreadIndex / 8;
        uint firstWarpLane_nodeG = groupNodeIndex_G * 8;

        uint groupChildNodeIndex_E = groupThreadIndex & 7;
        uint hitType = groupNodePairHitInfos[groupNodeIndex_G].hitType[groupChildNodeIndex_E];
        bool hasData = groupIsHasDataChildNode_E[groupChildNodeIndex_E];
        bool childNodeSampleAllHit = (hasData && hitType == 0x11) || !hasData;
        bool childNodeSampleNotHit = (hasData && hitType == 0x00) || !hasData;

        bool nodeSampleAllHit = childNodeSampleAllHit;
        bool nodeSampleNotHit = childNodeSampleNotHit;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            nodeSampleAllHit &= WaveReadLaneAt(nodeSampleAllHit, targetLane);
            nodeSampleNotHit &= WaveReadLaneAt(nodeSampleNotHit, targetLane);
        }
        nodeSampleAllHit = WaveReadLaneAt(nodeSampleAllHit, firstWarpLane_nodeG);
        uint allHitLabel = WavePrefixCountBits(nodeSampleAllHit) / 8;
        if (warpLane == firstWarpLane_nodeG) groupNodePairAllHitNodeIndex_G[allHitLabel] = groupNodeIndex_G;
        if (groupNodeIndex_G == currentGroupNodeGCount - 1 && warpLane == firstWarpLane_nodeG) groupNodePairAllHitCount = allHitLabel + (uint)nodeSampleAllHit;

        nodeSampleNotHit = WaveReadLaneAt(nodeSampleNotHit, firstWarpLane_nodeG);
        bool nodeSamplePartialHit = !nodeSampleAllHit && !nodeSampleNotHit;
        bool nodeSamplePartialHit_All_Not = childNodeSampleAllHit || childNodeSampleNotHit;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            nodeSamplePartialHit_All_Not &= WaveReadLaneAt(nodeSamplePartialHit_All_Not, targetLane);
        }
        nodeSamplePartialHit_All_Not = WaveReadLaneAt(nodeSamplePartialHit_All_Not, firstWarpLane_nodeG);
        if (nodeSamplePartialHit_All_Not) {
            uint childNodeIndex_E = nodeData_E.nodeIndex * 8 + groupChildNodeIndex_E;
            float childNodeMergeE = 0.0f;
            if ((hasData && hitType == 0x11) || !hasData) 
                childNodeMergeE = OctreeClusterDataBuffer_E[pushConst.currentLayer + 1][childNodeIndex_E].E;

            float mergeE = nodeData_E.E;
            for (int offset = 4; offset > 0; offset /= 2) {
                uint targetLane = warpLane + offset;
                mergeE += WaveReadLaneAt(mergeE, targetLane);
            }
            mergeE = WaveReadLaneAt(mergeE, firstWarpLane_nodeG);

            float probability = childNodeMergeE / mergeE;
            uint32_t sampleWarpLane = firstWarpLane_nodeG;
            float sampleProbability = 1.0f;
            if (warpLane == firstWarpLane_nodeG) {
                uint randomSeed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));
                sampleProbability = rand(randomSeed);
            }
            sampleProbability = WaveReadLaneAt(sampleProbability, firstWarpLane_nodeG);
            for (int offset = 0; offset < 8; ++offset) {
                sampleProbability -= WaveReadLaneAt(probability, firstWarpLane_nodeG + offset);
                if (sampleProbability <= 0.0f) {
                    sampleWarpLane += offset;
                    break;
                }
            }
            if (warpLane == sampleWarpLane) {
                if (pushConst.currentLayer == 0) {
                    OctreeNodeClusterData_E_FzbPG childNodeData = OctreeClusterDataBuffer_E[pushConst.currentLayer + 1][childNodeIndex_E];
                    OctreeNodeClusterData_E_FzbPG importantSampleNodeData_E = OctreeClusterDataBuffer_E[pushConst.octreeMaxLayer - OCTREE_CLUSTER_LAYER_FZBPG][childNodeData.importantSampleNodeIndex];
                    OctreeNodePairData_FzbPG nodePairData;
                    nodePairData.aabb = importantSampleNodeData_E.aabb;
                    nodePairData.pdf = childNodeData.pdf * probability;
                }else{
                    
                }
            }
        }else{

        }
    }
}