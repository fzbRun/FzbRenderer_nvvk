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

//----------------------------------------------getNodePairData-------------------------------------------------
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;

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
        GlobalInfoBuffer[0].cmd.x = GlobalInfoBuffer[0].indivisibleNodeCount_E * ((indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE);
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
    uint layerIndex;
    uint nodeIndex;
};
struct HitPayload_FzbWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
};

groupshared IndivisibleNodeData_E groupNodeData_E;

void traceRay(RayDesc ray, inout HitPayload_FzbWeight payload)
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

#ifdef ADAPTIVE_IMPORTANCE_SAMPLING
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodePairData)]] RWStructuredBuffer<OctreeNodePairData_FzbPG, ScalarDataLayout> OctreeNodePairDataBuffer;
struct ChildNodeInfo_E {
    AABB aabb;
    float3 normal;
    float E;
    float pdf;
}

#define MAX_INDIVISIBLE_NODE_G_COUNT HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG
groupshared IndivisibleNodeData_G groupNodeData_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_FzbWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * 8 * HITTEST_COUNT_PER_CHILDNODE_FZBPG];

groupshared ChildNodeInfo_E groupChildNodeInfos_E[8];
groupshared uint groupHasDataChildNodeCount;

[numthreads(HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int indivisibleNodeCount_E = groupIndivisibleNodeCount_E;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG;
    uint threadGroupCount_perNodeE = (threadCount_perNodeE + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE;
    uint nodeLabel_E = threadGroupIndex / threadGroupCount_perNodeE;
    if (groupThreadIndex < 8) {
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[nodeLabel_E];
        OctreeNodeClusterData_E_FzbPG clusterNodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        if (groupThreadIndex == 0) {
            IndivisibleNodeData_E indivisibleNodeData_E;
            indivisibleNodeData_E.nodeIndex = nodeIndex;
            indivisibleNodeData_E.E = clusterNodeData.E;
            indivisibleNodeData_E.normal = normalize(clusterNodeData.meanNormal.xyz);
            indivisibleNodeData_E.aabb = clusterNodeData.aabb;

            groupNodeData_E = indivisibleNodeData_E;
        }
        {
            // float3 voxelSize = pushConst.VGBVoxelSize.xyz * (1 << (3 * (pushConst.octreeMaxLayer - pushConst.currentLayer)));
            // float3 childVoxelStartPos =
            //     int3((clusterNodeData.aabb.minimum - pushConst.VGBStartPos_Size.xyz) / voxelSize) * voxelSize + pushConst.VGBStartPos_Size.xyz;
            // float3 childVoxelSize = voxelSize / 8;
            // uint3 nodeOffset = uint3(groupThreadIndex & 1u, (groupThreadIndex & 2u) >> 1, (groupThreadIndex & 4u) >> 2);
            // AABB childNodeAabb;
            // childNodeAabb.minimum = childVoxelStartPos + nodeOffset * childVoxelSize;
            // childNodeAabb.maximum = childNodeAabb.minimum + childVoxelSize;
            // childNodeAabb.minimum = max(childNodeAabb.minimum, clusterNodeData.aabb.minimum);
            // childNodeAabb.maximum = min(childNodeAabb.maximum, clusterNodeData.aabb.maximum);
        }
        OctreeNodeClusterData_E_FzbPG childNodeData = OctreeClusterDataBuffer_E[1][nodeIndex * 8 + groupThreadIndex];
        bool hasData = childNodeData.E > 0.0f;

        uint warpHasDataChildNodeCount = WaveActiveCountBits(hasData);
        uint warpHasDataChildNodeLabel = WavePrefixCountBits(hasData);
        if (warpLane == 0) groupHasDataChildNodeCount = warpHasDataChildNodeCount;

        if (hasData) {
            ChildNodeInfo_E childNodeInfo;
            childNodeInfo.aabb = childNodeData.aabb; // this aabb is importantSample result
            childNodeInfo.normal = normalize(childNodeData.meanNormal.xyz);
            childNodeInfo.E = childNodeData.E;
            childNodeInfo.pdf = childNodeData.pdf;
            groupChildNodeInfos_E[warpHasDataChildNodeLabel] = childNodeInfo;
        }
    }

    uint threadGroupIndex_inNodeE = threadGroupIndex % threadGroupCount_perNodeE;
    uint remainingThreadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG - threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE;
    uint threadCount_currentThreadGroup = remainingThreadCount_perNodeE >= HITTEST_CS_THREADGROUP_SIZE ? HITTEST_CS_THREADGROUP_SIZE : remainingThreadCount_perNodeE;

    uint currentGroupNodeStartLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE) / OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeEndLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE + threadCount_currentThreadGroup - 1) / OUTGOING_COUNT_FZBPG;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG clusterNodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G IndivisibleNodeData_G;
        IndivisibleNodeData_G.aabb = clusterNodeData.aabb;
        IndivisibleNodeData_G.layerIndex = nodeInfo.x;
        IndivisibleNodeData_G.nodeIndex = nodeInfo.y;

        uint octreeLayerNode = 1 << (3 * nodeInfo.x);
        uint normalIndex = nodeInfo.y / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        IndivisibleNodeData_G.nodeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= nodeLabel == 211;
        // if (printfAble) {
        //     printf("normal: %f %f %f\nnormalIndex: %d\n",
        //         clusterNodeData.meanNormal.x, clusterNodeData.meanNormal.y, clusterNodeData.meanNormal.z,
        //         normalIndex,
        //     );
        // }

        groupNodeData_G[groupThreadIndex] = IndivisibleNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    IndivisibleNodeData_E nodeData_E = groupNodeData_E;
    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, (uint)pushConst.frameIndex));

    uint hasDataChildNodeCount_E = groupHasDataChildNodeCount;
    uint hitTestCount_perNodeG = hasDataChildNodeCount_E * HITTEST_COUNT_PER_CHILDNODE_FZBPG;
    for (int hitTestIndex = groupThreadIndex; hitTestIndex < currentGoupNodeGCount * hitTestCount_perNodeG; hitTestIndex += HITTEST_CS_THREADGROUP_SIZE) {
        uint nodeLocalLabel_G = hitTestIndex / hitTestCount_perNodeG;
        uint nodeLabel_G = nodeLocalLabel_G + currentGroupNodeStartLabel_G;
        IndivisibleNodeData_G nodeData_G = groupNodeData_G[nodeLocalLabel_G];

        uint childNodeLabel = (hitTestIndex % hitTestCount_perNodeG) / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        ChildNodeInfo_E childNodeInfo_E = groupChildNodeInfos_E[childNodeLabel];

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (childNodeInfo_E.aabb.maximum - childNodeInfo_E.aabb.minimum) + childNodeInfo_E.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        samplePoint_E = getAABBPoint(childNodeInfo_E.aabb, samplePoint_E, sampleDir); // make samplePoint_E move to the aabb surface, prevent self shielding

        HitPayload_FzbWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
        hitTestAble &= dot(-sampleDir, nodeData_G.nodeNormal) > 0.0f;
        if (hitTestAble) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;
            traceRay(ray, hitPayload);
            if (hitPayload.hit > 0.0f) {
                bool check = checkOcclusion(samplePoint_E, hitPayload.hitPos, nodeData_G.aabb);
                if (checkPointInAABB2(hitPayload.hitPos, nodeData_G.aabb)) {
                    check &= dot(hitPayload.normal, -sampleDir) > 0.0f;
                    check &= dot(hitPayload.normal, nodeData_G.nodeNormal) > 0.707f;
                }
                if (!check) hitPayload.hit = -2.0f;
            }
        }
        groupHitPayloads[hitTestIndex] = hitPayload;

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= nodeLabel_G == pushConst.sampleNodeLabel_G - 1;
        ////printfAble &= nodeLabel_E == 15;
        // if (printfAble) {
        //     printf("childLabel: %d, E: %f\nhitTestIndex: %d, hit: %f, hitTestAble: %d, checkOcclusion: %d,hitNormal: %f %f %f, hitPos: %f %f %f\nnormal_G:  %f %f %f\nsamplePoint_E: %f %f %f, samplePoint_G: %f %f  %f, sampleDir: %f  %f %f\nchildNodeAabb:   %f %f  %f %f %f %f\nnodeGAabb: %f %f  %f %f %f %f\n\n",
        //            childNodeLabel, childNodeInfo_E.E,
        //            hitTestIndex % HITTEST_COUNT_PER_CHILDNODE_FZBPG, hitPayload.hit,
        //            hitTestAble, checkOcclusion(samplePoint_E, hitPayload.hitPos, nodeData_G.aabb),
        //            hitPayload.normal.x, hitPayload.normal.y, hitPayload.normal.z,
        //            hitPayload.hitPos.x, hitPayload.hitPos.y, hitPayload.hitPos.z,
        //            nodeData_G.nodeNormal.x, nodeData_G.nodeNormal.y, nodeData_G.nodeNormal.z,
        //            samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
        //            samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
        //            sampleDir.x, sampleDir.y, sampleDir.z,
        //            childNodeInfo_E.aabb.minimum.x, childNodeInfo_E.aabb.minimum.y, childNodeInfo_E.aabb.minimum.z,
        //            childNodeInfo_E.aabb.maximum.x, childNodeInfo_E.aabb.maximum.y, childNodeInfo_E.aabb.maximum.z,
        //            nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
        //            nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
        //     );
        // }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex >= threadCount_currentThreadGroup) return;

    if (warpIndex < currentGoupNodeGCount && warpLane < hitTestCount_perNodeG) {
        uint childNodeLabel = warpLane / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        float childNodeE = groupChildNodeInfos_E[childNodeLabel].E;

        uint childNodeFirstWarpLane = warpLane / HITTEST_COUNT_PER_CHILDNODE_FZBPG * HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        uint hitTestIndex_inChildNode = childNodeFirstWarpLane + warpLane % HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        bool hit = groupHitPayloads[warpIndex * hitTestCount_perNodeG + hitTestIndex_inChildNode].hit == 1.0f;
        uint hitCount = uint(hit);
        for (int offset = HITTEST_COUNT_PER_CHILDNODE_FZBPG / 2; offset > 0; offset /= 2) {
            uint targetLane = warpLane + offset;
            hitCount += WaveReadLaneAt(hitCount, targetLane);
        }
        if (warpLane == childNodeFirstWarpLane) {
            childNodeE *= (float)hitCount / HITTEST_COUNT_PER_CHILDNODE_FZBPG;

            float mergeE = WaveActiveSum(childNodeE);
            float probability = childNodeE / max(mergeE, 1e-16f);

            float selectProbablity = rand(seed);
            selectProbablity = WaveReadLaneFirst(selectProbablity);
            uint sampleChildLabel = 0;
            for (int childNodeTargetLabel = 0; childNodeTargetLabel < hasDataChildNodeCount_E; ++childNodeTargetLabel) {
                float other_probablity = WaveReadLaneAt(probability, childNodeTargetLabel * HITTEST_COUNT_PER_CHILDNODE_FZBPG);
                if (selectProbablity <= other_probablity) {
                    sampleChildLabel = childNodeTargetLabel;
                    break;
                }
                selectProbablity -= other_probablity;
            }

            if (childNodeLabel == sampleChildLabel) {
                ChildNodeInfo_E childNodeInfo = groupChildNodeInfos_E[childNodeLabel];
                probability *= childNodeInfo.pdf;
                OctreeNodePairData_FzbPG nodePairData = { childNodeInfo.aabb, probability };

                uint nodeLabel_G = currentGroupNodeStartLabel_G + warpIndex;
                uint nodePairIndex = nodeLabel_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG + nodeData_E.nodeIndex;
                OctreeNodePairDataBuffer[nodePairIndex] = nodePairData;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    uint nodeLocalIndex_G = groupThreadIndex / OUTGOING_COUNT_FZBPG;
    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);
    float weightSum = 0.0f;
    for (int childNodeLabel = 0; childNodeLabel < hasDataChildNodeCount_E; ++childNodeLabel) {
        float weight = 0.0f;
        for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT_PER_CHILDNODE_FZBPG; ++hitTestIndex) {
            uint hitPayloadIndex = nodeLocalIndex_G * hitTestCount_perNodeG + childNodeLabel * HITTEST_COUNT_PER_CHILDNODE_FZBPG + hitTestIndex;
            HitPayload_FzbWeight hitPayload = groupHitPayloads[hitPayloadIndex];
            if (hitPayload.hit <= 0.0f) continue;

            float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
            if (cosine <= 0.0f) continue;

            float3 bsdf = float3(0.0f);
            if (hitPayload.material.type == MaterialType::Diffuse) {
                bsdf = getBSDF_Diffuse(hitPayload.material.albedo, -hitPayload.sampleDir, hitPayload.normal);
            } else {
                float3 tangent, bitangent;
                orthonormalBasis(hitPayload.normal, tangent, bitangent);
                float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

                bool isExt = true;
                bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
            }

            float3 weight3 = bsdf * cosine;
            weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);
        }
        weight *= groupChildNodeInfos_E[childNodeLabel].E / HITTEST_COUNT_PER_CHILDNODE_FZBPG;
        weightSum += weight;

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= currentGroupNodeStartLabel_G + nodeLocalIndex_G == 48;
        // printfAble &= nodeLabel_E == 15;
        // if (printfAble) {
        //     printf("weight: %f\n", weightSum);
        // }
    }
    if (weightSum > 0.0f) {
        uint nodeLabel_G = currentGroupNodeStartLabel_G + nodeLocalIndex_G;
        uint nodeIndex_E = nodeData_E.nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                           nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG + (nodeIndex_E + 56);
        OctreeNodePairWeightBuffer[weightIndex] = weightSum;
    }

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G;
}
#else
#define MAX_INDIVISIBLE_NODE_G_COUNT HITTEST_CS_THREADGROUP_SIZE / OUTGOING_COUNT_FZBPG
groupshared IndivisibleNodeData_G groupNodeData_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_FzbWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT_FZBPG];

[numthreads(HITTEST_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_octreeNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    int indivisibleNodeCount_E = groupIndivisibleNodeCount_E;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG;
    uint threadGroupCount_perNodeE = (threadCount_perNodeE + HITTEST_CS_THREADGROUP_SIZE - 1) / HITTEST_CS_THREADGROUP_SIZE;
    uint nodeLabel_E = threadGroupIndex / threadGroupCount_perNodeE;
    if (groupThreadIndex == 0) {
        uint nodeIndex = IndivisibleNodeInfoBuffer_E[nodeLabel_E];
        OctreeNodeClusterData_E_FzbPG clusterNodeData = OctreeClusterDataBuffer_E[0][nodeIndex];

        IndivisibleNodeData_E indivisibleNodeData_E;
        indivisibleNodeData_E.nodeIndex = nodeIndex;
        indivisibleNodeData_E.E = clusterNodeData.E;
        indivisibleNodeData_E.normal = normalize(clusterNodeData.meanNormal.xyz);
        indivisibleNodeData_E.aabb = clusterNodeData.aabb;

        groupNodeData_E = indivisibleNodeData_E;
    }

    uint threadGroupIndex_inNodeE = threadGroupIndex % threadGroupCount_perNodeE;
    uint remainingThreadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT_FZBPG - threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE;
    uint threadCount_currentThreadGroup = remainingThreadCount_perNodeE >= HITTEST_CS_THREADGROUP_SIZE ? HITTEST_CS_THREADGROUP_SIZE : remainingThreadCount_perNodeE;

    uint currentGroupNodeStartLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE) / OUTGOING_COUNT_FZBPG;
    uint currentGroupNodeEndLabel_G = (threadGroupIndex_inNodeE * HITTEST_CS_THREADGROUP_SIZE + threadCount_currentThreadGroup - 1) / OUTGOING_COUNT_FZBPG;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_G;
        uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[nodeLabel];
        OctreeNodeClusterData_G_FzbPG clusterNodeData = OctreeClusterDataBuffer_G[nodeInfo.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo.y];

        IndivisibleNodeData_G IndivisibleNodeData_G;
        IndivisibleNodeData_G.aabb = clusterNodeData.aabb;

        uint octreeLayerNode = 1 << (3 * nodeInfo.x);
        uint normalIndex = nodeInfo.y / octreeLayerNode;
        float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
        IndivisibleNodeData_G.nodeNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
        IndivisibleNodeData_G.nodeNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

        groupNodeData_G[groupThreadIndex] = IndivisibleNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    IndivisibleNodeData_E nodeData_E = groupNodeData_E;

    for (int hitTestIndex = groupThreadIndex; hitTestIndex < currentGoupNodeGCount * HITTEST_COUNT_FZBPG; hitTestIndex += HITTEST_CS_THREADGROUP_SIZE) {
        uint nodeLabel_G_group = hitTestIndex / HITTEST_COUNT_FZBPG;
        IndivisibleNodeData_G nodeData_G = groupNodeData_G[nodeLabel_G_group];

        uint seed = xxhash32(uint3(threadIndex, hitTestIndex, (uint)pushConst.frameIndex));

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) + nodeData_E.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        samplePoint_E = getAABBPoint(nodeData_E.aabb, samplePoint_E, sampleDir); // make samplePoint_E move to the aabb surface, prevent self shielding

        HitPayload_FzbWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;

        bool hitTestAble = dot(nodeData_E.normal, sampleDir) > 0.0f;
        hitTestAble &= dot(-sampleDir, nodeData_G.nodeNormal) > 0.0f;
        if (hitTestAble) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
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
        groupHitPayloads[hitTestIndex] = hitPayload;

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= nodeLabel_G % indivisibleNodeCount_G == 87;
        // printfAble &= nodeLabel_E == 30;
        ////printfAble &= i % HITTEST_COUNT_FZBPG == 0;
        // if (printfAble) {
        //     printf("nodeLabel_E: %d hit: %f\nhitPos: %f %f %f, hitNormal: %f %f %f\nsampleDir: %f %f %f\naabb_E: %f %f %f %f %f %f\naabb_G: %f %f %f %f %f %f\nnodeNormal_G: %f %f %f\n\n",
        //            nodeLabel_E, hitPayload.hit,
        //            hitPayload.hitPos.x, hitPayload.hitPos.y, hitPayload.hitPos.z,
        //            hitPayload.normal.x, hitPayload.normal.y, hitPayload.normal.z,
        //            sampleDir.x, sampleDir.y, sampleDir.z,
        //            nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
        //            nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
        //            nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
        //            nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
        //            nodeData_G.nodeNormal.x, nodeData_G.nodeNormal.y, nodeData_G.nodeNormal.z,
        //     );
        // }
    }
    GroupMemoryBarrierWithGroupSync();
    if (groupThreadIndex >= threadCount_currentThreadGroup) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT_FZBPG - 1);
    float3 outgoing = mul(getOutgoing_FzbPG(outgoingIndex, OUTGOING_COUNT_FZBPG), pushConst.randomRotateMatrix);
    uint nodeLabel_G_group = groupThreadIndex / OUTGOING_COUNT_FZBPG;
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT_FZBPG; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT_FZBPG + hitTestIndex;
        HitPayload_FzbWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        if (cosine <= 0.0f) continue;

        float3 bsdf = float3(0.0f);
        if (hitPayload.material.type == MaterialType::Diffuse) {
            bsdf = getBSDF_Diffuse(hitPayload.material.albedo, -hitPayload.sampleDir, hitPayload.normal);
        } else {
            float3 tangent, bitangent;
            orthonormalBasis(hitPayload.normal, tangent, bitangent);
            float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

            bool isExt = true;
            bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        }

        float3 weight3 = bsdf * cosine;
        weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);
    }
    weight = weight * nodeData_E.E / HITTEST_COUNT_FZBPG;

    if (weight > 0.0f) {
        uint nodeLabel_G = currentGroupNodeStartLabel_G + nodeLabel_G_group;
        uint nodeIndex_E = nodeData_E.nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                           nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG + (nodeIndex_E + 56);
        OctreeNodePairWeightBuffer[weightIndex] = weight;
    }

    if (threadIndex == 0) {
        // uint threadTotalCount = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG;
        // GlobalInfoBuffer[0].cmd.x = (threadTotalCount + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
        GlobalInfoBuffer[0].cmd.x = OUTGOING_COUNT_FZBPG * indivisibleNodeCount_G;
    }
}
#endif

groupshared float groupWeights[CLUSTER_LAYER_NODECOUNT_E_FZBPG / 8];
[numthreads(CLUSTER_LAYER_NODECOUNT_E_FZBPG, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupIndivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    }
    GroupMemoryBarrierWithGroupSync();
    uint layerIndex = pushConst.currentLayer;
    uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;

    uint outgoingIndex = threadGroupIndex / indivisibleNodeCount_G; // outgoing Index
    uint nodeLabel_G = threadGroupIndex % indivisibleNodeCount_G;

    uint weightStartIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                       nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG;
    uint weightIndex = weightStartIndex + OctreeLayerStartIndex_FzbPG[OCTREE_CLUSTER_LAYER_FZBPG] + groupThreadIndex;
    float weight = OctreeNodePairWeightBuffer[weightIndex];

    uint warplane = groupThreadIndex & 31;
    uint blockFirstWarplane = warplane / 8 * 8;
    float weightSum = weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = warplane + offset;
        weightSum += WaveReadLaneAt(weightSum, targetLane);
    }
    weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
    float probability = weight / max(weightSum, 1e-20f);
    if (probability > 0.0f) OctreeNodePairWeightBuffer[weightIndex] = probability;
    if (warplane == blockFirstWarplane) groupWeights[groupThreadIndex / 8] = weightSum;
    GroupMemoryBarrierWithGroupSync();

    // bool printfAble = pushConst.frameIndex == 1;
    // printfAble &= outgoingIndex == 57;
    // printfAble &= nodeLabel_G == 0;
    // if (printfAble) {
    //     printf("layerIndex: 2, nodeIndex_E: %d\nweight:%f, probability: %f\n\n", groupThreadIndex, weight, probability);
    // }

    for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG - 1; layerIndex >= 0; --layerIndex) {
        uint layerNodeCount = OctreeLayerNodeCount_FzbPG[layerIndex];
        if (groupThreadIndex >= layerNodeCount) return; // after votla

        weight = groupWeights[groupThreadIndex];
        weightSum = weight;
        for (int offset = 4; offset > 0; offset /= 2) {
            uint targetLane = warplane + offset;
            weightSum += WaveReadLaneAt(weightSum, targetLane);
        }
        weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
        probability = weight / max(weightSum, 1e-20f);

        // if (printfAble) {
        //     printf("layerIndex: %d, nodeIndex_E: %d\nweight:%f, probability: %f\n\n", layerIndex, groupThreadIndex, weight, probability);
        // }

        if (probability > 0.0f) {
            weightIndex = weightStartIndex + OctreeLayerStartIndex_FzbPG[layerIndex] + groupThreadIndex;
            OctreeNodePairWeightBuffer[weightIndex] = probability;
        }

        if (layerIndex > 0) {
            if (warplane == blockFirstWarplane) groupWeights[groupThreadIndex / 8] = weightSum;
            GroupMemoryBarrierWithGroupSync();
        }
    }
}
//------------------------------------------------debug---------------------------------------------------------
#ifndef NDEBUG
struct VSout_OctreeNodePairHitTestResult {
    float4 sv_position : SV_Position;
    float4 color;
};
struct PSout_OctreeNodePairHitTestResult
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout_OctreeNodePairHitTestResult vertexMain_OctreeNodePairHitTestResult(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout_OctreeNodePairHitTestResult output;
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    uint indivisibleNodeCount_E = GlobalInfoBuffer[0].indivisibleNodeCount_E;
    uint indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    if (instanceIndex >= indivisibleNodeCount_E + 1 ||
        pushConst.sampleNodeLabel_G <= 0 ||
        pushConst.sampleNodeLabel_G >= indivisibleNodeCount_G + 1) return output;

    uint nodeLabel_G = pushConst.sampleNodeLabel_G - 1;
    uint2 nodeInfo_G = IndivisibleNodeInfoBuffer_G[nodeLabel_G];
    OctreeNodeClusterData_G_FzbPG nodeData_G = OctreeClusterDataBuffer_G[nodeInfo_G.x - OCTREE_CLUSTER_LAYER_FZBPG][nodeInfo_G.y];

    float3 nodeCenter = (nodeData_G.aabb.minimum + nodeData_G.aabb.maximum) * 0.5f;
    float3 viewDir = normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter);
    float3 outgoing = mul(pushConst.randomRotateMatrix, viewDir);
    uint outgoingIndex = inverseOutgoing_FzbPG(outgoing, OUTGOING_COUNT_FZBPG);
    uint weightStartIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG + nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG;

    AABB nodeAABB;
    if (instanceIndex == indivisibleNodeCount_E) {
        nodeAABB = nodeData_G.aabb;
        output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);

        if (pushConst.frameIndex == 1 && vertexIndex == 0) printf("\nnormal: %f %f %f\n", nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z);
    } else {
        uint nodeLabel_E = instanceIndex;
        uint nodeIndex_E = IndivisibleNodeInfoBuffer_E[instanceIndex];
        float nodePdf = 1.0f;
        #ifdef ADAPTIVE_IMPORTANCE_SAMPLING
        uint nodePairIndex = nodeLabel_G * CLUSTER_LAYER_NODECOUNT_E_FZBPG + nodeIndex_E;
        OctreeNodePairData_FzbPG nodePairData = OctreeNodePairDataBuffer[nodePairIndex];
        nodeAABB = nodePairData.aabb;
        nodePdf = nodePairData.pdf;
        #else
        OctreeNodeData_E_FzbPG nodeData_E = ClusterlayerDataBuffer_E[nodeIndex_E];
        nodeAABB = nodeData_E.aabb;
        nodePdf = nodeData_E.pdf;
        #endif

        float layerPdf = 1.0f;
        for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG; layerIndex >= 0; --layerIndex) {
            uint weightIndex = weightStartIndex + nodeIndex_E + OctreeLayerStartIndex_FzbPG[layerIndex];
            layerPdf *= OctreeNodePairWeightBuffer[weightIndex];
            nodeIndex_E /= 8;
        }

        // bool printfAble = pushConst.frameIndex == 1;
        // printfAble &= vertexIndex == 0;
        // printfAble &= nodeLabel_G == 48;
        // printfAble &= outgoingIndex == 12;
        // printfAble &= nodeLabel_E == 15;
        // if (printfAble){
        //     printf("nodeIndex_E: %d\naabb: %f %f %f %f %f %f\nnodePdf: %f, layerPdf: %f\n\n",
        //            IndivisibleNodeInfoBuffer_E[instanceIndex],
        //            nodeAABB.minimum.x, nodeAABB.minimum.y, nodeAABB.minimum.z,
        //            nodeAABB.maximum.x, nodeAABB.maximum.y, nodeAABB.maximum.z,
        //            nodePdf, layerPdf
        //     );
        // }

        float pdf = layerPdf * nodePdf;
        output.color = float4(0.0f, pdf * 10.0f, 0.0f, 1.0f);
    }
    if (nodeAABB.minimum.x > 1e10f) return output;

    float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

    output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
    output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
    output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
    output.sv_position.xyz += nodeAABB.minimum;

    output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);

    return output;
}
[shader("pixel")]
PSout_OctreeNodePairHitTestResult fragmentMain_OctreeNodePairHitTestResult(VSout_OctreeNodePairHitTestResult input) {
    return { input.color };
}
#endif