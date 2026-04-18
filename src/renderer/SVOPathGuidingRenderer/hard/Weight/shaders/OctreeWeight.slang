#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/SVOPathGuidingRenderer/hard/Weight/shaderio.h"
#ifndef NDEBUG
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"
#endif

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eOctreeArray_G)]] RWStructuredBuffer<OctreeNodeData_G, ScalarDataLayout> OctreeArray_G[];
[[vk::binding(StaticBindingPoints_SVOWeight::eNodeData_E)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> NodeData_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eTreeGlobalInfo)]] StructuredBuffer<OctreeGlobalInfo> OctreeGlobalInfoBuffer;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eIndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeights)]] RWStructuredBuffer<float> SVOWeights;
[[vk::binding(StaticBindingPoints_SVOWeight::eIndivisibleNodeNearbyNodeInfos)]] RWStructuredBuffer<IndivisibleNodeNearbyNodeInfo> IndivisibleNodeNearbyNodeInfos;

struct IndivisibleNodeData_E {
    float3 normal;
    float E;
    float3x3 TBN;
    AABB aabb;
    uint nodeIndex;
};
struct IndivisibleNodeData_G {
    float3 normal;
    float3x3 TBN;
    AABB aabb;
};
struct HitPayload_SVOWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float E;
    float3 sampleDir;
    float weight;
};

struct IndivisibleNodeData2_G {
    float3 normal;
    uint materialIndex;
    AABB aabb;
}
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared uint groupSVOLayerMaxLayer_G;

#define MAX_INDIVISIBLE_NODE_E_COUNT 2 // if indivisibleNodeCount_G > (THREADGROUP_SIZE / OUTGOING_COUNT), indivisibleNodeCount_E <= 2
groupshared IndivisibleNodeData_E groupNodeDatas_E[MAX_INDIVISIBLE_NODE_E_COUNT];
#define MAX_INDIVISIBLE_NODE_G_COUNT GETWEIGHT_CS_THREADGROUP_SIZE / OUTGOING_COUNT
groupshared IndivisibleNodeData_G groupNodeDatas_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_SVOWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT];

groupshared IndivisibleNodeData2_G groupNodeInfo;
groupshared int groupNearbyNodeDistanceIs[NEARBY_NODE_COUNT];
groupshared int2 groupNearbyNodeInfos[NEARBY_NODE_COUNT];

groupshared uint2 groupIndivisibleNodeInfos[GETNEARBYNODES2_CS_THREADGROUP_SIZE / 32];

inline void fillGroupshared(uint groupThreadIndex) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupIndivisibleNodeCount_G = globalInfo.indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = globalInfo.indivisibleNodeCount_E;
    }
    WaveActiveAllTrue(true);
}
void traceRay(RayDesc ray, inout HitPayload_SVOWeight payload, AABB nodeAABB_G)
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

        if (!checkPointInAABB(payload.hitPos, nodeAABB_G)) {
            payload.hit = -1.0f;
            return;
        }

        payload.hit = 1.0f;

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
    }
    else payload.hit = 0.0f;
}
//--------------------------------------------getIndivisibleNodeInfo-----------------------------------------
[numthreads(INITWEIGHT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0)
        groupIndivisibleNodeCount_G = OctreeGlobalInfoBuffer[0].indivisibleNodeCount_G;
    GroupMemoryBarrierWithGroupSync();

    int threadTotalCount = OUTGOING_COUNT * groupIndivisibleNodeCount_G * OCTREE_NODECOUNT_E;
    if (threadIndex >= threadTotalCount) return;
    SVOWeights[threadIndex] = 0.0f;

    if (threadIndex == 0) {
        OctreeGlobalInfo octreeGlobalInfo = OctreeGlobalInfoBuffer[0];

        SVOWeightGlobalInfo globalInfo;
        globalInfo.indivisibleNodeCount_G = octreeGlobalInfo.indivisibleNodeCount_G;
        globalInfo.indivisibleNodeCount_E = octreeGlobalInfo.indivisibleNodeCount_E;
        globalInfo.layerInfos_G = octreeGlobalInfo.layerInfos_G;

        globalInfo.cmd.x = (octreeGlobalInfo.indivisibleNodeCount_E * octreeGlobalInfo.indivisibleNodeCount_G * OUTGOING_COUNT + GETWEIGHT_CS_THREADGROUP_SIZE - 1) / GETWEIGHT_CS_THREADGROUP_SIZE;
        globalInfo.cmd.y = 1; globalInfo.cmd.z = 1;

        GlobalInfo[0] = globalInfo;
    }
}

[numthreads(GETWEIGHT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT;
    uint currentGroupNodeStartLabel_E = threadGroupIndex * GETWEIGHT_CS_THREADGROUP_SIZE / threadCount_perNodeE;
    uint currentGroupNodeEndLabel_E = ((threadGroupIndex + 1) * GETWEIGHT_CS_THREADGROUP_SIZE - 1) / threadCount_perNodeE;
    uint currentGroupNodeECount = currentGroupNodeEndLabel_E - currentGroupNodeStartLabel_E + 1;
    if (groupThreadIndex < currentGroupNodeECount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_E;
        uint nodeIndex = IndivisibleNodeInfos_E[nodeLabel];
        OctreeNodeData_E nodeData = NodeData_E[nodeIndex];

        IndivisibleNodeData_E groupNodeData_E;
        groupNodeData_E.normal = normalize(nodeData.meanNormal.xyz);
        groupNodeData_E.nodeIndex = nodeIndex;
        groupNodeData_E.E = nodeData.E;
        groupNodeData_E.aabb = nodeData.aabb;

        float3 tangent, bitangent;
        orthonormalBasis(groupNodeData_E.normal, tangent, bitangent);
        groupNodeData_E.TBN = float3x3(tangent, bitangent, groupNodeData_E.normal);

        groupNodeDatas_E[groupThreadIndex] = groupNodeData_E;
    }

    uint currentGroupNodeStartLabel_G = threadGroupIndex * GETWEIGHT_CS_THREADGROUP_SIZE / OUTGOING_COUNT;
    uint currentGroupNodeEndLabel_G = ((threadGroupIndex + 1) * GETWEIGHT_CS_THREADGROUP_SIZE - 1) / OUTGOING_COUNT;
    uint currentGoupNodeGCount = currentGroupNodeEndLabel_G - currentGroupNodeStartLabel_G + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeLabel = (groupThreadIndex + currentGroupNodeStartLabel_G) % indivisibleNodeCount_G;
        uint2 nodeInfo = IndivisibleNodeInfos_G[nodeLabel];
        OctreeNodeData_G nodeData = OctreeArray_G[nodeInfo.x][nodeInfo.y];

        IndivisibleNodeData_G groupNodeData_G;
        groupNodeData_G.normal = normalize(nodeData.meanNormal.xyz);
        groupNodeData_G.aabb = nodeData.aabb;

        float3 tangent, bitangent;
        orthonormalBasis(groupNodeData_G.normal, tangent, bitangent);
        groupNodeData_G.TBN = float3x3(tangent, bitangent, groupNodeData_G.normal);

        groupNodeDatas_G[groupThreadIndex] = groupNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    for (int i = groupThreadIndex; i < currentGoupNodeGCount * HITTEST_COUNT; i += GETWEIGHT_CS_THREADGROUP_SIZE) {
        uint nodeLabel_G_group = i / HITTEST_COUNT;
        uint nodeLabel_G = nodeLabel_G_group + currentGroupNodeStartLabel_G;

        uint nodeLabel_E = nodeLabel_G / indivisibleNodeCount_G;
        uint nodeLabel_E_group = nodeLabel_E - currentGroupNodeStartLabel_E;

        IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        IndivisibleNodeData_G nodeData_G = groupNodeDatas_G[nodeLabel_G_group];

        uint hitTestIndex = nodeLabel_G_group * HITTEST_COUNT + i % HITTEST_COUNT;
        uint seed = xxhash32(uint3(threadIndex, hitTestIndex, pushConst.frameIndex));

        float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_E = randXYZ * (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) + nodeData_E.aabb.minimum;

        randXYZ = float3(rand(seed), rand(seed), rand(seed));
        float3 samplePoint_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);

        HitPayload_SVOWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;
        hitPayload.E = nodeData_E.E;
        hitPayload.weight = 0.0f;

        bool normalCheck = dot(sampleDir, nodeData_E.normal) > 0.0f && dot(-sampleDir, nodeData_G.normal) > 0.0f;
        // if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        //     if (dot(outgoing, sampleDir) <= 0.0f) continue;
        // }
        if (normalCheck) {
            RayDesc ray;
            ray.Direction = sampleDir;
            ray.Origin = samplePoint_E + sampleDir * 0.00001f;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;

            traceRay(ray, hitPayload, nodeData_G.aabb);
            hitPayload.weight = 0.000001f;
        }

        //if (pushConst.frameIndex == 0) {
        //    bool printfAble = nodeData_G.nodeLabel == 9;
        //    printfAble = printfAble && abs(nodeData_E.aabb.maximum.y - 1.95) <= 0.0001f && abs(nodeData_E.aabb.minimum.y - 1.95) <= 0.0001f;
        //    if (printfAble) {
        //        printf("\nnodeIndex_E: %d, hit: %f\nsamplePoint_E: %f %f %f, samplePoint_G: %f %f %f\n",
        //               nodeData_E.nodeIndex, hitPayload.hit,
        //               samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
        //               samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
        //        );
        //    }
        //}

        groupHitPayloads[hitTestIndex] = hitPayload;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupIndivisibleNodeCount_E * indivisibleNodeCount_G * OUTGOING_COUNT) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT - 1);
    float3 outgoing = mul(getOutgoing(outgoingIndex, OUTGOING_COUNT), pushConst.randomRotateMatrix);
    uint nodeLabel_E_group = threadIndex / (indivisibleNodeCount_G * OUTGOING_COUNT) - currentGroupNodeStartLabel_E; // groupIndex
    uint nodeLabel_G_group = threadIndex / OUTGOING_COUNT - currentGroupNodeStartLabel_G;       // groupIndex
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT + hitTestIndex;
        HitPayload_SVOWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        weight += hitPayload.weight;

        //if (pushConst.frameIndex == 0) {
        //    bool printfAble = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G == 8;
        //    printfAble = printfAble && outgoingIndex == 41;
        //    IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        //    if (printfAble) {
        //        printf("nodeIndex_E: %d, hit: %f E: %f\naabb: %f %f %f %f  %f %f\n\n",
        //            nodeData_E.nodeIndex, hitPayload.hit,
        //            nodeData_E.E,
        //            nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
        //            nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
        //        );
        //    }
        //}

        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        bool isExt = true;
        // if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        //     if (cosine <= 0.0f || dot(outgoing, hitPayload.normal) <= 0.0f) continue;
        // } else if (cosine < 0.0f) isExt = false;

        float3 tangent, bitangent;
        orthonormalBasis(hitPayload.normal, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

        float3 weight3 = hitPayload.E * cosine;
        #ifdef WEIGHT_WITH_MATERIAL
        float3 bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        weight3 *= bsdf;
        #endif
        weight += max(0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z, 0.0f);

        //if (pushConst.frameIndex == 0) {
        //    bool printfAble = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G == 8;
        //    printfAble = printfAble && outgoingIndex == 41;
        //    IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        //    //printfAble = printfAble && nodeData_E.nodeIndex == 186;
        //    if (printfAble) {
        //        printf("nodeIndex_E: %d, hit: %f\nweight: %f, E: %f, cosine: %f, bsdf: %f %f %f\naabb: %f %f %f %f %f %f\n\n",
        //            nodeData_E.nodeIndex, hitPayload.hit,
        //            0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z,
        //            nodeData_E.E, cosine, bsdf.x, bsdf.y, bsdf.z,
        //            nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
        //            nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
        //        );
        //    }
        //}
    }
    //if (pushConst.frameIndex == 0) {
    //    bool printfAble = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G == 8;
    //    printfAble = printfAble && outgoingIndex == 41;
    //    IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
    //    if (printfAble) {
    //        printf("nodeIndex_E: %d weight: %f, E: %f\naabb: %f %f %f %f %f %f\n\n",
    //                nodeData_E.nodeIndex, weight, nodeData_E.E,
    //                nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
    //                nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
    //            );
    //    }
    //}

    if (weight > 0.0f) {
        uint nodeLabel_G = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G;
        uint nodeIndex_E = groupNodeDatas_E[nodeLabel_E_group].nodeIndex;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E +
                           nodeLabel_G * OCTREE_NODECOUNT_E + (nodeIndex_E + 56);
        SVOWeights[weightIndex] = weight;
    }

    if (threadIndex == 0) {
        uint threadTotalCount = OUTGOING_COUNT * indivisibleNodeCount_G * NODECOUNT_E;
        GlobalInfo[0].cmd.x = (threadTotalCount + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
    }
}

[numthreads(GETPROBABILITY_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();
    uint layerIndex = pushConst.currentLayer_E;
    uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    uint layerNodeCount_E = OctreeLayerInfo_E[layerIndex];
    if (threadIndex >= OUTGOING_COUNT * indivisibleNodeCount_G * layerNodeCount_E) return;

    uint outgoingIndex = threadIndex / (indivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
    uint indexInA = threadIndex - outgoingIndex * (indivisibleNodeCount_G * layerNodeCount_E);
    uint nodeLable_G = indexInA / layerNodeCount_E;                               // SVO_G nodeIndex
    uint nodeIndex_currentLayer_E = indexInA - nodeLable_G * layerNodeCount_E; // SVO_E currentLayer nodeIndex

    uint weightStartIndex = outgoingIndex * indivisibleNodeCount_G * OCTREE_NODECOUNT_E +
                       nodeLable_G * OCTREE_NODECOUNT_E;
    uint weightIndex = weightStartIndex + OctreeLayerStartIndex_E[layerIndex] + nodeIndex_currentLayer_E;

    float weight = SVOWeights[weightIndex];
    uint warplane = groupThreadIndex & (WARP_SIZE - 1);
    uint blockFirstWarplane = warplane / 8 * 8;
    float weightSum = weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = warplane + offset;
        weightSum += WaveReadLaneAt(weightSum, targetLane);
    }
    weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
    float probability = weight / max(weightSum, 1e-20f);
    if (probability > 0.0f) SVOWeights[weightIndex] = probability;

    if (layerIndex > 0) {
        if ((warplane & 7u) == 0 && weightSum > 0.0f) {
            uint fatherWeightIndex = weightStartIndex + OctreeLayerStartIndex_E[layerIndex - 1] + nodeIndex_currentLayer_E / 8;
            SVOWeights[fatherWeightIndex] = weightSum;
        }

        if (threadIndex == 0)
            GlobalInfo[0].cmd.x = (OUTGOING_COUNT * groupIndivisibleNodeCount_G * OctreeLayerInfo_E[layerIndex - 1] + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
    }
    /*
    if (pushConst.frameIndex == 0 && pushConst.currentLayer_E == 2) {
        bool printfAble = nodeLable_G == 8;
        printfAble = printfAble && outgoingIndex == 41;
        printfAble = printfAble && nodeIndex_currentLayer_E >= 184 && nodeIndex_currentLayer_E <= 191;
        if (printfAble) {
            printf("nodeIndex_E: %d\nweight: %f, probability: %f\n\n",
                   nodeIndex_currentLayer_E,
                   weight, probability
                );
        }
    }
    if (pushConst.frameIndex == 0 && pushConst.currentLayer_E == 1) {
        bool printfAble = nodeLable_G == 8;
        printfAble = printfAble && outgoingIndex == 41;
        printfAble = printfAble && nodeIndex_currentLayer_E >= 16 && nodeIndex_currentLayer_E <= 23;
        if (printfAble) {
            printf("nodeIndex_E: %d\nweight: %f, probability: %f\n\n",
                   nodeIndex_currentLayer_E,
                   weight, probability
                );
        }
    }
    if (pushConst.frameIndex == 0 && pushConst.currentLayer_E == 0) {
        bool printfAble = nodeLable_G == 8;
        printfAble = printfAble && outgoingIndex == 41;
        printfAble = printfAble && nodeIndex_currentLayer_E >= 0 && nodeIndex_currentLayer_E <= 8;
        if (printfAble) {
            printf("nodeIndex_E: %d\nweight: %f, probability: %f\n\n",
                   nodeIndex_currentLayer_E,
                   weight, probability
                );
        }
    }
    */
}
//---------------------------------------------getNearbyNode---------------------------------------------------
[numthreads(GETNEARBYNODES_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex < NEARBY_NODE_COUNT) {
        groupNearbyNodeDistanceIs[groupThreadIndex] = 2147483647;
        groupNearbyNodeInfos[groupThreadIndex] = uint2(0);
    }
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;
    uint indivisibleNodeLabel1 = threadGroupIndex / threadGroupCount_perIndivisibleNode;
    if (groupThreadIndex == 0) {
        uint2 nodeInfos = IndivisibleNodeInfos_G[indivisibleNodeLabel1];
        OctreeNodeData_G nodeData = OctreeArray_G[nodeInfos.x][nodeInfos.y];
        IndivisibleNodeData2_G nodeInfo;
        nodeInfo.normal = normalize(nodeData.meanNormal.xyz);
        nodeInfo.aabb = nodeData.aabb;
        #ifdef CLUSTER_WITH_MATERIAL
        nodeInfo.materialIndex = nodeData.materialIndex;
        #endif
        groupNodeInfo = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();
    uint threadGroupOffset = threadGroupIndex % threadGroupCount_perIndivisibleNode;
    uint nodeLabel2 = threadGroupOffset * GETNEARBYNODES_CS_THREADGROUP_SIZE + groupThreadIndex;
    if (nodeLabel2 >= indivisibleNodeCount) return;

    IndivisibleNodeData2_G nodeData1 = groupNodeInfo;

    uint2 nodeInfo2 = IndivisibleNodeInfos_G[nodeLabel2];
    OctreeNodeData_G nodeData2 = OctreeArray_G[nodeInfo2.x][nodeInfo2.y];

    float3 nodeCenter1 = (nodeData1.aabb.maximum + nodeData1.aabb.minimum) * 0.5f;
    float distance = PointToAABBDistance(nodeCenter1, nodeData2.aabb);
    distance *= 1.0f / max(dot(nodeData1.normal, normalize(nodeData2.meanNormal.xyz)), 0.00001f);
    //distance *= nodeData1.materialIndex == nodeData2.materialIndex ? 1.0f : 100000.0f;

    uint randomSeed = pushConst.frameIndex;
    distance += rand(randomSeed) * 0.0001f;

    if (indivisibleNodeLabel1 == nodeLabel2) distance = 1e30f;
    int distanceI = FloatToOrderedInt(distance);

    //if (pushConst.frameIndex == 0 && indivisibleNodeLabel1 == pushConst.sampleNodeLabel) {
    //    printf("nodeLabel2: %d, distance: %f\nnormal1: %f %f %f, normal2: %f %f %f\n",
    //           nodeLabel2, distance,
    //           nodeData1.normal.x, nodeData1.normal.y, nodeData1.normal.z,
    //           nodeData2.meanNormal.x, nodeData2.meanNormal.y, nodeData2.meanNormal.z
    //    );
    //}

    uint warpLane = groupThreadIndex & 31;
    int warpMinDistanceI = WaveActiveMin(distanceI);
    for (int i = 0; i < NEARBY_NODE_COUNT; ++i) {
        if (warpLane == 0) InterlockedMin<int>(groupNearbyNodeDistanceIs[i], warpMinDistanceI);
        GroupMemoryBarrierWithGroupSync();

        int groupMinDistanceI = groupNearbyNodeDistanceIs[i];
        uint minDistanceLabel = WavePrefixCountBits(distanceI == groupMinDistanceI);
        if (distanceI == groupMinDistanceI && minDistanceLabel == 0) { // maybe multiple nodes have the same distance, but it is rare
            groupNearbyNodeInfos[i] = nodeInfo2;
            distanceI = 2147483647;

            //if (pushConst.frameIndex == 0 && indivisibleNodeLabel1 == pushConst.sampleNodeLabel - 1) {
            //    printf("nodeIndex2: %d, distance: %f\nnormal1: %f %f %f, normal2: %f %f %f\n",
            //           nodeIndex2, distance,
            //           nodeData1.normal.x, nodeData1.normal.y, nodeData1.normal.z,
            //           nodeData2.meanNormal.x, nodeData2.meanNormal.y, nodeData2.meanNormal.z
            //    );
            //}
        }
        warpMinDistanceI = WaveActiveMin(distanceI);
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) {
        uint infoIndex = indivisibleNodeLabel1 * threadGroupCount_perIndivisibleNode + threadGroupOffset;
        IndivisibleNodeNearbyNodeInfos[infoIndex].nearbyNodeInfos = groupNearbyNodeInfos;
        for (int i = 0; i < NEARBY_NODE_COUNT; ++i)
            IndivisibleNodeNearbyNodeInfos[infoIndex].nearbyNodeDistances[i] = OrderedIntToFloat(groupNearbyNodeDistanceIs[i]);

        //if (pushConst.frameIndex == 0 && indivisibleNodeLabel1 == 20) {
        //    for (int i = 0; i < NEARBY_NODE_COUNT; ++i) {
        //        printf("nodeIndex2: %d, distance: %f\n",
        //                groupNearbyNodeIndics[i], OrderedIntToFloat(groupNearbyNodeDistanceIs[i]),
        //            );
        //    }
        //}
    }

    if (threadIndex == 0) {
        DispatchIndirectCommand cmd2;
        cmd2.x = (indivisibleNodeCount * 32 + GETNEARBYNODES2_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES2_CS_THREADGROUP_SIZE;
        cmd2.y = 1; cmd2.z = 1;

        if (threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT > 32) {
            printf("Error: Too many threads per indivisible node for getNearbyNodes. currentThreadCount: %u,    maxThreadCount: %u\n",
                   threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT, 32);
            cmd2.x = 0; cmd2.y = 0; cmd2.z = 0;
        }

        GlobalInfo[0].cmd2 = cmd2;
    }
}
[numthreads(GETNEARBYNODES2_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getNearbyNodes2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();
    int indivisibleNodeCount = groupIndivisibleNodeCount_G;
    uint threadGroupIndivisibleNodeStartLabel = threadGroupIndex * (GETNEARBYNODES2_CS_THREADGROUP_SIZE / 32);
    if (groupThreadIndex < GETNEARBYNODES2_CS_THREADGROUP_SIZE / 32) {
        uint groupIndivisibleNodeLabel = threadGroupIndivisibleNodeStartLabel + groupThreadIndex;
        if (groupIndivisibleNodeLabel < indivisibleNodeCount) {
            uint2 indivisibleNodeInfo = IndivisibleNodeInfos_G[groupIndivisibleNodeLabel];
            groupIndivisibleNodeInfos[groupThreadIndex] = indivisibleNodeInfo;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    uint indivisibleNodeLabel = threadIndex / 32;
    if (indivisibleNodeLabel >= indivisibleNodeCount) return;

    uint threadGroupCount_perIndivisibleNode = (indivisibleNodeCount + GETNEARBYNODES_CS_THREADGROUP_SIZE - 1) / GETNEARBYNODES_CS_THREADGROUP_SIZE;

    uint warpLane = groupThreadIndex & 31;
    uint warpIndex = groupThreadIndex / 32;

    float distance = 1e30f; int2 nearbyNodeIndex = -1;
    if (warpLane < threadGroupCount_perIndivisibleNode * NEARBY_NODE_COUNT) {
        uint infoIndex = indivisibleNodeLabel * threadGroupCount_perIndivisibleNode + warpLane / NEARBY_NODE_COUNT;
        distance = IndivisibleNodeNearbyNodeInfos[infoIndex].nearbyNodeDistances[warpLane % NEARBY_NODE_COUNT];
        nearbyNodeIndex = IndivisibleNodeNearbyNodeInfos[infoIndex].nearbyNodeInfos[warpLane % NEARBY_NODE_COUNT];
    }
    uint2 nodeInfo = groupIndivisibleNodeInfos[warpIndex];
    for (int i = 0; i < NEARBY_NODE_COUNT; ++i) {
        float minDistance = WaveActiveMin(distance);
        //if (minDistance > 1.0f) {
        //    if (warpLane == 0) {
        //        for (int j = i; j < NEARBY_NODE_COUNT; ++j) SVO_G[nodeIndex].nearbyNodeIndices[j] = -1;
        //    }
        //    break;
        //}
        uint minDistanceLabel = WavePrefixCountBits(distance == minDistance);
        if (distance == minDistance && minDistanceLabel == 0) {
            OctreeArray_G[nodeInfo.x][nodeInfo.y].nearbyNodeInfos[i] = nearbyNodeIndex;
            distance = 1e30f;

            //if (pushConst.frameIndex == 0 && indivisibleNodeLabel == pushConst.sampleNodeLabel) {
            //    printf("nodeIndex1: %d, nodeIndex2: %d, distance: %f\n",
            //        nodeIndex, nearbyNodeIndex, minDistance,
            //    );
            //}
        }
    }
}

#ifndef NDEBUG
//------------------------------------------------debug-------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getSampleNodeInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    if (globalInfo.indivisibleNodeCount_G == 0) return;

    OctreeLayerInfo layerInfos_G[MAX_OCTREE_LAYER];
    for (int i = 0; i <= MAX_OCTREE_LAYER; ++i) layerInfos_G[i] = GlobalInfo[0].layerInfos_G[i];

    /*
    float3 samplePos = pushConst.samplePos;
    SVONodeData_G nodeData;
    bool hitIndivisibleNode = false;
    uint nodeIndex = 0, nodeOffset = 0, nodeLabelSum = 0;
    for (int normalIndex = 0; normalIndex < 6; ++normalIndex) {
        bool hitNode = false;
        nodeIndex = normalIndex * 8; uint nodeSum = 0; nodeLabelSum = 0;
        for (int layerIndex = 1; layerIndex <= SVOMaxLayer; ++layerIndex) {
            for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                nodeData = SVO_G[nodeIndex + nodeOffset];
                if (checkPointInAABB(samplePos, nodeData.aabb)) {
                    if (nodeData.indivisible == 1) hitIndivisibleNode = true;
                    else {
                        nodeSum += layerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
                        nodeIndex = nodeSum + (nodeData.label - 1) * 8;
                        nodeLabelSum += layerInfos_G[layerIndex].indivisibleNodeCount;
                    }
                    hitNode = true;
                    break;
                }
            }
            if (!hitNode || hitIndivisibleNode) break;
        }
        if (hitIndivisibleNode) break;
    }
    */

    uint nodeLabel = pushConst.sampleNodeLabel;
    uint2 nodeInfo = IndivisibleNodeInfos_G[nodeLabel - 1];
    OctreeNodeData_G nodeData = OctreeArray_G[nodeInfo.x][nodeInfo.y];
    bool hitIndivisibleNode = nodeData.indivisible == 1;

    if (hitIndivisibleNode) GlobalInfo[0].sampelNodeAABB = nodeData.aabb;
    else {
        nodeData.aabb.minimum.x = 3.402823466e+38F;
        nodeData.aabb.minimum.y = 3.402823466e+38F;
        nodeData.aabb.minimum.z = 3.402823466e+38F;
        nodeData.aabb.maximum.x = -3.402823466e+38F;
        nodeData.aabb.maximum.y = -3.402823466e+38F;
        nodeData.aabb.maximum.z = -3.402823466e+38F;
        GlobalInfo[0].sampelNodeAABB = nodeData.aabb;
    }
    GlobalInfo[0].sampelNodeLabel = nodeLabel;

    if (pushConst.frameIndex == 0) {
        AABB aabb = GlobalInfo[0].sampelNodeAABB;
        printf("layerIndex: %d, nodeIndex: %d, sampleNodeLabel: %d\nsampleNodeAABB:\n %f %f %f\n%f %f %f\nnormal: %f   %f %f\n",
               nodeInfo.x, nodeInfo.y, nodeLabel,
            aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
               aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
            nodeData.meanNormal.x, nodeData.meanNormal.y, nodeData.meanNormal.z,
        );

        float3 nodeCenter = (aabb.maximum + aabb.minimum) * 0.5f;
        float3 outgoing = mul(pushConst.randomRotateMatrix, normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter));
        uint outgoingIndex = inverseOutgoing(outgoing, OUTGOING_COUNT);
        printf("outgoingIndex: %d\n", outgoingIndex);
    }
}

struct VSout {
    float4 sv_position : SV_Position;
    float3 worldPos;
    float3 worldNormal;
    float2 uv;
    uint randomSeed;
};
struct PSout
{
    float4 color : SV_Target;
};
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID)
{
    VSout output;

    int instanceIndex = pushConst.instanceIndex;
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    float3 normalMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, vertexIndex);
    float3 normal = mul(Inverse3x3((float3x3)instance.transform), normalMesh);
    normal = normalize(normal);

    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];
    if (material.materialMapIndex.y >= 0) {
        output.uv = getAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, vertexIndex);
        output.uv.y = 1.0f - output.uv.y;
    }

    output.sv_position = mul(pos, sceneInfo.viewProjMatrix);
    output.worldPos = pos.xyz;
    output.worldNormal = normal;
    output.randomSeed = xxhash32(uint3(vertexIndex, instanceIndex, pushConst.frameIndex));
    return output;
}
[shader("pixel")]
PSout fragmentMain(VSout input) {
    float3 pos = input.worldPos;
    float3 color;

    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    AABB sampleNodeAABB = globalInfo.sampelNodeAABB;
    if (checkPointInAABB(pos.xyz, sampleNodeAABB)) return { float4(1.0f, 0.0f, 0.0f, 1.0f) };

    if (globalInfo.indivisibleNodeCount_G == 0 || sampleNodeAABB.minimum.x > 100000.0f) return { float4(0.0f) };

    float3 nodeCenter_G = (globalInfo.sampelNodeAABB.maximum + globalInfo.sampelNodeAABB.minimum) * 0.5f;

    float3 view = normalize(pushConst.sceneInfoAddress[0].cameraPosition - pos);
    float probability = rand(input.randomSeed);
    // uint outgoingIndex = inverseOutgoing(view, OUTGOING_COUNT, probabality);
    uint outgoingIndex = inverseOutgoing(mul(pushConst.randomRotateMatrix, view), OUTGOING_COUNT);
    uint weightIndex = outgoingIndex * globalInfo.indivisibleNodeCount_G * OCTREE_NODECOUNT_E +
                       (globalInfo.sampelNodeLabel - 1) * OCTREE_NODECOUNT_E;

    uint normalIndex = getNormalIndex(input.worldNormal);

    OctreeNodeData_E nodeData;
    bool hitNode = false;
    uint nodeIndex = normalIndex * 64;
    float pdf = 1.0f;
    for (int offset = 0; offset < 64; ++offset) {
        nodeData = NodeData_E[nodeIndex + offset];
        if (checkPointInAABB(input.worldPos, nodeData.aabb)) {
            pdf *= SVOWeights[weightIndex + 56 + nodeIndex + offset];
            nodeIndex = nodeIndex + offset;
            hitNode = true;
            break;
        }
    }
    if (hitNode) {
        pdf *= SVOWeights[weightIndex + 8 + nodeIndex / 8];
        pdf *= SVOWeights[weightIndex + normalIndex];

        // if (nodeIndex == 339 && pushConst.frameIndex == 0) {
        //     for (int layerIndex = OCTREE_CLUSTER_LAYER; layerIndex >= 0; --layerIndex) {
        //         float pdf = SVOWeights[weightIndex + OctreeLayerStartIndex_E[layerIndex] + nodeIndex];
        //         printf("layerIndex: %d, nodeIndex: %d, pdf: %f\n", layerIndex, nodeIndex, pdf);
        //         nodeIndex /= 8;
        //     }
        //
        //     return { float4(10.0f, 10.0f, 0.0f, 1.0f) };
        // }
        //if (nodeIndex / 8 == 23) return { float4(10.0f, 10.0f, 0.0f, 1.0f) };
    }
    else return { float4(0.0f) };

    pdf *= nodeData.pdf;

    //if (pdf > 0.005f) {
    //    if (pushConst.frameIndex == 2) {
    //        printf("nodeIndex: %d, pdf: %f, nodePdf: %f, E: %f\naabb: %f %f %f %f %f %f\n\n",
    //            nodeIndex, pdf, nodeData.pdf, nodeData.E,
    //            nodeData.aabb.minimum.x, nodeData.aabb.minimum.y, nodeData.aabb.minimum.z,
    //            nodeData.aabb.maximum.x, nodeData.aabb.maximum.y, nodeData.aabb.maximum.z,
    //        );
    //    }
    //    return { float4(pdf * 10.0f, pdf * 10.0f, 0.0f, 1.0f) };
    //}


    return { float4(0.0f, pdf * 100.0f, 0.0f, 1.0f) };
}

struct VSout_nearby {
    float4 sv_position : SV_Position;
    float4 color;
};
[shader("vertex")]
VSout_nearby vertexMain_nearby(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VSout_nearby output;
    output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    return output;

    /*
    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    if (pushConst.sampleNodeLabel - 1 >= globalInfo.indivisibleNodeCount_G) return output;

    uint2 sampleNodeInfo = IndivisibleNodeInfos_G[pushConst.sampleNodeLabel - 1];
    OctreeNodeData_G nodeData = OctreeArray_G[sampleNodeInfo.x][sampleNodeInfo.y];
    bool show = instanceIndex == sampleNodeIndex;
    if (show) output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    else output.color = float4(0.0f, 1.0f, 0.0f, 1.0f);

    if (instanceIndex == 0 && vertexIndex == 0 && pushConst.frameIndex == 0) {
        printf("sampleNodeIndex: %d\n", sampleNodeIndex);
        for (int i = 0; i < NEARBY_NODE_COUNT; ++i) printf("nearbyNodeIndex: %d\n", nodeData.nearbyNodeIndices[i]);
    }

    uint nodeIndex = instanceIndex;
    for (int i = 0; i < NEARBY_NODE_COUNT; ++i) {
        if (nodeIndex == nodeData.nearbyNodeIndices[i]) {
            show = true;
            nodeData = SVO_G[nodeIndex];
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
    */
}
[shader("pixel")]
PSout fragmentMain_nearby(VSout_nearby input) {
    PSout output;
    output.color = input.color;
    return output;
}
#endif