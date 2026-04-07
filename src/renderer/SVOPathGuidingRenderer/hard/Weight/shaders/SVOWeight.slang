#include "renderer/SVOPathGuidingRenderer/hard/Weight/shaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#ifndef NDEBUG
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"
#endif

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eNodeData_E)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> NodeData_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOGlobalInfo)]] StructuredBuffer<SVOGlobalInfo_SVOPG> SVOGlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G)]] RWStructuredBuffer<uint> SVO_IndivisibleNodeInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_E)]] RWStructuredBuffer<uint> SVO_IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeights)]] RWStructuredBuffer<float> SVOWeights;

struct IndivisibleNodeData_E {
    float3 normal;
    float3x3 TBN;
    float3 irradiance;
    uint nodeIndex;
    AABB aabb;
};
struct IndivisibleNodeData_G {
    float3 normal;
    float3x3 TBN;
    uint nodeLabel;
    uint materialType;
    AABB aabb;
};
struct HitPayload_SVOWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
    float3 sampleDir;
    float3 irradiance;
};
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_SVO_LAYER];
groupshared uint groupSVOLayerMaxLayer_G;
groupshared uint groupTotalNodeCount_G;
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupWarpIndivisibleNodeCount_E[(OCTREE_NODECOUNT_E + 31) / 32];

groupshared uint groupIndivisibleNodeCount_E;

#define MAX_INDIVISIBLE_NODE_E_COUNT 2 // if indivisibleNodeCount_G > (THREADGROUP_SIZE / OUTGOING_COUNT), indivisibleNodeCount_E <= 2
groupshared IndivisibleNodeData_E groupNodeDatas_E[MAX_INDIVISIBLE_NODE_E_COUNT];
#define MAX_INDIVISIBLE_NODE_G_COUNT GETWEIGHT_CS_THREADGROUP_SIZE / OUTGOING_COUNT
groupshared IndivisibleNodeData_G groupNodeDatas_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_SVOWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT];

inline void fillGroupshared(uint groupThreadIndex) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupSVOLayerMaxLayer_G = globalInfo.SVOMaxLayer_G;
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
[numthreads(GETINDIVISIBLENODEINFO_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getIndivisibleNode(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < WARP_SIZE) {
        SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];
        if (groupThreadIndex < MAX_SVO_LAYER)
            groupSVOLayerInfos_G[groupThreadIndex] = svoGlobalInfo.layerInfos_G[groupThreadIndex];

        if (groupThreadIndex == 0) {
            for (int i = 0; i < MAX_SVO_LAYER; ++i) {
                if (svoGlobalInfo.layerInfos_G[i].divisibleNodeCount == 0) {
                    groupSVOLayerMaxLayer_G = i;
                    break;
                }
            }
        }
        WaveActiveAllTrue(true);

        uint indivisibleNodeCount_G = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_G) indivisibleNodeCount_G = svoGlobalInfo.layerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);

        if (groupThreadIndex == 0) {
            groupTotalNodeCount_G = svoGlobalInfo.totalNodeCount_G;
            groupIndivisibleNodeCount_G = indivisibleNodeCount_G;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
    if (groupIndivisibleNodeCount_G > SVOIndivisibleNodeCount_G) {
        if (threadIndex == 0) {
            printf("Error: G indivisibleNodeCount_G exceeds the maximum node count. maxNodeCount: %u,    currentNodeCount: %u\n",
                   SVOIndivisibleNodeCount_G, GlobalInfo[0].indivisibleNodeCount_G);
            GlobalInfo[0].indivisibleNodeCount_G = 0;
            GlobalInfo[0].indivisibleNodeCount_E = 0;
            GlobalInfo[0].cmd.x = 0;
        }
        return;
    }
    if (threadIndex < groupTotalNodeCount_G) {
        SVONodeData_G nodeData = SVO_G[threadIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (indivisible) {
            int maxLayerIndex = groupSVOLayerMaxLayer_G;
            int nodeIndex = threadIndex, labelOffset = 0;
            for (int layerIndex = 1; layerIndex <= maxLayerIndex; ++layerIndex) {
                uint layerNodeCount = groupSVOLayerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
                if (nodeIndex < layerNodeCount) break;
                nodeIndex -= layerNodeCount;
                labelOffset += groupSVOLayerInfos_G[layerIndex].indivisibleNodeCount;
            }
            SVO_IndivisibleNodeInfos_G[labelOffset + nodeData.label - 1] = threadIndex;
        }
    }

    if (threadGroupIndex == 0) {
        uint warpIndex = threadIndex / 32;
        uint warpLane = threadIndex & 31;

        uint warpCount = 12;    //(384 + 31) / 32;
        bool indivisible = false;
        uint warpIndivisibleNodeCount;
        if (threadIndex < warpCount * 32) {
            if (threadIndex < 384) {
                SVONodeData_E nodeData = NodeData_E[threadIndex];
                bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
                indivisible = hasData && nodeData.indivisible == 1;
            }
            warpIndivisibleNodeCount = WaveActiveCountBits(indivisible);
            if (warpLane == 0) groupWarpIndivisibleNodeCount_E[warpIndex] = warpIndivisibleNodeCount;
        }
        GroupMemoryBarrierWithGroupSync();
        if (threadIndex < warpCount * 32) {
            uint nodeWarpLabel = WavePrefixCountBits(indivisible);

            uint nodeGroupLabel = 0;
            if (warpLane < warpIndex) nodeGroupLabel = groupWarpIndivisibleNodeCount_E[warpLane];
            nodeGroupLabel = WaveActiveSum(nodeGroupLabel);

            if(indivisible) SVO_IndivisibleNodeInfos_E[nodeWarpLabel + nodeGroupLabel] = threadIndex;

            if (warpIndex == warpCount - 1 && warpLane == 0) {
                DispatchIndirectCommand cmd;
                cmd.x = (OUTGOING_COUNT * indivisibleNodeCount_G * OCTREE_NODECOUNT_E + INITWEIGHT_CS_THREADGROUP_SIZE - 1) / INITWEIGHT_CS_THREADGROUP_SIZE;
                cmd.y = 1; cmd.z = 1;

                SVOWeightGlobalInfo globalInfo;
                globalInfo.cmd = cmd;
                globalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
                globalInfo.indivisibleNodeCount_E = nodeGroupLabel + warpIndivisibleNodeCount;
                globalInfo.SVOMaxLayer_G = groupSVOLayerMaxLayer_G;
                for (int layerIndex = 0; layerIndex < MAX_SVO_LAYER; ++layerIndex)
                    globalInfo.layerInfos_G[layerIndex] = groupSVOLayerInfos_G[layerIndex];
                GlobalInfo[0] = globalInfo;

                #ifndef NDEBUG
                if (pushConst.frameIndex == 0) {
                    printf( "getIndivisibleNode_G cmd:%d %d %d\nindivisibleNodeCount: G: %d  E: %d\nSVOMaxLayer: G: %d\n",
                        cmd.x, cmd.y, cmd.z,
                        indivisibleNodeCount_G, globalInfo.indivisibleNodeCount_E,
                        groupSVOLayerMaxLayer_G,
                    );
                    for (int i = 0; i <= groupSVOLayerMaxLayer_G; ++i) {
                        printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               groupSVOLayerInfos_G[i].divisibleNodeCount, groupSVOLayerInfos_G[i].indivisibleNodeCount
                        );
                    }
                }
                #endif
            }
        }
    }
}

[numthreads(INITWEIGHT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();

    int threadTotalCount = OUTGOING_COUNT * groupIndivisibleNodeCount_G * OCTREE_NODECOUNT_E;
    if (threadIndex >= threadTotalCount) return;
    SVOWeights[threadIndex] = 0.0f;

    if (threadIndex == 0)
        GlobalInfo[0].cmd.x = (groupIndivisibleNodeCount_E * groupIndivisibleNodeCount_G * OUTGOING_COUNT + GETWEIGHT_CS_THREADGROUP_SIZE - 1) / GETWEIGHT_CS_THREADGROUP_SIZE;
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
        uint nodeIndex = SVO_IndivisibleNodeInfos_E[nodeLabel];
        SVONodeData_E nodeData = NodeData_E[nodeIndex];

        IndivisibleNodeData_E groupNodeData_E;
        groupNodeData_E.normal = normalize(nodeData.meanNormal.xyz);
        groupNodeData_E.nodeIndex = nodeIndex;
        groupNodeData_E.irradiance = nodeData.irradiance;
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
        uint nodeIndex = SVO_IndivisibleNodeInfos_G[nodeLabel];
        SVONodeData_G nodeData = SVO_G[nodeIndex];

        IndivisibleNodeData_G groupNodeData_G;
        groupNodeData_G.nodeLabel = nodeLabel + 1;
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

        float3 halfSize_E = (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) * 0.5f;
        float3 nodeCenter_E = nodeData_E.aabb.minimum + halfSize_E;
        float3 halfSize_G = (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) * 0.5f;
        float3 nodeCenter_G = nodeData_G.aabb.minimum + halfSize_G;

        float3 samplePoint_E = sample_hemisphere_mean(seed);
        samplePoint_E = mul(samplePoint_E, nodeData_E.TBN); // slang is right multiplication
        samplePoint_E = getAABBPoint(samplePoint_E, nodeCenter_E, halfSize_E);

        float3 samplePoint_G = sample_hemisphere_mean(seed);
        samplePoint_G = mul(samplePoint_G, nodeData_G.TBN);
        samplePoint_G = getAABBPoint(samplePoint_G, nodeCenter_G, halfSize_G);

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);

        HitPayload_SVOWeight hitPayload;
        hitPayload.hit = -1.0f;
        hitPayload.sampleDir = sampleDir;
        hitPayload.irradiance = nodeData_E.irradiance;

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
        }

        groupHitPayloads[hitTestIndex] = hitPayload;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupIndivisibleNodeCount_E * indivisibleNodeCount_G * OUTGOING_COUNT) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT - 1);
    float3 outgoing = fibSpherePoint(outgoingIndex, OUTGOING_COUNT);
    uint nodeLabel_E_group = threadIndex / (indivisibleNodeCount_G * OUTGOING_COUNT) - currentGroupNodeStartLabel_E; // groupIndex
    uint nodeLabel_G_group = threadIndex / OUTGOING_COUNT - currentGroupNodeStartLabel_G;       // groupIndex
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT + hitTestIndex;
        HitPayload_SVOWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        bool isExt = true;
        // if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        //     if (cosine <= 0.0f || dot(outgoing, hitPayload.normal) <= 0.0f) continue;
        // } else if (cosine < 0.0f) isExt = false;

        float3 tangent, bitangent;
        orthonormalBasis(hitPayload.normal, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

        float3 bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        float3 weight3 = hitPayload.irradiance * cosine * bsdf; // notIgnoreRatio is made by irradiance length, so use length here;
        weight += 0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z;
    }

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

    //if (pushConst.frameIndex == 0) {
    //    if (outgoingIndex == 41 && (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G == 65) {
    //        printf("nodeIndex: %d, weight: %f\n",
    //            groupNodeDatas_E[nodeLabel_E_group].nodeIndex, weight
    //        );
    //    }
    //}
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
    float probability = weight / max(weightSum, 0.00001f);
    if (probability > 0.0f) SVOWeights[weightIndex] = probability;

    //if (pushConst.frameIndex == 0) {
    //    if (outgoingIndex == 41 && nodeLable_G == 65) {
    //        printf("layerIndex: %d, nodeIndex: %d, probability: %f\n",
    //            pushConst.currentLayer_E, nodeIndex_currentLayer_E, probability
    //        );
    //    }
    //}


    if (layerIndex > 0) {
        uint fatherWeightIndex = weightStartIndex + OctreeLayerStartIndex_E[layerIndex - 1] + nodeIndex_currentLayer_E / 8;
        if ((warplane & 7u) == 0 && weightSum > 0.0f) SVOWeights[fatherWeightIndex] = weightSum;

        if (threadIndex == 0)
            GlobalInfo[0].cmd.x = (OUTGOING_COUNT * groupIndivisibleNodeCount_G * OctreeLayerInfo_E[layerIndex - 1] + GETPROBABILITY_CS_THREADGROUP_SIZE - 1) / GETPROBABILITY_CS_THREADGROUP_SIZE;
    }
}
#ifndef NDEBUG
//------------------------------------------------debug-------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getSampleNodeInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    if (globalInfo.indivisibleNodeCount_G == 0) return;

    int SVOMaxLayer = globalInfo.SVOMaxLayer_G;
    SVOLayerInfo layerInfos_G[MAX_SVO_LAYER];
    for (int i = 0; i <= SVOMaxLayer; ++i) layerInfos_G[i] = SVOGlobalInfo[0].layerInfos_G[i];

    SVONodeData_G nodeData;
    bool hitIndivisibleNode = false;
    uint nodeIndex = 0, nodeOffset = 0, nodeLabelSum = 0;
    for (int normalIndex = 0; normalIndex < 6; ++normalIndex) {
        bool hitNode = false;
        nodeIndex = normalIndex * 8; uint nodeSum = 0; nodeLabelSum = 0;
        for (int layerIndex = 1; layerIndex <= SVOMaxLayer; ++layerIndex) {
            for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                nodeData = SVO_G[nodeIndex + nodeOffset];
                if (checkPointInAABB(pushConst.samplePos, nodeData.aabb)) {
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
    GlobalInfo[0].sampelNodeLabel = nodeData.label + nodeLabelSum;

    if (pushConst.frameIndex == 0) {
        AABB aabb = GlobalInfo[0].sampelNodeAABB;
        printf("nodeIndex: %d, sampleNodeLabel: %d\nsampleNodeAABB:\n %f %f %f\n%f %f %f\nnormal: %f %f %f\n",
               nodeIndex + nodeOffset, nodeData.label + nodeLabelSum,
            aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
               aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
            nodeData.meanNormal.x, nodeData.meanNormal.y, nodeData.meanNormal.z,
        );

        float3 nodeCenter = (aabb.maximum + aabb.minimum) * 0.5f;
        uint outgoingIndex = inverseSF(normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter), OUTGOING_COUNT);
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
    // uint outgoingIndex = inverseSF(view, OUTGOING_COUNT, probabality);
    uint outgoingIndex = inverseSF(view, OUTGOING_COUNT);
    uint weightIndex = outgoingIndex * globalInfo.indivisibleNodeCount_G * OCTREE_NODECOUNT_E +
                       (globalInfo.sampelNodeLabel - 1) * OCTREE_NODECOUNT_E;

    SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];

    uint normalIndex = getNormalIndex(input.worldNormal);

    SVONodeData_E nodeData;
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
    }
    else return { float4(0.0f) };

    probability *= nodeData.pdf;
    return { float4(0.0f, probability * 10.0f, 0.0f, 1.0f) };
}
#endif