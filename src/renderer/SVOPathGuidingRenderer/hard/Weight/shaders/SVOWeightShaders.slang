#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#ifndef NDEBUG
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"
#endif

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOGlobalInfo_SVOWeight)]] StructuredBuffer<SVOGlobalInfo_SVOPG> SVOGlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight)]] RWStructuredBuffer<uint> SVO_IndivisibleNodeInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_E_SVOWeight)]] RWStructuredBuffer<uint> SVO_IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight)]] RWStructuredBuffer<float> SVOWeights;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeightSums_SVOWeight)]] RWStructuredBuffer<float> SVOWeightSums;

struct SVONodeInfo_G {
    AABB aabb;
    uint32_t indivisible;
    uint32_t label;
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
groupshared uint groupSVOLayerMaxLayer_G;
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_OCTREE_DEPTH];
groupshared SVOLayerInfo groupSVOLayerInfos_E[MAX_OCTREE_DEPTH];
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared uint groupTotalNodeCount_G;
groupshared uint groupTotalNodeCount_E;

#define MAX_INDIVISIBLE_NODE_E_COUNT 2 // if indivisibleNodeCount_G > (THREADGROUP_SIZE / OUTGOING_COUNT), indivisibleNodeCount_E <= 2
groupshared IndivisibleNodeData_E groupNodeDatas_E[MAX_INDIVISIBLE_NODE_E_COUNT];
#define MAX_INDIVISIBLE_NODE_G_COUNT THREADGROUP_SIZE / OUTGOING_COUNT
groupshared IndivisibleNodeData_G groupNodeDatas_G[MAX_INDIVISIBLE_NODE_G_COUNT];
groupshared HitPayload_SVOWeight groupHitPayloads[MAX_INDIVISIBLE_NODE_G_COUNT * HITTEST_COUNT];

groupshared uint groupLayerNodeCount_E;;
groupshared uint groupPreviousLayerNodeCountSum_E;
groupshared uint groupCurrentLayerNodeCountSum_E;

inline void fillGroupshared(uint groupThreadIndex) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupSVOLayerMaxLayer_G = globalInfo.SVOMaxLayer_G;
        groupIndivisibleNodeCount_G = globalInfo.indivisibleNodeCount_G;
        groupIndivisibleNodeCount_E = globalInfo.indivisibleNodeCount_E;
        groupTotalNodeCount_E = globalInfo.totalNodeCount_E;
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
        Instance instance = sceneInfo.instances[instanceIndex];              // Instance data
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
//--------------------------------------------------compute shader-----------------------------------------
//computeMain_getIndivisibleNode_G get SVO_G's indivisibleNode's info for reduce thread wastage(except divisibleNode)
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getIndivisibleNode(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < WARP_SIZE) {
        SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];
        if (groupThreadIndex < MAX_OCTREE_DEPTH) {
            groupSVOLayerInfos_G[groupThreadIndex] = svoGlobalInfo.layerInfos_G[groupThreadIndex];
            groupSVOLayerInfos_E[groupThreadIndex] = svoGlobalInfo.layerInfos_E[groupThreadIndex];
        }

        if (groupThreadIndex == 0) {
            for (int i = 1; i < 8; ++i) {
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

        uint indivisibleNodeCount_E = 0;
        if (groupThreadIndex <= pushConst.clusterLayer_E) indivisibleNodeCount_E = svoGlobalInfo.layerInfos_E[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_E = WaveActiveSum(indivisibleNodeCount_E);

        if (groupThreadIndex == 0) {
            groupIndivisibleNodeCount_G = indivisibleNodeCount_G;
            groupIndivisibleNodeCount_E = indivisibleNodeCount_E;
            groupTotalNodeCount_G = svoGlobalInfo.totalNodeCount_G;
            groupTotalNodeCount_E = svoGlobalInfo.totalNodeCount_E;

            if (threadIndex == 0) {

                DispatchIndirectCommand cmd;
                cmd.x = (OUTGOING_COUNT * indivisibleNodeCount_G * svoGlobalInfo.totalNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
                cmd.y = 1; cmd.z = 1;

                SVOWeightGlobalInfo globalInfo;
                globalInfo.cmd = cmd;
                globalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
                globalInfo.indivisibleNodeCount_E = indivisibleNodeCount_E;
                globalInfo.SVOMaxLayer_G = groupSVOLayerMaxLayer_G;
                globalInfo.totalNodeCount_E = svoGlobalInfo.totalNodeCount_E;
                for (int layerIndex = 0; layerIndex < MAX_OCTREE_DEPTH; ++layerIndex) {
                    globalInfo.layerInfos_G[layerIndex] = svoGlobalInfo.layerInfos_G[layerIndex];
                    globalInfo.layerInfos_E[layerIndex] = svoGlobalInfo.layerInfos_E[layerIndex];
                }
                GlobalInfo[0] = globalInfo;

                #ifndef NDEBUG
                if (pushConst.frameIndex == 0) {
                    printf("getIndivisibleNode_G\ncmd:%d %d %d\nindivisibleNodeCount: G: %d  E: %d\nSVOMaxLayer: G:  %d  E: %d\ntotalNodeCount_E: %d\n",
                           cmd.x, cmd.y, cmd.z,
                        indivisibleNodeCount_G, indivisibleNodeCount_E,
                        groupSVOLayerMaxLayer_G, pushConst.clusterLayer_E,
                        svoGlobalInfo.totalNodeCount_E
                    );
                    for (int i = 0; i <= groupSVOLayerMaxLayer_G; ++i) {
                        printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               groupSVOLayerInfos_G[i].divisibleNodeCount, groupSVOLayerInfos_G[i].indivisibleNodeCount
                        );
                    }
                    for (int i = 0; i <= pushConst.clusterLayer_E; ++i) {
                        printf("E layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               groupSVOLayerInfos_E[i].divisibleNodeCount, groupSVOLayerInfos_E[i].indivisibleNodeCount
                        );
                    }
                }
                #endif
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndivisibleNodeCount_G > SVOIndivisibleNodeCount_G) {
        if (threadIndex == 0) {
            printf( "Error: G indivisibleNodeCount_G exceeds the maximum node count. maxNodeCount: %u,   currentNodeCount: %u\n",
                SVOIndivisibleNodeCount_G, GlobalInfo[0].indivisibleNodeCount_G);
            GlobalInfo[0].indivisibleNodeCount_G = 0;
            GlobalInfo[0].indivisibleNodeCount_E = 0;
            GlobalInfo[0].cmd.x = 0;
        }
        return;
    }
    if (groupIndivisibleNodeCount_E > SVOIndivisibleNodeCount_E) {
        if (threadIndex == 0) {
            printf("Error: G indivisibleNodeCount_E exceeds the maximum node count. maxNodeCount: %u,    currentNodeCount: %u\n",
                SVOIndivisibleNodeCount_E, GlobalInfo[0].indivisibleNodeCount_E);
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

    if (threadIndex < groupTotalNodeCount_E) {
        SVONodeData_E nodeData = SVO_E[threadIndex];
        bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (indivisible) {
            int maxLayerIndex = pushConst.clusterLayer_E;
            int nodeIndex = threadIndex, labelOffset = 0;
            for (int layerIndex = 1; layerIndex <= maxLayerIndex; ++layerIndex) {
                uint layerNodeCount = groupSVOLayerInfos_E[layerIndex - 1].divisibleNodeCount * 8;
                if (nodeIndex < layerNodeCount) break;
                nodeIndex -= layerNodeCount;
                labelOffset += groupSVOLayerInfos_E[layerIndex].indivisibleNodeCount;
            }
            SVO_IndivisibleNodeInfos_E[labelOffset + nodeData.label - 1] = threadIndex;
        }
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();

    int threadTotalCount = OUTGOING_COUNT * groupIndivisibleNodeCount_G * (groupTotalNodeCount_E + 8);
    if (threadIndex >= threadTotalCount) return;
    SVOWeights[threadIndex] = 0.0f;

    if (threadIndex >= threadTotalCount / 8) return;
    SVOWeightSums[threadIndex] = 0.0f;

    if (threadIndex == 0)
        GlobalInfo[0].cmd.x = (groupIndivisibleNodeCount_E * groupIndivisibleNodeCount_G * OUTGOING_COUNT  + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_G = groupIndivisibleNodeCount_G;

    uint threadCount_perNodeE = indivisibleNodeCount_G * OUTGOING_COUNT;
    uint currentGroupNodeStartLabel_E = threadGroupIndex * THREADGROUP_SIZE / threadCount_perNodeE;
    uint currentGroupNodeEndLabel_E = ((threadGroupIndex + 1) * THREADGROUP_SIZE - 1) / threadCount_perNodeE;
    uint currentGroupNodeECount = currentGroupNodeEndLabel_E - currentGroupNodeStartLabel_E + 1;
    if (groupThreadIndex < currentGroupNodeECount) {
        uint nodeLabel = groupThreadIndex + currentGroupNodeStartLabel_E;
        uint nodeIndex = SVO_IndivisibleNodeInfos_E[nodeLabel];
        SVONodeData_E nodeData = SVO_E[nodeIndex];

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

    uint currentGroupNodeStartLabel_G = threadGroupIndex * THREADGROUP_SIZE / OUTGOING_COUNT;
    uint currentGroupNodeEndLabel_G = ((threadGroupIndex + 1) * THREADGROUP_SIZE - 1) / OUTGOING_COUNT;
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

    for (int i = groupThreadIndex; i < currentGoupNodeGCount * HITTEST_COUNT; i += THREADGROUP_SIZE) {
        uint nodeLabel_G_group = i / HITTEST_COUNT;
        uint nodeLabel_G = nodeLabel_G_group + currentGroupNodeStartLabel_G;

        uint nodeLabel_E = nodeLabel_G / indivisibleNodeCount_G;
        uint nodeLabel_E_group = nodeLabel_E - currentGroupNodeStartLabel_E;

        IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        IndivisibleNodeData_G nodeData_G = groupNodeDatas_G[nodeLabel_G_group];

        uint hitTestIndex = nodeLabel_E_group * HITTEST_COUNT + i % HITTEST_COUNT;
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

        //nodeLabel_G = nodeLabel_G % indivisibleNodeCount_G;
        //if (pushConst.frameIndex == 0 && nodeLabel_G + 1 == 137 && nodeLabel_E == 75) {
        //    printf("hit: %f, samplePos_E: %f %f %f, samplePos_G: %f %f %f, sampleDir: %f %f %f\n",
        //        hitPayload.hit,
        //        samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
        //        samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
        //        sampleDir.x, sampleDir.y, sampleDir.z,
        //    );
        //}
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupIndivisibleNodeCount_E * indivisibleNodeCount_G * OUTGOING_COUNT) return;

    uint outgoingIndex = groupThreadIndex & (OUTGOING_COUNT - 1);
    float3 outgoing = fibSpherePoint(outgoingIndex, OUTGOING_COUNT);
    uint nodeLabel_E_group = threadIndex / (indivisibleNodeCount_G * OUTGOING_COUNT) - currentGroupNodeStartLabel_E; // groupIndex
    uint nodeLabel_G_group = threadIndex / OUTGOING_COUNT - currentGroupNodeStartLabel_G;      //groupIndex
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT; ++hitTestIndex) {
        uint hitPayloadIndex = nodeLabel_G_group * HITTEST_COUNT + hitTestIndex;
        HitPayload_SVOWeight hitPayload = groupHitPayloads[hitPayloadIndex];
        if (hitPayload.hit <= 0.0f) continue;

        float cosine = dot(-hitPayload.sampleDir, hitPayload.normal);
        bool isExt = true;
        //if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        //    if (cosine <= 0.0f || dot(outgoing, hitPayload.normal) <= 0.0f) continue;
        //} else if (cosine < 0.0f) isExt = false;

        float3 tangent, bitangent;
        orthonormalBasis(hitPayload.normal, tangent, bitangent);
        float3x3 TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

        float3 bsdf = getBSDF(hitPayload.material, -hitPayload.sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        float3 weight3 = hitPayload.irradiance * cosine * bsdf; // notIgnoreRatio is made by irradiance length, so use length here;
        weight += 0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z;

        uint nodeLabel_G = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G;
        uint nodeLabel_E = nodeLabel_E_group + currentGroupNodeStartLabel_E;
        if (pushConst.frameIndex == 0 && nodeLabel_G + 1 == 137 && nodeLabel_E == 75 && outgoingIndex == 41) {
            printf("outgoingIndex: %d, hitTestIndex: %d, hit: %f, weight: %f, irradiance: %f %f %f, cosine: %f, bsdf: %f %f %f\n",
                outgoingIndex, hitTestIndex, hitPayload.hit,
                0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z,
                hitPayload.irradiance.x, hitPayload.irradiance.y, hitPayload.irradiance.z,
                cosine,
                bsdf.x, bsdf.y, bsdf.z,
            );
        }
    }

    {
        IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[nodeLabel_E_group];
        IndivisibleNodeData_G nodeData_G = groupNodeDatas_G[nodeLabel_G_group];

        bool printfAble = pushConst.frameIndex == 0;
        printfAble &= ((threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G) + 1 == 137;
        printfAble &= outgoingIndex == 41;
        printfAble &= nodeLabel_E_group + currentGroupNodeStartLabel_E == 75;
        if (printfAble) {
            printf("nodeLabel_E: %d, nodeIndex: %d, aabb: %f %f %f %f %f %f, normal_G: %f %f %f\n",
                nodeLabel_E_group + currentGroupNodeStartLabel_E,
                nodeData_E.nodeIndex,
                nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
                nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
                nodeData_E.normal.x, nodeData_E.normal.y, nodeData_E.normal.z
            );
        }
    }

    if (weight > 0.0f) {
        uint nodeLabel_G = (threadIndex / OUTGOING_COUNT) % indivisibleNodeCount_G;
        uint nodeIndex_E = groupNodeDatas_E[nodeLabel_E_group].nodeIndex;
        int totalNodeCount_E = groupTotalNodeCount_E + 8;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * totalNodeCount_E +
                           nodeLabel_G * totalNodeCount_E + (nodeIndex_E + 8);
        SVOWeights[weightIndex] = weight;
    }

    if (threadIndex == 0) {
        uint SVOMaxLayerNodeCount = SVOGlobalInfo[0].layerInfos_E[pushConst.clusterLayer_E - 1].divisibleNodeCount * 8;
        uint threadTotalCount = OUTGOING_COUNT * indivisibleNodeCount_G * SVOMaxLayerNodeCount;
        GlobalInfo[0].cmd.x = (threadTotalCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    int currentLayerIndex = pushConst.currentLayer_E;
    int maxLayer_E = pushConst.clusterLayer_E;
    if (currentLayerIndex > 0) {
        if (groupThreadIndex == 0) {
            SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];

            uint layerNodeCount_E = svoGlobalInfo.layerInfos_E[currentLayerIndex - 1].divisibleNodeCount * 8;
            groupLayerNodeCount_E = layerNodeCount_E;

            uint currentLayerNodeCountSum_E = groupTotalNodeCount_E;
            for (int i = maxLayer_E; i > currentLayerIndex; --i)
                currentLayerNodeCountSum_E -= svoGlobalInfo.layerInfos_E[i - 1].divisibleNodeCount * 8;
            groupCurrentLayerNodeCountSum_E = currentLayerNodeCountSum_E;

            groupPreviousLayerNodeCountSum_E = currentLayerNodeCountSum_E - layerNodeCount_E;
        }
        GroupMemoryBarrierWithGroupSync();

        uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
        uint layerNodeCount_E = groupLayerNodeCount_E;
        if (threadIndex >= OUTGOING_COUNT * indivisibleNodeCount_G * layerNodeCount_E) return;

        uint outgoingIndex = threadIndex / (indivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
        uint indexInA = threadIndex - outgoingIndex * (indivisibleNodeCount_G * layerNodeCount_E);
        uint nodeLable_G = indexInA / layerNodeCount_E;                            // SVO_G nodeIndex
        uint nodeSVOIndex_currentLayer_E = indexInA - nodeLable_G * layerNodeCount_E; // SVO_E currentLayer nodeIndex

        uint nodeSVOIndex = groupPreviousLayerNodeCountSum_E + nodeSVOIndex_currentLayer_E;
        SVONodeData_E nodeData_E = SVO_E[nodeSVOIndex];
        bool hasData = nodeData_E.irradiance.x + nodeData_E.irradiance.y + nodeData_E.irradiance.z > 0.0f;
        bool indivisible = nodeData_E.indivisible == 1;
        uint nodeLabel_E = nodeData_E.label;

        uint weightCount_perSVO_E = groupTotalNodeCount_E + 8;
        uint nodeWeightIndex = nodeSVOIndex + 8;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * weightCount_perSVO_E +
                           nodeLable_G * weightCount_perSVO_E + nodeWeightIndex;
        float weight = 0.0f;
        if (hasData) {
            if (indivisible == 0) {
                uint childNodeWeightIndex = groupCurrentLayerNodeCountSum_E + (nodeLabel_E - 1) * 8 + 8;
                uint weightSumIndex = (weightIndex - nodeWeightIndex + childNodeWeightIndex) / 8;
                weight = SVOWeightSums[weightSumIndex];
            } else weight = SVOWeights[weightIndex];
        }

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

        if ((warplane & 7u) == 0 && weightSum > 0.0f) SVOWeightSums[weightIndex / 8] = weightSum;

        /*
        {
            bool printfAble = pushConst.frameIndex == 0;
            printfAble &= nodeLable_G + 1 == 137;
            printfAble &= outgoingIndex == 41;

            bool printfAble1 = printfAble;  // && nodeSVOIndex >= 200 && nodeSVOIndex <= 207; // nodeIndex == 202;
            if (printfAble1) {
                printf("currentLayerIndex: %d, nodeIndex_E: %d, weight: %f, probability: %f, weightIndex: %d, weightSumIndex: %d\n",
                       currentLayerIndex, nodeSVOIndex, weight, probability, weightIndex, weightIndex / 8
                );
            }

            bool printfAble2 = printfAble;  // && nodeSVOIndex == 23;
            if (printfAble2) {
                uint childNodeWeightIndex = groupCurrentLayerNodeCountSum_E + (nodeLabel_E - 1) * 8 + 8;
                uint weightSumIndex = (weightIndex - nodeWeightIndex + childNodeWeightIndex) / 8;
                printf("currentLayerIndex: %d, nodeIndex_E: %d, weight: %f, probability: %f, weightIndex: %d, weightSumIndex: %d\n",
                       currentLayerIndex, nodeSVOIndex, weight, probability, weightIndex, weightSumIndex,
                );
            }
        }
        */
    } else {
        uint indivisibleNodeCount_G = groupIndivisibleNodeCount_G;
        const uint layerNodeCount_E = 8;
        if (threadIndex >= OUTGOING_COUNT * indivisibleNodeCount_G * layerNodeCount_E) return;

        uint outgoingIndex = threadIndex / (indivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
        uint indexInA = threadIndex - outgoingIndex * (indivisibleNodeCount_G * layerNodeCount_E);
        uint nodeLable_G = indexInA / layerNodeCount_E; // SVO_G nodeIndex
        uint nodeSVOIndex_currentLayer_E = indexInA - nodeLable_G * layerNodeCount_E;

        uint weightCount_perSVO_E = groupTotalNodeCount_E + 8;
        uint nodeWeightIndex = nodeSVOIndex_currentLayer_E;
        uint weightIndex = outgoingIndex * indivisibleNodeCount_G * weightCount_perSVO_E +
                           nodeLable_G * weightCount_perSVO_E + nodeWeightIndex;
        float weight = 0.0f;
        if (nodeWeightIndex < 6) {
            uint childNodeWeightIndex = nodeWeightIndex * 8 + 8;
            uint weightSumIndex = (weightIndex - nodeWeightIndex + childNodeWeightIndex) / 8;
            weight = SVOWeightSums[weightSumIndex];
        }

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

        /*
        {
            bool printfAble = pushConst.frameIndex == 0;
            printfAble &= nodeLable_G + 1 == 137;
            printfAble &= outgoingIndex == 41;

            bool printfAble3 = printfAble && currentLayerIndex == 0;
            if (printfAble3) {
                uint childNodeWeightIndex = nodeWeightIndex * 8 + 8;
                uint weightSumIndex = (weightIndex - nodeWeightIndex + childNodeWeightIndex) / 8;
                printf("currentLayerIndex: %d, nodeIndex_E: %d, weight: %f, probability: %f, weightIndex: %d,  weightSumIndex: %d\n",
                       currentLayerIndex, nodeWeightIndex, weight, probability, weightIndex, weightSumIndex
                );
            }
        }
        */
    }

    if (threadIndex == 0 && currentLayerIndex > 0) {
        uint upperLayerNodeCount_E = currentLayerIndex == 1 ? 8 : SVOGlobalInfo[0].layerInfos_E[currentLayerIndex - 2].divisibleNodeCount * 8;
        GlobalInfo[0].cmd.x = (OUTGOING_COUNT * groupIndivisibleNodeCount_G * upperLayerNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    }
}

#ifndef NDEBUG
//------------------------------------------------debug-------------------------------------------------
/*
This debug shader will highlight (in red) the node_G where the samplePos set in pushConstant is located.
Based on the set outgoing, the vertices in the scene will be colored according to the probability of the node_E they belong to,
thus displaying the selection of node_G where the samplePos is located during pathGuiding
*/
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getSampleNodeInfo(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    if (globalInfo.indivisibleNodeCount_G == 0) return;

    int SVOMaxLayer = globalInfo.SVOMaxLayer_G;
    SVOLayerInfo layerInfos_G[MAX_OCTREE_DEPTH];
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

    if (globalInfo.indivisibleNodeCount_G == 0 ||  sampleNodeAABB.minimum.x > 100000.0f) return { float4(0.0f) };

    float3 nodeCenter_G = (globalInfo.sampelNodeAABB.maximum + globalInfo.sampelNodeAABB.minimum) * 0.5f;

    float3 view = normalize(pushConst.sceneInfoAddress[0].cameraPosition - pos);
    float probability = rand(input.randomSeed);
    //uint outgoingIndex = inverseSF(view, OUTGOING_COUNT, probabality);
    uint outgoingIndex = inverseSF(view, OUTGOING_COUNT);

    // uint outgoingIndex = inverseSF(normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter_G), OUTGOING_COUNT);
    uint totalNodeCount_E = globalInfo.totalNodeCount_E + 8;
    uint weightIndex = outgoingIndex * globalInfo.indivisibleNodeCount_G * totalNodeCount_E +
                       (globalInfo.sampelNodeLabel - 1) * totalNodeCount_E;

    SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];

    uint normalIndex = getNormalIndex(input.worldNormal);

    SVONodeData_E nodeData;
    bool hitNode = false, hitIndivisibleNode = false;
    uint nodeIndex = normalIndex * 8, nodeSum = 0, nodeOffset = 0;
    float pdf = SVOWeights[weightIndex + normalIndex];
    for (int layerIndex = 1; layerIndex <= pushConst.clusterLayer_E; ++layerIndex) {
        for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            nodeData = SVO_E[nodeIndex + nodeOffset];
            if (checkPointInAABB(input.worldPos, nodeData.aabb)) {
                pdf *= SVOWeights[weightIndex + nodeIndex + nodeOffset + 8];
                nodeSum += svoGlobalInfo.layerInfos_E[layerIndex - 1].divisibleNodeCount * 8;

                if (nodeData.indivisible == 1) hitIndivisibleNode = true;
                else nodeIndex = nodeSum + (nodeData.label - 1) * 8;

                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode || pdf == 0.0f) break;
    }
    if (!hitIndivisibleNode) return { float4(0.0f) };

    probability *= nodeData.pdf;
    return { float4(0.0f, probability * 10.0f, 0.0f, 1.0f) };
}
#endif