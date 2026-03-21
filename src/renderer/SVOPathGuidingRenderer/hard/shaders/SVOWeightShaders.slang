#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight)]] StructuredBuffer<SVOLayerInfo> SVOLayerInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight)]] StructuredBuffer<SVOLayerInfo> SVOLayerInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight)]] RWStructuredBuffer<SVOIndivisibleNodeInfo> SVO_IndivisibleNodeInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_E_SVOWeight)]] RWStructuredBuffer<SVOIndivisibleNodeInfo> SVO_IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight)]] RWStructuredBuffer<float> SVOWeights;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeightSums_SVOWeight)]] RWStructuredBuffer<float> SVOWeightSums;

struct SVONodeInfo_G {
    AABB aabb;
    uint32_t indivisible;
    uint32_t label;
}

struct HitPayload_SVOWeight {
    BSDFMaterial material;
    float3 hitPos;
    float hit;
    float3 normal;
}

//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared uint groupSVOLayerMaxLayer_G;
groupshared uint groupSVOLayerMaxLayer_E;
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_OCTREE_DEPTH];
groupshared SVOLayerInfo groupSVOLayerInfos_E[MAX_OCTREE_DEPTH];
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared uint groupTotalNodeCount_G;
groupshared uint groupTotalNodeCount_E;
groupshared uint groupMaxLayerNodeCount_E;

groupshared IndivisibleNodeData_E groupNodeDatas_E[(THREADGROUP_SIZE + OUTGOING_COUNT - 1) / (OUTGOING_COUNT) + 1];
groupshared IndivisibleNodeData_G groupNodeDatas_G[(THREADGROUP_SIZE + OUTGOING_COUNT - 1) / (OUTGOING_COUNT) + 1];

groupshared SVOLayerInfo groupParentLayerInfo_E;
groupshared uint groupPreviousLayerNodeCountSum_E; // except current layer

inline void fillGroupshared(uint groupThreadIndex) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupSVOLayerMaxLayer_G = globalInfo.SVOMaxLayer_G;
        groupSVOLayerMaxLayer_E = globalInfo.SVOMaxLayer_E;
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
        if (groupThreadIndex < 8) {
            groupSVOLayerInfos_G[groupThreadIndex] = SVOLayerInfos_G[groupThreadIndex];
            groupSVOLayerInfos_E[groupThreadIndex] = SVOLayerInfos_E[groupThreadIndex];
        }

        if (groupThreadIndex == 0) {
            for (int i = 1; i < 8; ++i) {
                if (groupSVOLayerInfos_G[i].divisibleNodeCount == 0) {
                    groupSVOLayerMaxLayer_G = i;
                    break;
                }
            }
            for (int i = 1; i < 8; ++i) {
                if (groupSVOLayerInfos_E[i].divisibleNodeCount == 0) {
                    groupSVOLayerMaxLayer_E = i;
                    break;
                }
            }
        }
        WaveActiveAllTrue(true);

        uint indivisibleNodeCount_G = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_G) indivisibleNodeCount_G = groupSVOLayerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);

        uint totalNodeCount_G = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_G - 1) totalNodeCount_G = groupSVOLayerInfos_G[groupThreadIndex].divisibleNodeCount * 8;
        totalNodeCount_G = WaveActiveSum(totalNodeCount_G);

        uint indivisibleNodeCount_E = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_E) indivisibleNodeCount_E = groupSVOLayerInfos_E[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_E = WaveActiveSum(indivisibleNodeCount_E);

        uint totalNodeCount_E = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_E - 1) totalNodeCount_E = groupSVOLayerInfos_E[groupThreadIndex].divisibleNodeCount * 8;
        totalNodeCount_E = WaveActiveSum(totalNodeCount_E);

        if (groupThreadIndex == 0) {
            groupIndivisibleNodeCount_G = indivisibleNodeCount_G;
            groupIndivisibleNodeCount_E = indivisibleNodeCount_E;
            groupTotalNodeCount_G = totalNodeCount_G;
            groupTotalNodeCount_E = totalNodeCount_E;

            if (threadIndex == 0) {

                DispatchIndirectCommand cmd;
                cmd.x = (OUTGOING_COUNT * indivisibleNodeCount_G * totalNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
                cmd.y = 1; cmd.z = 1;

                SVOWeightGlobalInfo globalInfo;
                globalInfo.cmd = cmd;
                globalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
                globalInfo.indivisibleNodeCount_E = indivisibleNodeCount_E;
                globalInfo.SVOMaxLayer_G = groupSVOLayerMaxLayer_G;
                globalInfo.SVOMaxLayer_E = groupSVOLayerMaxLayer_E;
                globalInfo.totalNodeCount_E = totalNodeCount_E;

                #ifndef NDEBUG
                if (pushConst.frameIndex == 1) {
                    printf("getIndivisibleNode_G\ncmd:%d %d %d\nindivisibleNodeCount: G: %d  E: %d\nSVOMaxLayer: G: %d  E: %d\ntotalNodeCount_E: %d\n",
                           cmd.x, cmd.y, cmd.z,
                        indivisibleNodeCount_G, indivisibleNodeCount_E,
                        groupSVOLayerMaxLayer_G, groupSVOLayerMaxLayer_E,
                        totalNodeCount_E
                    );
                    for (int i = 0; i <= groupSVOLayerMaxLayer_G; ++i) {
                        printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               groupSVOLayerInfos_G[i].divisibleNodeCount, groupSVOLayerInfos_G[i].indivisibleNodeCount
                        );
                    }
                    for (int i = 0; i <= groupSVOLayerMaxLayer_E; ++i) {
                        printf("E layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               groupSVOLayerInfos_E[i].divisibleNodeCount, groupSVOLayerInfos_E[i].indivisibleNodeCount
                        );
                    }
                }
                #endif


                GlobalInfo[0] = globalInfo;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex < groupTotalNodeCount_G) {
        uint nodeLayer = 1; uint nodeIndex = threadIndex; uint nodeLabel = 0;
        for (; nodeLayer <= groupSVOLayerMaxLayer_G; ++nodeLayer) {
            uint layerNodeCount = groupSVOLayerInfos_G[nodeLayer - 1].divisibleNodeCount * 8;
            if (nodeIndex >= layerNodeCount) nodeIndex -= layerNodeCount;
            else break;
            nodeLabel += groupSVOLayerInfos_G[nodeLayer].indivisibleNodeCount;
        }

        SVONodeData_G nodeData = SVO_G[nodeLayer][nodeIndex];
        bool hasData = nodeData.meanNormal.w > 0.0f;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (indivisible) {
            SVOIndivisibleNodeInfo nodeInfo;
            nodeInfo.layerIndex = nodeLayer;
            nodeInfo.nodeIndex = nodeIndex;

            nodeLabel += nodeData.label - 1;
            SVO_IndivisibleNodeInfos_G[nodeLabel] = nodeInfo;
        }
    }

    if (threadIndex < groupTotalNodeCount_E) {
        uint nodeLayer = 1; uint nodeIndex = threadIndex; uint nodeLabel = 0;
        for (; nodeLayer <= groupSVOLayerMaxLayer_E; ++nodeLayer) {
            uint layerNodeCount = groupSVOLayerInfos_E[nodeLayer - 1].divisibleNodeCount * 8;
            if (nodeIndex >= layerNodeCount) nodeIndex -= layerNodeCount;
            else break;
            nodeLabel += groupSVOLayerInfos_E[nodeLayer].indivisibleNodeCount;
        }

        SVONodeData_E nodeData = SVO_E[nodeLayer][nodeIndex];
        bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (indivisible) {
            SVOIndivisibleNodeInfo nodeInfo;
            nodeInfo.layerIndex = nodeLayer;
            nodeInfo.nodeIndex = nodeIndex;

            nodeLabel += nodeData.label - 1;
            SVO_IndivisibleNodeInfos_E[nodeLabel] = nodeInfo;
        }
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex == 0)
        groupMaxLayerNodeCount_E = SVOLayerInfos_E[groupSVOLayerMaxLayer_E - 1].divisibleNodeCount * 8;
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex >= OUTGOING_COUNT * groupIndivisibleNodeCount_G * groupTotalNodeCount_E) return;
    SVOWeights[threadIndex] = 0.0f;

    if (threadIndex == 0)
        GlobalInfo[0].cmd.x = (groupIndivisibleNodeCount_E * OUTGOING_COUNT * groupIndivisibleNodeCount_G + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex < 8) {
        groupSVOLayerInfos_G[groupThreadIndex] = SVOLayerInfos_G[groupThreadIndex];
        groupSVOLayerInfos_E[groupThreadIndex] = SVOLayerInfos_E[groupThreadIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    uint threadCount_perNodeE = groupIndivisibleNodeCount_G * OUTGOING_COUNT;
    uint currentGroupNodeEStartIndex = threadGroupIndex * THREADGROUP_SIZE / threadCount_perNodeE;
    uint currentGroupNodeEEndIndex = ((threadGroupIndex + 1) * THREADGROUP_SIZE - 1) / threadCount_perNodeE;
    uint currentGoupNodeECount = currentGroupNodeEEndIndex - currentGroupNodeEStartIndex + 1;
    if (groupThreadIndex < currentGoupNodeECount) {
        SVOIndivisibleNodeInfo nodeInfo = SVO_IndivisibleNodeInfos_E[groupThreadIndex + currentGroupNodeEStartIndex];
        SVONodeData_E nodeData = SVO_E[nodeInfo.layerIndex][nodeInfo.nodeIndex];

        IndivisibleNodeData_E groupNodeData_E;
        groupNodeData_E.normal = nodeData.meanNormal.xyz;
        uint nodeIndex = nodeInfo.nodeIndex;
        for (int i = 0; i <= (int)nodeInfo.layerIndex - 2; ++i) nodeIndex += groupSVOLayerInfos_E[i].divisibleNodeCount * 8;
        groupNodeData_E.nodeIndex = nodeIndex;
        groupNodeData_E.irradiance = length(nodeData.irradiance);
        groupNodeData_E.aabb = nodeData.aabb;

        groupNodeDatas_E[groupThreadIndex] = groupNodeData_E;
    }

    uint threadCount_perNodeG = OUTGOING_COUNT;
    uint currentGroupNodeGStartIndex = threadGroupIndex * THREADGROUP_SIZE / threadCount_perNodeG;
    uint currentGroupNodeGEndIndex = ((threadGroupIndex + 1) * THREADGROUP_SIZE - 1) / threadCount_perNodeG;
    uint currentGoupNodeGCount = currentGroupNodeGEndIndex - currentGroupNodeGStartIndex + 1;
    if (groupThreadIndex < currentGoupNodeGCount) {
        uint nodeGIndex = (currentGroupNodeGStartIndex + groupThreadIndex) % groupIndivisibleNodeCount_G;
        SVOIndivisibleNodeInfo nodeInfo = SVO_IndivisibleNodeInfos_G[nodeGIndex];
        SVONodeData_G nodeData = SVO_G[nodeInfo.layerIndex][nodeInfo.nodeIndex];

        IndivisibleNodeData_G groupNodeData_G;
        uint nodeLabel = nodeData.label - 1;
        for (int i = 1; i < nodeInfo.layerIndex; ++i) nodeLabel += groupSVOLayerInfos_G[i].indivisibleNodeCount;
        groupNodeData_G.nodeLabel = nodeLabel;
        groupNodeData_G.normal = nodeData.meanNormal.xyz;
        groupNodeData_G.aabb = nodeData.aabb;

        groupNodeDatas_G[groupThreadIndex] = groupNodeData_G;
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex == 0) {
        uint SVOMaxLayerNodeCount = groupSVOLayerInfos_E[groupSVOLayerMaxLayer_E - 1].divisibleNodeCount * 8;
        uint threadTotalCount = OUTGOING_COUNT * groupIndivisibleNodeCount_G * SVOMaxLayerNodeCount;
        GlobalInfo[0].cmd.x = (threadTotalCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    }
    if (threadIndex >= groupIndivisibleNodeCount_E * groupIndivisibleNodeCount_G * OUTGOING_COUNT) return;

    uint groupNodeEIndex = threadIndex / threadCount_perNodeE - currentGroupNodeEStartIndex;
    IndivisibleNodeData_E nodeData_E = groupNodeDatas_E[groupNodeEIndex];

    uint groupNodeGIndex = ((threadIndex / threadCount_perNodeG) - currentGroupNodeGStartIndex) % groupIndivisibleNodeCount_G;
    IndivisibleNodeData_G nodeData_G = groupNodeDatas_G[groupNodeGIndex];

    float3 meanNormal_E = nodeData_E.normal; float normalLength_E = length(meanNormal_E); bool singleSide_E = normalLength_E >= 0.707;
    float3 meanNormal_G = nodeData_G.normal; float normalLength_G = length(meanNormal_G); bool singleSide_G = normalLength_G >= 0.707;

    meanNormal_E = normalize(meanNormal_E);
    meanNormal_G = normalize(meanNormal_G);

    float3 tangent, bitangent;
    orthonormalBasis(meanNormal_E, tangent, bitangent);
    float3x3 TBN_E = float3x3(tangent, bitangent, meanNormal_E);

    orthonormalBasis(meanNormal_G, tangent, bitangent);
    float3x3 TBN_G = float3x3(tangent, bitangent, meanNormal_G);

    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, pushConst.frameIndex));

    uint outgoingIndex = threadIndex % OUTGOING_COUNT;
    float3 outgoing = fibSpherePoint(outgoingIndex, OUTGOING_COUNT);

    float3 halfSize_E = (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) * 0.5f;
    float3 nodeCenter_E = nodeData_E.aabb.minimum + halfSize_E;
    float3 halfSize_G = (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) * 0.5f;
    float3 nodeCenter_G = nodeData_G.aabb.minimum + halfSize_G;
    RayDesc ray;
    float weight = 0.0f;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT; ++hitTestIndex) {
        float3 samplePoint_E = singleSide_E ? sample_hemisphere_mean(seed) : sample_sphere_mean(seed);
        samplePoint_E = mul(samplePoint_E, TBN_E); // slang is right multiplication
        samplePoint_E = getAABBPoint(samplePoint_E, nodeCenter_E, halfSize_E);

        float3 samplePoint_G = singleSide_G ? sample_hemisphere_mean(seed) : sample_sphere_mean(seed);
        samplePoint_G = mul(samplePoint_G, TBN_G);
        samplePoint_G = getAABBPoint(samplePoint_G, nodeCenter_G, halfSize_G);

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);

        if (singleSide_E && dot(sampleDir, meanNormal_E) <= 0.0f) continue;
        //if (singleSide_G && dot(-sampleDir, meanNormal_G) <= 0.0f) continue;
        //if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        //    if (dot(outgoing, sampleDir) <= 0.0f) continue;
        //}

        ray.Direction = sampleDir;
        ray.Origin = samplePoint_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;

        HitPayload_SVOWeight hitPayload;
        traceRay(ray, hitPayload, nodeData_G.aabb);

        //uint nodeGIndex = (threadIndex / threadCount_perNodeG) % groupIndivisibleNodeCount_G;
        //uint nodeEIndex = threadIndex / threadCount_perNodeE;
        //if (pushConst.frameIndex == 1 && outgoingIndex == 26 && nodeGIndex == 46) {
        //    printf("hit: %f, singleSide_E: %d, singleSide_G: %d\nmeanNormal_E: %f %f %f, meanNormal_G: %f %f %f\nsamplePoint_E: %f %f %f, samplePoint_G: %f %f %f,     sampleDir: %f %f  %f\nhitPos: %f %f %f\n\n",
        //           hitPayload.hit, singleSide_E, singleSide_G,
        //        meanNormal_E.x, meanNormal_E.y, meanNormal_E.z,
        //        meanNormal_G.x, meanNormal_G.y, meanNormal_G.z,
        //        samplePoint_E.x, samplePoint_E.y, samplePoint_E.z,
        //        samplePoint_G.x, samplePoint_G.y, samplePoint_G.z,
        //        sampleDir.x, sampleDir.y, sampleDir.z,
        //        hitPayload.hitPos.x, hitPayload.hitPos.y, hitPayload.hitPos.z
        //    );
        //}

        if (hitPayload.hit <= 0.0f) continue;
        float cosine = dot(-sampleDir, hitPayload.normal);

        bool isExt = true;
        if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
            //if (cosine <= 0.0f || dot(outgoing, hitPayload.normal) <= 0.0f) continue;
            if (cosine <= 0.0f) continue;
        } else if (dot(hitPayload.normal, -ray.Direction) < 0.0f) isExt = false;

        orthonormalBasis(hitPayload.normal, tangent, bitangent);
        TBN_G = float3x3(tangent, bitangent, hitPayload.normal);

        float3 bsdf = getBSDF(hitPayload.material, -sampleDir, hitPayload.normal, outgoing, TBN_G, isExt);
        float3 weight3 = nodeData_E.irradiance * cosine * bsdf; // notIgnoreRatio is made by irradiance length, so use length here;
        weight += 0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z;
    }
    weight /= HITTEST_COUNT;

    if (weight == 0.0f) return;

    uint nodeIndex_E = nodeData_E.nodeIndex;
    uint nodeLable_G = nodeData_G.nodeLabel;    //has -1
    uint weightIndex = outgoingIndex * groupIndivisibleNodeCount_G * groupTotalNodeCount_E +
                       nodeLable_G * groupTotalNodeCount_E + nodeIndex_E;
    SVOWeights[weightIndex] = weight;
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {  
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex == 0) {
        int currentLayerIndex = (int)groupSVOLayerMaxLayer_E - pushConst.countdown;
        if (currentLayerIndex <= 0) return;

        groupParentLayerInfo_E = SVOLayerInfos_E[currentLayerIndex - 1];

        uint previousLayerNodeCountSum_E = groupTotalNodeCount_E;
        for (int i = groupSVOLayerMaxLayer_E; i >= currentLayerIndex; --i)
            previousLayerNodeCountSum_E -= SVOLayerInfos_E[i - 1].divisibleNodeCount * 8;
        groupPreviousLayerNodeCountSum_E = previousLayerNodeCountSum_E;
    }
    GroupMemoryBarrierWithGroupSync();

    int currentLayerIndex = (int)groupSVOLayerMaxLayer_E - pushConst.countdown;
    if (currentLayerIndex <= 0) return;

    uint layerNodeCount_E = groupParentLayerInfo_E.divisibleNodeCount * 8;
    if (threadIndex >= OUTGOING_COUNT * groupIndivisibleNodeCount_G * layerNodeCount_E) return;

    uint outgoingIndex = threadIndex / (groupIndivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
    uint indexInA = threadIndex - outgoingIndex * (groupIndivisibleNodeCount_G * layerNodeCount_E);
    uint nodeLable_G = indexInA / layerNodeCount_E;              // SVO_G nodeIndex
    uint nodeIndex_E = indexInA - nodeLable_G * layerNodeCount_E; // SVO_E currentLayer nodeIndex

    SVONodeData_E nodeData_E = SVO_E[currentLayerIndex][nodeIndex_E];

    float weight = 0.0f;
    uint weightIndex = outgoingIndex * groupIndivisibleNodeCount_G * groupTotalNodeCount_E +
                       nodeLable_G * groupTotalNodeCount_E +
                       groupPreviousLayerNodeCountSum_E + nodeIndex_E;
    uint weightSumArraySize = pushConst.layerNodeMaxCount_E;
    if (nodeData_E.indivisible == 0) { // divisible node's weight is sum of childNode's weight
        uint weightSumIndex = outgoingIndex * groupIndivisibleNodeCount_G * weightSumArraySize +
                              nodeLable_G * weightSumArraySize + nodeData_E.label - 1;
        weight = SVOWeightSums[weightSumIndex];
    } else weight = SVOWeights[weightIndex];

    uint warplane = groupThreadIndex & (WARP_SIZE - 1);
    uint blockFirstWarplane = warplane / 8 * 8;
    float weightSum = weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = warplane + offset;
        weightSum += WaveReadLaneAt(weightSum, targetLane);
    }
    weightSum = WaveReadLaneAt(weightSum, blockFirstWarplane);
    float probability = weight / max(weightSum, 0.00001f);
    SVOWeights[weightIndex] = probability;

    uint fatherNodeLabel = nodeIndex_E / 8;
    fatherNodeLabel += outgoingIndex * groupIndivisibleNodeCount_G * weightSumArraySize + nodeLable_G * weightSumArraySize;
    if ((warplane & 7u) == 0) SVOWeightSums[fatherNodeLabel] = weightSum;
    
    if (threadIndex == 0) {
        if (currentLayerIndex > 1) {
            uint upperLayerNodeCount_E = SVOLayerInfos_E[currentLayerIndex - 2].divisibleNodeCount * 8;
            GlobalInfo[0].cmd.x = (OUTGOING_COUNT * groupIndivisibleNodeCount_G * upperLayerNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        } else GlobalInfo[0].cmd.x = 1;
    }

    //if (pushConst.frameIndex == 1 && outgoingIndex == 17 && nodeLable_G == 46) {
    //    printf("layerIndex: %d, nodeIndex: %d, probability: %f, weight: %f\n", currentLayerIndex, nodeIndex_E, probability, weight);
    //}
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
    int SVOMaxLayer = globalInfo.SVOMaxLayer_G;
    SVOLayerInfo layerInfos_G[8];
    for (int i = 0; i < SVOMaxLayer; ++i) layerInfos_G[i] = SVOLayerInfos_G[i];

    SVONodeData_G nodeData;
    int layerIndex = 1, nodeIndex = 0, nodeLabel = 0;
    bool hitNode = false, hitIndivisibleNode = false;
    for (; layerIndex <= SVOMaxLayer; ++layerIndex) {
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            nodeData = SVO_G[layerIndex][nodeIndex + nodeOffset];
            if (checkPointInAABB(pushConst.samplePos, nodeData.aabb)) {
                if (nodeData.indivisible == 1) {
                    hitIndivisibleNode = true;
                    nodeLabel += nodeData.label - 1;
                } else nodeIndex = (nodeData.label - 1) * 8;
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode) break;
        nodeLabel += layerInfos_G[layerIndex].indivisibleNodeCount;
    }

    if (hitIndivisibleNode) GlobalInfo[0].sampelNodeAABB = nodeData.aabb;
    else {
        AABB aabb;
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;
        GlobalInfo[0].sampelNodeAABB = aabb;
    }
    GlobalInfo[0].sampelNodeLabel = nodeLabel;

    if (pushConst.frameIndex == 1) {
        AABB aabb = GlobalInfo[0].sampelNodeAABB;
        printf("sampleNodeLabel: %d, sampleNodeAABB:\n %f %f %f\n%f %f %f\n", 
            nodeLabel,
            aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
            aabb.maximum.x, aabb.maximum.y, aabb.maximum.z);

        float3 nodeCenter = (aabb.maximum + aabb.minimum) * 0.5f;
        uint outgoingIndex = inverseSF(normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter), OUTGOING_COUNT);
        printf("outgoingIndex: %d\n", outgoingIndex);
    }

    //if (pushConst.frameIndex == 1) {
    //    for (int i = 0; i < 8; ++i) {
    //        AABB aabb = SVO_G[1][i].aabb;
    //        printf("layerIndex: 1, nodeIndex: %d, aabb: %f %f %f %f %f %f\n",
    //            i,
    //            aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
    //            aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
    //        );
    //    }
    //}
}

struct VSout {
    float4 sv_position : SV_Position;
    float3 worldPos;
    float2 uv;
    uint randomSeed;
}
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

    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];
    if (material.materialMapIndex.y >= 0) {
        output.uv = getAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, vertexIndex);
        output.uv.y = 1.0f - output.uv.y;
    }

    output.sv_position = mul(pos, sceneInfo.viewProjMatrix);
    output.worldPos = pos.xyz;
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

    float3 nodeCenter_G = (globalInfo.sampelNodeAABB.maximum + globalInfo.sampelNodeAABB.minimum) * 0.5f;

    float3 view = normalize(pushConst.sceneInfoAddress[0].cameraPosition - pos);
    float probabality = rand(input.randomSeed);
    uint outgoingIndex = inverseSF(view, OUTGOING_COUNT, probabality);

    //uint outgoingIndex = inverseSF(normalize(pushConst.sceneInfoAddress[0].cameraPosition - nodeCenter_G), OUTGOING_COUNT);
    uint weightIndex = outgoingIndex * globalInfo.indivisibleNodeCount_G * globalInfo.totalNodeCount_E +
                       globalInfo.sampelNodeLabel * globalInfo.totalNodeCount_E;

    SVONodeData_E nodeData;
    float probability = 1.0f;
    uint layerIndex = 1, nodeIndex = 0; int nodeOffset = 0;
    bool hitNode = false, hitIndivisibleNode = false;
    for (; layerIndex <= globalInfo.SVOMaxLayer_E; ++layerIndex) {
        for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            nodeData = SVO_E[layerIndex][nodeIndex + nodeOffset];
            if (checkPointInAABB(input.worldPos, nodeData.aabb)) {
                probability *= SVOWeights[weightIndex + nodeIndex + nodeOffset];
                weightIndex += SVOLayerInfos_E[layerIndex - 1].divisibleNodeCount * 8;

                if (nodeData.indivisible == 1) {
                    hitIndivisibleNode = true;
                } else nodeIndex = (nodeData.label - 1) * 8;
                hitNode = true;

                break;
            }
        }
        if (!hitNode || hitIndivisibleNode || probability == 0.0f) break;
    }
    if (!hitIndivisibleNode) return { float4(0.0f) };

    // if (nodeIndex + nodeOffset == 107 && layerIndex == 3) return { float4(0.0f, 0.0f, 1.0f, 1.0f) };
    // if (nodeData.label - 1 == 45 && layerIndex == 3) return { float4(0.0f, 0.0f, 1.0f, 1.0f) };

    probability *= nodeData.notIgnoreRatio;
    return { float4(0.0f, probability * 10.0f, 0.0f, 1.0f) };
}
#endif