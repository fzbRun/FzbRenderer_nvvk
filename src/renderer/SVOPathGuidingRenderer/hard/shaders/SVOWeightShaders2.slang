#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOGlobalInfo_SVOWeight)]] StructuredBuffer<SVOGlobalInfo_SVOPG> SVOGlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_E_SVOWeight)]] RWStructuredBuffer<uint> SVO_IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_HitIndivisibleNodePairInfos_SVOWeight)]] RWStructuredBuffer<uint2> SVO_HitIndivisibleNodePairInfos;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_HitIndivisibleNodePairCount_SVOWeight)]] RWStructuredBuffer<uint> SVO_HitIndivisibleNodePairCount;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_HitDivisibleNodePairInfos_SVOWeight)]] RWStructuredBuffer<uint4> SVO_HitDivisibleNodePairInfos;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_HitDivisibleNodePairCount_SVOWeight)]] RWStructuredBuffer<uint2> SVO_HitDivisibleNodePairCount;
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
};

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
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared SVOWeightGlobalInfo groupGlobalInfo;

groupshared uint groupHitDivisibleNodePairCount;
groupshared uint groupPreviousLayerNodeCountSum_G;
groupshared uint groupIndivisibleNodePairCount;
groupshared uint groupDivisibleNodePairCount;
groupshared uint groupIndivisibleNodePairLabel;
groupshared uint groupDivisibleNodePairLabel;

groupshared uint groupFatherLayerDivisibleNodeCount_E;
groupshared uint groupCurrentLayerNodeCountSum_E;
groupshared uint groupPreviousLayerNodeCountSum_E;

groupshared uint groupHitIndivisibleNodeCount_G;
groupshared IndivisibleNodeData_E groupNodeData_E[(THREADGROUP_SIZE + OUTGOING_COUNT - 1) / (OUTGOING_COUNT) + 1];
groupshared IndivisibleNodeData_G groupNodeData_G[(THREADGROUP_SIZE + OUTGOING_COUNT - 1) / (OUTGOING_COUNT) + 1];
//--------------------------------------------------compute shader-----------------------------------------
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getIndivisibleNode(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < WARP_SIZE) {
        SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];

        int SVOLayerMaxLayer_G = MAX_OCTREE_DEPTH;
        int SVOLayerMaxLayer_E = MAX_OCTREE_DEPTH;
        if (groupThreadIndex < MAX_OCTREE_DEPTH) {
            groupGlobalInfo.layerInfos_E[groupThreadIndex] = svoGlobalInfo.layerInfos_E[groupThreadIndex];

            if (groupThreadIndex == 0) {
                for (int i = 1; i < MAX_OCTREE_DEPTH; ++i) {
                    if (svoGlobalInfo.layerInfos_G[i].divisibleNodeCount == 0) {
                        SVOLayerMaxLayer_G = i;
                        break;
                    }
                }

                for (int i = 1; i < MAX_OCTREE_DEPTH; ++i) {
                    if (svoGlobalInfo.layerInfos_E[i].divisibleNodeCount == 0) {
                        SVOLayerMaxLayer_E = i;
                        break;
                    }
                }
                groupGlobalInfo.SVOMaxLayer_E = SVOLayerMaxLayer_E;
            }
        }
        SVOLayerMaxLayer_G = WaveReadLaneAt(SVOLayerMaxLayer_G, 0);
        SVOLayerMaxLayer_E = WaveReadLaneAt(SVOLayerMaxLayer_E, 0);

        uint indivisibleNodeCount_G = 0;
        if (groupThreadIndex <= SVOLayerMaxLayer_G) indivisibleNodeCount_G = svoGlobalInfo.layerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);

        uint indivisibleNodeCount_E = 0;
        if (groupThreadIndex <= SVOLayerMaxLayer_E) indivisibleNodeCount_E = svoGlobalInfo.layerInfos_E[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_E = WaveActiveSum(indivisibleNodeCount_E);

        if (groupThreadIndex == 0) {
            groupGlobalInfo.indivisibleNodeCount_G = indivisibleNodeCount_G;
            groupGlobalInfo.indivisibleNodeCount_E = indivisibleNodeCount_E;
            groupGlobalInfo.totalNodeCount_E = svoGlobalInfo.totalNodeCount_G;

            if (threadIndex == 0) {
                SVOWeightGlobalInfo globalInfo = groupGlobalInfo;
                for (int i = 0; i < MAX_OCTREE_DEPTH; ++i)
                    globalInfo.layerInfos_G[i] = svoGlobalInfo.layerInfos_G[i];
                globalInfo.SVOMaxLayer_G = SVOLayerMaxLayer_G;

                DispatchIndirectCommand cmd;
                cmd.x = (OUTGOING_COUNT * indivisibleNodeCount_G * svoGlobalInfo.totalNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
                cmd.y = 1; cmd.z = 1;
                globalInfo.cmd = cmd;
                GlobalInfo[0] = globalInfo;

                #ifndef NDEBUG
                if (pushConst.frameIndex == 0) {
                    printf("getIndivisibleNode_G\ncmd:%d %d %d\nindivisibleNodeCount: G: %d  E: %d\nSVOMaxLayer: G:       %d  E: %d\ntotalNodeCount_E: %d\n",
                           cmd.x, cmd.y, cmd.z,
                        indivisibleNodeCount_G, indivisibleNodeCount_E,
                        SVOLayerMaxLayer_G, SVOLayerMaxLayer_E,
                        svoGlobalInfo.totalNodeCount_E
                    );
                    for (int i = 0; i <= SVOLayerMaxLayer_G; ++i) {
                        printf("G layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               svoGlobalInfo.layerInfos_G[i].divisibleNodeCount, svoGlobalInfo.layerInfos_G[i].indivisibleNodeCount
                        );
                    }
                    for (int i = 0; i <= SVOLayerMaxLayer_E; ++i) {
                        printf("E layer%d: divisibleNodeCount: %d   indivisibleNodeCount: %d\n", i,
                               svoGlobalInfo.layerInfos_E[i].divisibleNodeCount, svoGlobalInfo.layerInfos_E[i].indivisibleNodeCount
                        );
                    }
                }
                #endif
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupGlobalInfo.indivisibleNodeCount_G > SVOIndivisibleNodeCount_G) {
        if (threadIndex == 0) {
            printf("Error: G indivisibleNodeCount_G exceeds the maximum node count. maxNodeCount: %u,     currentNodeCount: %u\n",
                   SVOIndivisibleNodeCount_G, GlobalInfo[0].indivisibleNodeCount_G);
            GlobalInfo[0].indivisibleNodeCount_G = 0;
            GlobalInfo[0].indivisibleNodeCount_E = 0;
            GlobalInfo[0].cmd.x = 0;
        }
        return;
    }
    if (groupGlobalInfo.indivisibleNodeCount_E > SVOIndivisibleNodeCount_E) {
        if (threadIndex == 0) {
            printf("Error: G indivisibleNodeCount_E exceeds the maximum node count. maxNodeCount: %u,      currentNodeCount: %u\n",
                   SVOIndivisibleNodeCount_E, GlobalInfo[0].indivisibleNodeCount_E);
            GlobalInfo[0].indivisibleNodeCount_G = 0;
            GlobalInfo[0].indivisibleNodeCount_E = 0;
            GlobalInfo[0].cmd.x = 0;
        }
        return;
    }

    if (threadIndex < groupGlobalInfo.totalNodeCount_E) {
        SVONodeData_E nodeData = SVO_E[threadIndex];
        bool hasData = nodeData.irradiance.x + nodeData.irradiance.y + nodeData.irradiance.z > 0.0f;
        bool indivisible = hasData && nodeData.indivisible == 1;
        if (indivisible) {
            int maxLayerIndex = groupGlobalInfo.SVOMaxLayer_E;
            int nodeIndex = threadIndex, labelOffset = 0;
            for (int layerIndex = 1; layerIndex <= maxLayerIndex; ++layerIndex) {
                uint layerNodeCount = groupGlobalInfo.layerInfos_E[layerIndex - 1].divisibleNodeCount * 8;
                if (nodeIndex < layerNodeCount) break;
                nodeIndex -= layerNodeCount;
                labelOffset += groupGlobalInfo.layerInfos_E[layerIndex].indivisibleNodeCount;
            }
            SVO_IndivisibleNodeInfos_E[labelOffset + nodeData.label - 1] = threadIndex;
        }
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupGlobalInfo = GlobalInfo[0];
    GroupMemoryBarrierWithGroupSync();

    int indivisibleNodeCount_E = groupGlobalInfo.indivisibleNodeCount_E;
    if (threadIndex == 0) {
        GlobalInfo[0].cmd.x = (indivisibleNodeCount_E * 6 + THREADGROUP_SIZE3 - 1) / THREADGROUP_SIZE3;
        SVO_HitIndivisibleNodePairCount[0] = 0;
        SVO_HitDivisibleNodePairCount[0] = indivisibleNodeCount_E * 6; // all indivisibleNodeE hit the root node of G;
    }
    if (threadIndex < indivisibleNodeCount_E * 6)
        SVO_HitDivisibleNodePairInfos[threadIndex] = uint4(threadIndex / 6, threadIndex % 6, 0, 0);

    int threadTotalCount = OUTGOING_COUNT * groupGlobalInfo.indivisibleNodeCount_G * groupGlobalInfo.totalNodeCount_E;
    if (threadIndex >= threadTotalCount) return;
    SVOWeights[threadIndex] = 0.0f;

    if (threadIndex >= threadTotalCount / 8) return;
    SVOWeightSums[threadIndex] = 0.0f;
}

[numthreads(THREADGROUP_SIZE3, 1, 1)]
[shader("compute")]
void computeMain_getHitNodeInfos(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupGlobalInfo = globalInfo;
        groupHitDivisibleNodePairCount = SVO_HitDivisibleNodePairCount[pushConst.currentLayer_G].x;
        uint previousLayerNodeCountSum_G = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentLayer_G - 1; ++layerIndex)
            previousLayerNodeCountSum_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
        groupPreviousLayerNodeCountSum_G = previousLayerNodeCountSum_G;

        groupIndivisibleNodePairCount = 0;
        groupDivisibleNodePairCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHitDivisibleNodePairCount * 8) return;  //after volta

    uint warplane = groupThreadIndex & (WARP_SIZE - 1);
    uint blocklane = groupThreadIndex & 7;
    uint blockFirstWarplane = warplane / 8 * 8;

    uint nodeLabel_E = 0;
    uint nodeLabel_G = 0;
    if (blocklane == 0) {
        uint pairInfoIndex = threadIndex / 8;
        uint2 hitDivisibleNodePairInfo = SVO_HitDivisibleNodePairInfos[pairInfoIndex].xy;
        nodeLabel_E = hitDivisibleNodePairInfo.x;
        nodeLabel_G = hitDivisibleNodePairInfo.y;
    }
    nodeLabel_E = WaveReadLaneAt(nodeLabel_E, blockFirstWarplane);
    nodeLabel_G = WaveReadLaneAt(nodeLabel_G, blockFirstWarplane);

    uint nodeIndex_currentLayer_G = (nodeLabel_G - 1) * 8 + blocklane;
    uint nodeIndex_G = groupPreviousLayerNodeCountSum_G + nodeIndex_currentLayer_G;
    SVONodeData_G nodeData_G = SVO_G[nodeIndex_G]; // here also can reduce by warp
    bool hasData_G = nodeData_G.meanNormal.w > 0.0f;
    bool indivisible_G = hasData_G && nodeData_G.indivisible == 1;
    bool divisible_G = hasData_G && nodeData_G.indivisible == 1;

    uint indivisibleNodePairlabel_warp = WavePrefixCountBits(indivisible_G);
    uint indivisibleNodePairCount_warp = WaveActiveCountBits(indivisible_G);
    uint indivisibleNodePairlabel_group = 0;
    if (warplane == 0) InterlockedAdd<uint>(groupIndivisibleNodePairCount, indivisibleNodePairCount_warp, indivisibleNodePairlabel_group);
    indivisibleNodePairlabel_group = WaveReadLaneFirst(indivisibleNodePairlabel_group);

    uint divisibleNodePairlabel_warp = WavePrefixCountBits(indivisible_G);
    uint divisibleNodePairCount_warp = WaveActiveCountBits(indivisible_G);
    uint divisibleNodePairlabel_group = 0;
    if (warplane == 0) InterlockedAdd<uint>(groupDivisibleNodePairCount, divisibleNodePairCount_warp, divisibleNodePairlabel_group);
    divisibleNodePairlabel_group = WaveReadLaneFirst(divisibleNodePairlabel_group);

    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) {
        InterlockedAdd<uint>(SVO_HitIndivisibleNodePairCount[0], groupIndivisibleNodePairCount, groupIndivisibleNodePairLabel);
        InterlockedAdd<uint>(SVO_HitDivisibleNodePairCount[pushConst.currentLayer_G].y, groupDivisibleNodePairCount, groupDivisibleNodePairLabel);
    }
    GroupMemoryBarrierWithGroupSync();

    if (indivisible_G) {
        uint2 hitIndivisibleNodePairInfo = uint2(nodeLabel_E, nodeIndex_G);
        SVO_HitIndivisibleNodePairInfos[groupIndivisibleNodePairLabel + indivisibleNodePairlabel_group + indivisibleNodePairlabel_warp] = hitIndivisibleNodePairInfo;
    }
    if (divisible_G) {
        uint2 hitDivisibleNodePairInfo = uint2(nodeLabel_E, nodeIndex_G);
        SVO_HitDivisibleNodePairInfos[groupDivisibleNodePairLabel + divisibleNodePairlabel_group + divisibleNodePairlabel_warp].zw = hitDivisibleNodePairInfo;
    }

    if (threadIndex == 0)
        GlobalInfo[0].cmd.x = (groupGlobalInfo.indivisibleNodeCount_E * groupGlobalInfo.layerInfos_G[pushConst.currentLayer_G].divisibleNodeCount + THREADGROUP_SIZE3 - 1) / THREADGROUP_SIZE3;
}

[numthreads(THREADGROUP_SIZE3, 1, 1)]
[shader("compute")]
void computeMain_divisibleNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupGlobalInfo = globalInfo;
        groupDivisibleNodePairCount = SVO_HitDivisibleNodePairCount[pushConst.currentLayer_G].y;
        uint previousLayerNodeCountSum_G = 0;
        for (int layerIndex = 0; layerIndex < pushConst.currentLayer_G - 1; ++layerIndex)
            previousLayerNodeCountSum_G += globalInfo.layerInfos_G[layerIndex].divisibleNodeCount * 8;
        groupPreviousLayerNodeCountSum_G = previousLayerNodeCountSum_G;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupDivisibleNodePairCount) return; // after volta

    uint2 pairInfo = SVO_HitIndivisibleNodePairInfos[threadIndex];

    uint indivisibleNodeIndex_E = SVO_IndivisibleNodeInfos_E[pairInfo.x];
    SVONodeData_E nodeData_E = SVO_E[indivisibleNodeIndex_E];
    bool singleSide_E = length(nodeData_E.meanNormal.xyz) >= 0.707;
    float3 normal_E = normalize(nodeData_E.meanNormal.xyz);
    float irradianceValue = length(nodeData_E.irradiance);

    float3 tangent, bitangent;
    orthonormalBasis(normal_E, tangent, bitangent);
    float3x3 TBN_E = float3x3(tangent, bitangent, normal_E);

    uint warplane = groupThreadIndex & (WARP_SIZE - 1);
    uint blocklane = threadIndex & 7;

    uint nodeIndex_currentLayer_G = (pairInfo.y - 1) * 8 + blocklane;
    uint nodeIndex_G = groupPreviousLayerNodeCountSum_G + nodeIndex_currentLayer_G;
    SVONodeData_G nodeData_G = SVO_G[nodeIndex_G]; // here also can reduce by warp

    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, pushConst.frameIndex));

    float3 halfSize_E = (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) * 0.5f;
    float3 nodeCenter_E = nodeData_E.aabb.minimum + halfSize_E;
    float3 halfSize_G = (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) * 0.5f;
    float3 nodeCenter_G = nodeData_G.aabb.minimum + halfSize_G;

    RayDesc ray;
    bool hit = false;
    for (int hitTestIndex = 0; hitTestIndex < HITTEST_COUNT; ++hitTestIndex) {
        float3 samplePoint_E = singleSide_E ? sample_hemisphere_mean(seed) : sample_sphere_mean(seed);
        samplePoint_E = mul(samplePoint_E, TBN_E); // slang is right multiplication
        samplePoint_E = getAABBPoint(samplePoint_E, nodeCenter_E, halfSize_E);

        float3 samplePoint_G = sample_sphere_mean(seed);
        samplePoint_G = getAABBPoint(samplePoint_G, nodeCenter_G, halfSize_G);

        float3 sampleDir = normalize(samplePoint_G - samplePoint_E);
        if (singleSide_E && dot(sampleDir, normal_E) <= 0.0f) continue;

        ray.Direction = sampleDir;
        ray.Origin = samplePoint_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;

        HitPayload_SVOWeight hitPayload;
        traceRay(ray, hitPayload, nodeData_G.aabb);

        if (hitPayload.hit > 0.0f) {
            hit = true;
            break;
        } else continue;
    }

    uint hitCount_warp = WaveActiveCountBits(hit);
    uint hitLabel_warp = WavePrefixCountBits(hit);
    uint hitLabel_group = 0;
    if (warplane == 0) InterlockedAdd<uint>(groupHitDivisibleNodePairCount, hitCount_warp, hitLabel_group);
    hitLabel_group = WaveReadLaneFirst(hitLabel_group);
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex == 0)
        InterlockedAdd<uint>(SVO_HitDivisibleNodePairCount[pushConst.currentLayer_G + 1].x, 1, groupDivisibleNodePairLabel);
    GroupMemoryBarrierWithGroupSync();

    if (hit) {
        uint2 hitDivisibleNodePairInfo = uint2(pairInfo.x, nodeData_G.label);
        SVO_HitDivisibleNodePairInfos[groupDivisibleNodePairLabel + hitLabel_group + hitLabel_warp].xy = hitDivisibleNodePairInfo;
    }

    if (threadIndex == 0) {
        uint totalHitDivisibleNodePairCount = groupGlobalInfo.indivisibleNodeCount_E * groupGlobalInfo.layerInfos_G[pushConst.currentLayer_G].divisibleNodeCount * 8;
        GlobalInfo[0].cmd.x = (totalHitDivisibleNodePairCount + THREADGROUP_SIZE3 - 1) / THREADGROUP_SIZE3;
    }
}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_inDivisibleNodeHitTest(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        groupGlobalInfo = GlobalInfo[0];
        groupHitIndivisibleNodeCount_G = SVO_HitIndivisibleNodePairCount[0];
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHitIndivisibleNodeCount_G) return;

    uint2 pairInfo = SVO_HitIndivisibleNodePairInfos[threadIndex];
    uint nodeIndex_E = SVO_IndivisibleNodeInfos_E[pairInfo.x];
    SVONodeData_E nodeData_E = SVO_E[nodeIndex_E];
    SVONodeData_G nodeData_G = SVO_G[pairInfo.y];

    float3 meanNormal_E = nodeData_E.meanNormal.xyz; bool singleSide_E = length(meanNormal_E) >= 0.707; meanNormal_E = normalize(meanNormal_E);
    float3 meanNormal_G = nodeData_G.meanNormal.xyz; bool singleSide_G = length(meanNormal_G) >= 0.707; meanNormal_G = normalize(meanNormal_G);

    float3x3 TBN_E, TBN_G;
    float3 tangent, bitangent;
    orthonormalBasis(meanNormal_E, tangent, bitangent);
    TBN_E = float3x3(tangent, bitangent, meanNormal_E);
    orthonormalBasis(meanNormal_G, tangent, bitangent);
    TBN_G = float3x3(tangent, bitangent, meanNormal_G);

    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, pushConst.frameIndex));

}

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) {
        SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
        groupGlobalInfo = globalInfo;
        int currentLayerIndex = (int)globalInfo.SVOMaxLayer_E - pushConst.countdown;
        if (currentLayerIndex <= 0) return;

        SVOGlobalInfo_SVOPG svoGlobalInfo = SVOGlobalInfo[0];

        groupFatherLayerDivisibleNodeCount_E = svoGlobalInfo.layerInfos_E[currentLayerIndex - 1].divisibleNodeCount;

        uint currentLayerNodeCountSum_E = globalInfo.totalNodeCount_E;
        for (int i = globalInfo.SVOMaxLayer_E; i > currentLayerIndex; --i)
            currentLayerNodeCountSum_E -= svoGlobalInfo.layerInfos_E[i - 1].divisibleNodeCount * 8;
        groupCurrentLayerNodeCountSum_E = currentLayerNodeCountSum_E;

        groupPreviousLayerNodeCountSum_E = currentLayerNodeCountSum_E - svoGlobalInfo.layerInfos_E[currentLayerIndex - 1].divisibleNodeCount * 8;
    }
    GroupMemoryBarrierWithGroupSync();

    int currentLayerIndex = (int)groupGlobalInfo.SVOMaxLayer_E - pushConst.countdown;
    if (currentLayerIndex <= 0) return;

    uint indivisibleNodeCount_G = groupGlobalInfo.indivisibleNodeCount_G;
    uint layerNodeCount_E = groupFatherLayerDivisibleNodeCount_E * 8;
    if (threadIndex >= OUTGOING_COUNT * indivisibleNodeCount_G * layerNodeCount_E) return;

    uint outgoingIndex = threadIndex / (indivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
    uint indexInA = threadIndex - outgoingIndex * (indivisibleNodeCount_G * layerNodeCount_E);
    uint nodeLable_G = indexInA / layerNodeCount_E;                            // SVO_G nodeIndex
    uint nodeIndex_currentLayer_E = indexInA - nodeLable_G * layerNodeCount_E; // SVO_E currentLayer nodeIndex

    int previousLayerNodeCountSum_E = groupPreviousLayerNodeCountSum_E;
    uint nodeIndex = previousLayerNodeCountSum_E + nodeIndex_currentLayer_E;
    SVONodeData_E nodeData_E = SVO_E[nodeIndex];

    uint nodeTotalCount_E = groupGlobalInfo.totalNodeCount_E;

    float weight = 0.0f;
    uint weightIndex = outgoingIndex * indivisibleNodeCount_G * nodeTotalCount_E +
                       nodeLable_G * nodeTotalCount_E +
                       nodeIndex;
    if (nodeData_E.indivisible == 0) {
        uint childNodeIndex = groupCurrentLayerNodeCountSum_E + (nodeData_E.label - 1) * 8;
        uint weightSumIndex = (weightIndex - nodeIndex + childNodeIndex) / 8;
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
    if (probability > 0.0f) SVOWeights[weightIndex] = probability;

    if (currentLayerIndex > 1) {
        if ((warplane & 7u) == 0 && weightSum > 0.0f) SVOWeightSums[weightIndex / 8] = weightSum;

        if (threadIndex == 0) {
            uint upperLayerNodeCount_E = SVOGlobalInfo[0].layerInfos_E[currentLayerIndex - 2].divisibleNodeCount * 8;
            GlobalInfo[0].cmd.x = (OUTGOING_COUNT * indivisibleNodeCount_G * upperLayerNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        }
    } else if (threadIndex == 0) GlobalInfo[0].cmd.x = 0;
}