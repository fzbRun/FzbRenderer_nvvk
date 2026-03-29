#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight)]] StructuredBuffer<SVOLayerInfo> SVOLayerInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight)]] StructuredBuffer<SVOLayerInfo> SVOLayerInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eGlobalInfo_SVOWeight)]] RWStructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_E_SVOWeight)]] RWStructuredBuffer<SVOIndivisibleNodeInfo> SVO_IndivisibleNodeInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeightSampleCounts_SVOWeight)]] RWStructuredBuffer<uint32_t> SVOWeightCounts;
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
    float3x3 TBN;
}

//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared uint groupSVOLayerMaxLayer_G;
groupshared uint groupSVOLayerMaxLayer_E;
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_OCTREE_DEPTH];
groupshared SVOLayerInfo groupSVOLayerInfos_E[MAX_OCTREE_DEPTH];
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupIndivisibleNodeCount_E;
groupshared uint groupTotalNodeCount_E;
groupshared uint groupMaxLayerNodeCount_E;
groupshared SVONodeData_E groupNodeC[(THREADGROUP_SIZE + OUTGOING_COUNT * HITTEST_COUNT - 1) / (OUTGOING_COUNT * HITTEST_COUNT)];
groupshared uint groupNodeC_SVONodeIndex_E[(THREADGROUP_SIZE + OUTGOING_COUNT * HITTEST_COUNT - 1) / (OUTGOING_COUNT * HITTEST_COUNT)];
groupshared SVONodeInfo_G groupNodeB_first3Layer[SVONodeCount_G_Layer1 + SVONodeCount_G_Layer2 + SVONodeCount_G_Layer3]; // 18.25 KB
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

void traceRay(RayDesc ray, inout HitPayload_SVOWeight payload)
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
        payload.hit = 1.0f;

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));

        payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        float3 tangent, bitangent;
        orthonormalBasis(payload.normal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.normal); // column matrix
    }
    else payload.hit = 0.0f;
}

inline bool checkPointInAABB(float3 pos, AABB aabb) {
    return all((pos - aabb.minimum >= -0.00001f) && (aabb.maximum - pos >= -0.0001f));
}
bool hitSVONodeTest_useSharedMemory(inout uint layerIndex, inout uint nodeIndex, inout uint layerNodeSum, inout uint nodeLabel, inout bool hitIndivisibleNode, float3 hitPos) {
    bool hitNode = false;
    for (; layerIndex <= 3; ++layerIndex) {
        hitNode = false;
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            SVONodeInfo_G nodeInfo_G = groupNodeB_first3Layer[layerNodeSum + nodeIndex + nodeOffset];
            if (checkPointInAABB(hitPos, nodeInfo_G.aabb)) {
                if (nodeInfo_G.indivisible == 1) {
                    hitIndivisibleNode = true;
                    nodeIndex += nodeOffset;
                    nodeLabel += nodeInfo_G.label - 1;
                } else {
                    nodeIndex = (nodeInfo_G.label - 1) * 8;
                    nodeLabel += groupSVOLayerInfos_G[layerIndex - 1].indivisibleNodeCount;
                }
                layerNodeSum += groupSVOLayerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode) break;
    }
    return hitNode;
}
bool hitSVONodeTest(inout uint layerIndex, inout uint nodeIndex, inout uint layerNodeSum, inout uint nodeLabel, inout bool hitIndivisibleNode, float3 hitPos) {
    bool hitNode = false;
    for (; layerIndex <= groupSVOLayerMaxLayer_G; ++layerIndex) {
        hitNode = false;
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            SVONodeData_G nodeData = SVO_G[layerIndex][nodeIndex + nodeOffset];
            if (checkPointInAABB(hitPos, nodeData.aabb)) {
                if (nodeData.indivisible == 1) {
                    hitIndivisibleNode = true;
                    nodeIndex += nodeOffset;
                    nodeLabel += nodeData.label - 1;
                } else {
                    nodeIndex = (nodeData.label - 1) * 8;
                    nodeLabel += groupSVOLayerInfos_G[layerIndex - 1].indivisibleNodeCount;
                }
                layerNodeSum += groupSVOLayerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode) break;
    }
    return hitNode;
}
float3 getBSDF(BSDFMaterial material, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    float cosineON = dot(outgoing, normal);
    float3 bsdf = float3(0.0f);
    if (material.type == MaterialType::Diffuse) bsdf = M_1_PI * dot(incidence, normal);
    else if (material.type == MaterialType::Conductor) {
        float3 h = normalize(incidence + outgoing);
        bsdf = cosineON > 0.0f && dot(h, normal) < 0.01f ? 1.0f : 0.0f;
    }
    else if (material.type == MaterialType::Dielectric) {
        float eta = material.eta.x;
        if (cosineON < 0) {
            cosineON = -cosineON;
            normal = -normal;
            eta = 1.0f / eta;
        }
        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineON); // cosineON = cosineOH
        float F_oneChanel = 0.299 * F.x + 0.587 * F.y + 0.114 * F.z;

        if (dot(incidence, normal) > 0.0f) {
            float3 h = normalize(incidence + outgoing);
            if (abs(dot(h, normal)) > 0.99f) bsdf = F;
        } else {
            float3 h = normalize(eta * outgoing + incidence);
            if (abs(dot(h, normal)) > 0.99f) bsdf = 1.0f - F;
        }
    } else if (material.type == MaterialType::RoughConductor) {
        if (cosineON <= 0.0f) return bsdf;
        float3 h = normalize(incidence + outgoing);
        float3 h_tangentSpace = mul(TBN, h);
        if (h_tangentSpace.z == 0.0f) return bsdf;

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        if (cosineOH <= 0) return bsdf;

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / h_tangentSpace.z;

        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineOH);

        float3 incidence_tangentSpace = mul(TBN, incidence);
        if (incidence_tangentSpace.z <= 0) return bsdf;

        float G1_outgoing, G1_incidence;
        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(material.roughness));
        if (G2 <= 0.0f) return bsdf;

        bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    } else if (material.type == MaterialType::RoughDielectric) {
        float eta = material.eta.x;
        if (cosineON < 0) {
            cosineON = -cosineON;
            normal = -normal;
            eta = 1.0f / eta;
        }

        bool refraction = dot(incidence, normal) < 0.0f;
        float3 h;
        if (refraction) h = normalize(incidence + normal * eta);
        else h = normalize(incidence + outgoing);

        float3 h_tangentSpace = mul(TBN, h);
        if (h_tangentSpace.z <= 0.0f) return bsdf;

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        if (cosineOH <= 0.0f) return bsdf;

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / max(h_tangentSpace.z, 0.001f);

        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineOH);

        h_tangentSpace = faceforward(h_tangentSpace, outgoing_tangentSpace, -h_tangentSpace); // dot(I, nerf) < 0 return n;else return -n
        float3 incidence_tangentSpace = mul(TBN, incidence);

        float G1_outgoing, G1_incidence;
        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(material.roughness));
        if(G2 <= 0.0f) return bsdf;
        if (refraction) {
            float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
            bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
        } else
            bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    }
    return bsdf;
}
//--------------------------------------------------compute shader-----------------------------------------
//computeMain_getIndivisibleNode_G get SVO_G's indivisibleNode's info for reduce thread wastage(except divisibleNode)
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getIndivisibleNode_G(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
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

        uint indivisibleNodeCount_E = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_E) indivisibleNodeCount_E = groupSVOLayerInfos_E[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_E = WaveActiveSum(indivisibleNodeCount_E);

        uint totalNodeCount_E = 0;
        if (groupThreadIndex <= groupSVOLayerMaxLayer_E - 1) totalNodeCount_E = groupSVOLayerInfos_E[groupThreadIndex].divisibleNodeCount * 8;
        totalNodeCount_E = WaveActiveSum(totalNodeCount_E);

        if (groupThreadIndex == 0) {
            groupIndivisibleNodeCount_G = indivisibleNodeCount_G;
            groupIndivisibleNodeCount_E = indivisibleNodeCount_E;
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
                    printf("getIndivisibleNode_G\ncmd:%d %d %d\nindivisibleNodeCount: G: %d  E: %d\nSVOMaxLayer: G: %d  E:     %d\ntotalNodeCount_E: %d\n",
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

    if (threadIndex >= groupIndivisibleNodeCount_E) {   //threadCount = bufferSize, so will not overflow
        SVOIndivisibleNodeInfo nodeInfo = { 0, 0 };
        SVO_IndivisibleNodeInfos_E[threadIndex] = nodeInfo;
    }
    if (threadIndex >= groupTotalNodeCount_E) return;
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

[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_initWeights(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex == 0)
        groupMaxLayerNodeCount_E = SVOLayerInfos_E[groupSVOLayerMaxLayer_E - 1].divisibleNodeCount * 8;
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex >= OUTGOING_COUNT * groupIndivisibleNodeCount_G * groupTotalNodeCount_E) return;
    SVOWeightCounts[threadIndex] = 0;
    SVOWeights[threadIndex] = 0.0f;

    //if (threadIndex >= OUTGOING_COUNT * groupIndivisibleNodeCount_G * groupMaxLayerNodeCount_E) return;
    //SVOWeightSums[threadIndex] = 0.0f;

    if (threadIndex == 0)
        GlobalInfo[0].cmd.x = (groupIndivisibleNodeCount_E * OUTGOING_COUNT * HITTEST_COUNT + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
}

/*
computeMain_getWeights will get the weight from node A B C
A is the the node where the starting point of the emission vector is located，the SVO_G's indivisible node
B is the SV _G's indivisible node;
C is the SVO_E's node;

but If we consider all A, that is,
all indivisibleNodes of SVO_G, the computational and storage resources required for weight calculation would be exces ively large. 
Therefore, we only consider N outgoing vectors, and finally perform interpolation among these N outgoing vectors during pathGuiding

Similarly, we do not perform collision detection for every B and C. Instead, we let each C emit M rays outward and calculate the weight
for each hit on B; this is because we do not need to consider the weight of invisible B and C

Therefore, we require a total of ACount * BCount * CCount threads,
with each thread emitting a ray and storing the obtained weight in the corresponding location
*/
[numthreads(THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    fillGroupshared(groupThreadIndex);
    if (groupThreadIndex < 8) {
        groupSVOLayerInfos_G[groupThreadIndex] = SVOLayerInfos_G[groupThreadIndex];
        groupSVOLayerInfos_E[groupThreadIndex] = SVOLayerInfos_E[groupThreadIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    uint startIndex = 0;
    for (int layerIndex = 1; layerIndex <= 3; ++layerIndex) {
        uint layerNodeCount_G = groupSVOLayerInfos_G[layerIndex - 1].divisibleNodeCount * 8;
        for (int i = groupThreadIndex; i < layerNodeCount_G; i += THREADGROUP_SIZE) {
            SVONodeData_G nodeData = SVO_G[layerIndex][i];
            SVONodeInfo_G nodeInfo;
            nodeInfo.indivisible = nodeData.indivisible;
            nodeInfo.label = nodeData.label;
            nodeInfo.aabb = nodeData.aabb;
            groupNodeB_first3Layer[startIndex + groupThreadIndex] = nodeInfo;
        }
        startIndex += layerNodeCount_G;
    }

    const uint groupNodeCCount = (THREADGROUP_SIZE + OUTGOING_COUNT * HITTEST_COUNT - 1) / (OUTGOING_COUNT * HITTEST_COUNT);
    const uint nodeCGroupCount = OUTGOING_COUNT * HITTEST_COUNT / THREADGROUP_SIZE;
    if (groupThreadIndex < groupNodeCCount) {
        uint nodeCIndex = nodeCGroupCount <= 1 ? 
                          threadGroupIndex * groupNodeCCount + groupThreadIndex : 
                          threadGroupIndex / nodeCGroupCount;
        SVOIndivisibleNodeInfo nodeInfo_E = SVO_IndivisibleNodeInfos_E[nodeCIndex];
        SVONodeData_E nodeData_E = SVO_E[nodeInfo_E.layerIndex][nodeInfo_E.nodeIndex];
        groupNodeC[groupThreadIndex] = nodeData_E;

        uint nodeEIndex = nodeInfo_E.nodeIndex;
        for (int layerIndex = 0; layerIndex < nodeInfo_E.layerIndex - 1; ++layerIndex) nodeEIndex += groupSVOLayerInfos_E[layerIndex].divisibleNodeCount * 8;
        groupNodeC_SVONodeIndex_E[groupThreadIndex] = nodeEIndex;
    }
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex == 0) {
        uint SVOMaxLayerNodeCount = groupSVOLayerInfos_E[groupSVOLayerMaxLayer_E - 1].divisibleNodeCount * 8;
        uint threadTotalCount = OUTGOING_COUNT * groupIndivisibleNodeCount_G * SVOMaxLayerNodeCount;
        GlobalInfo[0].cmd.x = (threadTotalCount + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    }
    if (threadIndex >= groupIndivisibleNodeCount_E * OUTGOING_COUNT * HITTEST_COUNT) return;

    uint nodeCGroupIndex = groupThreadIndex / (OUTGOING_COUNT * HITTEST_COUNT);
    SVONodeData_E nodeData_E = groupNodeC[nodeCGroupIndex];

    float3 meanNormal = nodeData_E.meanNormal.xyz;
    float frontSampleCount = HITTEST_COUNT * (0.5f + 0.5f * length(meanNormal));
    meanNormal = normalize(meanNormal);
    float3 bitangent = abs(meanNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 tangent = cross(bitangent, meanNormal);
    bitangent = cross(meanNormal, tangent);
    float3x3 TBN = float3x3(tangent, bitangent, meanNormal);

    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, pushConst.frameIndex));

    uint hitTestIndex = threadIndex & (HITTEST_COUNT - 1);

    // mean sample hemisphere
    float r1 = rand(seed); float r2 = rand(seed);
    float cosTheta = r1; float phi = 2.0f * M_PI * r2;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float3 sampleDir = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    sampleDir = mul(sampleDir, TBN); // slang is right multiplication
    if (hitTestIndex >= frontSampleCount) sampleDir = -sampleDir;

    RayDesc ray;
    ray.Direction = sampleDir;
    // get startPoint from aabb
    ray.Origin = (nodeData_E.aabb.minimum + nodeData_E.aabb.maximum) * 0.5f;
    float3 halfSize = (nodeData_E.aabb.maximum - nodeData_E.aabb.minimum) * 0.5f;
    float3 invDir = 1.0f / max(abs(sampleDir), 0.001f);
    float3 tCandidate = halfSize * invDir;
    float t = min(min(tCandidate.x, tCandidate.y), tCandidate.z);
    ray.Origin += sampleDir * t + ray.Direction * 0.0001f;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_SVOWeight hitPayload;
    traceRay(ray, hitPayload);
    if (hitPayload.hit == 0.0f) return;
    float cosine = dot(-sampleDir, hitPayload.normal);

    uint outgoingIndex = (threadIndex & (OUTGOING_COUNT * HITTEST_COUNT - 1)) / HITTEST_COUNT;
    float3 outgoing = fibSpherePoint(outgoingIndex, OUTGOING_COUNT);

    if (hitPayload.material.type != MaterialType::Dielectric && hitPayload.material.type != MaterialType::RoughDielectric) {
        if (cosine <= 0.0f || dot(outgoing, hitPayload.normal) <= 0.0f) return;
    }

    uint layerIndex = 1; uint nodeIndex = 0; uint layerNodeSum = 0; uint nodeLabel = 0;
    bool hitIndivisibleNode = false;
    bool hitNode = hitSVONodeTest_useSharedMemory(layerIndex, nodeIndex, layerNodeSum, nodeLabel, hitIndivisibleNode, hitPayload.hitPos);
    if (hitNode) {
        if (!hitIndivisibleNode)
            hitSVONodeTest(layerIndex, nodeIndex, layerNodeSum, nodeLabel, hitIndivisibleNode, hitPayload.hitPos);
        if (hitIndivisibleNode) {
            float3 bsdf = getBSDF(hitPayload.material, -sampleDir, hitPayload.normal, outgoing, TBN);
            float3 weight3 = length(nodeData_E.irradiance.xyz) * cosine * bsdf; // notIgnoreRatio is made by irradiance length, so use length here;
            float weight = 0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z;

            uint nodeEIndex = groupNodeC_SVONodeIndex_E[nodeCGroupIndex];
            uint resultIndex = outgoingIndex * groupIndivisibleNodeCount_G * groupTotalNodeCount_E +
                               nodeLabel * groupTotalNodeCount_E + nodeEIndex;

            InterlockedAdd(SVOWeightCounts[resultIndex], 1);
            InterlockedAdd<float>(SVOWeights[resultIndex], weight);
        }
    }
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

    uint nodeAIndex = threadIndex / (groupIndivisibleNodeCount_G * layerNodeCount_E); // outgoing Index
    uint indexInA = threadIndex - nodeAIndex * (groupIndivisibleNodeCount_G * layerNodeCount_E);
    uint nodeBIndex = indexInA / layerNodeCount_E;              // SVO_G nodeIndex
    uint nodeCIndex = indexInA - nodeBIndex * layerNodeCount_E; // SVO_E currentLayer nodeIndex

    SVONodeData_E nodeData_E = SVO_E[currentLayerIndex][nodeCIndex];

    float weight = 0.0f;
    uint weightIndex = nodeAIndex * groupIndivisibleNodeCount_G * groupTotalNodeCount_E +
                       nodeBIndex * groupTotalNodeCount_E +
                       groupPreviousLayerNodeCountSum_E + nodeCIndex;
    uint weightSumArraySize = pushConst.sizes[groupSVOLayerMaxLayer_E - 1];
    if (nodeData_E.indivisible == 0) { // divisible node's weight is sum of childNode's weight
        uint weightSumIndex = nodeAIndex * groupIndivisibleNodeCount_G * weightSumArraySize +
                              nodeBIndex * weightSumArraySize + nodeData_E.label - 1;
        weight = SVOWeightSums[weightSumIndex];
    } else {
        weight = SVOWeights[weightIndex];
        uint sampleCount = SVOWeightCounts[weightIndex];
        weight /= max(sampleCount, 1);
    }

    uint warplane = groupThreadIndex & (WARP_SIZE - 1);
    float weightSum = weight;
    for (int offset = 4; offset > 0; offset /= 2) {
        uint targetLane = warplane + offset;
        weightSum += WaveReadLaneAt(weightSum, targetLane);
    }
    float prbability = weight / max(weightSum, 1.0f);
    SVOWeights[weightIndex] = prbability;

    uint fatherNodeLabel = nodeCIndex / 8;
    fatherNodeLabel += nodeAIndex * groupIndivisibleNodeCount_G * weightSumArraySize + nodeBIndex * weightSumArraySize;
    if ((warplane & 7u) == 0) SVOWeightSums[fatherNodeLabel] = weightSum;
    
    if (threadIndex == 0) {
        if (currentLayerIndex > 1) {
            uint upperLayerNodeCount_E = SVOLayerInfos_E[currentLayerIndex - 2].divisibleNodeCount * 8;
            GlobalInfo[0].cmd.x = (OUTGOING_COUNT * groupIndivisibleNodeCount_G * upperLayerNodeCount_E + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
        } else GlobalInfo[0].cmd.x = 1;
    }
    /*
    if (pushConst.frameIndex == 1 && nodeAIndex == 11 && nodeBIndex == 128 && pushConst.countdown == 0 && nodeCIndex >= 24 && nodeCIndex <= 31) {
        printf("nodeInde_E: %d, groupPreviousLayerNodeCountSum_E: %d, nodeCIndex: %d weight: %f, weightSum: %f    fatherNodeLabel: %d\n",
               groupPreviousLayerNodeCountSum_E + nodeCIndex,     
               groupPreviousLayerNodeCountSum_E, nodeCIndex, weight,
               weightSum, fatherNodeLabel);
    }
    if (pushConst.frameIndex == 1 && nodeAIndex == 11 && nodeBIndex == 128 && pushConst.countdown == 1 && nodeCIndex == 3) {
        printf("indivisible: %d, weight: %f, weightSumIndex: %d\n", nodeData_E.indivisible, weight,
                nodeAIndex * groupIndivisibleNodeCount_G * weightSumArraySize + nodeData_E.label - 1);
    }
    */
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
                if (pushConst.frameIndex == 1) {
                    printf("layerIndex; %d,  nodeIndex: %d, indivisible: %d\n", layerIndex, nodeIndex + nodeOffset, nodeData.indivisible);
                }

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

    if (hitIndivisibleNode)
        GlobalInfo[0] .sampelNodeAABB = nodeData.aabb;
    else {
        AABB aabb;
        aabb.minimum.x = 3.402823466e+38F;
        aabb.minimum.y = 3.402823466e+38F;
        aabb.minimum.z = 3.402823466e+38F;
        aabb.maximum.x = -3.402823466e+38F;
        aabb.maximum.y = -3.402823466e+38F;
        aabb.maximum.z = -3.402823466e+38F;
        GlobalInfo[0].sampelNodeAABB = aabb;
        GlobalInfo[0].sampelNodeLabel = nodeLabel;
    }

    if (pushConst.frameIndex == 1) {
        AABB aabb = GlobalInfo[0].sampelNodeAABB;
        printf("sampleNodeAABB:\n %f %f %f\n%f %f %f", 
            aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
            aabb.maximum.x, aabb.maximum.y, aabb.maximum.z);
    }
        
}

struct VSout {
    float4 sv_position : SV_Position;
    float3 worldPos;
    float2 uv;
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
    return output;
}
[shader("pixel")]
PSout fragmentMain(VSout input) {
    float3 pos = input.worldPos;
    float3 color;

    SVOWeightGlobalInfo globalInfo = GlobalInfo[0];
    AABB sampleNodeAABB = globalInfo.sampelNodeAABB;
    if (checkPointInAABB(pos.xyz, sampleNodeAABB)) return { float4(1.0f, 0.0f, 0.0f, 1.0f) };

    uint outgoingIndex = inverseSF(normalize(pushConst.sceneInfoAddress[0].cameraPosition - input.worldPos), OUTGOING_COUNT);
    uint weightIndex = outgoingIndex * globalInfo.indivisibleNodeCount_G * globalInfo.totalNodeCount_E +
                       globalInfo.sampelNodeLabel * globalInfo.totalNodeCount_E;

    SVONodeData_E nodeData;
    float probability = 1.0f;
    uint layerIndex = 1, nodeIndex = 0;
    bool hitNode = false, hitIndivisibleNode = false;
    for (; layerIndex <= globalInfo.SVOMaxLayer_E; ++layerIndex) {
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
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
        if (!hitNode || hitIndivisibleNode) break;
    }
    if (!hitIndivisibleNode) return { float4(0.0f) };

    probability *= nodeData.pdf;
    return { float4(0.0f, probability, 0.0f, 1.0f) };
}
#endif