#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"

[[vk::push_constant]] ConstantBuffer<SVOWeightPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_G_SVOWeight)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_E_SVOWeight)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E[];
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_G_SVOWeight)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOLayerInfos_E_SVOWeight)]] RWStructuredBuffer<SVOLayerInfo> SVOLayerInfos_E;
[[vk::binding(StaticBindingPoints_SVOWeight::eDispatchIndirect_SVOWeight)]] RWStructuredBuffer<DispatchIndirectCommand> DispatchCommand;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVO_IndivisibleNodeInfos_G_SVOWeight)]] RWStructuredBuffer<SVOIndivisibleNodeInfo> SVO_IndivisibleNodeInfos_G;
[[vk::binding(StaticBindingPoints_SVOWeight::eHitTestResult_SVOWeight)]] RWStructuredBuffer<HitTestResult> HitTestResults;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeightSampleCounts_SVOWeight)]] RWStructuredBuffer<uint> SVOWeightCounts;
[[vk::binding(StaticBindingPoints_SVOWeight::eSVOWeights_SVOWeight)]] RWStructuredBuffer<float> SVOWeights;

struct HitPayload_SVOWeight {
    float3 hitPos;
    float hit;
}

struct SVONodeInfo_E {
    uint32_t indivisible;
    uint32_t label;
    AABB aabb;
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
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();       // Inverse transform matrix

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
    }
    else payload.hit = 0.0f;
}

inline bool checkPointInAABB(float3 pos, AABB aabb) {
    return all((pos >= aabb.minimum) && (pos <= aabb.maximum));
}
//--------------------------------------------------compute shader-----------------------------------------
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_OCTREE_DEPTH];
[numthreads(THREADGROUP_SIZE2, 1, 1)]
[shader("compute")]
void computeMain_getIndivisibleNode_G(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex < 8) groupSVOLayerInfos_G[groupThreadIndex] = SVOLayerInfos_G[groupThreadIndex];
    GroupMemoryBarrierWithGroupSync();

    uint nodeLayer = 1;
    uint nodeIndex = threadIndex;
    uint nodeLabel = 0;
    for (; nodeLayer < pushConst.maxDepth_G; ++nodeLayer) {
        uint layerNodeCount = pushConst.sizes[nodeLayer];
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

    if (threadIndex < WARP_SIZE) {
        uint indivisibleNodeCount_G = 0;
        if (threadIndex < pushConst.maxDepth_G) indivisibleNodeCount_G = groupSVOLayerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);
        if (threadIndex == 0) DispatchCommand[0].x = (indivisibleNodeCount_G * HITTEST_COUNT + THREADGROUP_SIZE2 - 1) / THREADGROUP_SIZE2;
    }
}

groupshared SVOLayerInfo groupSVOLayerInfos_E[MAX_OCTREE_DEPTH];
groupshared uint groupIndivisibleNodeCount_G;
groupshared uint groupNodeCount_E;
groupshared SVONodeData_G groupSVONodeB[THREADGROUP_SIZE2 / HITTEST_COUNT];
groupshared SVONodeInfo_E groupNodeInfo_E_first3Layer[SVONodeCount_E_Layer1 + SVONodeCount_E_Layer2 + SVONodeCount_E_Layer3];   //18.25 KB
bool hitSVONodeTest_useSharedMemory(inout uint layerIndex, inout uint nodeIndex, inout uint layerNodeSum, inout bool hitIndivisibleNode, float3 hitPos) {
    bool hitNode = false;
    for (; layerIndex < 3; ++layerIndex) {
        hitNode = false;
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            SVONodeInfo_E nodeInfo_E = groupNodeInfo_E_first3Layer[layerNodeSum + nodeIndex + nodeOffset];
            if (checkPointInAABB(hitPos, nodeInfo_E.aabb)) {
                if (nodeInfo_E.indivisible == 1) {
                    hitIndivisibleNode = true;
                    nodeIndex += nodeOffset;
                } else nodeIndex = (nodeInfo_E.label - 1) * 8;
                layerNodeSum += groupSVOLayerInfos_E[layerIndex - 1].indivisibleNodeCount * 8;
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode) break;
    }
    return hitNode;
}
bool hitSVONodeTest(inout uint layerIndex, inout uint nodeIndex, inout uint layerNodeSum, inout bool hitIndivisibleNode, float3 hitPos) {
    bool hitNode = false;
    for (; layerIndex < pushConst.maxDepth_E; ++layerIndex) {
        hitNode = false;
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            SVONodeData_E nodeData = SVO_E[layerIndex][nodeIndex + nodeOffset];
            if (checkPointInAABB(hitPos, nodeData.aabb)) {
                if (nodeData.indivisible == 1) {
                    hitIndivisibleNode = true;
                    nodeIndex += nodeOffset;
                } else nodeIndex = (nodeData.label - 1) * 8;
                layerNodeSum += groupSVOLayerInfos_E[layerIndex - 1].indivisibleNodeCount * 8;
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisibleNode) break;
    }
    return hitNode;
}
float3 getBSDF(uint materialIndex, float3 incidence, float3 normal, float3 outgoing, float3x3 TBN) {
    BSDFMaterial material = pushConst.sceneInfoAddress[0].materials[materialIndex];

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
        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineON);     //cosineON = cosineOH
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
        if (h_tangentSpace.z <= 0.001f) return bsdf;
        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / h_tangentSpace.z;

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);
        float3 F = schlickFresnel(material.albedo, float3(1.0F), cosineOH);

        float3 incidence_tangentSpace = mul(TBN, incidence);

        float G1_outgoing, G1_incidence;
        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(material.roughness));

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
        if (h_tangentSpace.z < 0.0f) h_tangentSpace = -h_tangentSpace;
        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / material.roughness), h_tangentSpace);
        float D = D_costhetaNH / max(h_tangentSpace.z, 0.001f);

        float3 outgoing_tangentSpace = mul(TBN, outgoing);
        float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);
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
        if (refraction) {
            float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
            bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
        } else
            bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
    }
    return bsdf;
}
[numthreads(THREADGROUP_SIZE2, 1, 1)]
[shader("compute")]
void computeMain_getWeights(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    uint threadGroupBCount = THREADGROUP_SIZE2 / HITTEST_COUNT;
    if (groupThreadIndex < threadGroupBCount) {
        SVOIndivisibleNodeInfo nodeInfo_G = SVO_IndivisibleNodeInfos_G[threadGroupIndex * threadGroupBCount + groupThreadIndex];
        groupSVONodeB[groupThreadIndex] = SVO_G[nodeInfo_G.layerIndex][nodeInfo_G.nodeIndex];
    }
    if (groupThreadIndex < WARP_SIZE) {
        if (groupThreadIndex < 8) {
            groupSVOLayerInfos_G[groupThreadIndex] = SVOLayerInfos_G[groupThreadIndex];
            groupSVOLayerInfos_E[groupThreadIndex] = SVOLayerInfos_E[groupThreadIndex];
        }

        uint indivisibleNodeCount_G = 0;
        if (groupThreadIndex < pushConst.maxDepth_G) indivisibleNodeCount_G = groupSVOLayerInfos_G[groupThreadIndex].indivisibleNodeCount;
        indivisibleNodeCount_G = WaveActiveSum(indivisibleNodeCount_G);
        if(groupThreadIndex == 0) groupIndivisibleNodeCount_G = indivisibleNodeCount_G;

        uint nodeCount_E = 0;
        if (groupThreadIndex < pushConst.maxDepth_E - 1) nodeCount_E = groupSVOLayerInfos_G[groupThreadIndex].indivisibleNodeCount * 8;
        nodeCount_E = WaveActiveSum(nodeCount_E);
        if (groupThreadIndex == 0) groupNodeCount_E = nodeCount_E;
    }
    if (groupThreadIndex < SVONodeCount_E_Layer1) {
        SVONodeData_E nodeData = SVO_E[1][groupThreadIndex];
        SVONodeInfo_E nodeInfo;
        nodeInfo.indivisible = nodeData.indivisible;
        nodeInfo.label = nodeData.label;
        nodeInfo.aabb = nodeData.aabb;
        groupNodeInfo_E_first3Layer[groupThreadIndex] = nodeInfo;
    }
    if (groupThreadIndex > WARP_SIZE && groupThreadIndex < WARP_SIZE + SVONodeCount_E_Layer2) {
        uint nodeIndex = SVONodeCount_E_Layer1 + groupThreadIndex - WARP_SIZE;
        SVONodeData_E nodeData = SVO_E[2][nodeIndex];
        SVONodeInfo_E nodeInfo;
        nodeInfo.indivisible = nodeData.indivisible;
        nodeInfo.label = nodeData.label;
        nodeInfo.aabb = nodeData.aabb;
        groupNodeInfo_E_first3Layer[nodeIndex] = nodeInfo;
    }
    for (int i = groupThreadIndex; i < SVONodeCount_E_Layer3; i += THREADGROUP_SIZE2) {
        uint nodeIndex = SVONodeCount_E_Layer1 + SVONodeCount_E_Layer2 + i;
        SVONodeData_E nodeData = SVO_E[2][nodeIndex];
        SVONodeInfo_E nodeInfo;
        nodeInfo.indivisible = nodeData.indivisible;
        nodeInfo.label = nodeData.label;
        nodeInfo.aabb = nodeData.aabb;
        groupNodeInfo_E_first3Layer[nodeIndex] = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupIndivisibleNodeCount_G * HITTEST_COUNT) return;

    uint nodeBLane = groupThreadIndex / HITTEST_COUNT;
    uint nodeCLane = groupThreadIndex % HITTEST_COUNT;
    SVONodeData_G nodeData = groupSVONodeB[nodeBLane];

    float3 meanNormal = nodeData.meanNormal.xyz;
    float frontSampleCount = HITTEST_COUNT * (0.5f + 0.5f * length(meanNormal));
    meanNormal = normalize(meanNormal);
    float3 bitangent = abs(meanNormal.y) < 0.999 ? float3(0.0, 1.0, 0.0) : float3(0.0, 0.0, 1.0);
    float3 tangent = cross(bitangent, meanNormal);
    bitangent = cross(meanNormal, tangent);
    float3x3 TBN = float3x3(tangent, bitangent, meanNormal);

    uint seed = xxhash32(uint3(threadIndex, groupThreadIndex, pushConst.frameIndex));

    RayDesc ray;
    HitPayload_SVOWeight hitPayload;
    // mean sample hemisphere
    float r1 = rand(seed); float r2 = rand(seed);
    float cosTheta = 1.0f - r1; float phi = 2.0f * M_PI * r2;
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);
    float3 sampleDir = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    sampleDir = mul(sampleDir, TBN); // slang is right multiplication
    if (nodeCLane >= frontSampleCount) sampleDir = -sampleDir;

    ray.Direction = sampleDir;
    // get startPoint from aabb
    ray.Origin = (nodeData.aabb.minimum + nodeData.aabb.maximum) * 0.5f;
    float3 halfSize = (nodeData.aabb.maximum - nodeData.aabb.minimum) * 0.5f;
    float3 invDir = 1.0f / max(abs(sampleDir), 0.001f);
    float3 tCandidate = halfSize * invDir;
    float t = min(min(tCandidate.x, tCandidate.y), tCandidate.z);
    ray.Origin += sampleDir * t + ray.Direction * 0.0001f;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    traceRay(ray, hitPayload);
    if (hitPayload.hit == 0.0f) return;
    float cosine = abs(dot(sampleDir, meanNormal));

    uint layerIndex = 1; uint nodeIndex = 0; uint layerNodeSum = 0;
    bool hitIndivisibleNode = false;
    bool hitNode = hitSVONodeTest_useSharedMemory(layerIndex, nodeIndex, layerNodeSum, hitIndivisibleNode, hitPayload.hitPos);
    if (hitNode) {
        if (!hitIndivisibleNode)
            hitSVONodeTest(layerIndex, nodeIndex, layerNodeSum, hitIndivisibleNode, hitPayload.hitPos);
        if (hitIndivisibleNode) {
            for (int outgoingIndex = 0; outgoingIndex < OUTGOING_COUNT; ++outgoingIndex) {
                float3 outgoing = fibSpherePoint(outgoingIndex, OUTGOING_COUNT);
                float3 bsdf = getBSDF(nodeData.materialIndex, sampleDir, meanNormal, outgoing, TBN);
                float3 weight3 = cosine * bsdf;
                float weight = 0.299 * weight3.x + 0.587 * weight3.y + 0.114 * weight3.z;

                uint resultIndex = outgoingIndex * groupIndivisibleNodeCount_G * groupNodeCount_E + layerNodeSum + nodeIndex;
                InterlockedAdd(SVOWeightCounts[resultIndex], 1);
                InterlockedAdd<float>(SVOWeights[resultIndex], weight);
            }
        }
    }
}

[numthreads(THREADGROUP_SIZE2, 1, 1)]
[shader("compute")]
void computeMain_getProbability(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {

}
