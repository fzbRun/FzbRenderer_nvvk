#pragma once
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "renderer/VRPG/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<VRPGPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_VRPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_VRPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(StaticBindingPoints_VRPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];
[[vk::binding(StaticBindingPoints_VRPG::eCandidateSubTreeData)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG, ScalarDataLayout> candidateSubTreeDataBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eCandidateLeafNodeData_VRPG)]] RWStructuredBuffer<CandidateLeafNodeData_VRPG, ScalarDataLayout> candidateLeafNodeDataBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_VRPG, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_VRPG, ScalarDataLayout> GlobalInfoBuffer;
#ifdef SpatialReuse_VRPG
[[vk::binding(StaticBindingPoints_VRPG::eCandidateSampleAABB)]] RWStructuredBuffer<AABB, ScalarDataLayout> candidateSampleAABBBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eCandidateSampleData2)]] RWStructuredBuffer<CandidateSampleData_VRPG, ScalarDataLayout> candidateSampleData2Buffer;
#endif
[[vk::binding(StaticBindingPoints_VRPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(StaticBindingPoints_VRPG::eNearbyNodeInfos)]] RWStructuredBuffer<OctreeNearbyNodeInfo_VRPG, ScalarDataLayout> NearbyNodeInfoBuffer;
[[vk::binding(StaticBindingPoints_VRPG::eReservoir)]] StructuredBuffer<ReservoirSample_VRPG, ScalarDataLayout> ReservoirBuffer;

typedef CallablePayload HitPayload_VRPG;

struct VRSampleData {
    float pdf;
    float wSum;
    float3 sampleDir;
    int nodeLabel_G;
    int outgoingIndex;
    AABB aabb;
    int normalIndex;
};

//----------------------------------------------Sample Function----------------------------------------------
bool SphericalRectangleSample(AABB aabb, float3 hitPos, inout uint randomSeed, inout float3 sampleDir, inout float pdf, uint2 threadIndex) {
    float3 distance = aabb.maximum - aabb.minimum;
    float3 nodeDirection = (aabb.maximum + aabb.minimum) * 0.5f - hitPos;

    float3 edge0 = float3(distance.x, 0.0f, 0.0f);
    float3 edge1 = float3(0.0f, distance.y, 0.0f);
    float3 edge2 = float3(0.0f, 0.0f, distance.z);

    float3 face0StartPos = aabb.minimum;
    face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
    float3 face1StartPos = aabb.minimum;
    face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
    float3 face2StartPos = aabb.minimum;
    face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f; // in back

    Quadrilateral_initialization quad0 = initQuadrilateral(hitPos, face0StartPos, edge2, edge1);
    quad0.S = isnan(quad0.S) ? 0.0f : quad0.S;
    Quadrilateral_initialization quad1 = initQuadrilateral(hitPos, face1StartPos, edge0, edge2);
    quad1.S = isnan(quad1.S) ? 0.0f : quad1.S;
    Quadrilateral_initialization quad2 = initQuadrilateral(hitPos, face2StartPos, edge0, edge1);
    quad2.S = isnan(quad2.S) ? 0.0f : quad2.S;

    float sphericalRectangleSampleProbability = 1.0f;
    float selectFaceProbability = rand(randomSeed);
    float randomU = rand(randomSeed); float randomV = rand(randomSeed);

    // The occurrence of this all-0 situation indicates that hitPos is located on the sampling plane, and we degenerate to BSDF sampling
    if (quad0.S + quad1.S + quad2.S == 0.0f) return false;
    else {
        float3 sphereQuadrilateralSolidAngle = float3(quad0.S, quad1.S, quad2.S);
        sphereQuadrilateralSolidAngle /= quad0.S + quad1.S + quad2.S;
        Quadrilateral_initialization sampleQuad;
        if (selectFaceProbability <= sphereQuadrilateralSolidAngle.x) {
            sampleQuad = quad0;
            sphericalRectangleSampleProbability *= sphereQuadrilateralSolidAngle.x;
        }
        else if (selectFaceProbability <= 1.0f - sphereQuadrilateralSolidAngle.z) {
            sampleQuad = quad1;
            sphericalRectangleSampleProbability *= sphereQuadrilateralSolidAngle.y;
        }
        else {
            sampleQuad = quad2;
            sphericalRectangleSampleProbability *= sphereQuadrilateralSolidAngle.z;
        }

        pdf *= sphericalRectangleSampleProbability;
        sampleDir = normalize(sphericalRectangleSample(sampleQuad, randomU, randomV, pdf)); // p *= 1 / quad.s

        // if (pushConst.frameIndex == 1 && all(threadIndex == uint2(740, 62))) {
        //     printf("pdf: %f, faceSelectPdf: %f, srPdf: %f\naabb: %f %f %f %f %f %f\n\n",
        //         pdf, sphericalRectangleSampleProbability, 1 / sampleQuad.S,
        //         aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
        //         aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
        //     );
        // }
    }

    return true;
}
int getNodeIndex_E(float3 hitPos, float3 hitNormal) {
    int3 voxelIndex = int3((hitPos - pushConst.lightVGBStartPos_Size.xyz) / pushConst.lightVGBVoxelSize.xyz);
    int voxelCount = 1 << pushConst.lightMaxOctreeLayer;
    int voxelTotalCount = voxelCount * voxelCount * voxelCount;

    uint normalIndex = getNormalIndex(hitNormal);
    int nodeIndex_E = voxelTotalCount * normalIndex;

    while (voxelTotalCount > 1) {
        voxelCount = voxelCount / 2;
        voxelTotalCount = voxelTotalCount / 8;

        uint3 offset = voxelIndex / voxelCount;

        nodeIndex_E += offset.z * 4 * voxelTotalCount;
        voxelIndex.z -= offset.z * voxelCount;

        nodeIndex_E += offset.y * 2 * voxelTotalCount;
        voxelIndex.y -= offset.y * voxelCount;

        nodeIndex_E += offset.x * voxelTotalCount;
        voxelIndex.x -= offset.x * voxelCount;
    }

    return nodeIndex_E;
}

int getGlobalGLabelFromInfo(int2 info) {
    if (info.x < 0) return -1;
    int label = 0;
    for (int l = 0; l < info.x; ++l) {
        label += GlobalInfoBuffer[0].layerInfos_G[l].indivisibleNodeCount;
    }
    label += (OctreeNodeInfoBuffer_G[info.x][info.y].label_indivisible >> 1) - 1;
    return label;
}

// Uniform disk sample in the tangent plane of the current G cluster, then
// choose the candidate G cluster containing the jittered point.
int sampleJitteredClusterG(int currentNodeLabel, int currentLayer, int currentNodeIndex, float3 hitPos, float materialRoughness, inout uint randomSeed) {
    OctreeNodeData_G_VRPG currentNode = OctreeNodeDataBuffer_G[currentLayer][currentNodeIndex];
    float3 normal = normalize(currentNode.meanNormal.xyz + float3(1e-8f));

    float3 tangent, bitangent;
    orthonormalBasis(normal, tangent, bitangent);

    // Smooth surfaces have a narrow BSDF lobe: keep the jitter disk small.
    // Rough surfaces can afford a wider spatial blend.
    float roughness = clamp(materialRoughness, 0.0f, 1.0f);
    float roughnessScale = 0.1f + 0.9f * roughness;
    float radius = max(length(currentNode.aabb.maximum - currentNode.aabb.minimum) *
                       pushConst.jitterRadiusScale * roughnessScale, 1e-5f);
    float r = radius * sqrt(rand(randomSeed));
    float theta = 2.0f * M_PI * rand(randomSeed);
    float3 jitteredPos = hitPos + (cos(theta) * tangent + sin(theta) * bitangent) * r;

    int selectedLabel = currentNodeLabel;
    float bestCenterDistance = 1e30f;
    bool foundInside = false;

    {
        float3 extent = currentNode.aabb.maximum - currentNode.aabb.minimum;
        float tolerance = max(1e-4f, max(extent.x, max(extent.y, extent.z)) * 0.01f);
        bool inside = all(jitteredPos >= currentNode.aabb.minimum - tolerance) &&
                      all(jitteredPos <= currentNode.aabb.maximum + tolerance);
        if (inside) {
            float3 center = (currentNode.aabb.minimum + currentNode.aabb.maximum) * 0.5f;
            bestCenterDistance = length(jitteredPos - center);
            foundInside = true;
        }
    }

    int2 nearbyNodeInfos[NEARBY_NODE_COUNT_VRPG] = NearbyNodeInfoBuffer[currentNodeLabel].nearbyNodeInfos;
    int nearbyNodeLimit = clamp(VRPG_NEARBY_NODE_COUNT, 0, NEARBY_NODE_COUNT_VRPG);
    for (int i = 0; i < nearbyNodeLimit; ++i) {
        int2 nearbyInfo = nearbyNodeInfos[i];
        if (nearbyInfo.x < 0) continue;

        OctreeNodeData_G_VRPG nearbyNode = OctreeNodeDataBuffer_G[nearbyInfo.x][nearbyInfo.y];
        float3 extent = nearbyNode.aabb.maximum - nearbyNode.aabb.minimum;
        float tolerance = max(1e-4f, max(extent.x, max(extent.y, extent.z)) * 0.01f);
        bool inside = all(jitteredPos >= nearbyNode.aabb.minimum - tolerance) &&
                      all(jitteredPos <= nearbyNode.aabb.maximum + tolerance);
        if (!inside) continue;

        float3 center = (nearbyNode.aabb.minimum + nearbyNode.aabb.maximum) * 0.5f;
        float centerDistance = length(jitteredPos - center);
        if (centerDistance < bestCenterDistance) {
            bestCenterDistance = centerDistance;
            selectedLabel = getGlobalGLabelFromInfo(nearbyInfo);
            foundInside = true;
        }
    }

    // The disk can extend outside the union of all candidate AABBs. In that
    // case, do not silently fall back to the current cluster: select the
    // candidate whose AABB center is closest to the sampled point.
    if (!foundInside) {
        float3 currentCenter = (currentNode.aabb.minimum + currentNode.aabb.maximum) * 0.5f;
        bestCenterDistance = length(jitteredPos - currentCenter);
        selectedLabel = currentNodeLabel;
        for (int i = 0; i < nearbyNodeLimit; ++i) {
            int2 nearbyInfo = nearbyNodeInfos[i];
            if (nearbyInfo.x < 0) continue;
            OctreeNodeData_G_VRPG nearbyNode = OctreeNodeDataBuffer_G[nearbyInfo.x][nearbyInfo.y];
            float3 center = (nearbyNode.aabb.minimum + nearbyNode.aabb.maximum) * 0.5f;
            float centerDistance = length(jitteredPos - center);
            if (centerDistance < bestCenterDistance) {
                bestCenterDistance = centerDistance;
                selectedLabel = getGlobalGLabelFromInfo(nearbyInfo);
            }
        }
    }

    return selectedLabel >= 0 ? selectedLabel : currentNodeLabel;
}

bool getNodeAABB_E_VRPG(inout VRSampleData sampleData, inout HitPayload_VRPG payload, uint2 threadIndex) {
    if (payload.material.type == uint(MaterialType::Dielectric) ||
        payload.material.type == uint(MaterialType::RoughDielectric)) return false;

    float3 geoNormal = payload.isExt ? payload.hitNormal : -payload.hitNormal;
    uint normalIndex = getNormalIndex(geoNormal);
    float3 hitPos = payload.hitPos;

    OctreeNodeInfo_G_VRPG nodeInfo_G = OctreeNodeInfoBuffer_G[0][normalIndex];
    int nodeLabel_G = int(nodeInfo_G.label_indivisible >> 1) - 1;
    bool hasData = nodeLabel_G >= 0;
    if (!hasData) return false;

    bool indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;
    bool hitIndivisible = false;
    int layerIndex_G = 0; int nodeIndex_G = normalIndex;
    if (indivisible) hitIndivisible = true;
    else {
        nodeLabel_G = GlobalInfoBuffer[0].layerInfos_G[0].indivisibleNodeCount;

        float3 nodeSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - 1));
        float3 nodeStartPos = pushConst.VGBStartPos_Size.xyz;

        for (layerIndex_G = 1; layerIndex_G <= pushConst.maxOctreeLayer; ++layerIndex_G) {
            int3 nodeIndex3 = min(int3((hitPos - nodeStartPos) / nodeSize), 1);
            int subNodeIndex = nodeIndex3.z * 4 + nodeIndex3.y * 2 + nodeIndex3.x;
            nodeIndex_G = nodeIndex_G * 8 + subNodeIndex;

            hasData = true;
            indivisible = false;
            int nodeLayerLabel = 0;
            nodeInfo_G = OctreeNodeInfoBuffer_G[layerIndex_G][nodeIndex_G];

            nodeLayerLabel = int(nodeInfo_G.label_indivisible >> 1) - 1;
            hasData = nodeLayerLabel >= 0;
            indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;

            if (!hasData) break;
            if (indivisible) {
                nodeLabel_G += nodeLayerLabel;
                hitIndivisible = true;
                break;
            }
            nodeLabel_G += GlobalInfoBuffer[0].layerInfos_G[layerIndex_G].indivisibleNodeCount;
            nodeStartPos += nodeIndex3 * nodeSize;
            nodeSize /= 2.0f;
        }
    }
    if (!hitIndivisible) return false;

    #if VRPG_JITTER_SAMPLING == 1
    {
        float materialRoughness = payload.material.roughness;
        if (payload.material.type == uint(MaterialType::Diffuse))
            materialRoughness = 1.0f;
        nodeLabel_G = sampleJitteredClusterG(nodeLabel_G, layerIndex_G, nodeIndex_G,
                                             hitPos, materialRoughness, payload.randomSeed);
    }
    #endif

    float weightSampleProbability = rand(payload.randomSeed);
    float3 outgoing = mul(pushConst.randomRotateMatrix, payload.outgoing);
    float3 hitNormal = mul(pushConst.randomRotateMatrix, payload.hitNormal);
    int outgoingIndex = inverseOutgoing_VRPG(outgoing, OUTGOING_COUNT_VRPG, weightSampleProbability, hitNormal);

    uint indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;

    // Spatiotemporal reservoir: one selected sample per (outgoing, G).
    #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
    {
        int reservoirIndex = nodeLabel_G * OUTGOING_COUNT_VRPG + outgoingIndex;
        ReservoirSample_VRPG reservoir = ReservoirBuffer[reservoirIndex];
        if (reservoir.weight <= 0.0f || reservoir.p_hat <= 0.0f) return false;

        sampleData.wSum = reservoir.weight;
        sampleData.aabb = reservoir.aabb;
        sampleData.normalIndex = reservoir.normalIndex;
        sampleData.nodeLabel_G = nodeLabel_G;
        sampleData.outgoingIndex = outgoingIndex;
        sampleData.pdf = reservoir.p_hat / reservoir.weight;
        return SphericalRectangleSample(sampleData.aabb, payload.hitPos, payload.randomSeed,
                                        sampleData.sampleDir, sampleData.pdf);
    }
    #endif

    int candidateNodeStartIndex = outgoingIndex * indivisibleNodeCount_G * Candidate_Samples_Count + nodeLabel_G * Candidate_Samples_Count;
    float wSum = candidateSamplePrefixWSumBuffer[candidateNodeStartIndex + Candidate_Samples_Count - 1];
    if (wSum == 0.0f) return false;

    float randomNumber = rand(payload.randomSeed) * wSum;
    int lo = 0;
    int hi = Candidate_Samples_Count - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (randomNumber < candidateSamplePrefixWSumBuffer[candidateNodeStartIndex + mid]) hi = mid;
        else lo = mid + 1;
    }
    int candidateIndex = lo;

    sampleData.wSum = wSum;
    #ifdef SpatialReuse_VRPG
    #if (VRPG_SPATIAL_REUSE || VRPG_TEMPORAL_REUSE)
    sampleData.pdf = candidateSampleData2Buffer[candidateNodeStartIndex + candidateIndex].p_hat / wSum;
    sampleData.aabb = candidateSampleAABBBuffer[candidateNodeStartIndex + candidateIndex];
    #else
    sampleData.pdf = candidateSampleDataBuffer[candidateNodeStartIndex + candidateIndex].p_hat / wSum;
    sampleData.aabb = candidateLeafNodeDataBuffer[nodeLabel_G * Candidate_Samples_Count + candidateIndex].aabb;
    #endif
    #else
    sampleData.pdf = candidateSampleDataBuffer[candidateNodeStartIndex + candidateIndex].p_hat / wSum;
    sampleData.aabb = candidateLeafNodeDataBuffer[nodeLabel_G * Candidate_Samples_Count + candidateIndex].aabb;
    #endif
    
    sampleData.nodeLabel_G = nodeLabel_G;
    sampleData.outgoingIndex = outgoingIndex;

    int subTreeNodeIndex_layer = candidateSubTreeDataBuffer[nodeLabel_G * Candidate_Samples_Count + candidateIndex].subTreeNodeIndex_layer;
    int subTreeLayer = subTreeNodeIndex_layer & 15;
    int subTreeIndex = subTreeNodeIndex_layer >> 4;
    int subTreeLayerNodeCount = (1 << (3 * subTreeLayer));
    sampleData.normalIndex = subTreeIndex / subTreeLayerNodeCount;

    return SphericalRectangleSample(sampleData.aabb, payload.hitPos, payload.randomSeed, sampleData.sampleDir, sampleData.pdf);
}

float getPgPdf_SPReuse(float wSum, float3 bsdfSamplePos, float3 bsdfSampleNormal, float3 hitPos, float3 sampleDir, float3 bsdf_cosine, uint2 threadIndex) {
    uint nodeIndex_E = getNodeIndex_E(hitPos, bsdfSampleNormal);
    OctreeNodeData_E_VRPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.lightMaxOctreeLayer][nodeIndex_E];
    float cosine_Light = max(dot(bsdfSampleNormal, -sampleDir), 0.0f);
    float d = distance(bsdfSamplePos, hitPos);
    float3 f = nodeData_E.radiance * bsdf_cosine * cosine_Light / max(d * d, 1e-3f);
    float p_hat = dot(f, 1.0f);
    if (p_hat == 0.0f) return 0.0f;
    float pdf = p_hat / wSum;

    // getPdf from Sphere sample
    {
        float3 distance = nodeData_E.aabb.maximum - nodeData_E.aabb.minimum;
        float3 nodeDirection = (nodeData_E.aabb.maximum + nodeData_E.aabb.minimum) * 0.5f - hitPos;

        float3 edge0 = float3(distance.x, 0.0f, 0.0f);
        float3 edge1 = float3(0.0f, distance.y, 0.0f);
        float3 edge2 = float3(0.0f, 0.0f, distance.z);

        float3 face0StartPos = nodeData_E.aabb.minimum;
        face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
        float3 face1StartPos = nodeData_E.aabb.minimum;
        face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
        float3 face2StartPos = nodeData_E.aabb.minimum;
        face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f; // in back

        Quadrilateral_initialization quad0 = initQuadrilateral(hitPos, face0StartPos, edge2, edge1);
        quad0.S = isnan(quad0.S) ? 0.0f : quad0.S;
        Quadrilateral_initialization quad1 = initQuadrilateral(hitPos, face1StartPos, edge0, edge2);
        quad1.S = isnan(quad1.S) ? 0.0f : quad1.S;
        Quadrilateral_initialization quad2 = initQuadrilateral(hitPos, face2StartPos, edge0, edge1);
        quad2.S = isnan(quad2.S) ? 0.0f : quad2.S;

        float solidAngleSum = quad0.S + quad1.S + quad2.S;
        if (solidAngleSum == 0.0f) pdf = 0.0f;
        else pdf /= solidAngleSum;
    }

    return pdf;
}

float getPgPdf2(float wSum, int label_G, int outgoingIndex, float3 bsdfSamplePos, float3 bsdfSampleNormal, float3 hitPos, float3 sampleDir, float3 bsdf_cosine) {
    int gCount = int(GlobalInfoBuffer[0].indivisibleNodeCount_G);
    if (label_G < 0 || label_G >= gCount ||
        outgoingIndex < 0 || outgoingIndex >= OUTGOING_COUNT_VRPG)
        return 0.0f;

    uint nodeIndex_E = getNodeIndex_E(hitPos, bsdfSampleNormal);

    float ESum = 0.0f; float wi_ori = 0.0f;
    for (int i = 0; i < Candidate_Samples_Count; ++i) {
        int subTreeIndex = label_G * Candidate_Samples_Count + i;
        CandidateSubTreeInfo_VRPG subTreeInfo = candidateSubTreeDataBuffer[subTreeIndex];
        int subTreeNodeIndex_layer = subTreeInfo.subTreeNodeIndex_layer;
        if (subTreeNodeIndex_layer == -1) break;

        subTreeIndex = subTreeNodeIndex_layer >> 4;
        int subTreeLayer = subTreeNodeIndex_layer & 15;
        int subTreeLeafNodeCount = (1 << (3 * (pushConst.lightMaxOctreeLayer - subTreeLayer)));
        int subTreeLeafNodeStartIndex = subTreeLeafNodeCount * subTreeIndex;
        int subTreeLeafNodeEndIndex = subTreeLeafNodeStartIndex + subTreeLeafNodeCount;
        if (nodeIndex_E >= subTreeLeafNodeStartIndex && nodeIndex_E < subTreeLeafNodeEndIndex) {
            ESum = subTreeInfo.ESum;
            int candidateIndex = outgoingIndex * GlobalInfoBuffer[0].indivisibleNodeCount_G * Candidate_Samples_Count + label_G * Candidate_Samples_Count + i;
            wi_ori = candidateSampleDataBuffer[candidateIndex].w;
            break;
        }
    }
    if (ESum == 0.0f) return 0.0f;

    OctreeNodeData_E_VRPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.lightMaxOctreeLayer][nodeIndex_E];
    float p = dot(nodeData_E.radiance, 1.0f) / ESum;
    if (p == 0.0f) return 0.0f;

    float cosine_Light = max(dot(bsdfSampleNormal, -sampleDir), 0.0f);
    float d = distance(bsdfSamplePos, hitPos);
    float3 f = nodeData_E.radiance * bsdf_cosine * cosine_Light / max(d * d, 1e-3f);
    float p_hat = dot(f, 1.0f);

    float wi = p_hat / p;
    wSum = wSum + wi - wi_ori;
    float pdf = p_hat / max(wSum, 1e-3f);
    if (pdf == 0.0f) return 0.0f;

    // getPdf from Sphere sample
    {
        float3 distance = nodeData_E.aabb.maximum - nodeData_E.aabb.minimum;
        float3 nodeDirection = (nodeData_E.aabb.maximum + nodeData_E.aabb.minimum) * 0.5f - hitPos;

        float3 edge0 = float3(distance.x, 0.0f, 0.0f);
        float3 edge1 = float3(0.0f, distance.y, 0.0f);
        float3 edge2 = float3(0.0f, 0.0f, distance.z);

        float3 face0StartPos = nodeData_E.aabb.minimum;
        face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
        float3 face1StartPos = nodeData_E.aabb.minimum;
        face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
        float3 face2StartPos = nodeData_E.aabb.minimum;
        face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f; // in back

        Quadrilateral_initialization quad0 = initQuadrilateral(hitPos, face0StartPos, edge2, edge1);
        quad0.S = isnan(quad0.S) ? 0.0f : quad0.S;
        Quadrilateral_initialization quad1 = initQuadrilateral(hitPos, face1StartPos, edge0, edge2);
        quad1.S = isnan(quad1.S) ? 0.0f : quad1.S;
        Quadrilateral_initialization quad2 = initQuadrilateral(hitPos, face2StartPos, edge0, edge1);
        quad2.S = isnan(quad2.S) ? 0.0f : quad2.S;

        float solidAngleSum = quad0.S + quad1.S + quad2.S;
        if (solidAngleSum == 0.0f) pdf = 0.0f;
        else pdf /= solidAngleSum;
    }

    return pdf;
}