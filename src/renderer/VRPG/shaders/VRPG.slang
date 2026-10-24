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

#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_VRPG::eDepthImage, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
[[vk::binding(StaticBindingPoints_VRPG::ePGValueImage, 0)]] RWTexture2D<float4> pgValueImage;
[[vk::binding(StaticBindingPoints_VRPG::ePGValue2Image, 0)]] RWTexture2D<float4> pgValue2Image;
[[vk::binding(StaticBindingPoints_VRPG::ePGVarianceImage, 0)]] RWTexture2D<float4> pgVarianceImage;
#endif

typedef CallablePayload HitPayload_VRPG;

struct VRSampleData {
    float pdf;
    float wSum;
    float3 sampleDir;

    int nodeLabel_G;
    int outgoingIndex;

    AABB aabb;
    int normalIndex;
}
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared OctreeGlobalInfo_VRPG groupGlobalInfo;
groupshared OctreeNodeInfo_G_VRPG groupOctreeNodeInfo_G[3510];
//----------------------------------------------Hit Function----------------------------------------------
bool traceRay(RayDesc ray, inout HitPayload_VRPG payload)
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
        uint primitiveIndex = q.CommittedPrimitiveIndex();

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex]; // Instance data
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
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

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));

        // payload.hitNormal = normalize(mul(worldToObject, geoNormal).xyz);
        payload.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(payload.hitNormal, -ray.Direction);
        if (cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
            payload.material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;

        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal); // column matrix

        if (payload.material.materialMapIndex.x >= 0 || payload.material.materialMapIndex.y >= 0 || payload.material.materialMapIndex.z >= 0) {
            int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
            float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
            texCoords.y = 1.0f - texCoords.y;
            getTextueData(payload, texCoords);
        }

        return true;
    }
    return false;
}
float3 RayMiss(float3 rayDirection) {
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}
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
    uint nodeSum = 0;
    for (int l = 0; l < info.x; ++l) {
        label += groupGlobalInfo.layerInfos_G[l].indivisibleNodeCount;
        nodeSum += 6 * (1u << (l * 3));
    }
    if (info.x < 4)
        label += (groupOctreeNodeInfo_G[nodeSum + info.y].label_indivisible >> 1) - 1;
    else
        label += (OctreeNodeInfoBuffer_G[info.x][info.y].label_indivisible >> 1) - 1;
    return label;
}

// Uniform disk sample in the tangent plane of the current G cluster, then
// choose the candidate G cluster containing the jittered point.
int sampleJitteredClusterG(int currentNodeLabel, int currentLayer, int currentNodeIndex,
                           float3 hitPos, float materialRoughness, inout uint randomSeed) {
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
    int nearbyNodeLimit = clamp(pushConst.nearbyNodeCount, 0, NEARBY_NODE_COUNT_VRPG);
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

    OctreeNodeInfo_G_VRPG nodeInfo_G = groupOctreeNodeInfo_G[normalIndex];
    int nodeLabel_G = int(nodeInfo_G.label_indivisible >> 1) - 1;
    bool hasData = nodeLabel_G >= 0;
    if (!hasData) return false;

    bool indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;
    bool hitIndivisible = false;
    int layerIndex_G = 0; int nodeIndex_G = normalIndex;
    if (indivisible) hitIndivisible = true;
    else {
        nodeLabel_G = groupGlobalInfo.layerInfos_G[0].indivisibleNodeCount;

        float3 nodeSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - 1));
        float3 nodeStartPos = pushConst.VGBStartPos_Size.xyz;

        int nodeSum = 6; int layerNodeCount = 6;
        for (layerIndex_G = 1; layerIndex_G <= pushConst.maxOctreeLayer; ++layerIndex_G) {
            int3 nodeIndex3 = min(int3((hitPos - nodeStartPos) / nodeSize), 1);
            int subNodeIndex = nodeIndex3.z * 4 + nodeIndex3.y * 2 + nodeIndex3.x;
            nodeIndex_G = nodeIndex_G * 8 + subNodeIndex;

            hasData = true;
            indivisible = false;
            int nodeLayerLabel = 0;
            if (layerIndex_G < 4) {
                nodeInfo_G = groupOctreeNodeInfo_G[nodeSum + nodeIndex_G];
                layerNodeCount *= 8;
                nodeSum += layerNodeCount;
            } else nodeInfo_G = OctreeNodeInfoBuffer_G[layerIndex_G][nodeIndex_G];

            nodeLayerLabel = int(nodeInfo_G.label_indivisible >> 1) - 1;
            hasData = nodeLayerLabel >= 0;
            indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;

            if (!hasData) break;
            if (indivisible) {
                nodeLabel_G += nodeLayerLabel;
                hitIndivisible = true;
                break;
            }
            nodeLabel_G += groupGlobalInfo.layerInfos_G[layerIndex_G].indivisibleNodeCount;
            nodeStartPos += nodeIndex3 * nodeSize;
            nodeSize /= 2.0f;
        }
    }
    if (!hitIndivisible) return false;

    if (pushConst.ditherSample == 1) {
        float materialRoughness = payload.material.roughness;
        if (payload.material.type == uint(MaterialType::Diffuse))
            materialRoughness = 1.0f;
        nodeLabel_G = sampleJitteredClusterG(nodeLabel_G, layerIndex_G, nodeIndex_G,
                                             hitPos, materialRoughness, payload.randomSeed);
    }

    float weightSampleProbability = rand(payload.randomSeed);
    float3 outgoing = mul(pushConst.randomRotateMatrix, payload.outgoing);
    float3 hitNormal = mul(pushConst.randomRotateMatrix, payload.hitNormal);
    int outgoingIndex = inverseOutgoing_VRPG(outgoing, OUTGOING_COUNT_VRPG, weightSampleProbability, hitNormal);

    uint indivisibleNodeCount_G = groupGlobalInfo.indivisibleNodeCount_G;

    // Spatiotemporal reservoir: one selected sample per (outgoing, G).
    if (pushConst.spatialRuse == 1) {
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
    sampleData.pdf = (pushConst.spatialRuse == 1 ? candidateSampleData2Buffer[candidateNodeStartIndex + candidateIndex].p_hat : candidateSampleDataBuffer[candidateNodeStartIndex + candidateIndex].p_hat) / wSum;
    sampleData.aabb = pushConst.spatialRuse == 1 ? candidateSampleAABBBuffer[candidateNodeStartIndex + candidateIndex] : 
                                             candidateLeafNodeDataBuffer[nodeLabel_G * Candidate_Samples_Count + candidateIndex].aabb;
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
            int candidateIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * Candidate_Samples_Count + label_G * Candidate_Samples_Count + i;
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
//---------------------------------------------PathGuiding-----------------------------------------
/*
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VRPG(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    threadIndex += pushConst.dispatchOffset;
    uint groupThreadIndex_linear = groupThreadIndex.y * FZB_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = FZB_PATHGUIDING_THREADGROUP_SIZE_X * FZB_PATHGUIDING_THREADGROUP_SIZE_Y;
    if (groupThreadIndex_linear == 0) groupGlobalInfo = GlobalInfoBuffer[0];
    for (int i = groupThreadIndex_linear; i < 3510; i += threadGroupThreadTotalCount) {
        int layerIndex = 0; int nodeIndex = i; int layerNodeCount = 6;
        for (; layerIndex < 4; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        groupOctreeNodeInfo_G[i] = OctreeNodeInfoBuffer_G[layerIndex][nodeIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_VRPG payload;
    payload.randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    #ifndef NDEBUG
    float3 pgValue = 0.0f;
    float3 pgValue2 = 0.0f;
    #endif

    for (int sampleIndex = 0; sampleIndex < pushConst.spp; ++sampleIndex) {
        float r1 = rand(payload.randomSeed);
        float r2 = rand(payload.randomSeed);
        float2 subpixel_jitter = pushConst.frameIndex == 0 && sampleIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

        const float2 pixelCenter = launchID + subpixel_jitter;
        const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        ray.Origin = sceneInfo.cameraPosition;
        ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

        payload.bsdf = float3(1.0f);
        payload.pdf = 1.0f;
        payload.outgoing = -ray.Direction;
        payload.isExt = true;

        float3 bsdf_cosine_pdf = float3(1.0f);

        uint bounceDepth = 0; bool useBsdfSample = false, hit = true;
        while (bounceDepth < pushConst.maxDepth) {
            if (!useBsdfSample) hit = traceRay(ray, payload);
            useBsdfSample = false;

            #ifndef NDEBUG
            if (bounceDepth == 0 && sampleIndex == 0) {
                float3 hitPos = payload.hitPos + 0.01f * ray.Direction;
                float4 clipPos = mul(float4(hitPos, 1.0f), sceneInfo.viewProjMatrix);
                float depth = clipPos.z / clipPos.w; // any hitPoint must in our cameraView
                depthImage[int2(launchID)] = depth;
            } else if (bounceDepth == MISS_DEPTH) depthImage[int2(launchID)] = 1.0f;
            #endif

            if (!hit) {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                bounceDepth = MISS_DEPTH;
            } else {
                //----------------------------------------------------------------------------------------------
                PGSampleData pgSampleData;
                bool useVRPG = getNodeAABB_E_VRPG(pgSampleData, payload, threadIndex);

                float3 pgSample_bsdf_cosine = 0.0f;
                float pgSample_bsdfPdf = 0.0f;
                float pgSample_weight = 0.0f;
                if (useVRPG) {
                    float cosine = abs(dot(payload.hitNormal, pgSampleData.sampleDir));
                    pgSample_bsdf_cosine = cosine * getBSDF(payload.material, pgSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    pgSample_bsdfPdf = getPdf(payload.material, pgSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    pgSample_weight = pgSampleData.pdf / max(pgSampleData.pdf + pgSample_bsdfPdf, 1e-16f);
                }
                //----------------------------------------------------------------------------------------------
                getDirectionLight(payload, sceneInfo);
                BSDF_SAMPLE(payload);
                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;

                float3 bsdfSampleDir = payload.incidence;
                float3 bsdfSample_bsdf_cosine = payload.bsdf * abs(dot(bsdfSampleDir, payload.hitNormal));
                float bsdfSample_bsdfPdf = payload.pdf;

                float3 hitNormal = payload.hitNormal;
                float3 hitPos = payload.hitPos;
                ray.Origin = offsetRay(hitPos, hitNormal);
                ray.Direction = bsdfSampleDir;
                hit = traceRay(ray, payload);       //after here, payload is changed to next hitPos's payload!!!!!!

                float bsdfSample_pgPdf = 0.0f;
                if (useVRPG && hit) {
                    if (pushConst.spatialRuse == 1) {
                        bsdfSample_pgPdf = getPgPdf_SPReuse(pgSampleData.wSum,
                            payload.hitPos, payload.hitNormal, hitPos,
                            bsdfSampleDir, bsdfSample_bsdf_cosine, threadIndex);
                    } else {
                        bsdfSample_pgPdf = getPgPdf2(pgSampleData.wSum, pgSampleData.nodeLabel_G, pgSampleData.outgoingIndex,
                            payload.hitPos, payload.hitNormal, hitPos, bsdfSampleDir, bsdfSample_bsdf_cosine);
                    }
                }
                    
                float bsdfSample_weight = bsdfSample_bsdfPdf / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);

                #ifndef NDEBUG
                if (bounceDepth > 0) {
                    float3 pgValue_temp = (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
                    pgValue += pgValue_temp;
                    pgValue2 += pgValue_temp * pgValue_temp;
                }
                #endif
                //----------------------------------------------------MIS------------------------------------------
                float pgSample_selectProbability = pgSample_weight / max(pgSample_weight + bsdfSample_weight, 1e-16f);
                float randomNumber = rand(payload.randomSeed);
                if (randomNumber <= pgSample_selectProbability) {
                    payload.incidence = pgSampleData.sampleDir;
                    bsdf_cosine_pdf *= pgSample_weight * pgSample_bsdf_cosine / pgSampleData.pdf / max(pgSample_selectProbability, 1e-16f);
                } else {
                    useBsdfSample = true;
                    payload.incidence = bsdfSampleDir;
                    bsdf_cosine_pdf *= bsdfSample_weight * bsdfSample_bsdf_cosine / bsdfSample_bsdfPdf / max(1.0f - pgSample_selectProbability, 1e-16f);
                }
                //----------------------------------------------------------------------------------------------
                bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
                if (end) break;

                randomNumber = rand(payload.randomSeed);
                if (randomNumber >= RR) break;
                bsdf_cosine_pdf /= RR;

                payload.outgoing = -payload.incidence;
                if (!useBsdfSample) {
                    ray.Direction = payload.incidence;
                    ray.Origin = offsetRay(hitPos, hitNormal);
                }

                ++bounceDepth;
            }
        }
    }
    accumulatedRadiance /= pushConst.spp;

    #ifndef NDEBUG
    uint n = max(pushConst.spp, 1);
    pgValue /= n;
    pgValue2 /= n;
    if (pushConst.frameIndex > 0) {
        pgValue = (pgValue + pgValueImage[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;
        pgValue2 = (pgValue2 + pgValue2Image[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;

        pgVarianceImage[threadIndex].x = luminance(pgValue2 - pgValue * pgValue);
    }
    pgValueImage[threadIndex].xyz = pgValue;
    pgValue2Image[threadIndex].xyz = pgValue2;
    #endif

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0){
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}
*/
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_VRPG(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    threadIndex += pushConst.dispatchOffset;
    uint groupThreadIndex_linear = groupThreadIndex.y * FZB_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = FZB_PATHGUIDING_THREADGROUP_SIZE_X * FZB_PATHGUIDING_THREADGROUP_SIZE_Y;
    if (groupThreadIndex_linear == 0) groupGlobalInfo = GlobalInfoBuffer[0];
    for (int i = groupThreadIndex_linear; i < 3510; i += threadGroupThreadTotalCount) {
        int layerIndex = 0; int nodeIndex = i; int layerNodeCount = 6;
        for (; layerIndex < 4; ++layerIndex) {
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            layerNodeCount *= 8;
        }
        groupOctreeNodeInfo_G[i] = OctreeNodeInfoBuffer_G[layerIndex][nodeIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_VRPG payload;
    HitPayload_VRPG vrPayload;
    HitPayload_VRPG bsdfPayload;

    uint pgSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    #ifndef NDEBUG
    float3 pgValue = 0.0f;
    float3 pgValue2 = 0.0f;
    #endif

    for (int sampleIndex = 0; sampleIndex < pushConst.spp; ++sampleIndex) {
        float r1 = rand(pgSeed);
        float r2 = rand(pgSeed);
        float2 subpixel_jitter = pushConst.frameIndex == 0 && sampleIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

        const float2 pixelCenter = launchID + subpixel_jitter;
        const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
        float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
        viewCoords /= viewCoords.w;

        ray.Origin = sceneInfo.cameraPosition;
        ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

        payload.bsdf = float3(1.0f);
        payload.pdf = 1.0f;
        payload.outgoing = -ray.Direction;
        payload.isExt = true;
        payload.randomSeed = pgSeed;

        float3 bsdf_cosine_pdf = float3(1.0f);
        uint bounceDepth = 0;
        bool hit = traceRay(ray, payload);

        while (bounceDepth < pushConst.maxDepth) {
            #ifndef NDEBUG
            if (bounceDepth == 0 && sampleIndex == 0) {
                if (hit) {
                    float3 hitPos = payload.hitPos + 0.01f * ray.Direction;
                    float4 clipPos = mul(float4(hitPos, 1.0f), sceneInfo.viewProjMatrix);
                    depthImage[int2(launchID)] = clipPos.z / clipPos.w;
                }
                else {
                    depthImage[int2(launchID)] = 1.0f;
                }
            }
            #endif

            if (!hit) {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                break;
            }

            //-------------------------------------------------- PG candidate --------------------------------------------------
            payload.randomSeed = pgSeed;

            VRSampleData vrSampleData;
            bool getVrSample = getNodeAABB_E_VRPG(vrSampleData, payload, threadIndex);
            pgSeed = payload.randomSeed;

            vrPayload = payload;
            vrPayload.randomSeed = pgSeed;

            bool vrHit = false;
            bool validSample = false;

            if (getVrSample) {
                vrPayload.incidence = vrSampleData.sampleDir;

                ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                ray.Direction = vrSampleData.sampleDir;
                vrHit = traceRay(ray, vrPayload);

                if (vrHit) {
                    validSample = checkPointInAABB(vrPayload.hitPos, vrSampleData.aabb);
                    validSample = validSample && (getNormalIndex(vrPayload.hitNormal) == vrSampleData.normalIndex);
                    validSample = validSample && (vrSampleData.pdf > 0.0f);
                }
            }

            //-------------------------------------------------- BSDF candidate -------------------------------------------------
            bsdfPayload = payload;
            bsdfPayload.randomSeed = pgSeed;

            getDirectionLight(bsdfPayload, sceneInfo);
            BSDF_SAMPLE(bsdfPayload);

            // Direct lighting at the current hit is independent of which outgoing direction is selected.
            accumulatedRadiance += (bsdfPayload.material.emissive + bsdfPayload.directionLightRadiance) * bsdf_cosine_pdf;

            float3 bsdfSampleDir = bsdfPayload.incidence;
            float3 bsdfSample_bsdf_cosine = bsdfPayload.bsdf * abs(dot(bsdfSampleDir, bsdfPayload.hitNormal));
            float bsdfSample_bsdfPdf = bsdfPayload.pdf;
            if (!(bsdfSample_bsdfPdf >= 0.0f) || isinf(bsdfSample_bsdfPdf)) bsdfSample_bsdfPdf = 0.0f;

            ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
            ray.Direction = bsdfSampleDir;
            bool bsdfHit = traceRay(ray, bsdfPayload);
            pgSeed = bsdfPayload.randomSeed;

            //-------------------------------------------------- MIS weights ----------------------------------------------------
            float3 pgSample_bsdf_cosine = float3(0.0f);
            float pgSample_bsdfPdf = 0.0f;
            float pgSample_weight = 0.0f;

            if (validSample) {
                float cosine = abs(dot(payload.hitNormal, vrSampleData.sampleDir));
                pgSample_bsdf_cosine = cosine * getBSDF(payload.material, vrSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                pgSample_bsdfPdf = getPdf(payload.material, vrSampleData.sampleDir, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                if (!(pgSample_bsdfPdf >= 0.0f) || isinf(pgSample_bsdfPdf)) pgSample_bsdfPdf = 0.0f;
                pgSample_weight = vrSampleData.pdf / max(vrSampleData.pdf + pgSample_bsdfPdf, 1e-16f);
            }

            float bsdfSample_pgPdf = 0.0f;
            if (validSample && bsdfHit) {
                if (pushConst.spatialRuse == 1) {
                    bsdfSample_pgPdf = getPgPdf_SPReuse(vrSampleData.wSum,
                        bsdfPayload.hitPos, bsdfPayload.hitNormal, payload.hitPos,
                        bsdfSampleDir, bsdfSample_bsdf_cosine, threadIndex);
                } else {
                    bsdfSample_pgPdf = getPgPdf2(vrSampleData.wSum, vrSampleData.nodeLabel_G, vrSampleData.outgoingIndex,
                        bsdfPayload.hitPos, bsdfPayload.hitNormal, payload.hitPos,
                        bsdfSampleDir, bsdfSample_bsdf_cosine);
                }
                if (!(bsdfSample_pgPdf >= 0.0f) || isinf(bsdfSample_pgPdf)) bsdfSample_pgPdf = 0.0f;
            }

            float bsdfSample_weight = bsdfSample_bsdfPdf / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);

            #ifndef NDEBUG
            if (bounceDepth > 0) {
                float3 pgValue_temp = (bsdfPayload.material.emissive + bsdfPayload.directionLightRadiance) * bsdf_cosine_pdf;
                pgValue += pgValue_temp;
                pgValue2 += pgValue_temp * pgValue_temp;
            }
            #endif

            //-------------------------------------------------- choose sample -------------------------------------------------
            // A technique whose contribution or MIS weight is zero cannot contribute; do not waste a path on it.
            bool pgAvailable = validSample && (pgSample_weight > 0.0f) && any(pgSample_bsdf_cosine > 0.0f);
            bool bsdfAvailable = (bsdfSample_bsdfPdf > 0.0f) && (bsdfSample_weight > 0.0f) && any(bsdfSample_bsdf_cosine > 0.0f);
            if (!pgAvailable && !bsdfAvailable) break;

            float pgSample_selectProbability = pgAvailable ? (bsdfAvailable ? 0.5f : 1.0f) : 0.0f;
            float randomNumber = rand(pgSeed);

            if (randomNumber < pgSample_selectProbability) {
                payload = vrPayload;
                payload.incidence = vrSampleData.sampleDir;
                bsdf_cosine_pdf *= pgSample_weight * pgSample_bsdf_cosine / max(vrSampleData.pdf, 1e-16f) / max(pgSample_selectProbability, 1e-16f);
                hit = vrHit;
            }
            else {
                payload = bsdfPayload;
                bsdf_cosine_pdf *= bsdfSample_weight * bsdfSample_bsdf_cosine / max(bsdfSample_bsdfPdf, 1e-16f) / max(1.0f - pgSample_selectProbability, 1e-16f);
                hit = bsdfHit;
            }

            if (!hit) {
                // Let the next iteration add the sky contribution with the selected direction.
                ray.Direction = payload.incidence;
            }

            bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
            if (end) break;

            randomNumber = rand(pgSeed);
            if (randomNumber >= RR) break;
            bsdf_cosine_pdf /= RR;

            payload.randomSeed = pgSeed;
            payload.outgoing = -payload.incidence;
            ++bounceDepth;
        }
    }
    accumulatedRadiance /= pushConst.spp;

    #ifndef NDEBUG
    uint n = max(pushConst.spp, 1);
    pgValue /= n;
    pgValue2 /= n;
    if (pushConst.frameIndex > 0) {
        pgValue = (pgValue + pgValueImage[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;
        pgValue2 = (pgValue2 + pgValue2Image[threadIndex].xyz * (pushConst.frameIndex - 1)) / pushConst.frameIndex;

        pgVarianceImage[threadIndex].x = luminance(pgValue2 - pgValue * pgValue);
    }
    pgValueImage[threadIndex].xyz = pgValue;
    pgValue2Image[threadIndex].xyz = pgValue2;
    #endif

    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0) {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}