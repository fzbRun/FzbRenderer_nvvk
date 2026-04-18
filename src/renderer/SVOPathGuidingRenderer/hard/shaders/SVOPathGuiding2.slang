#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/Weight/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<SVOPathGuidingPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOPG::eSVO_G)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(StaticBindingPoints_SVOPG::eNodeData_E)]] StructuredBuffer<OctreeNodeData_E, ScalarDataLayout> NodeData_E;
[[vk::binding(StaticBindingPoints_SVOPG::eGlobalInfo)]] StructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOPG::eWeights)]] StructuredBuffer<float> SVOWeights;
#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_SVOPG::eDepthImage, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
#endif

typedef CallablePayload HitPayload_SVOPG;
struct NodeInfo_G {
    uint label_indivisible;
    AABB aabb;
};
struct NodeInfo_E {
    float pdf;
    AABB aabb;
};
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_SVO_LAYER];
groupshared GlobalInfo_SOVPG groupGlobalInfo;

#if SVOSize_G > 1000
#define GroupNodeInfoCount_G 1000
#else
#define GroupNodeInfoCount_G SVOSize_G
#endif
groupshared NodeInfo_G groupNodeInfos_G[GroupNodeInfoCount_G];  //28KB
groupshared NodeInfo_E groupNodeInfos_E[NODECOUNT_E];           //12KB
//---------------------------------------------Function-----------------------------------------
bool traceRay(RayDesc ray, inout HitPayload_SVOPG payload)
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

        payload.hitNormal = normalize(mul(worldToObject, geoNormal).xyz);
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
bool SphericalRectangleSample(AABB aabb, float3 hitPos, inout uint randomSeed, inout float3 sampleDir, inout float pdf) {
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
        sampleDir = sphericalRectangleSample(sampleQuad, randomU, randomV, pdf);
    }

    //bool printfAble = all(abs(hitPos - float3(-0.974075, 1.990000, -0.309935)) < 0.00001f);
    //printfAble = printfAble && pushConst.frameIndex == 0;
    //if (printfAble) {
    //    printf("pdf: %f, srspdf: %f quad.S: %f %f %f\nhitPos: %f %f %f\naabb: %f %f %f %f %f %f\n\n", 
    //        pdf, sphericalRectangleSampleProbability,
    //        quad0.S, quad1.S, quad2.S,
    //        hitPos.x, hitPos.y, hitPos.z,
    //        aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
    //        aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
    //     );
    //}

    return true;
}
bool getNodeAABB_E_SVOPG(inout AABB aabb, inout HitPayload_SVOPG payload, inout float pdf,
    inout uint nodeLabel_G, inout uint outgoingIndex) 
{
    float3 geoNormal = payload.isExt ? payload.hitNormal : -payload.hitNormal;
    uint normalIndex = getNormalIndex(geoNormal);

    float3 hitPos = payload.hitPos;

    SVONodeData_G nodeData_G;
    NodeInfo_G nodeInfo_G = groupNodeInfos_G[normalIndex];
    nodeLabel_G = (nodeInfo_G.label_indivisible >> 1) - 1;
    bool indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;
    bool hitIndivisible = false, hitNode = false;
    if (!indivisible) {
        uint nodeLayerLabel = nodeLabel_G;
        nodeLabel_G = groupSVOLayerInfos_G[1].indivisibleNodeCount;
        uint nodeIndexSum_G = 8;
        uint nodeIndex_G = nodeLayerLabel * 8 + nodeIndexSum_G;
        uint layerIndex_G = 2;
        for (; layerIndex_G <= groupGlobalInfo.SVOMaxLayer_G && nodeIndex_G < GroupNodeInfoCount_G; ++layerIndex_G) {
            hitNode = false;
            for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                nodeInfo_G = groupNodeInfos_G[nodeIndex_G + nodeOffset];
                nodeLayerLabel = (nodeInfo_G.label_indivisible >> 1) - 1;
                if (checkPointInAABB(hitPos, nodeInfo_G.aabb)) {
                    if ((nodeInfo_G.label_indivisible & 1u) == 1) {
                        hitIndivisible = true;
                        nodeLabel_G += nodeLayerLabel;
                    } else {
                        nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                        nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                        nodeIndex_G = nodeIndexSum_G + nodeLayerLabel * 8;
                    }
                    hitNode = true;
                    break;
                }
            }
            if (!hitNode) {
                float weights[8]; float maxWeight = 0.0f;
                for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                    nodeInfo_G = groupNodeInfos_G[nodeIndex_G + nodeOffset];
                    if (nodeInfo_G.aabb.minimum.x > 1e10f) {
                        weights[nodeOffset] = 0.0f;
                        continue;
                    }

                    float distance = PointToAABBDistance(hitPos, nodeInfo_G.aabb);
                    weights[nodeOffset] = 1.0f / distance;
                    maxWeight = max(maxWeight, weights[nodeOffset]);
                }
                for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                    if (maxWeight == weights[nodeOffset]) {
                        nodeInfo_G = groupNodeInfos_G[nodeIndex_G + nodeOffset];
                        nodeLayerLabel = (nodeInfo_G.label_indivisible >> 1) - 1;
                        if ((nodeInfo_G.label_indivisible & 1u) == 1) {
                            hitIndivisible = true;
                            nodeLabel_G += nodeLayerLabel;
                        } else {
                            nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                            nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                            nodeIndex_G = nodeIndexSum_G + nodeLayerLabel * 8;
                        }
                        hitNode = true;
                        break;
                    }
                }
            }
            if (hitIndivisible) break;
        }
        if (hitNode && !hitIndivisible) {
            for (; layerIndex_G <= groupGlobalInfo.SVOMaxLayer_G; ++layerIndex_G) {
                hitNode = false;
                for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                    nodeData_G = SVO_G[nodeIndex_G + nodeOffset];
                    if (checkPointInAABB(hitPos, nodeData_G.aabb)) {
                        if (nodeData_G.indivisible == 1) {
                            hitIndivisible = true;
                            nodeLabel_G += nodeData_G.label - 1;
                        } else {
                            nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                            nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                            nodeIndex_G = nodeIndexSum_G + (nodeData_G.label - 1) * 8;
                        }
                        hitNode = true;
                        break;
                    }
                }
                if (!hitNode) {
                    float weights[8]; float maxWeight = 0.0f;
                    for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                        nodeData_G = SVO_G[nodeIndex_G + nodeOffset];
                        if (nodeData_G.aabb.minimum.x > 1e10f) {
                            weights[nodeOffset] = 0.0f;
                            continue;
                        }

                        float distance = PointToAABBDistance(hitPos, nodeData_G.aabb);
                        weights[nodeOffset] = 1.0f / distance;
                        maxWeight = max(maxWeight, weights[nodeOffset]);
                    }
                    for(int nodeOffset = 0; nodeOffset < 8; ++nodeOffset){
                        if (maxWeight == weights[nodeOffset]) {
                            nodeData_G = SVO_G[nodeIndex_G + nodeOffset];
                            if (nodeData_G.indivisible == 1) {
                                hitIndivisible = true;
                                nodeLabel_G += nodeData_G.label - 1;
                            } else {
                                nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                                nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                                nodeIndex_G = nodeIndexSum_G + (nodeData_G.label - 1) * 8;
                            }
                            hitNode = true;
                            break;
                        }
                    }
                }
                if (hitIndivisible) break;
            }
        }
    } else hitIndivisible = true;
    if (!hitIndivisible) return false;

    float weightSampleProbability = rand(payload.randomSeed);
    float3 outgoing = mul(pushConst.randomRotateMatrix, payload.outgoing);
    outgoingIndex = inverseOutgoing(outgoing, OUTGOING_COUNT, weightSampleProbability);
    pdf *= weightSampleProbability;
    //uint outgoingIndex = inverseSF(payload.outgoing, OUTGOING_COUNT);

    uint weightStartIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * OCTREE_NODECOUNT_E +
                            nodeLabel_G * OCTREE_NODECOUNT_E;

    uint nodeIndex_E = 0, nodeOffset; bool getNode = false;
    for (uint layerIndex_E = 0; layerIndex_E <= OCTREE_CLUSTER_LAYER; ++layerIndex_E) {
        getNode = false;
        float randomNumber = rand(payload.randomSeed);
        for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            uint weightIndex = weightStartIndex + nodeIndex_E + nodeOffset;
            float probability = SVOWeights[weightIndex];
            if (randomNumber <= probability) {
                pdf *= probability;
                if (layerIndex_E != OCTREE_CLUSTER_LAYER)
                    nodeIndex_E = OctreeLayerStartIndex_E[layerIndex_E + 1] + (nodeIndex_E - OctreeLayerStartIndex_E[layerIndex_E] + nodeOffset) * 8;
                getNode = true;
                break;
            }
            randomNumber -= probability;
        }
        if (getNode == false) break;
    }
    if (getNode == false) return false;
    nodeIndex_E = nodeIndex_E + nodeOffset - OctreeLayerStartIndex_E[OCTREE_CLUSTER_LAYER];

    NodeInfo_E nodeInfo_E = groupNodeInfos_E[nodeIndex_E];
    aabb = nodeInfo_E.aabb;
    pdf *= nodeInfo_E.pdf;

    //bool printfAble = all(abs(hitPos - float3(-0.974075, 1.990000, -0.309935)) < 0.00001f);
    //printfAble = printfAble && pushConst.frameIndex == 0;
    //if (printfAble) {
    //    printf("pdf: %f, nodeInfo_E.pdf: %f\nhitPos: %f %f %f\naabb: %f %f %f %f %f %f\n\n", 
    //        payload.pdf, nodeInfo_E.pdf,
    //        hitPos.x, hitPos.y, hitPos.z,
    //        aabb.minimum.x, aabb.minimum.y, aabb.minimum.z,
    //        aabb.maximum.x, aabb.maximum.y, aabb.maximum.z,
    //    );
    //}
    return true;
}

int getNodeIndex_E(float3 hitPos, float3 hitNormal) {
    float3 voxelSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - OCTREE_CLUSTER_LAYER));
    int3 voxelIndex = int3((hitPos - pushConst.VGBStartPos_Size.xyz) / voxelSize);
    int voxelCount = 1 << OCTREE_CLUSTER_LAYER;
    int voxelTotalCount = voxelCount * voxelCount * voxelCount;

    int voxelIndexU = 0;
    while (voxelTotalCount > 1) {
        voxelCount = voxelCount / 2;
        voxelTotalCount = voxelTotalCount / 8;

        uint3 offset = voxelIndex / voxelCount;

        voxelIndexU += offset.z * 4 * voxelTotalCount;
        voxelIndex.z -= offset.z * voxelCount;

        voxelIndexU += offset.y * 2 * voxelTotalCount;
        voxelIndex.y -= offset.y * voxelCount;

        voxelIndexU += offset.x * voxelTotalCount;
        voxelIndex.x -= offset.x * voxelCount;
    }
    uint normalIndex = getNormalIndex(hitNormal);
    return voxelTotalCount * normalIndex + voxelIndexU;
}
float getPgPdf(uint nodeLabel_G, uint outgoingIndex, float3 hitPos, float3 hitNormal, float3 viewPos) {
    uint weightStartIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * OCTREE_NODECOUNT_E + nodeLabel_G * OCTREE_NODECOUNT_E;

    float pg_Pdf = 1.0f;
    uint nearbyNodeIndex_E = getNodeIndex_E(hitPos, hitNormal);
    uint nodeIndex = nearbyNodeIndex_E;
    for (int layerIndex = OCTREE_CLUSTER_LAYER; layerIndex >= 0; --layerIndex) {
        uint weightIndex = weightStartIndex + OctreeLayerStartIndex_E[layerIndex] + nodeIndex;
        pg_Pdf *= SVOWeights[weightIndex];
        nodeIndex /= 8;
    }

    NodeInfo_E nodeInfo_E = groupNodeInfos_E[nearbyNodeIndex_E];
    pg_Pdf *= nodeInfo_E.pdf;
    //if (checkPointInAABB(hitPos, nodeInfo_E.aabb)) pg_Pdf *= nodeInfo_E.pdf;
    //else pg_Pdf *= (1.0f - nodeInfo_E.pdf) / pow(8, pushConst.maxOctreeLayer - OCTREE_CLUSTER_LAYER);

    // getPdf from Sphere sample
    {
        float3 distance = nodeInfo_E.aabb.maximum - nodeInfo_E.aabb.minimum;
        float3 nodeDirection = (nodeInfo_E.aabb.maximum + nodeInfo_E.aabb.minimum) * 0.5f - viewPos;

        float3 edge0 = float3(distance.x, 0.0f, 0.0f);
        float3 edge1 = float3(0.0f, distance.y, 0.0f);
        float3 edge2 = float3(0.0f, 0.0f, distance.z);

        float3 face0StartPos = nodeInfo_E.aabb.minimum;
        face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
        float3 face1StartPos = nodeInfo_E.aabb.minimum;
        face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
        float3 face2StartPos = nodeInfo_E.aabb.minimum;
        face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f; // in back

        Quadrilateral_initialization quad0 = initQuadrilateral(viewPos, face0StartPos, edge2, edge1);
        quad0.S = isnan(quad0.S) ? 0.0f : quad0.S;
        Quadrilateral_initialization quad1 = initQuadrilateral(viewPos, face1StartPos, edge0, edge2);
        quad1.S = isnan(quad1.S) ? 0.0f : quad1.S;
        Quadrilateral_initialization quad2 = initQuadrilateral(viewPos, face2StartPos, edge0, edge1);
        quad2.S = isnan(quad2.S) ? 0.0f : quad2.S;

        float solidAngleSum = quad0.S + quad1.S + quad2.S;
        if (solidAngleSum == 0.0f) {}
        else {
            //float3 aabbCenter = (nodeInfo_E.aabb.minimum + nodeInfo_E.aabb.maximum) * 0.5f;
            //float3 faceXYZ = abs(hitPos - aabbCenter);
            //int faceIndex = faceXYZ.x > faceXYZ.y ? (faceXYZ.x > faceXYZ.z ? 0 : 2) : (faceXYZ.y > faceXYZ.z ? 1 : 2);

            // float3 sphereQuadrilateralSolidAngle = float3(quad0.S, quad1.S, quad2.S);
            // float3 sphereQuadrilateralSolidAngleProbability = sphereQuadrilateralSolidAngle / solidAngleSum;
            // pg_Pdf *= sphereQuadrilateralSolidAngleProbability[faceIndex] / sphereQuadrilateralSolidAngle[faceIndex];
            pg_Pdf /= solidAngleSum;
        }
    }
    return pg_Pdf;
}
//---------------------------------------------PathGuiding-----------------------------------------
[numthreads(SVO_PATHGUIDING_THREADGROUP_SIZE_X, SVO_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_SVOPathGuiding(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndex_linear = groupThreadIndex.y * SVO_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = SVO_PATHGUIDING_THREADGROUP_SIZE_X * SVO_PATHGUIDING_THREADGROUP_SIZE_Y;
    if (groupThreadIndex_linear < MAX_SVO_LAYER) {
        SVOWeightGlobalInfo weightGlobalInfo = GlobalInfo[0];
        groupSVOLayerInfos_G[groupThreadIndex_linear] = weightGlobalInfo.layerInfos_G[groupThreadIndex_linear];
        if (groupThreadIndex_linear == 0) {
            GlobalInfo_SOVPG globalInfo;
            globalInfo.SVOMaxLayer_G = weightGlobalInfo.SVOMaxLayer_G;
            globalInfo.indivisibleNodeCount_G = weightGlobalInfo.indivisibleNodeCount_G;
            groupGlobalInfo = globalInfo;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (int i = groupThreadIndex_linear; i < GroupNodeInfoCount_G; i += threadGroupThreadTotalCount) {
        SVONodeData_G nodeData_G = SVO_G[i];
        NodeInfo_G nodeInfo_G;
        uint label = nodeData_G.label;
        bool indivisible = (nodeData_G.meanNormal.w > 0.0f) && nodeData_G.indivisible == 1;
        nodeInfo_G.label_indivisible = (label << 1) | (indivisible ? 1u : 0u);
        nodeInfo_G.aabb = nodeData_G.aabb;
        groupNodeInfos_G[i] = nodeInfo_G;
    }
    for (int i = groupThreadIndex_linear; i < NODECOUNT_E; i += threadGroupThreadTotalCount) {
        OctreeNodeData_E nodeData = NodeData_E[i];
        NodeInfo_E nodeInfo;
        nodeInfo.pdf = nodeData.pdf;
        nodeInfo.aabb = nodeData.aabb;
        groupNodeInfos_E[i] = nodeInfo;
    }
    GroupMemoryBarrierWithGroupSync();

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    uint seed = xxhash32(uint3(threadIndex, pushConst.frameIndex));
    float r1 = rand(seed);
    float r2 = rand(seed);
    float2 subpixel_jitter = pushConst.frameIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 clipCoords = pixelCenter / launchSize * 2.0 - 1.0;
    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    RayDesc ray;
    ray.Origin = sceneInfo.cameraPosition; // mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_SVOPG payload;
    payload.randomSeed = seed;
    payload.bsdf = float3(1.0f);
    payload.pdf = 1.0f;
    payload.outgoing = -ray.Direction;
    payload.isExt = true;

    float3 bsdf_cosine_pdf = float3(1.0f);

    uint nodeLabel_G = 0;
    uint outgoingIndex = 0;

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    uint bounceDepth = 0;
    while (bounceDepth < pushConst.maxDepth) {
        bool hit = traceRay(ray, payload);

        #ifndef NDEBUG
        if (bounceDepth == 0) {
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
            accumulatedRadiance += payload.material.emissive * bsdf_cosine_pdf;

            float3 pgSample;
            float pgSample_pgPdf = 1.0f;
            float pgSample_bsdfPdf = 1.0f;
            float pgSample_weight = 1.0f;
            float3 pgSample_bsdf = 0.0f;
            {
                AABB nodeAABB_E;
                bool useSVOPG = getNodeAABB_E_SVOPG(nodeAABB_E, payload, pgSample_pgPdf, nodeLabel_G, outgoingIndex);
                if (useSVOPG) useSVOPG = SphericalRectangleSample(nodeAABB_E, payload.hitPos, payload.randomSeed, pgSample, pgSample_pgPdf);
                if (useSVOPG) {
                    pgSample_bsdf = getBSDF(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    // if (dot(payload.incidence, payload.hitNormal) * dot(payload.outgoing, payload.hitNormal) < 0.0f) payload.isExt = !payload.isExt;

                    pgSample_bsdfPdf = getPdf(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                }
                pgSample_weight = pgSample_pgPdf / (pgSample_pgPdf + pgSample_bsdfPdf);

                // bool useSVOPG = getNodeAABB_E_jittering_SVOPG(nodeAABB_E, payload);
                // if (useSVOPG == false) accumulatedRadiance = float3(10.0f, 0.0f, 0.0f);
                // else if (useSVOPG == true) accumulatedRadiance = float3(0.0f, 10.0f, 0.0f);
                // break;
            }

            float3 bsdfSample;
            float bsdfSample_pgPdf = 1.0f;
            float bsdfSample_bsdfPdf = 1.0f;
            float bsdfSample_weight = 0.0f;
            float3 bsdfSample_bsdf = 0.0f;
            {
                BSDF_SAMPLE(payload);
                bsdfSample = payload.incidence;
                bsdfSample_bsdf = payload.bsdf;

                RayDesc ray_bsdfSample;
                ray_bsdfSample.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                ray_bsdfSample.Direction = bsdfSample;
                ray_bsdfSample.TMin = 0.001;
                ray_bsdfSample.TMax = INFINITE;

                HitPayload_SVOPG payload_bsdfSample;
                bool hit = traceRay(ray_bsdfSample, payload_bsdfSample);
                if (hit) {
                    bsdfSample_pgPdf = getPgPdf(nodeLabel_G, outgoingIndex, payload_bsdfSample.hitPos, payload_bsdfSample.hitNormal, ray_bsdfSample.Origin);
                    bsdfSample_bsdfPdf = payload.pdf;
                    bsdfSample_weight = bsdfSample_bsdfPdf / (bsdfSample_pgPdf + bsdfSample_bsdfPdf);
                }
            }

            //MIS
            {
                float probabilitySum = max(pgSample_weight + bsdfSample_weight, 1e-16f);
                float pgSample_selectProbability = pgSample_weight / probabilitySum;
                float randomNumber = rand(payload.randomSeed);
                if (randomNumber <= pgSample_selectProbability) {
                    payload.incidence = pgSample;
                    bsdf_cosine_pdf *= pgSample_weight / pgSample_pgPdf * pgSample_bsdf;
                } else {
                    payload.incidence = bsdfSample;
                    bsdf_cosine_pdf *= bsdfSample_weight / bsdfSample_bsdfPdf * bsdfSample_bsdf;
                }
            }

            float cosine = dot(payload.hitNormal, payload.incidence);
            if (payload.material.type == MaterialType::Dielectric || payload.material.type == MaterialType::RoughDielectric) cosine = abs(cosine);
            bsdf_cosine_pdf *= cosine;
            payload.outgoing = -payload.incidence;

            bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf <= 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
            if (end) break;

            float randomNumber = rand(payload.randomSeed);
            if (randomNumber >= RR) break;
            bsdf_cosine_pdf /= RR;

            ray.Direction = payload.incidence;
            ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);

            ++bounceDepth;
        }
    }

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}