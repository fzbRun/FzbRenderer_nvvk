#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"
#include "feature/SceneDivision/SparseVoxelOctree/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<SVOPathGuidingPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_SVOPG::eSVO_G_SVOPG)]] StructuredBuffer<SVONodeData_G, ScalarDataLayout> SVO_G;
[[vk::binding(StaticBindingPoints_SVOPG::eSVO_E_SVOPG)]] StructuredBuffer<SVONodeData_E, ScalarDataLayout> SVO_E;
[[vk::binding(StaticBindingPoints_SVOPG::eGlobalInfo_SVOPG)]] StructuredBuffer<SVOWeightGlobalInfo> GlobalInfo;
[[vk::binding(StaticBindingPoints_SVOPG::eWeights_SVOPG)]] StructuredBuffer<float> SVOWeights;
#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_SVOPG::eDepthImage_SVOPG, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
#endif

typedef CallablePayload HitPayload_SVOPG;

groupshared SVOLayerInfo groupSVOLayerInfos_G[MAX_OCTREE_DEPTH];
groupshared SVOLayerInfo groupSVOLayerInfos_E[MAX_OCTREE_DEPTH];
groupshared GlobalInfo_SOVPG groupGlobalInfo;

#define GroupNodeInfoCount_G 800
#define GroupNodeInfoCount_E SVOSize_E
groupshared SVONodeInfo_G_SVOPG groupNodeInfos_G[GroupNodeInfoCount_G]; // 32KB
groupshared SVONodeInfo_E_SVOPG groupNodeInfos_E[GroupNodeInfoCount_E]; // 25KB
groupshared float groupNodePdfs_E[SVOIndivisibleNodeCount_E];

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
    face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f;	//in back

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

        sampleDir = sphericalRectangleSample(sampleQuad, randomU, randomV, sphericalRectangleSampleProbability);
        pdf *= sphericalRectangleSampleProbability;
    }
    return true;
}
bool getNodeAABB_E_SVOPG(inout AABB aabb, inout HitPayload_SVOPG payload) {
    float3 geoNormal = payload.isExt ? payload.hitNormal : -payload.hitNormal;
    uint normalIndex = getNormalIndex(geoNormal);

    uint nodeIndex_G = normalIndex * 8, nodeLabel_G = 0, nodeIndexSum_G = 0, layerIndex_G = 1;
    bool hitIndivisible = false, hitNode = false;
    for (; layerIndex_G <= groupGlobalInfo.SVOMaxLayer_G && nodeIndex_G < GroupNodeInfoCount_G; ++layerIndex_G) {
        hitNode = false;
        for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            SVONodeInfo_G_SVOPG nodeInfo_G = groupNodeInfos_G[nodeIndex_G + nodeOffset];
            if (checkPointInAABB(payload.hitPos, nodeInfo_G.aabb)) {
                if ((nodeInfo_G.label_indivisible & 1u) == 1){
                    hitIndivisible = true;
                    nodeLabel_G += nodeInfo_G.label_indivisible  >> 1;
                } else {
                    nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                    nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                    nodeIndex_G = nodeIndexSum_G + ((nodeInfo_G.label_indivisible >> 1) - 1) * 8;
                }
                hitNode = true;
                break;
            }
        }
        if (!hitNode || hitIndivisible) break;
    }
    if (hitNode && !hitIndivisible) {
        for (; layerIndex_G <= groupGlobalInfo.SVOMaxLayer_G; ++layerIndex_G) {
            hitNode = false;
            for (int nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
                SVONodeData_G nodeData_G = SVO_G[nodeIndex_G + nodeOffset];
                if (checkPointInAABB(payload.hitPos, nodeData_G.aabb)) {
                    if (nodeData_G.indivisible == 1) {
                        hitIndivisible = true;
                        nodeLabel_G += nodeData_G.label;
                    } else {
                        nodeLabel_G += groupSVOLayerInfos_G[layerIndex_G].indivisibleNodeCount;
                        nodeIndexSum_G += groupSVOLayerInfos_G[layerIndex_G - 1].divisibleNodeCount * 8;
                        nodeIndex_G = nodeIndexSum_G + (nodeData_G.label - 1) * 8;
                    }
                    hitNode = true;
                    break;
                }
            }
            if (!hitNode || hitIndivisible) break;
        }
    }
    if (!hitIndivisible) return false;

    float pdf = 1.0f;
    float weightSampleProbability = rand(payload.randomSeed);
    uint outgoingIndex = inverseSF(payload.outgoing, OUTGOING_COUNT, weightSampleProbability);
    pdf *= weightSampleProbability;
    //uint outgoingIndex = inverseSF(payload.outgoing, OUTGOING_COUNT);

    uint totalNodeCount_E = groupGlobalInfo.totalNodeCount_E;
    uint weightStartIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * totalNodeCount_E +
                            (nodeLabel_G - 1) * totalNodeCount_E;

    hitIndivisible = false, hitNode = false;
    uint layerIndex_E = 1, nodeIndex_E = 0, nodeIndexSum_E = 0, nodeLabel_E = 0;
    for (; layerIndex_E <= groupGlobalInfo.SVOMaxLayer_E; ++layerIndex_E) {
        hitNode = false;
        float randomNumber = rand(payload.randomSeed);
        for (uint nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            uint weightIndex = weightStartIndex + nodeIndex_E + nodeOffset;
            float probability = SVOWeights[weightIndex];

            if (randomNumber <= probability) {
                pdf *= probability;
                SVONodeInfo_E_SVOPG nodeInfo_E = groupNodeInfos_E[nodeIndex_E + nodeOffset];
                if ((nodeInfo_E.label_indivisible & 1u) == 1) {
                    aabb = nodeInfo_E.aabb;
                    nodeLabel_E += nodeInfo_E.label_indivisible >> 1;
                    hitIndivisible = true;
                } else {
                    nodeIndexSum_E += groupSVOLayerInfos_E[layerIndex_E - 1].divisibleNodeCount * 8;
                    nodeIndex_E = nodeIndexSum_E + ((nodeInfo_E.label_indivisible >> 1) - 1) * 8;
                    nodeLabel_E += groupSVOLayerInfos_E[layerIndex_E].indivisibleNodeCount;
                }
                hitNode = true;
                break;
            }
            randomNumber -= probability;
        }
        if (!hitNode || hitIndivisible) break;
    }
    if (!hitIndivisible) return false;

    payload.pdf = pdf * groupNodePdfs_E[nodeLabel_E - 1];
    return true;
}

void BSDFSample(inout HitPayload_SVOPG payload) {
    if (payload.material.type == MaterialType::Diffuse) PopulateCallablePayload_DiffuseMaterial(payload);
    else if (payload.material.type == MaterialType::Conductor) {
        payload.bsdf = payload.material.albedo;
        payload.pdf = 1.0f;
        payload.incidence = reflect(-payload.outgoing, payload.hitNormal);
    } else if (payload.material.type == MaterialType::Dielectric) {
        PopulateCallablePayload_DielectricMaterial(payload);
    } else if (payload.material.type == MaterialType::RoughConductor) {
        PopulateCallablePayload_RoughConductorMaterial(payload);
    } else if (payload.material.type == MaterialType::RoughDielectric) {
        PopulateCallablePayload_RoughDielectricMaterial(payload);
    }
}
//----------------------------------------------------CS----------------------------------------------
[numthreads(SVO_PATHGUIDING_THREADGROUP_SIZE_X, SVO_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_SVOPathGuiding(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndex_linear = groupThreadIndex.y * SVO_PATHGUIDING_THREADGROUP_SIZE_X + groupThreadIndex.x;
    uint threadGroupThreadTotalCount = SVO_PATHGUIDING_THREADGROUP_SIZE_X * SVO_PATHGUIDING_THREADGROUP_SIZE_Y;
    if (groupThreadIndex_linear < MAX_OCTREE_DEPTH) {
        SVOWeightGlobalInfo weightGlobalInfo = GlobalInfo[0];
        groupSVOLayerInfos_G[groupThreadIndex_linear] = weightGlobalInfo.layerInfos_G[groupThreadIndex_linear];
        groupSVOLayerInfos_E[groupThreadIndex_linear] = weightGlobalInfo.layerInfos_E[groupThreadIndex_linear];
        if (groupThreadIndex_linear == 0) {
            GlobalInfo_SOVPG globalInfo;
            globalInfo.SVOMaxLayer_G = weightGlobalInfo.SVOMaxLayer_G;
            globalInfo.SVOMaxLayer_E = weightGlobalInfo.SVOMaxLayer_E;
            globalInfo.indivisibleNodeCount_G = weightGlobalInfo.indivisibleNodeCount_G;
            globalInfo.totalNodeCount_E = weightGlobalInfo.totalNodeCount_E;
            groupGlobalInfo = globalInfo;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    for (int i = groupThreadIndex_linear; i < GroupNodeInfoCount_G; i += threadGroupThreadTotalCount) {
        SVONodeData_G nodeData_G = SVO_G[i];
        SVONodeInfo_G_SVOPG nodeInfo_G;
        uint label = nodeData_G.label;
        bool indivisible = (nodeData_G.meanNormal.w > 0.0f) && nodeData_G.indivisible == 1;
        nodeInfo_G.label_indivisible = (label << 1) | (indivisible ? 1u : 0u);
        nodeInfo_G.aabb = nodeData_G.aabb;
        groupNodeInfos_G[i] = nodeInfo_G;
    }
    for (int i = groupThreadIndex_linear; i < GroupNodeInfoCount_E; i += threadGroupThreadTotalCount) {
        SVONodeData_E nodeData_E = SVO_E[i];
        SVONodeInfo_E_SVOPG nodeInfo_E;
        uint label = nodeData_E.label;
        bool indivisible = any(nodeData_E.irradiance > 0.0f) && nodeData_E.indivisible == 1;
        nodeInfo_E.label_indivisible = (label << 1) | (indivisible ? 1u : 0u);
        nodeInfo_E.aabb = nodeData_E.aabb;
        groupNodeInfos_E[i] = nodeInfo_E;

        uint nodeLabelSum = 0, nodeIndex = i;
        for (int layerIndex = 1; layerIndex < MAX_OCTREE_DEPTH; ++layerIndex) {
            uint layerNodeCount = groupSVOLayerInfos_E[layerIndex - 1].divisibleNodeCount * 8;
            if (nodeIndex < layerNodeCount) break;
            nodeIndex -= layerNodeCount;
            nodeLabelSum += groupSVOLayerInfos_E[layerIndex].indivisibleNodeCount;
        }
        if (indivisible) groupNodePdfs_E[nodeLabelSum + label - 1] = nodeData_E.pdf;
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

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

    uint bounceDepth = 0;
    while (bounceDepth < pushConst.maxDepth && (bsdf_cosine_pdf.x + bsdf_cosine_pdf.y + bsdf_cosine_pdf.z > 0.0f)) {
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
            AABB nodeAABB_E;
            bool useSVOPG = getNodeAABB_E_SVOPG(nodeAABB_E, payload);
            if (useSVOPG) useSVOPG = SphericalRectangleSample(nodeAABB_E, payload.hitPos, payload.randomSeed, payload.incidence, payload.pdf);
            if (useSVOPG) {
                payload.bsdf = getBSDF(payload.material, payload.incidence, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                if (dot(payload.incidence, payload.hitNormal) * dot(payload.outgoing, payload.hitNormal) < 0.0f) payload.isExt = !payload.isExt;
            }
            //else BSDFSample(payload);
            else break;

            //int useSVOPG = getNodeAABB_E_SVOPG(nodeAABB_E, payload);
            //if (useSVOPG == 2) accumulatedRadiance = float3(10.0f, 0.0f, 0.0f);
            //else if (useSVOPG == 1) accumulatedRadiance = float3(0.0f, 10.0f, 0.0f);
            //else accumulatedRadiance = float3(0.0f, 0.0f, 10.0f);
            //break;

            float cosine = dot(payload.hitNormal, payload.incidence);
            if (payload.material.type == MaterialType::Dielectric || payload.material.type == MaterialType::RoughDielectric) cosine = abs(cosine);
            bsdf_cosine_pdf *= payload.bsdf * cosine / payload.pdf;
            payload.outgoing = -payload.incidence;

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