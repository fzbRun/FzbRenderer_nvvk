#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "renderer/FzbPathGuidingRenderer/shaders/FzbPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<FzbPathGuidingPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_FzbPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];
[[vk::binding(StaticBindingPoints_FzbPG::eCandidateNodeData_E)]] RWStructuredBuffer<CandidateNodeData_E_FzbPG, ScalarDataLayout> candidateNodeDataBuffer_E;
[[vk::binding(StaticBindingPoints_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_FzbPG::eDepthImage, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
[[vk::binding(StaticBindingPoints_FzbPG::ePGValueImage, 0)]] RWTexture2D<float4> pgValueImage;
[[vk::binding(StaticBindingPoints_FzbPG::ePGValue2Image, 0)]] RWTexture2D<float4> pgValue2Image;
[[vk::binding(StaticBindingPoints_FzbPG::ePGVarianceImage, 0)]] RWTexture2D<float4> pgVarianceImage;
#endif

typedef CallablePayload HitPayload_FzbPG;
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared OctreeGlobalInfo_FzbPG groupGlobalInfo;
groupshared OctreeNodeInfo_G_FzbPG groupOctreeNodeInfo_G[3510];
//----------------------------------------------Hit Function----------------------------------------------
bool traceRay(RayDesc ray, inout HitPayload_FzbPG payload)
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
    float3 voxelSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - OCTREE_CLUSTER_LAYER_FZBPG));
    int3 voxelIndex = int3((hitPos - pushConst.VGBStartPos_Size.xyz) / voxelSize);
    int voxelCount = 1 << OCTREE_CLUSTER_LAYER_FZBPG;
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

bool getNodeAABB_E_FzbPG(inout AABB aabb, inout HitPayload_FzbPG payload, inout float nodeEPdf, inout float jitterPdf,
                         inout int nodeLabel_G, inout uint outgoingIndex, uint2 threadIndex)
{
    float3 geoNormal = payload.isExt ? payload.hitNormal : -payload.hitNormal;
    uint normalIndex = getNormalIndex(geoNormal);
    float3 hitPos = payload.hitPos;

    OctreeNodeInfo_G_FzbPG nodeInfo_G = groupOctreeNodeInfo_G[normalIndex];
    nodeLabel_G = int(nodeInfo_G.label_indivisible >> 1) - 1;
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

    float weightSampleProbability = rand(payload.randomSeed);
    float3 outgoing = mul(pushConst.randomRotateMatrix, payload.outgoing);
    float3 hitNormal = mul(pushConst.randomRotateMatrix, payload.hitNormal);
    outgoingIndex = inverseOutgoing_FzbPG(outgoing, OUTGOING_COUNT_FZBPG, weightSampleProbability, hitNormal);
    //jitterPdf *= weightSampleProbability;

    uint indivisibleNodeCount_G = groupGlobalInfo.indivisibleNodeCount_G;
    uint candidateNodeStartIndex = outgoingIndex * indivisibleNodeCount_G * Candidate_Samples_Count + nodeLabel_G * Candidate_Samples_Count;

    float randomNumber = rand(payload.randomSeed);
    float sampleP_hat = 0.0f; float weightSum = 0.0f;
    for (int i = 0; i < Candidate_Samples_Count; ++i) {
        CandidateNodeData_E_FzbPG candidateNodeData = candidateNodeDataBuffer_E[candidateNodeStartIndex + i];
        weightSum += candidateNodeData.w;

        if (candidateNodeData.w / max(weightSum, 1e-6f) >= rand(payload.randomSeed)) {
            aabb = candidateNodeData.aabb;
            sampleP_hat = candidateNodeData.p_hat;
        }
    }
    if (weightSum == 0.0f) return false;

    nodeEPdf = sampleP_hat / max(weightSum, 1e-10f); // pdf = 1 / W = 1 / ( weightSum / p_hat);
    return true;
}

float getPgPdf(uint nodeLabel_G, uint outgoingIndex, float3 hitPos, float3 hitNormal, float3 viewPos, uint2 threadIndex) {
    uint nodeIndex_E = getNodeIndex_E(hitPos, hitNormal);
    OctreeNodeData_E_FzbPG nodeData_E = OctreeNodeDataBuffer_E[pushConst.maxOctreeLayer][nodeIndex_E];

    uint indivisibleNodeCount_G = groupGlobalInfo.indivisibleNodeCount_G;
    uint candidateNodeStartIndex = outgoingIndex * indivisibleNodeCount_G * Candidate_Samples_Count + nodeLabel_G * Candidate_Samples_Count;
    float p_hat = 0.0f;
    float wSum = 0.0f;
    for (int i = 0; i < Candidate_Samples_Count; ++i) {
        CandidateNodeData_E_FzbPG candidateNodeData = candidateNodeDataBuffer_E[candidateNodeStartIndex + i];
        if (candidateNodeData.subTreeNodeIndex_layer == -1) continue;

        int subTreeNodeIndex = candidateNodeData.subTreeNodeIndex_layer >> 4;
        int subTreeLayer = candidateNodeData.subTreeNodeIndex_layer & 15;
        int leafNodeCount = (1 << (3 * (pushConst.maxOctreeLayer - subTreeLayer)));
        int leafNodeStartIndex = subTreeNodeIndex * (1 << (3 * (pushConst.maxOctreeLayer - subTreeLayer)));
        if (nodeIndex_E >= leafNodeStartIndex && nodeIndex_E < leafNodeStartIndex + leafNodeCount)
            p_hat = dot(candidateNodeData.radiance, 1.0f) / max(dot(nodeData_E.radiance, 1.0f), 1e-6f) * candidateNodeData.p_hat;

        wSum += candidateNodeData.w;
    }
    float pg_Pdf = p_hat / max(wSum, 1e-10f);
    if (pg_Pdf == 0.0f) return pg_Pdf;

    // getPdf from Sphere sample
    {
        float3 distance = nodeData_E.aabb.maximum - nodeData_E.aabb.minimum;
        float3 nodeDirection = (nodeData_E.aabb.maximum + nodeData_E.aabb.minimum) * 0.5f - viewPos;

        float3 edge0 = float3(distance.x, 0.0f, 0.0f);
        float3 edge1 = float3(0.0f, distance.y, 0.0f);
        float3 edge2 = float3(0.0f, 0.0f, distance.z);

        float3 face0StartPos = nodeData_E.aabb.minimum;
        face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
        float3 face1StartPos = nodeData_E.aabb.minimum;
        face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
        float3 face2StartPos = nodeData_E.aabb.minimum;
        face2StartPos.z += nodeDirection.z < 0 ? distance.z : 0.0f; // in back

        Quadrilateral_initialization quad0 = initQuadrilateral(viewPos, face0StartPos, edge2, edge1);
        quad0.S = isnan(quad0.S) ? 0.0f : quad0.S;
        Quadrilateral_initialization quad1 = initQuadrilateral(viewPos, face1StartPos, edge0, edge2);
        quad1.S = isnan(quad1.S) ? 0.0f : quad1.S;
        Quadrilateral_initialization quad2 = initQuadrilateral(viewPos, face2StartPos, edge0, edge1);
        quad2.S = isnan(quad2.S) ? 0.0f : quad2.S;

        float solidAngleSum = quad0.S + quad1.S + quad2.S;
        if (solidAngleSum == 0.0f) {}
        else pg_Pdf /= solidAngleSum;
    }

    return pg_Pdf;
}
//---------------------------------------------PathGuiding-----------------------------------------
[numthreads(FZB_PATHGUIDING_THREADGROUP_SIZE_X, FZB_PATHGUIDING_THREADGROUP_SIZE_Y, 1)]
[shader("compute")]
void computeMain_FzbPathGuiding(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
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

    HitPayload_FzbPG payload;
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

        int nodeLabel_G = 0;
        uint outgoingIndex = 0;

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
                float jitterPdf = 1.0f;

                float3 pgSample;
                float pgSample_pgPdf = 0.0f;
                float pgSample_bsdfPdf = 0.0f;
                float3 pgSample_bsdf = float3(0.0f);
                AABB nodeAABB_E;
                bool useFzbPG = getNodeAABB_E_FzbPG(nodeAABB_E, payload, pgSample_pgPdf, jitterPdf, nodeLabel_G, outgoingIndex, threadIndex);

                if (useFzbPG) {
                    useFzbPG = SphericalRectangleSample(nodeAABB_E, payload.hitPos, payload.randomSeed, pgSample, pgSample_pgPdf, threadIndex);
                    if (useFzbPG) {
                        pgSample_bsdf = getBSDF(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                        pgSample_bsdfPdf = getPdf(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                    }
                }
                if (!useFzbPG){
                    pgSample_pgPdf = 0.0f;
                    pgSample_bsdfPdf = 0.0f;
                }
                float pgSample_weight = 1.0f / max(pgSample_pgPdf + pgSample_bsdfPdf, 1e-16f);

                getDirectionLight(payload, sceneInfo);
                BSDF_SAMPLE(payload);

                accumulatedRadiance += (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;

                float3 bsdfSample = payload.incidence;
                float3 bsdfSample_bsdf = payload.bsdf;
                float bsdfSample_bsdfPdf = payload.pdf;

                float3 hitNormal = payload.hitNormal;
                float3 hitPos = payload.hitPos;
                ray.Origin = offsetRay(hitPos, hitNormal);
                ray.Direction = bsdfSample;
                hit = traceRay(ray, payload);       //after here, payload is changed to next hitPos's payload!!!!!!

                float bsdfSample_pgPdf = 0.0f;
                if (useFzbPG && hit) {
                    bsdfSample_pgPdf = getPgPdf(nodeLabel_G, outgoingIndex, payload.hitPos, payload.hitNormal, hitPos, threadIndex);
                    //bsdfSample_pgPdf *= jitterPdf;
                }
                float bsdfSample_weight = 1.0f / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);

                #ifndef NDEBUG
                if (bounceDepth > 0) {
                    float3 pgValue_temp = (payload.material.emissive + payload.directionLightRadiance) * bsdf_cosine_pdf;
                    pgValue += pgValue_temp;
                    pgValue2 += pgValue_temp * pgValue_temp;
                }
                #endif

                // MIS
                //float probabilitySum = max(pgSample_pgPdf * pgSample_bsdfPdf + bsdfSample_bsdfPdf, 1e-16f);
                //float pgSample_selectProbability = pgSample_pgPdf * pgSample_bsdfPdf / probabilitySum;
                float probabilitySum = max(pgSample_pgPdf * pgSample_weight + bsdfSample_bsdfPdf * bsdfSample_weight, 1e-16f);
                float pgSample_selectProbability = pgSample_pgPdf * pgSample_weight / probabilitySum;
                float randomNumber = rand(payload.randomSeed);
                if (randomNumber <= pgSample_selectProbability) {
                    payload.incidence = pgSample;
                    bsdf_cosine_pdf *= pgSample_weight * pgSample_bsdf / pgSample_selectProbability;
                } else {
                    useBsdfSample = true;
                    payload.incidence = bsdfSample;
                    bsdf_cosine_pdf *= bsdfSample_weight * bsdfSample_bsdf / (1.0f - pgSample_selectProbability);
                }

                float cosine = dot(hitNormal, payload.incidence);
                bsdf_cosine_pdf *= cosine;

                //bool end = any(bsdf_cosine_pdf <= 0.0f) || all(bsdf_cosine_pdf <= 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
                bool end = any(bsdf_cosine_pdf < 0.0f) || all(bsdf_cosine_pdf == 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
                if (end) break;

                //randomNumber = rand(payload.randomSeed);
                //if (randomNumber >= RR) break;
                //bsdf_cosine_pdf /= RR;

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