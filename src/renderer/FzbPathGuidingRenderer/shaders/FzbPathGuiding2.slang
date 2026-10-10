#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"
#include "renderer/FzbPathGuidingRenderer/shaders/FzbPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<FzbPathGuidingPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_FzbPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(StaticBindingPoints_FzbPG::eCandidateNodeData_E)]] RWStructuredBuffer<CandidateNodeData_E_FzbPG, ScalarDataLayout> candidateNodeDataBuffer_E;
[[vk::binding(StaticBindingPoints_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

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
    jitterPdf *= weightSampleProbability;

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