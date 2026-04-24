#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/FzbPathGuidingRenderer/FzbPathGuidingShaderio.h"
#include "renderer/FzbPathGuidingRenderer/Octree/OctreeShaderio_FzbPG.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<FzbPathGuidingPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeDataBuffer_G[];
[[vk::binding(StaticBindingPoints_FzbPG::eClusterLayerData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> ClusterlayerDataBuffer_E;
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeNodePairVisibleData)]] RWStructuredBuffer<OctreeNodePairVisibleData_FzbPG, ScalarDataLayout> OctreeNodePairVisibleDataBuffer;
[[vk::binding(StaticBindingPoints_FzbPG::eOctreeNodePairWeight)]] RWStructuredBuffer<float> OctreeNodePairWeightBuffer;
[[vk::binding(StaticBindingPoints_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;
#ifndef NDEBUG
[[vk::binding(StaticBindingPoints_FzbPG::eDepthImage, 0)]] RWTexture2D<float> depthImage; // VK_FORMAT_D32_SFLOAT
#endif

typedef CallablePayload HitPayload_SVOPG;
//---------------------------------------------groupsharedMemory-----------------------------------------
groupshared OctreeGlobalInfo_FzbPG groupGlobalInfo;

groupshared OctreeNodeData_G_FzbPG groupOctreeNodeData_G[3510];                               // 13.7KB  6 + 48 + 384 + 3072 = 3510
groupshared OctreeNodeData_E_FzbPG groupClusterLayerNodeData_E[CLUSTER_LAYER_NODECOUNT_E_FZBPG];   //1.71KB
//-------------------------------------------------Function----------------------------------------------
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
bool getNodeAABB_E_SVOPG(inout AABB aabb, inout HitPayload_SVOPG payload, inout float nodeEPdf, inout float jitterPdf,
                         inout int nodeLabel_G, inout uint outgoingIndex)
{
    float3 geoNormal = payload.isExt ? payload.hitNormal : -payload.hitNormal;
    uint normalIndex = getNormalIndex(geoNormal);
    float3 hitPos = payload.hitPos;

    bool printfAble = pushConst.frameIndex == 1;
    printfAble &= all(abs(hitPos - float3(0.188881, 0.000000, - 4.067429)) < 0.0001f);

    OctreeNodeData_G_FzbPG nodeInfo_G = groupOctreeNodeData_G[normalIndex];
    nodeLabel_G = int(nodeInfo_G.label_indivisible >> 1) - 1;
    bool hasData = nodeLabel_G >= 0;
    if (!hasData) return false;

    bool indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;
    bool hitIndivisible = false;
    int layerIndex = 0; int nodeIndex = nodeLabel_G;
    if (indivisible) hitIndivisible = true;
    else {
        nodeLabel_G = groupGlobalInfo.layerInfos_G[0].indivisibleNodeCount;

        float3 nodeSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - 1));
        float3 nodeStartPos = pushConst.VGBStartPos_Size.xyz;

        int nodeSum = 6; int layerNodeCount = 6;
        for (layerIndex = 1; layerIndex <= pushConst.maxOctreeLayer; ++layerIndex) {
            int3 nodeIndex3 = min(int3((hitPos - nodeStartPos) / nodeSize), 1);
            int subNodeIndex = nodeIndex3.z * 4 + nodeIndex3.y * 2 + nodeIndex3.x;
            nodeIndex = nodeIndex * 8 + subNodeIndex;

            hasData = true;
            indivisible = false;
            int nodeLayerLabel = 0;
            if (layerIndex < 4) {
                nodeInfo_G = groupOctreeNodeData_G[nodeSum + nodeIndex];
                layerNodeCount *= 8;
                nodeSum += layerNodeCount;
            } else nodeInfo_G = OctreeDataBuffer_G[layerIndex][nodeIndex];

            nodeLayerLabel = int(nodeInfo_G.label_indivisible >> 1) - 1;
            hasData = nodeLayerLabel >= 0;
            indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;

            if (!hasData) break;
            if (indivisible) {
                nodeLabel_G += nodeLayerLabel;
                hitIndivisible = true;
                break;
            }
            nodeLabel_G += groupGlobalInfo.layerInfos_G[layerIndex].indivisibleNodeCount;
            nodeStartPos += nodeIndex3 * nodeSize;
            nodeSize /= 2.0f;
        }
    }
    if (!hitIndivisible) return false;

    #ifdef NEARBYNODE_JITTER
    {
        if (layerIndex < 4) nodeData_G = OctreeArray_G[layerIndex][nodeIndex];
        float weights[NEARBY_NODE_COUNT + 1];
        weights[0] = 1.0f / getDistanceToAABB(hitPos, nodeData_G.aabb);
        float weightSum = weights[0];
        int2 nearbyNodeInfos[NEARBY_NODE_COUNT] = OctreeNearbyNodeInfos[nodeLabel_G].nearbyNodeInfos;
        int nearbyNodeCount = 0;
        for (; nearbyNodeCount < NEARBY_NODE_COUNT; ++nearbyNodeCount) {
            int2 nearbyNodeInfo = nearbyNodeInfos[nearbyNodeCount];
            if (nearbyNodeInfo.x == -1) break;
            OctreeNodeData_G nearbyNodeData = OctreeArray_G[nearbyNodeInfo.x][nearbyNodeInfo.y];
            weights[nearbyNodeCount + 1] = max(dot(payload.hitNormal, normalize(nearbyNodeData.meanNormal.xyz)), 0.0f) / getDistanceToAABB(hitPos, nearbyNodeData.aabb);
            weightSum += weights[nearbyNodeCount + 1];
        }
        float nearbyNodeSampleProbability = rand(payload.randomSeed);
        for (int nearbyNodeIndex = 0; nearbyNodeIndex < nearbyNodeCount + 1; ++nearbyNodeIndex) {
            float selectProbability = weights[nearbyNodeIndex] / weightSum;
            if (nearbyNodeSampleProbability <= selectProbability) {
                if (nearbyNodeIndex > 0) {
                    int2 nearbyNodeInfo = nearbyNodeInfos[nearbyNodeIndex - 1];
                    nodeLabel_G = OctreeArray_G[nearbyNodeInfo.x][nearbyNodeInfo.y].label - 1;
                    for (int l = 0; l < nearbyNodeInfo.x; ++l) nodeLabel_G += groupOctreeLayerInfos_G[l].indivisibleNodeCount;
                }
                jitterPdf *= selectProbability; // this pdf shoud add to bsdfSample
                break;
            }
            nearbyNodeSampleProbability -= selectProbability;
        }
    }
    #endif

    float weightSampleProbability = rand(payload.randomSeed);
    float3 outgoing = mul(pushConst.randomRotateMatrix, payload.outgoing);
    float3 hitNormal = mul(pushConst.randomRotateMatrix, payload.hitNormal);    //这里每天思考一下问题有没有
    outgoingIndex = inverseOutgoing_FzbPG(outgoing, OUTGOING_COUNT_FZBPG, weightSampleProbability, hitNormal);
    jitterPdf *= weightSampleProbability; // this pdf shoud add to bsdfSample
    // uint outgoingIndex = inverseSF(payload.outgoing, OUTGOING_COUNT);

    uint weightStartIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * OCTREE_NODECOUNT_E_FZBPG +
                            nodeLabel_G * OCTREE_NODECOUNT_E_FZBPG;

    uint nodeIndex_E = 0, nodeOffset; bool getNode = false;
    for (uint layerIndex_E = 0; layerIndex_E <= OCTREE_CLUSTER_LAYER_FZBPG; ++layerIndex_E) {
        getNode = false;
        float randomNumber = rand(payload.randomSeed);
        for (nodeOffset = 0; nodeOffset < 8; ++nodeOffset) {
            uint weightIndex = weightStartIndex + nodeIndex_E + nodeOffset;
            float probability = OctreeNodePairWeightBuffer[weightIndex];
            //if (printfAble) {
            //    printf("layerIndex: %d, nodeLabel_G: %d, outgoingIndex: %d, nodeIndex_E: %d\nrandomNumber: %f, probability: %f\n", 
            //        layerIndex_E, nodeLabel_G, outgoingIndex,  nodeIndex_E + nodeOffset,
            //        randomNumber, probability
            //    );
            //}
            if (randomNumber <= probability) {
                nodeEPdf *= probability;
                if (layerIndex_E != OCTREE_CLUSTER_LAYER_FZBPG)
                    nodeIndex_E = OctreeLayerStartIndex_FzbPG[layerIndex_E + 1] + (nodeIndex_E - OctreeLayerStartIndex_FzbPG[layerIndex_E] + nodeOffset) * 8;
                getNode = true;
                break;
            }
            randomNumber -= probability;
        }
        if (getNode == false) break;
    }

    if (getNode == false) {
        if (pushConst.frameIndex == 1)
            printf("nodeLabel_G: %d, outgoingIndex: %d, hitPos: %f %f %f\n", nodeLabel_G, outgoingIndex, hitPos.x, hitPos.y, hitPos.z);
        return false;
    }
    nodeIndex_E = nodeIndex_E + nodeOffset - OctreeLayerStartIndex_FzbPG[OCTREE_CLUSTER_LAYER_FZBPG];

    OctreeNodeData_E_FzbPG nodeInfo_E = groupClusterLayerNodeData_E[nodeIndex_E];
    uint nodePairIndex = nodeLabel_G * groupGlobalInfo.indivisibleNodeCount_E + nodeInfo_E.label - 1;
    OctreeNodePairVisibleData_FzbPG nodePairData = OctreeNodePairVisibleDataBuffer[nodePairIndex];
    aabb = nodePairData.aabb;
    nodeEPdf *= nodePairData.pdf;
    return true;
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
        sampleDir = sphericalRectangleSample(sampleQuad, randomU, randomV, pdf); // p *= 1 / quad.s
    }

    return true;
}
int getNodeIndex_E(float3 hitPos, float3 hitNormal) {
    float3 voxelSize = pushConst.VGBVoxelSize * (1 << (pushConst.maxOctreeLayer - OCTREE_CLUSTER_LAYER_FZBPG));
    int3 voxelIndex = int3((hitPos - pushConst.VGBStartPos_Size.xyz) / voxelSize);
    int voxelCount = 1 << OCTREE_CLUSTER_LAYER_FZBPG;
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
    uint weightStartIndex = outgoingIndex * groupGlobalInfo.indivisibleNodeCount_G * OCTREE_CLUSTER_LAYER_FZBPG + nodeLabel_G * OCTREE_CLUSTER_LAYER_FZBPG;

    float pg_Pdf = 1.0f;
    uint nodeIndex_E = getNodeIndex_E(hitPos, hitNormal);
    uint nodeIndex_temp = nodeIndex_E;
    for (int layerIndex = OCTREE_CLUSTER_LAYER_FZBPG; layerIndex >= 0; --layerIndex) {
        uint weightIndex = weightStartIndex + OctreeLayerStartIndex_FzbPG[layerIndex] + nodeIndex_temp;
        pg_Pdf *= OctreeNodePairWeightBuffer[weightIndex];
        nodeIndex_temp /= 8;
    }

    OctreeNodeData_E_FzbPG nodeInfo_E = groupClusterLayerNodeData_E[nodeIndex_E];
    uint nodePairIndex = nodeLabel_G * groupGlobalInfo.indivisibleNodeCount_E + nodeInfo_E.label - 1;
    OctreeNodePairVisibleData_FzbPG nodePairData = OctreeNodePairVisibleDataBuffer[nodePairIndex];
    pg_Pdf *= nodePairData.pdf;
    // if (checkPointInAABB(hitPos, nodeInfo_E.aabb)) pg_Pdf *= nodeInfo_E.pdf;
    // else pg_Pdf *= (1.0f - nodeInfo_E.pdf) / pow(8, pushConst.maxOctreeLayer - OCTREE_CLUSTER_LAYER);
    if (pg_Pdf == 0.0f) return pg_Pdf;

    // getPdf from Sphere sample
    {
        float3 distance = nodePairData.aabb.maximum - nodePairData.aabb.minimum;
        float3 nodeDirection = (nodePairData.aabb.maximum + nodePairData.aabb.minimum) * 0.5f - viewPos;

        float3 edge0 = float3(distance.x, 0.0f, 0.0f);
        float3 edge1 = float3(0.0f, distance.y, 0.0f);
        float3 edge2 = float3(0.0f, 0.0f, distance.z);

        float3 face0StartPos = nodePairData.aabb.minimum;
        face0StartPos.x += nodeDirection.x < 0 ? distance.x : 0.0f; // in left
        float3 face1StartPos = nodePairData.aabb.minimum;
        face1StartPos.y += nodeDirection.y < 0 ? distance.y : 0.0f; // in bottom
        float3 face2StartPos = nodePairData.aabb.minimum;
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
            // float3 aabbCenter = (nodeInfo_E.aabb.minimum + nodeInfo_E.aabb.maximum) * 0.5f;
            // float3 faceXYZ = abs(hitPos - aabbCenter);
            // int faceIndex = faceXYZ.x > faceXYZ.y ? (faceXYZ.x > faceXYZ.z ? 0 : 2) : (faceXYZ.y > faceXYZ.z ? 1 : 2);

            // float3 sphereQuadrilateralSolidAngle = float3(quad0.S, quad1.S, quad2.S);
            // float3 sphereQuadrilateralSolidAngleProbability = sphereQuadrilateralSolidAngle / solidAngleSum;
            // pg_Pdf *= sphereQuadrilateralSolidAngleProbability[faceIndex] / sphereQuadrilateralSolidAngle[faceIndex];
            pg_Pdf /= solidAngleSum;
        }
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
        groupOctreeNodeData_G[i] = OctreeDataBuffer_G[layerIndex][nodeIndex];
    }
    for (int i = groupThreadIndex_linear; i < CLUSTER_LAYER_NODECOUNT_E_FZBPG; i += threadGroupThreadTotalCount)
        groupClusterLayerNodeData_E[i] = ClusterlayerDataBuffer_E[i];
    GroupMemoryBarrierWithGroupSync();

    float2 launchID = (float2)threadIndex;
    float2 launchSize = (float2)pushConst.sceneSize;
    if (threadIndex.x >= pushConst.sceneSize.x || threadIndex.y >= pushConst.sceneSize.y) return;

    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_SVOPG payload;
    payload.randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex));

    float3 accumulatedRadiance = float3(0.0f);
    float RR = 0.8f;

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

        uint nodeLabel_G = 0;
        uint outgoingIndex = 0;

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

            uint bounceDepth = 0; bool useBsdfSample = false;
            while (bounceDepth < pushConst.maxDepth) {
                bool hit = true;
                if (!useBsdfSample) hit = traceRay(ray, payload);
                useBsdfSample = false;

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

                    float jitterPdf = 1.0f;

                    float3 pgSample;
                    float pgSample_pgPdf = 1.0f;
                    float pgSample_bsdfPdf = 1.0f;
                    float pgSample_weight = 0.0f;
                    float3 pgSample_bsdf = 0.0f;
                    bool useFzbPG = true;
                    {
                        AABB nodeAABB_E;
                        // bool useFzbPG = getNodeAABB_E_SVOPG(nodeAABB_E, payload, pgSample_pgPdf, jitterPdf, nodeLabel_G, outgoingIndex);
                        // if (useFzbPG) useFzbPG = SphericalRectangleSample(nodeAABB_E, payload.hitPos, payload.randomSeed, pgSample, pgSample_pgPdf);
                        // if (useFzbPG) {
                        //    pgSample_bsdf = getBSDF(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                        //    // if (dot(payload.incidence, payload.hitNormal) * dot(payload.outgoing, payload.hitNormal) < 0.0f) payload.isExt = !payload.isExt;
                        //
                        //    pgSample_bsdfPdf = getPdf(payload.material, pgSample, payload.hitNormal, payload.outgoing, payload.TBN, payload.isExt);
                        //}
                        //pgSample_weight = pgSample_pgPdf / max(pgSample_pgPdf + pgSample_bsdfPdf, 1e-16f);

                        useFzbPG = getNodeAABB_E_SVOPG(nodeAABB_E, payload, pgSample_pgPdf, jitterPdf, nodeLabel_G, outgoingIndex);
                        if (useFzbPG == false) accumulatedRadiance = float3(10.0f, 0.0f, 0.0f);
                        else if (useFzbPG == true) accumulatedRadiance = float3(0.0f, 10.0f, 0.0f);
                        break;
                    }

                    HitPayload_SVOPG payload_bsdfSample;
                    BSDF_SAMPLE(payload);
                    float3 bsdfSample = payload.incidence;
                    float3 bsdfSample_bsdf = payload.bsdf;
                    float bsdfSample_bsdfPdf = payload.pdf;

                    float bsdfSample_pgPdf = 0.0f;
                    if (useFzbPG){
                        RayDesc ray_bsdfSample;
                        ray_bsdfSample.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                        ray_bsdfSample.Direction = bsdfSample;
                        ray_bsdfSample.TMin = 0.001;
                        ray_bsdfSample.TMax = INFINITE;

                        bool hit = traceRay(ray_bsdfSample, payload_bsdfSample);
                        if (hit) {
                            bsdfSample_pgPdf = getPgPdf(nodeLabel_G, outgoingIndex, payload_bsdfSample.hitPos, payload_bsdfSample.hitNormal, ray_bsdfSample.Origin);
                            bsdfSample_pgPdf *= jitterPdf;
                        }
                    }
                    float bsdfSample_weight = bsdfSample_bsdfPdf / max(bsdfSample_pgPdf + bsdfSample_bsdfPdf, 1e-16f);

                    // MIS
                    {
                        float probabilitySum = max(pgSample_weight + bsdfSample_weight, 1e-16f);
                        float pgSample_selectProbability = pgSample_weight / probabilitySum;
                        float randomNumber = rand(payload.randomSeed);
                        if (randomNumber <= pgSample_selectProbability) {
                            payload.incidence = pgSample;
                            bsdf_cosine_pdf *= probabilitySum / pgSample_pgPdf * pgSample_bsdf;
                        } else {
                            useBsdfSample = true;
                            payload.incidence = bsdfSample;
                            bsdf_cosine_pdf *= probabilitySum / bsdfSample_bsdfPdf * bsdfSample_bsdf;
                        }
                    }

                    float cosine = dot(payload.hitNormal, payload.incidence);
                    if (payload.material.type == MaterialType::Dielectric || payload.material.type == MaterialType::RoughDielectric) cosine = abs(cosine);
                    bsdf_cosine_pdf *= cosine;

                    bool end = any(bsdf_cosine_pdf <= 0.0f) || all(bsdf_cosine_pdf <= 0.0f) || any(isnan(bsdf_cosine_pdf)) || any(isinf(bsdf_cosine_pdf));
                    if (end) break;

                    float randomNumber = rand(payload.randomSeed);
                    if (randomNumber >= RR) break;
                    bsdf_cosine_pdf /= RR;

                    if (useBsdfSample) {
                        payload_bsdfSample.randomSeed = payload.randomSeed;
                        payload_bsdfSample.outgoing = -payload.incidence;
                        payload = payload_bsdfSample;
                    } else {
                        payload.outgoing = -payload.incidence;
                        ray.Direction = payload.incidence;
                        ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                    }

                    ++bounceDepth;
                }
            }
        }
    }
    accumulatedRadiance /= pushConst.spp;

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.maxFrameCount > 1 && pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}