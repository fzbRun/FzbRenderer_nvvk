#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_VRPG> pushConst;
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_VRPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(BindingPoints_Octree_VRPG::eNearbyNodeInfos)]] RWStructuredBuffer<OctreeNearbyNodeInfo_VRPG, ScalarDataLayout> NearbyNodeInfoBuffer;
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_VRPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_VRPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_VRPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_VRPG::eAdjClusterInfo_G)]] RWStructuredBuffer<uint> AdjClusterInfoBuffer;

[[vk::binding(BindingPoints_Octree_VRPG::eClusterPairInfo_VRPG)]] RWStructuredBuffer<ClusterPairInfo_VRPG, ScalarDataLayout> clusterPairInfoBuffer;
[[vk::binding(BindingPoints_Octree_VRPG::eClusterPairGlobalInfo_VRPG)]] RWStructuredBuffer<ClusterPairGlobalInfo_VRPG, ScalarDataLayout> clusterPairGlobalInfoBuffer;
[[vk::binding(BindingPoints_Octree_VRPG::eClusetPairHitTestInfo)]] RWStructuredBuffer<ClusterPairHitInfo_VRPG, ScalarDataLayout> clusterPairHitTestInfoBuffer;

[[vk::binding(BindingPoints_Octree_VRPG::eCandidatePoolInfo_VRPG)]] RWStructuredBuffer<CandidatePoolInfo_VRPG, ScalarDataLayout> candidatePoolInfoBuffer;
[[vk::binding(BindingPoints_Octree_VRPG::eCandidatePoolNodeInfo_VRPG)]] RWStructuredBuffer<CandidatePoolNodeInfo_VRPG, ScalarDataLayout> candidatePoolNodeInfoBuffer;
//----------------------------------------------------------------------------------------------------------------------------------------------------
#define LIGHTCUT_RELATIVE_ERROR 0.0001f

struct LightCutNode {
    int layerIndex;
    int nodeIndex;
    float error;
    float I;
};

float3 getLightcutMUpperBound(OctreeNodeData_E_VRPG lightNode, BSDFMaterial material, float3 pos, float3 normal, float3 outgoing) {
    float3 maxCosine = MaxCosToAABB(pos, lightNode.meanNormal.xyz, lightNode.aabb.minimum, lightNode.aabb.maximum);

    float3 tangent, bitangent;
    orthonormalBasis(normal, tangent, bitangent);
    float3x3 TBN = float3x3(tangent, bitangent, normal);

    float3 maxBSDF_Cosine = -1.0f;
    for (int i = 0; i < 9; ++i) {
        float3 lightPos = lightNode.aabb.minimum + float3(
            (i & 1) == 1 ? lightNode.aabb.maximum.x : lightNode.aabb.minimum.x,
                                            (i & 2) == 2 ? lightNode.aabb.maximum.y : lightNode.aabb.minimum.y,
                                            (i & 4) == 4 ? lightNode.aabb.maximum.z : lightNode.aabb.minimum.z
        );
        if (i == 8) lightPos = (lightNode.aabb.maximum + lightNode.aabb.minimum) * 0.5f;
        float3 incidence = normalize(lightPos - pos);
        float3 brdf = getBSDF(material, incidence, normal, outgoing, TBN, true);
        maxBSDF_Cosine = max(maxBSDF_Cosine, brdf * maxCosine);
    }

    return maxBSDF_Cosine;
}
float getLightGUpperBound(OctreeNodeData_E_VRPG lightNode, float3 pos) {
    float3 minimum = pos - lightNode.aabb.maximum;
    float3 maximum = pos - lightNode.aabb.minimum;

    float cosThetaMax = MaxCosToAABB(float3(0, 0, 0), lightNode.meanNormal.xyz, minimum, maximum);
    cosThetaMax = clamp(cosThetaMax, -1.0, 1.0);

    float thetaMin = acos(cosThetaMax);
    if (thetaMin <= lightNode.halfAngle) return 1.0;
    else return max(0.0, cos(thetaMin - lightNode.halfAngle));
}

float3 getValidSamplePos(float3 endPos, float3 normal, AABB aabb) {
    const float3 n = normal;
    const float eps = 1e-4f;

    // Range of the AABB along the (axis aligned) normal, and the valid upper bound: the sample has to
    // satisfy dot(n, endPos - pos) > 0, i.e. dot(n, pos) < dot(n, endPos).
    const float uLo = min(dot(n, aabb.minimum), dot(n, aabb.maximum));
    const float uHi = max(dot(n, aabb.minimum), dot(n, aabb.maximum));
    const float uMax = max(min(uHi, dot(n, endPos) - eps), uLo);

    // Uniform sample inside the AABB, then move it along the normal so its coordinate lands in
    // [uLo, uMax]. Branchless: no per axis code and the tangent coordinates are left untouched.
    uint seed = xxhash32(uint3(asuint(endPos.x), asuint(endPos.y), asuint(endPos.z) ^ asuint(aabb.minimum.y)));
    const float3 rnd = float3(rand(seed), rand(seed), rand(seed));
    float3 pos = aabb.minimum + rnd * (aabb.maximum - aabb.minimum);

    const float u = dot(n, pos);
    const float t = clamp((u - uLo) / max(uHi - uLo, 1e-8f), 0.0f, 1.0f);
    return pos + (uLo + t * (uMax - uLo) - u) * n;
}

bool traceRay(RayDesc ray, float3 nodeNormal, AABB nodeAabb)
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
        Instance instance = sceneInfo.instances[instanceIndex]; // Instance data

        // Convert built-in barycentric coordinates to full barycentric coordinates
        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        // Process intersection data into world-space shading information
        // HitState hitState = getHitStateWithPositionFetch(barycentrics);
        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        float3 hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        // payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        float3 hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        // Make the geometric normal face the incoming ray. glTF triangle
        // winding is not guaranteed to match the normal partition; this is
        // shading orientation, not a post-hit rejection flip.
        if (dot(hitNormal, ray.Direction) > 0.0f) hitNormal = -hitNormal;

        if (dot(nodeNormal, GetPrincipalAxis(hitNormal)) != 1.0f || !checkPointInAABB2(hitPos, nodeAabb) || dot(hitNormal, -ray.Direction) <= 0.0f) return false;
    }
    else return false;

    return true;
}

struct HitInfo{
    float3 hitNormal;
    float3 hitPos;
}
bool traceRay2(RayDesc ray, float3 nodeNormal, AABB nodeAabb, inout HitInfo hitInfo)
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
        Instance instance = sceneInfo.instances[instanceIndex]; // Instance data

        // Convert built-in barycentric coordinates to full barycentric coordinates
        const float3 barycentrics =
            float3(1.0 - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        // Process intersection data into world-space shading information
        // HitState hitState = getHitStateWithPositionFetch(barycentrics);
        float3 triPos[3];
        triPos = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        float3 hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        // payload.normal = normalize(mul(worldToObject, geoNormal).xyz);
        float3 hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        // Make the geometric normal face the incoming ray. glTF triangle
        // winding is not guaranteed to match the normal partition; this is
        // shading orientation, not a post-hit rejection flip.
        if (dot(hitNormal, ray.Direction) > 0.0f) hitNormal = -hitNormal;

        hitInfo.hitNormal = hitNormal;
        hitInfo.hitPos = hitPos;
        if (dot(nodeNormal, GetPrincipalAxis(hitNormal)) != 1.0f || !checkPointInAABB2(hitPos, nodeAabb) || dot(hitNormal, -ray.Direction) <= 0.0f) return false;
    }
    else return false;

    return true;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
groupshared uint groupIndivisibleNodeCount_G;
//----------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_init(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    int indivisibleNodeIndex = threadIndex;
    if (indivisibleNodeIndex >= indivisibleNodeCount_G) return;

    candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = 1;

    int candidateNodeIndex = indivisibleNodeIndex * Candidate_Samples_Count + 0;
    CandidatePoolNodeInfo_VRPG nodeInfo;
    nodeInfo.valid = true;
    nodeInfo.layer_E = -1;
    nodeInfo.nodeIndex_E = 0;
    candidatePoolNodeInfoBuffer[candidateNodeIndex] = nodeInfo;

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = indivisibleNodeCount_G;

    //if (threadIndex == 0 && pushConst.frameIndex == 0) {
    //    int indivisibleNodeIndex = 226;
    //    uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[indivisibleNodeIndex];
    //    OctreeNodeData_G_VRPG nodeData_G = OctreeNodeDataBuffer_G[indivisibleNodeInfo.x][indivisibleNodeInfo.y];
    //    printf("ClusterIndex: %d\n normal: %f %f %f\n aabb: %f %f %f %f %f %f\n\n",
    //        indivisibleNodeIndex,
    //        nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z,
    //        nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
    //        nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
    //    );
    //}
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------------------------------------------------------
// Shared candidate pool expansion, used by both the expand only pass (_1) and the
// expand + emit pass (_2) so the two can never drift apart again.
//----------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_1(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    int indivisibleNodeIndex = threadGroupIndex;
    if (indivisibleNodeIndex >= indivisibleNodeCount_G) return;

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = indivisibleNodeCount_G;

    CandidatePoolInfo_VRPG candidatePoolInfo = candidatePoolInfoBuffer[indivisibleNodeIndex];
    if (groupThreadIndex >= candidatePoolInfo.candidateNodeCount) return;

    uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[indivisibleNodeIndex];
    OctreeNodeData_G_VRPG nodeData_G = OctreeNodeDataBuffer_G[indivisibleNodeInfo.x][indivisibleNodeInfo.y];
    float3 nodeCenterPos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    int candidateNodeIndex = indivisibleNodeIndex * Candidate_Samples_Count + groupThreadIndex;
    CandidatePoolNodeInfo_VRPG candidatePoolNodeInfo = candidatePoolNodeInfoBuffer[candidateNodeIndex];
    bool candidateNodeValid = candidatePoolNodeInfo.valid == 1;
    int candidateNodeCount = WaveActiveSum(candidateNodeValid);
    //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0 && groupThreadIndex == 0) {
    //    printf("\ncandidateNodeCount: %d\n", candidateNodeCount);
    //}
    if (candidateNodeCount == 0) {
        if (groupThreadIndex == 0) candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = 0;
        return;
    }

    uint seed = xxhash32(uint3(indivisibleNodeIndex, groupThreadIndex, (uint)pushConst.frameIndex));
    seed = WaveReadLaneFirst(seed);
    if (!candidateNodeValid) return;
    //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0){
    //    printf("candidateNodeInfo: layer: %d, index: %d\n", candidatePoolNodeInfo.layer_E, candidatePoolNodeInfo.nodeIndex_E);
    //}

    bool isLeafNode = candidatePoolNodeInfo.layer_E == pushConst.lightOctreeMaxLayer;
    int subNodeLayerIndex = candidatePoolNodeInfo.layer_E + 1;
    int subNodeStartIndex = candidatePoolNodeInfo.nodeIndex_E * 8;

    float BSum = 0.0f; float Bs[8]; uint validChildNodeCount = isLeafNode ? 1 : 0;
    if (!isLeafNode) {
        for (int i = 0; i < 8; ++i) {
            Bs[i] = 0.0f;
            if (subNodeLayerIndex == 0 && i >= 6) break;

            int subNodeIndex = subNodeStartIndex + i;
            OctreeNodeData_E_VRPG nodeData_E = OctreeNodeDataBuffer_E[subNodeLayerIndex][subNodeIndex];
            if (all(nodeData_E.radiance == 0.0f)) continue;

            if (length(nodeData_E.meanNormal.xyz) > 1e-6f) {
                float3 lightNormal = normalize(nodeData_E.meanNormal.xyz);
                float3 lightMin = nodeData_E.aabb.minimum;
                float3 lightMax = nodeData_E.aabb.maximum;
                float minProjection = dot(lightNormal, float3(
                    lightNormal.x >= 0.0f ? lightMin.x : lightMax.x,
                    lightNormal.y >= 0.0f ? lightMin.y : lightMax.y,
                    lightNormal.z >= 0.0f ? lightMin.z : lightMax.z));
                float gProjection = dot(lightNormal, nodeCenterPos_G);
                if (gProjection - minProjection <= 0.01f) continue;
            }

            float I = dot(nodeData_E.radiance, 1.0f);
            float3 MUpperBound = MaxCosToAABB(nodeCenterPos_G, nodeData_G.meanNormal.xyz, nodeData_E.aabb.minimum, nodeData_E.aabb.maximum);
            float GUpperBound = getLightGUpperBound(nodeData_E, nodeCenterPos_G);
            float3 center = (nodeData_E.aabb.maximum + nodeData_E.aabb.minimum) * 0.5f;
            float diagonalLength = length(nodeData_E.aabb.maximum - nodeData_E.aabb.minimum);
            float d = max(length(nodeCenterPos_G - center), diagonalLength + 1e-3f);
            float B = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);

            //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0) {
            //    printf("candidateNodeIndex: %d\n childNodeIndex: %d, nodeInfo:\n I: %f, M: %f %f %f, G: %f, d: %f\n pos_G: %f %f %f, normal_G: %f %f %f, aabb_E: %f %f %f %f %f %f\n",
            //        groupThreadIndex, i, I, MUpperBound.x, MUpperBound.y, MUpperBound.z, GUpperBound, d,
            //        nodeCenterPos_G.x, nodeCenterPos_G.y, nodeCenterPos_G.z,
            //        nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z,
            //        nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
            //        nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
            //    );
            //}

            if (B == 0.0f) continue;

            Bs[i] = B;
            BSum += B;
            ++validChildNodeCount;
        }
    }

    //if (indivisibleNodeIndex == 149 && pushConst.frameIndex == 0) {
    //    printf("candidateNodeIndex: %d, validChildNodeCount: %d\n", groupThreadIndex, validChildNodeCount);
    //}

    candidateNodeCount = WaveActiveSum(validChildNodeCount);
    int offsetInWarp = WavePrefixSum(validChildNodeCount);
    bool allSubTreeInPool = offsetInWarp + validChildNodeCount <= Candidate_Samples_Count;
    allSubTreeInPool = WaveActiveAllTrue(allSubTreeInPool);

    int warpLane = groupThreadIndex & 31;
    bool noExpand = isLeafNode;
    while (!allSubTreeInPool) {
        float minB = WaveActiveMin(BSum);
        uint mask = WaveActiveBallot(BSum == minB).x;
        uint victimBit = mask & (~mask + 1u);
        if ((1u << warpLane) == victimBit) { validChildNodeCount = 1; noExpand = true; BSum = 3.4e38f; }
        candidateNodeCount = WaveActiveSum(validChildNodeCount);
        offsetInWarp = WavePrefixSum(validChildNodeCount);
        allSubTreeInPool = WaveActiveAllTrue(offsetInWarp + validChildNodeCount <= Candidate_Samples_Count);
    }
    if (validChildNodeCount > 0 && offsetInWarp == 0) candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = candidateNodeCount;

    if (noExpand) {
        candidatePoolNodeInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + offsetInWarp] = candidatePoolNodeInfo;
        return;
    }

    candidatePoolNodeInfo.layer_E = subNodeLayerIndex;
    int validNodeIndex = 0;
    for (int i = 0; i < 8; ++i) {
        if (Bs[i] == 0.0f) continue;
        candidatePoolNodeInfo.nodeIndex_E = subNodeStartIndex + i;
        candidatePoolNodeInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + offsetInWarp + validNodeIndex] = candidatePoolNodeInfo;

        ++validNodeIndex;
    }
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
// Candidate_Samples_Count must equal 32
[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_2(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = 1;

    int indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    int indivisibleNodeIndex = threadGroupIndex;
    if (indivisibleNodeIndex >= indivisibleNodeCount_G) return;

    CandidatePoolInfo_VRPG candidatePoolInfo = candidatePoolInfoBuffer[indivisibleNodeIndex];
    if (groupThreadIndex >= candidatePoolInfo.candidateNodeCount) return;

    uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[indivisibleNodeIndex];
    OctreeNodeData_G_VRPG nodeData_G = OctreeNodeDataBuffer_G[indivisibleNodeInfo.x][indivisibleNodeInfo.y];
    float3 nodeCenterPos_G = (nodeData_G.aabb.maximum + nodeData_G.aabb.minimum) * 0.5f;

    int candidateNodeIndex = indivisibleNodeIndex * Candidate_Samples_Count + groupThreadIndex;
    CandidatePoolNodeInfo_VRPG candidatePoolNodeInfo = candidatePoolNodeInfoBuffer[candidateNodeIndex];
    bool candidateNodeValid = candidatePoolNodeInfo.valid == 1;
    int candidateNodeCount = WaveActiveSum(candidateNodeValid);
    //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0 && groupThreadIndex == 0) {
    //    printf("\ncandidateNodeCount: %d\n", candidateNodeCount);
    //}
    if (candidateNodeCount == 0) {
        if (groupThreadIndex == 0) candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = 0;
        return;
    }

    uint seed = xxhash32(uint3(indivisibleNodeIndex, groupThreadIndex, (uint)pushConst.frameIndex));
    seed = WaveReadLaneFirst(seed);
    if (!candidateNodeValid) return;
    //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0) {
    //    printf("candidateNodeIndex: %d, candidateNodeInfo: layer: %d, index: %d\n", groupThreadIndex, candidatePoolNodeInfo.layer_E, candidatePoolNodeInfo.nodeIndex_E);
    //}

    bool isLeafNode = candidatePoolNodeInfo.layer_E == pushConst.lightOctreeMaxLayer;
    int subNodeLayerIndex = candidatePoolNodeInfo.layer_E + 1;
    int subNodeStartIndex = candidatePoolNodeInfo.nodeIndex_E * 8;

    float BSum = 0.0f; float Bs[8]; uint validChildNodeCount = isLeafNode ? 1 : 0;
    if (!isLeafNode) {
        for (int i = 0; i < 8; ++i) {
            Bs[i] = 0.0f;
            if (subNodeLayerIndex == 0 && i >= 6) break;

            int subNodeIndex = subNodeStartIndex + i;
            OctreeNodeData_E_VRPG nodeData_E = OctreeNodeDataBuffer_E[subNodeLayerIndex][subNodeIndex];
            if (all(nodeData_E.radiance == 0.0f)) continue;

            if (length(nodeData_E.meanNormal.xyz) > 1e-6f) {
                float3 lightNormal = normalize(nodeData_E.meanNormal.xyz);
                float3 lightMin = nodeData_E.aabb.minimum;
                float3 lightMax = nodeData_E.aabb.maximum;
                float minProjection = dot(lightNormal, float3(
                    lightNormal.x >= 0.0f ? lightMin.x : lightMax.x,
                    lightNormal.y >= 0.0f ? lightMin.y : lightMax.y,
                    lightNormal.z >= 0.0f ? lightMin.z : lightMax.z));
                float gProjection = dot(lightNormal, nodeCenterPos_G);
                if (gProjection - minProjection <= 0.01f) continue;
            }

            float I = dot(nodeData_E.radiance, 1.0f);
            float3 MUpperBound = MaxCosToAABB(nodeCenterPos_G, nodeData_G.meanNormal.xyz, nodeData_E.aabb.minimum, nodeData_E.aabb.maximum);
            float GUpperBound = getLightGUpperBound(nodeData_E, nodeCenterPos_G);
            float3 center = (nodeData_E.aabb.maximum + nodeData_E.aabb.minimum) * 0.5f;
            float diagonalLength = length(nodeData_E.aabb.maximum - nodeData_E.aabb.minimum);
            float d = max(length(nodeCenterPos_G - center), diagonalLength + 1e-3f);
            float B = I * dot(MUpperBound, 1.0f) * GUpperBound / (d * d);
            if (B == 0.0f) continue;

            //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0) {
            //    printf("candidateNodeIndex: %d\n childNodeIndex: %d, nodeInfo:\n I: %f, M: %f %f %f, G: %f, d: %f\n pos_G: %f %f %f, normal_G: %f %f %f, aabb_E: %f %f %f %f %f %f\n",
            //        groupThreadIndex, i, I, MUpperBound.x, MUpperBound.y, MUpperBound.z, GUpperBound, d,
            //        nodeCenterPos_G.x, nodeCenterPos_G.y, nodeCenterPos_G.z,
            //        nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z,
            //        nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
            //        nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
            //    );
            //}

            Bs[i] = B;
            BSum += B;
            ++validChildNodeCount;
        }
    }

    candidateNodeCount = WaveActiveSum(validChildNodeCount);
    int offsetInWarp = WavePrefixSum(validChildNodeCount);
    bool allSubTreeInPool = offsetInWarp + validChildNodeCount <= Candidate_Samples_Count;
    allSubTreeInPool = WaveActiveAllTrue(allSubTreeInPool);
    int noExpandcandidateNodeIndex = 0; bool noExpand = isLeafNode;
    while (!allSubTreeInPool) {
        if (offsetInWarp == noExpandcandidateNodeIndex) {
            validChildNodeCount = 1;
            noExpand = true;
        }
        candidateNodeCount = WaveActiveSum(validChildNodeCount);
        offsetInWarp = WavePrefixSum(validChildNodeCount);
        allSubTreeInPool = WaveActiveAllTrue(offsetInWarp + validChildNodeCount <= Candidate_Samples_Count);
        ++noExpandcandidateNodeIndex;
    }
    int clusterPairOffset;
    if (validChildNodeCount > 0 && offsetInWarp == 0) {
        candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = candidateNodeCount;
        InterlockedAdd(clusterPairGlobalInfoBuffer[0].clusterPairCount, candidateNodeCount, clusterPairOffset);
    }
    clusterPairOffset = WaveReadLaneFirst(clusterPairOffset);

    ClusterPairInfo_VRPG clusterPairInfo;
    if (noExpand) {
        candidatePoolNodeInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + offsetInWarp] = candidatePoolNodeInfo;

        clusterPairInfo.layerIndex_G = indivisibleNodeInfo.x;
        clusterPairInfo.nodeIndex_G = indivisibleNodeInfo.y;
        clusterPairInfo.indivisibleNodeLabel = threadGroupIndex;
        clusterPairInfo.layerIndex_E = candidatePoolNodeInfo.layer_E;
        clusterPairInfo.nodeIndex_E = candidatePoolNodeInfo.nodeIndex_E;
        clusterPairInfo.candidateNodeIndex = offsetInWarp;
        clusterPairInfoBuffer[clusterPairOffset + offsetInWarp] = clusterPairInfo;

        return;
    }

    candidatePoolNodeInfo.layer_E = subNodeLayerIndex;
    int validNodeIndex = 0;
    for (int i = 0; i < 8; ++i) {
        if (Bs[i] == 0.0f) continue;
        candidatePoolNodeInfo.nodeIndex_E = subNodeStartIndex + i;
        candidatePoolNodeInfoBuffer[indivisibleNodeIndex * Candidate_Samples_Count + offsetInWarp + validNodeIndex] = candidatePoolNodeInfo;

        clusterPairInfo.layerIndex_G = indivisibleNodeInfo.x;
        clusterPairInfo.nodeIndex_G = indivisibleNodeInfo.y;
        clusterPairInfo.indivisibleNodeLabel = threadGroupIndex;
        clusterPairInfo.layerIndex_E = subNodeLayerIndex;
        clusterPairInfo.nodeIndex_E = subNodeStartIndex + i;
        clusterPairInfo.candidateNodeIndex = offsetInWarp + validNodeIndex;
        clusterPairInfoBuffer[clusterPairOffset + offsetInWarp + validNodeIndex] = clusterPairInfo;

        ++validNodeIndex;
    }
}

//--------------------------------------------------------------------------------------------------------------------------------------------------------
#define HitTestCount 4

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_3() {
    GlobalInfoBuffer[0].cmd.x = (clusterPairGlobalInfoBuffer[0].clusterPairCount * HitTestCount + 255) / 256;
}
//----------------------------------------------------------------------------------------------------------------------------------------------------
groupshared uint groupClusterPairCount;

[numthreads(256, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_4(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupClusterPairCount = clusterPairGlobalInfoBuffer[0].clusterPairCount;
    GroupMemoryBarrierWithGroupSync();
    int clusterPairCount = groupClusterPairCount;
    if (threadIndex >= clusterPairCount * HitTestCount) return;

    int clusterPairIndex = threadIndex / HitTestCount;
    int hitTestIndex = groupThreadIndex % HitTestCount; 
    ClusterPairInfo_VRPG clusterPairInfo = clusterPairInfoBuffer[clusterPairIndex];
    OctreeNodeData_G_VRPG nodeData_G = OctreeNodeDataBuffer_G[clusterPairInfo.layerIndex_G][clusterPairInfo.nodeIndex_G];
    OctreeNodeData_E_VRPG nodeData_E = OctreeNodeDataBuffer_E[clusterPairInfo.layerIndex_E][clusterPairInfo.nodeIndex_E];

    int clusterPairGlobalIndex = clusterPairInfo.indivisibleNodeLabel * Candidate_Samples_Count + clusterPairInfo.candidateNodeIndex;
    uint seed = xxhash32(uint3(clusterPairGlobalIndex, hitTestIndex, (uint)pushConst.randomFrameIndex));
    float3 randXYZ = float3(rand(seed), rand(seed), rand(seed));
    float3 nodeSamplePos_G = randXYZ * (nodeData_G.aabb.maximum - nodeData_G.aabb.minimum) + nodeData_G.aabb.minimum;
    float3 normal_G = GetPrincipalAxis(nodeData_G.meanNormal.xyz);

    float3 normal_E = GetPrincipalAxis(nodeData_E.meanNormal.xyz);
    float3 nodeSamplePos_E = getValidSamplePos(nodeSamplePos_G, normal_E, nodeData_E.aabb);
    float3 sampleDir = normalize(nodeSamplePos_G - nodeSamplePos_E);
    nodeSamplePos_E = getAABBPoint(nodeData_E.aabb, nodeSamplePos_E, sampleDir);

    bool hitTestAble = dot(normal_G, -sampleDir) > 0.0f; // dot(normal_E, sampleDir) > 0.0f;
    bool hit = false; HitInfo hitInfo;
    if (hitTestAble) {
        RayDesc ray;
        ray.Direction = sampleDir;
        ray.Origin = nodeSamplePos_E + sampleDir * 0.00001f;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        hit = traceRay2(ray, normal_G, nodeData_G.aabb, hitInfo);
    }
    //if (clusterPairInfo.indivisibleNodeLabel == 149 && pushConst.frameIndex == 0) {
    //    float hitType = hitTestAble ? (hit ? 1.0f : 0.0f) : -1.0f;
    //    //if (hitType == 0.0f) {
    //    //    // dot(nodeNormal, GetPrincipalAxis(hitNormal)) != 1.0f || !checkPointInAABB2(hitPos, nodeAabb) || dot(hitNormal, -ray.Direction) <= 0.0f
    //    //    hitType = dot(nodeNormal, GetPrincipalAxis(hitNormal)) != 1.0f ? -2.0f : 0.0f;
    //    //}
    //
    //    float3 normal_E = GetPrincipalAxis(nodeData_E.meanNormal.xyz);
    //    printf("layerIndex: %d, nodeIndex: %d, hitTestIndex: %d hit: %f\n sampleDir: %f %f %f SamplePos_E: %f %f %f SamplePos_G: %f %f %f\n normal_E: %f %f %f, normal_E_2: %f %f %f, aabb: %f %f %f %f %f %f\n normal_G: %f %f %f, aabb: %f %f %f %f %f %f\n\n",
    //        clusterPairInfo.layerIndex_E, clusterPairInfo.nodeIndex_E, hitTestIndex, hitType,
    //        sampleDir.x, sampleDir.y, sampleDir.z,
    //        nodeSamplePos_E.x, nodeSamplePos_E.y, nodeSamplePos_E.z,
    //        nodeSamplePos_G.x, nodeSamplePos_G.y, nodeSamplePos_G.z,
    //        nodeData_E.meanNormal.x, nodeData_E.meanNormal.y, nodeData_E.meanNormal.z,
    //        normal_E.x, normal_E.y, normal_E.z,
    //        nodeData_E.aabb.minimum.x, nodeData_E.aabb.minimum.y, nodeData_E.aabb.minimum.z,
    //        nodeData_E.aabb.maximum.x, nodeData_E.aabb.maximum.y, nodeData_E.aabb.maximum.z,
    //        nodeData_G.meanNormal.x, nodeData_G.meanNormal.y, nodeData_G.meanNormal.z,
    //        nodeData_G.aabb.minimum.x, nodeData_G.aabb.minimum.y, nodeData_G.aabb.minimum.z,
    //        nodeData_G.aabb.maximum.x, nodeData_G.aabb.maximum.y, nodeData_G.aabb.maximum.z,
    //    );
    //}

    int warpLane = groupThreadIndex & 31;
    for (int offset = HitTestCount / 2; offset > 0; offset /= 2) hit |= WaveReadLaneAt(hit, warpLane + offset) == 1;
    if (!hit && hitTestIndex == 0) candidatePoolNodeInfoBuffer[clusterPairGlobalIndex].valid = 0;

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = GlobalInfoBuffer[0].indivisibleNodeCount_G;
}
//--------------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain_getCandidateNodes_5(uint threadIndex: SV_DispatchThreadID, uint threadGroupIndex: SV_GroupID, uint groupThreadIndex: SV_GroupThreadID) {
    int indivisibleNodeCount_G = GlobalInfoBuffer[0].indivisibleNodeCount_G;
    int indivisibleNodeIndex = threadGroupIndex;
    if (indivisibleNodeIndex >= indivisibleNodeCount_G) return;

    CandidatePoolInfo_VRPG candidatePoolInfo = candidatePoolInfoBuffer[indivisibleNodeIndex];
    if (groupThreadIndex >= candidatePoolInfo.candidateNodeCount) return;

    uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[indivisibleNodeIndex];

    int candidateNodeIndex = indivisibleNodeIndex * Candidate_Samples_Count + groupThreadIndex;
    CandidatePoolNodeInfo_VRPG candidatePoolNodeInfo = candidatePoolNodeInfoBuffer[candidateNodeIndex];
    bool candidateNodeValid = candidatePoolNodeInfo.valid == 1;
    int candidateNodeCount = WaveActiveSum(candidateNodeValid);
    int validNodeIndex = WavePrefixSum(candidateNodeValid);
    //if (indivisibleNodeIndex == 226 && pushConst.frameIndex == 0 && groupThreadIndex == 0) {
    //    printf("\ncandidateNodeCount: %d\n", candidateNodeCount);
    //}

    int clusterPairOffset;
    if (groupThreadIndex == 0) {
        candidatePoolInfoBuffer[indivisibleNodeIndex].candidateNodeCount = candidateNodeCount;
        InterlockedAdd(clusterPairGlobalInfoBuffer[0].clusterPairCount, candidateNodeCount, clusterPairOffset);
    }
    clusterPairOffset = WaveReadLaneFirst(clusterPairOffset);

    ClusterPairInfo_VRPG clusterPairInfo;
    clusterPairInfo.layerIndex_G = indivisibleNodeInfo.x;
    clusterPairInfo.nodeIndex_G = indivisibleNodeInfo.y;
    clusterPairInfo.indivisibleNodeLabel = threadGroupIndex;
    clusterPairInfo.layerIndex_E = candidatePoolNodeInfo.layer_E;
    clusterPairInfo.nodeIndex_E = candidatePoolNodeInfo.nodeIndex_E;
    clusterPairInfo.candidateNodeIndex = validNodeIndex;
    clusterPairInfoBuffer[clusterPairOffset + validNodeIndex] = clusterPairInfo;

    if (threadIndex == 0) GlobalInfoBuffer[0].cmd.x = 1;
}