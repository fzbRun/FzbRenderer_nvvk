#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/FzbPathGuidingRenderer/shaders/FzbPGCommon.slang"
#include "renderer/FzbPathGuidingRenderer/Octree2/Octree2Shaderio_FzbPG.h"

[[vk::push_constant]] ConstantBuffer<ClusterPairDebugPushConstant_FzbPG> pushConst;

[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_FzbPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(BindingPoints_Octree_FzbPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_FzbPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_FzbPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_FzbPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_FzbPG, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo, ScalarDataLayout> candidateSubTreeInfoBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData, ScalarDataLayout> candidateLeafNodeDataBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_FzbPG, ScalarDataLayout> candidateSampleDataBuffer;

#ifdef SpatialReuse_FzbPG
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleAABB)]] RWStructuredBuffer<AABB, ScalarDataLayout> candidateSampleAABBBuffer;
[[vk::binding(BindingPoints_Octree_FzbPG::eCandidateSampleData2)]] RWStructuredBuffer<CandidateSampleData_FzbPG, ScalarDataLayout> candidateSampleData2Buffer;
#endif
//------------------------------------------------------------------------------------------
bool traceRay(RayDesc ray, inout CallablePayload payload)
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
        return true;
    }
    return false;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_GetDebugClusterG(uint threadIndex: SV_DispatchThreadID) {
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    const float2 pixelCenter = (float2)pushConst.pixelIndex + float2(0.5f, 0.5f);
    const float2 clipCoords = pixelCenter / pushConst.screenSize * 2.0 - 1.0;
    float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);
    viewCoords /= viewCoords.w;

    ray.Origin = sceneInfo.cameraPosition;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);

    CallablePayload payload;
    if (!traceRay(ray, payload)) {
        pushConst.globalInfoAddress.clusterIndex_G = -1;
        return;
    }

    uint normalIndex = getNormalIndex(payload.hitNormal);
    float3 hitPos = payload.hitPos;

    OctreeNodeInfo_G_FzbPG nodeInfo_G = OctreeNodeInfoBuffer_G[0][normalIndex];
    int nodeLabel_G = int(nodeInfo_G.label_indivisible >> 1) - 1;
    bool hasData = nodeLabel_G >= 0;
    if (!hasData) {
        pushConst.globalInfoAddress.clusterIndex_G = -1;
        return;
    }

    bool indivisible = (nodeInfo_G.label_indivisible & 1u) == 1;
    bool hitIndivisible = false;
    int layerIndex_G = 0; int nodeIndex_G = normalIndex;
    if (indivisible) hitIndivisible = true;
    else {
        nodeLabel_G = GlobalInfoBuffer[0].layerInfos_G[0].indivisibleNodeCount;

        float3 nodeSize = pushConst.VGBVoxelSize.xyz * (1 << (pushConst.octreeMaxLayer - 1));
        float3 nodeStartPos = pushConst.VGBStartPos_Size.xyz;

        for (layerIndex_G = 1; layerIndex_G <= pushConst.octreeMaxLayer; ++layerIndex_G) {
            int3 nodeIndex3 = min(int3((hitPos - nodeStartPos) / nodeSize), 1);
            int subNodeIndex = nodeIndex3.z * 4 + nodeIndex3.y * 2 + nodeIndex3.x;
            nodeIndex_G = nodeIndex_G * 8 + subNodeIndex;

            nodeInfo_G = OctreeNodeInfoBuffer_G[layerIndex_G][nodeIndex_G];

            int  nodeLayerLabel = int(nodeInfo_G.label_indivisible >> 1) - 1;
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
    if (!hitIndivisible) {
        pushConst.globalInfoAddress.clusterIndex_G = -1;
        return;
    }

    pushConst.globalInfoAddress.clusterIndex_G = nodeLabel_G;
    printf("clusterGIndex: %d\n", nodeLabel_G);
}

//------------------------------------------------------------------------------------------
struct VSout {
    float4 sv_position : SV_Position;
    bool debug_color;
    nointerpolation float4 color;
};
struct PSout
{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout output;

    bool show = pushConst.globalInfoAddress.clusterIndex_G >= 0;
    if (!show) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    uint2 nodeInfo = IndivisibleNodeInfoBuffer_G[pushConst.globalInfoAddress.clusterIndex_G];
    AABB nodeAABB = OctreeNodeDataBuffer_G[nodeInfo.x][nodeInfo.y].aabb;

    output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    if (pushConst.mode == 0) {}
    else if (pushConst.mode == 1) {
        float3 outgoing = getOutgoing_FzbPG(pushConst.outgoingIndex, OUTGOING_COUNT_FZBPG);
        float3 nodeCenter = (nodeAABB.maximum + nodeAABB.minimum) * 0.5f;
        output.sv_position.xyz = vertexIndex == 0 ? nodeCenter : nodeCenter + outgoing;
        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
        output.color.xyz = float3(0.0f, 0.0f, 1.0f);
        return output;
    } else {
        int candidateIndex = pushConst.outgoingIndex * GlobalInfoBuffer[0].indivisibleNodeCount_G * Candidate_Samples_Count + pushConst.globalInfoAddress.clusterIndex_G * Candidate_Samples_Count + instanceIndex;

        if (pushConst.spatialReuse == 0) {
            CandidateSampleData_FzbPG candidateNodeData = candidateSampleDataBuffer[candidateIndex];
            CandidateSubTreeInfo candidateSubTree = candidateSubTreeInfoBuffer[pushConst.globalInfoAddress.clusterIndex_G * Candidate_Samples_Count + instanceIndex];

            show = candidateSubTree.subTreeNodeIndex_layer >= 0;
            if (pushConst.mode2 == 1) show &= candidateNodeData.w > 0.0f;
            if (!show) {
                output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
                return output;
            }

            int layerIndex = candidateSubTree.subTreeNodeIndex_layer & 15;
            int nodeIndex = candidateSubTree.subTreeNodeIndex_layer >> 4;
            OctreeNodeData_E_FzbPG nodeData = OctreeNodeDataBuffer_E[layerIndex][nodeIndex];

            // nodeAABB = nodeData.aabb;
            // output.color.xyz = nodeData.radiance;

            CandidateLeafNodeData leafNodeData = candidateLeafNodeDataBuffer[pushConst.globalInfoAddress.clusterIndex_G * Candidate_Samples_Count + instanceIndex];
            nodeAABB = leafNodeData.aabb;
            output.color.xyz = leafNodeData.radiance;
        } else {
            #ifdef SpatialReuse_FzbPG
            CandidateSampleData_FzbPG candidateNodeData = candidateSampleData2Buffer[candidateIndex];

            if (pushConst.mode2 == 1) show = candidateNodeData.w > 0.0f;
            if (!show) {
                output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
                return output;
            }

            nodeAABB = candidateSampleAABBBuffer[candidateIndex];
            output.color.xyz = float3(10.0f, 0.0f, 0.0f);
            #endif
        }
       
    }

    float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

    output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
    output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
    output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
    output.sv_position.xyz += nodeAABB.minimum;

    output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);

    return output;
}

[shader("pixel")]
PSout fragmentMain(VSout input) {
    return { input.color };
}