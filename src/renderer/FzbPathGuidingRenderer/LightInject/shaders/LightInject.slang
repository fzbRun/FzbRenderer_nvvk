#include "renderer/FzbPathGuidingRenderer/LightInject/LightInjectShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

struct AABBI {
    int4 minimum;
    int4 maximum;
};
struct VGBVoxelData_SVOPG {
    float4 irradiance;
    float4 sumNormal_G;
    float4 sumNormal_E;
    AABBI aabbI;
};

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant_FzbPG> pushConst;
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eVGB)]] StructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eHasGeometryVoxelInfo)]] RWStructuredBuffer<HasGeometryVoxelInfo> HasGeometryVoxelInfoBuffer;
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eGlobalInfo)]] RWStructuredBuffer<LightInjectGlobalInfo> GlobalInfoBuffer;
//----------------------------------------------getHasGeometryVoxel--------------------------------------------
groupshared uint groupHasGeometryVoxelCount;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getHasGeometryVoxels(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHasGeometryVoxelCount = 0;
    GroupMemoryBarrierWithGroupSync();

    uint normalIndex = threadIndex / pushConst.voxelCount;
    uint voxelIndex = threadIndex & (pushConst.voxelCount - 1);
    bool hasData = VGBs[normalIndex][voxelIndex].sumNormal_G.w > 0.0f;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint warpHasGeometryVoxelCount = WaveActiveCountBits(hasData);
    uint hasGeometryVoxelLabel_inWarp = WavePrefixCountBits(hasData);

    uint hasGeometryVoxelLabel_inGroup = 0;
    if (warpLane == 0) InterlockedAdd<uint>(groupHasGeometryVoxelCount, warpHasGeometryVoxelCount, hasGeometryVoxelLabel_inGroup);
    GroupMemoryBarrierWithGroupSync();

    uint hasGeometryVoxelLabel_inArray = 0;
    if (groupThreadIndex == 0) InterlockedAdd<uint>(GlobalInfoBuffer[0].hasGeometryVoxelCount, groupHasGeometryVoxelCount, hasGeometryVoxelLabel_inArray);
    GroupMemoryBarrierWithGroupSync();

    uint hasGeometryVoxelIndex = hasGeometryVoxelLabel_inArray + hasGeometryVoxelLabel_inGroup + hasGeometryVoxelLabel_inWarp;
    HasGeometryVoxelInfoBuffer[hasGeometryVoxelIndex].voxelIndex_normalIndex = (voxelIndex << 4) | normalIndex;
}
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_setDispatchIndirectCommand(uint threadIndex: SV_DispatchThreadID) {
    DispatchIndirectCommand cmd;
    cmd.x = (GlobalInfoBuffer[0].hasGeometryVoxelCount + LIGHTINJECT_CS_THREADGROUP_SIZE - 1) / LIGHTINJECT_CS_THREADGROUP_SIZE;
    cmd.y = 1; cmd.z = 1;
    GlobalInfoBuffer[0].cmd = cmd;
}
//----------------------------------------------LightInject--------------------------------------------
struct LightInjectHitPayload {
    uint randomSeed;

    BSDFMaterial material;
    float3 radiance;

    float3 rayOrigin;
    float3 rayDirection;

    float3 hitPos;
    float3 hitNormal;

    uint bounceDepth;

    bool isExt;
}

typedef CallablePayload HitPayload_FzbPG;
bool traceRay(RayDesc ray, inout LightInjectHitPayload payload)
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
[numthreads(LIGHTINJECT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_LightInject(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHasGeometryVoxelCount = GlobalInfoBuffer[0].hasGeometryVoxelCount;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHasGeometryVoxelCount) return;

    uint2 voxelInfo = HasGeometryVoxelInfoBuffer[threadIndex].voxelIndex_normalIndex;
    VGBVoxelData_SVOPG voxelData = VGBs[voxelInfo.x][voxelInfo.y];
    AABB voxelAabb; {
        voxelAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
        voxelAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
        voxelAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
        voxelAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
        voxelAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
        voxelAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);}
    float3 voxelNormal = normalize(voxelData.sumNormal_G.xyz / voxelData.sumNormal_G.w);

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    HitPayload_FzbPG payload;

    for (int sampleIndex = 0; sampleIndex < LIGHTINJECT_SAMPLE_COUNT; ++sampleIndex) {
        uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex, sampleIndex));
        float3 randXYZ = float3(rand(randomSeed), rand(randomSeed), rand(randomSeed));
        float3 samplePos = randXYZ * (voxelAabb.maximum - voxelAabb.minimum) * 0.5f + voxelAabb.minimum;
        float3 sampleDir = sample_hemisphere_mean(randomSeed);

        ray.Origin = samplePos;
        ray.Direction = sampleDir;

        payload.randomSeed = randomSeed;
        payload.radiance = float3(0.0f);
        payload.bounceDepth = 0;
        payload.isExt = true; // first closestHit shader check
        payload.rayOrigin = ray.Origin;
        payload.rayDirection = ray.Direction;


    }
}