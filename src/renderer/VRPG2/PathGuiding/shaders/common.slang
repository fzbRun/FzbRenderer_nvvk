#pragma once
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG2/PathGuiding/PathGuidingShaderio.h"
#include "renderer/VRPG2/RIS/RISShaderio.h"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<PathGuidingPushConstant_VRPG2, ScalarDataLayout> pushConst;

[[vk::binding(PathGuidingBindingPoints_VRPG2::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(PathGuidingBindingPoints_VRPG2::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];
[[vk::binding(PathGuidingBindingPoints_VRPG2::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];
[[vk::binding(PathGuidingBindingPoints_VRPG2::eNearbyClusterInfos)]] RWStructuredBuffer<NearbyClusterInfo_VRPG, ScalarDataLayout> NearbyClusterInfoBuffer;

[[vk::binding(PathGuidingBindingPoints_VRPG2::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2, ScalarDataLayout> GlobalInfoBuffer;

[[vk::binding(PathGuidingBindingPoints_VRPG2::eCandidateSubTreeInfo)]] RWStructuredBuffer<CandidateSubTreeInfo_VRPG2, ScalarDataLayout> candidateSubTreeDataBuffer;
[[vk::binding(PathGuidingBindingPoints_VRPG2::eCandidateLeafNodeData)]] RWStructuredBuffer<CandidateLeafNodeData_VRPG2, ScalarDataLayout> candidateLeafNodeDataBuffer;
[[vk::binding(PathGuidingBindingPoints_VRPG2::eCandidateSampleData)]] RWStructuredBuffer<CandidateSampleData_VRPG2, ScalarDataLayout> candidateSampleDataBuffer;
[[vk::binding(PathGuidingBindingPoints_VRPG2::eCandidateSamplePrefixWSum)]] RWStructuredBuffer<float> candidateSamplePrefixWSumBuffer;
//-----------------------------------------------------------------------------------------------------------------------------------------
typedef CallablePayload HitPayload_VRPG;

struct VRPGGuideHit {
    bool hit;
    bool frontFacing;
    bool hasTexCoords;
    float3 hitPos;
    float3 hitNormal;
    float2 texCoords;
    BSDFMaterial material;
};

float3 getDirectionLight2(VRPGGuideHit hit, float3 outgoing, bool isExt, SceneInfo sceneInfo) {
    float3x3 TBN;
    orthonormalBasis(hit.hitNormal, TBN[0], TBN[1]);
    TBN[2] = hit.hitNormal;

    bool isDielectric = hit.material.type == uint(MaterialType::Dielectric) ||
                        hit.material.type == uint(MaterialType::RoughDielectric);
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int lightIndex = 0; lightIndex < lightNum; ++lightIndex) {
        Light light = sceneInfo.lights[lightIndex];
        if (light.type == uint(LightType::Direction)) {
            float3 sampleDir = -light.direction;
            float cosine = dot(sampleDir, hit.hitNormal);
            if (!isDielectric && cosine <= 0.001f) continue;

            const uint rayFlags = 0;
            RayDesc tempRay;
            tempRay.Direction = normalize(sampleDir);
            tempRay.Origin = hit.hitPos + 0.00001f * tempRay.Direction;
            tempRay.TMin = 0.001f;
            tempRay.TMax = INFINITE;
            HitTestPayload tempPayload;
            tempPayload.distance = MISS_DISTANCE;

            hitTestTraceRay(tempRay, tempPayload);
            if (tempPayload.distance != MISS_DISTANCE) continue;

            float3 bsdf = getBSDF(hit.material, sampleDir, hit.hitNormal, outgoing, TBN, isExt);
            return light.color * light.intensity * abs(cosine) * bsdf;
        }
    }

    return 0.0f;
}

VRPGGuideHit traceGuideRay(RayDesc ray)
{
    VRPGGuideHit result;
    result.hit = false;
    result.frontFacing = true;
    result.hasTexCoords = false;
    result.hitPos = float3(0.0f);
    result.hitNormal = float3(0.0f, 1.0f, 0.0f);
    result.texCoords = float2(0.0f);

    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return result;

    float2 bary2 = q.CommittedTriangleBarycentrics();
    float3 bary = float3(1.0f - bary2.x - bary2.y, bary2.x, bary2.y);
    float4x3 worldToObject = q.CommittedWorldToObject4x3();
    float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
    int instanceIndex = q.CommittedInstanceIndex();
    uint primitiveIndex = q.CommittedPrimitiveIndex();
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    result.material = sceneInfo.materials[instance.materialIndex];

    float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
    float3 localPos = triPos[0] * bary.x + triPos[1] * bary.y + triPos[2] * bary.z;
    result.hitPos = float3(mul(float4(localPos, 1.0), objectToWorld));
    float3 geoNormal = normalize(cross(triPos[1] - triPos[0], triPos[2] - triPos[0]));
    result.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
    float cosineON = dot(result.hitNormal, -ray.Direction);
    result.frontFacing = cosineON > 0.0f;
    if (cosineON <= 0.0f) result.hitNormal = -result.hitNormal;

    if (result.material.materialMapIndex.x >= 0 || result.material.materialMapIndex.y >= 0 || result.material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
        result.texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, bary);
        result.texCoords.y = 1.0f - result.texCoords.y;
        result.hasTexCoords = true;
        if (any(result.texCoords > 0.0f)) {
            int normalMapIndex = result.material.materialMapIndex.x;
            if (normalMapIndex >= 0) {
                float3 tangent, bitangent;
                orthonormalBasis(result.hitNormal, tangent, bitangent);
                float3x3 TBN = float3x3(tangent, bitangent, result.hitNormal);

                float3 localNormal = textures[normalMapIndex].SampleLevel(result.texCoords, 0).xyz;
                result.hitNormal = mul(localNormal, TBN);
            }
            getBSDFTextureData(result.material, result.texCoords);
        }
    }
    if (result.material.type != MaterialType::Dielectric && result.material.type != MaterialType::RoughDielectric)
        result.material.emissive *= result.frontFacing ? 1.0f : 0.0f;

    result.hit = true;
    return result;
}

HitPayload_VRPG makeGuidePayload(VRPGGuideHit hit, float3 guideDir, uint seed, SceneInfo sceneInfo)
{
    HitPayload_VRPG payload;
    payload.material = hit.material;
    payload.hitPos = hit.hitPos;
    payload.hitNormal = hit.hitNormal;
    payload.outgoing = -guideDir;
    payload.incidence = guideDir;
    payload.randomSeed = seed;
    payload.bsdf = float3(0.0f);
    payload.pdf = 1.0f;
    payload.isExt = hit.frontFacing;
    orthonormalBasis(payload.hitNormal, payload.TBN[0], payload.TBN[1]);
    payload.TBN[2] = payload.hitNormal;
    // if (hit.hasTexCoords)
    //     getTextueData(payload, hit.texCoords);
    if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
        payload.material.emissive *= hit.frontFacing ? 1.0f : 0.0f;
    return payload;
}

bool traceRay(RayDesc ray, inout HitPayload_VRPG payload)
{
    #ifndef NDEBUG
    if (any(isnan(ray.Origin)) || any(isinf(ray.Origin)) ||
        any(isnan(ray.Direction)) || any(isinf(ray.Direction)))
        return false;
    #endif
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        float2 barycentricCoords = q.CommittedTriangleBarycentrics();
        float4x3 worldToObject = q.CommittedWorldToObject4x3();
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
        int instanceIndex = q.CommittedInstanceIndex();
        uint primitiveIndex = q.CommittedPrimitiveIndex();
        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex];
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
        payload.material = sceneInfo.materials[instance.materialIndex];
        const float3 barycentrics = float3(1.0f - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);
        float3 triPos[3] = q.CommittedGetIntersectionTriangleVertexPositions();
        const float3 localPosition = triPos[0] * barycentrics.x + triPos[1] * barycentrics.y + triPos[2] * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));
        const float3 edge1 = triPos[1] - triPos[0];
        const float3 edge2 = triPos[2] - triPos[0];
        const float3 geoNormal = normalize(cross(edge1, edge2));
        payload.hitNormal = normalize(mul(geoNormal, transpose(worldToObject)).xyz);
        float cosineON = dot(payload.hitNormal, -ray.Direction);
        payload.isExt = cosineON > 0.0f;
        if (cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
            payload.material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;
        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal);
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

bool traceRay_test(RayDesc ray, inout HitPayload_VRPG payload)
{
    #ifndef NDEBUG
    if (any(isnan(ray.Origin)) || any(isinf(ray.Origin)) ||
        any(isnan(ray.Direction)) || any(isinf(ray.Direction)))
        return false;
    #endif
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
    {
        float2 barycentricCoords = q.CommittedTriangleBarycentrics();
        float4x3 worldToObject = q.CommittedWorldToObject4x3();
        float4x3 objectToWorld = q.CommittedObjectToWorld4x3();
        int instanceIndex = q.CommittedInstanceIndex();
        uint primitiveIndex = q.CommittedPrimitiveIndex();
        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        Instance instance = sceneInfo.instances[instanceIndex];
        Mesh mesh = sceneInfo.meshes[instance.meshIndex];
        payload.material = sceneInfo.materials[instance.materialIndex];
        const float3 barycentrics = float3(1.0f - barycentricCoords.x - barycentricCoords.y, barycentricCoords.x, barycentricCoords.y);

        const int3 idx = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);

        if (any(idx < 0) ||
    idx.x >= int(mesh.triMesh.positions.count) ||
    idx.y >= int(mesh.triMesh.positions.count) ||
            idx.z >= int(mesh.triMesh.positions.count))
        {
            return false;
        }

        BufferView posf = mesh.triMesh.positions;
        posf.byteStride = 4;

        const uint i0 = uint(idx.x) * 3u;
        const uint i1 = uint(idx.y) * 3u;
        const uint i2 = uint(idx.z) * 3u;

        const float3 v0 = float3(getAttribute<float>(mesh.dataBuffer, posf, i0 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i0 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i0 + 2u));
        const float3 v1 = float3(getAttribute<float>(mesh.dataBuffer, posf, i1 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i1 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i1 + 2u));
        const float3 v2 = float3(getAttribute<float>(mesh.dataBuffer, posf, i2 + 0u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i2 + 1u),
                                 getAttribute<float>(mesh.dataBuffer, posf, i2 + 2u));

        const float3 localPosition = v0 * barycentrics.x + v1 * barycentrics.y + v2 * barycentrics.z;
        payload.hitPos = float3(mul(float4(localPosition, 1.0), objectToWorld));

        const float3 geoNormal = normalize(cross(v1 - v0, v2 - v0));
        float3x3 W = float3x3(worldToObject[0].xyz, worldToObject[1].xyz, worldToObject[2].xyz);
        payload.hitNormal = normalize(mul(geoNormal, W));

        if (any(isnan(payload.hitNormal))) return false;
        return true;
    }
    return false;
}

float3 RayMiss(float3 rayDirection) {
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}
