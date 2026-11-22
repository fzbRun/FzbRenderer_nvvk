#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"
#include "renderer/VRPG2/LightInjection/LightInjectionShaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant_VRPG> pushConst;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eGeoClusterData)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> GeoClusterData;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eHasGeometryNodeInfo_VRPG)]] RWStructuredBuffer<HasGeometryNodeInfo_VRPG, ScalarDataLayout> HasGeometryNodeInfo_VRPGBuffer;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2> GlobalInfoBuffer;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eLightInjectionGlobalInfo)]] RWStructuredBuffer<LightInjectGlobalInfo_VRPG> LightInjectionGlobalInfoBuffer;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eLightNodeData)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData;
//--------------------------------------------------------------------------------------------
groupshared uint groupHasGeometryNodeCount;
groupshared uint groupHasGeometryNodeLabel_inArray;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_init(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHasGeometryNodeCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= pushConst.nodeCount * 6) return;

    ClusterData_Octree_VRPG geoClusterData = GeoClusterData[threadIndex];
    bool hasData = geoClusterData.meanNormal.w > 0.0f;

    LightNodeData_Octree_VRPG lightNodeData;
    lightNodeData.radiance = float3(0.0f);
    lightNodeData.halfAngle = M_PI;
    lightNodeData.meanNormal = geoClusterData.meanNormal;
    lightNodeData.aabb = geoClusterData.aabb;
    LightNodeData[threadIndex] = lightNodeData;

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint warphasGeometryNodeCount = WaveActiveCountBits(hasData);
    uint hasGeometryNodeLabel_inWarp = WavePrefixCountBits(hasData);

    uint hasGeometryNodeLabel_inGroup = 0;
    if (warpLane == 0) InterlockedAdd<uint>(groupHasGeometryNodeCount, warphasGeometryNodeCount, hasGeometryNodeLabel_inGroup);
    hasGeometryNodeLabel_inGroup = WaveReadLaneFirst(hasGeometryNodeLabel_inGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd<uint>(LightInjectionGlobalInfoBuffer[0].hasGeometryNodeCount, groupHasGeometryNodeCount, groupHasGeometryNodeLabel_inArray);
    GroupMemoryBarrierWithGroupSync();

    if (hasData) {
        uint hasGeometryNodeIndex = groupHasGeometryNodeLabel_inArray + hasGeometryNodeLabel_inGroup + hasGeometryNodeLabel_inWarp;
        HasGeometryNodeInfo_VRPGBuffer[hasGeometryNodeIndex].nodeIndex = threadIndex;
    }
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_dispatchLightInjection(uint threadIndex: SV_DispatchThreadID) {
    DispatchIndirectCommand cmd;
    cmd.x = (LightInjectionGlobalInfoBuffer[0].hasGeometryNodeCount + LIGHTINJECT_CS_THREADGROUP_SIZE - 1) / LIGHTINJECT_CS_THREADGROUP_SIZE;
    cmd.y = 1; cmd.z = 1;
    GlobalInfoBuffer[0].cmd = cmd;
}
//--------------------------------------------------------------------------------------------
struct LightInjectHitPayload {
    uint randomSeed;

    float3 hitPos = float3(0.0f);
    float3 hitNormal = float3(0.0f);
    float cosineON;
    float3x3 TBN;

    float3 albedo;
    float3 emissive;

    float3 incidence;

    bool hitLight;
    float3 lightRadiance;
}

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
        BSDFMaterial material = sceneInfo.materials[instance.materialIndex]; // Material properties

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
        payload.cosineON = dot(payload.hitNormal, -ray.Direction);
        if (payload.cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        if (material.type != MaterialType::Dielectric && material.type != MaterialType::RoughDielectric)
            material.emissive *= payload.cosineON > 0.0f ? 1.0f : 0.0f;

        float3 tangent, bitangent;
        orthonormalBasis(payload.hitNormal, tangent, bitangent);
        payload.TBN = float3x3(tangent, bitangent, payload.hitNormal); // column matrix

        if (any(material.materialMapIndex >= 0)) {
            int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, primitiveIndex);
            float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
            texCoords.y = 1.0f - texCoords.y;
            if (any(texCoords > 0.0f)) {
                int normalMapIndex = material.materialMapIndex.x;
                if (normalMapIndex >= 0) {
                    float3 localNormal = textures[normalMapIndex].SampleLevel(texCoords, 0).xyz;
                    payload.hitNormal = mul(localNormal, payload.TBN);
                }

                int albedoMapIndex = material.materialMapIndex.y;
                if (albedoMapIndex == int(AlbedoMapType::Checkerboard)) {
                    float2 uv = float2(texCoords.x * 8.0f, texCoords.y * 32.0f) * 2.0f;
                    float2 checker = floor(uv) % 2.0f;
                    float pattern = (checker.x + checker.y) % 2.0f;
                    material.albedo *= lerp(float3(0.8f), float3(0.2f), pattern);
                }
                else if (albedoMapIndex >= 0) material.albedo *= textures[albedoMapIndex].SampleLevel(texCoords, 0).xyz;

                int bsdfParamMapIndex = material.materialMapIndex.z;
                if (bsdfParamMapIndex >= 0) {
                    float4 bsdfParam = textures[bsdfParamMapIndex].SampleLevel(texCoords, 0);
                    material.eta = bsdfParam.xyz;
                    material.roughness = bsdfParam.w;
                }
            }
        }
        payload.albedo = material.albedo;
        payload.emissive = material.emissive;

        return true;
    }
    return false;
}

float3 RayMiss(float3 rayDirection) {
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1) return evalSimpleSky(sceneInfo.skySimpleParam, rayDirection);
    else return sceneInfo.backgroundColor;
}

bool traceRay_NEE(RayDesc ray, inout HitTestPayload payload) {
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(topLevelAS, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        // For non-opaque triangles, force them to be opaque (simplified alpha handling)
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            q.CommitNonOpaqueTriangleHit(); // Accept all intersections as solid
    }

    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        payload.distance = q.CommittedRayT();
        return true;
    }

    return false;
}
void initNEE(inout LightInjectHitPayload payload) {
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    uint lightCount = min(sceneInfo.numLights, 2);
    payload.lightRadiance = float3(0.0f);

    for (int lightIndex = 0; lightIndex < lightCount; ++lightIndex) {
        Light light = sceneInfo.lights[lightIndex];
        if (light.type == uint(LightType::Area)) {
            Quadrilateral quad;
            quad.startPos = light.pos;
            quad.normal = light.direction;
            quad.edge1 = light.edge1;
            quad.edge2 = light.edge2;
            float pdf = 1.0f;
            float3 lightSampleDir = sphereRectangleSample(quad, payload.hitPos, rand(payload.randomSeed), rand(payload.randomSeed), pdf);
            float distance = length(lightSampleDir);
            lightSampleDir = normalize(lightSampleDir);

            if (isnan(pdf) || isinf(pdf)) continue;
            if (dot(-lightSampleDir, light.direction) <= 0.001f) continue;
            if (dot(lightSampleDir, payload.hitNormal) <= 0.001f) continue;

            RayDesc ray;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;
            ray.Origin = payload.hitPos + 0.0001f * lightSampleDir;
            ray.Direction = lightSampleDir;

            HitTestPayload samplePayload;
            if (traceRay_NEE(ray, samplePayload)) {
                if (samplePayload.distance + 0.002f < distance) continue;

                float cosineIN = dot(lightSampleDir, payload.hitNormal);
                payload.lightRadiance += light.color * light.intensity * cosineIN;
            }
        } else if (light.type == uint(LightType::Direction)) {
            float3 lightSampleDir = -light.direction;

            if (dot(lightSampleDir, payload.hitNormal) <= 0.001f) continue;

            RayDesc ray;
            ray.TMin = 0.001;
            ray.TMax = INFINITE;
            ray.Origin = payload.hitPos + 0.0001f * lightSampleDir;
            ray.Direction = lightSampleDir;

            HitTestPayload samplePayload;
            if (!traceRay_NEE(ray, samplePayload)) {
                float cosineIN = dot(lightSampleDir, payload.hitNormal);
                payload.lightRadiance += light.color * light.intensity * cosineIN;
            }
        }
    }
}
//--------------------------------------------------------------------------------------------
[numthreads(LIGHTINJECT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_LightInject(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHasGeometryNodeCount = LightInjectionGlobalInfoBuffer[0].hasGeometryNodeCount;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHasGeometryNodeCount) return;

    HasGeometryNodeInfo_VRPG nodeInfo = HasGeometryNodeInfo_VRPGBuffer[threadIndex];
    LightNodeData_Octree_VRPG nodeData = LightNodeData[nodeInfo.nodeIndex];

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    float3 accumulatedRadiance = float3(0.0f);
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    float3 tangent, bitangent;
    orthonormalBasis(nodeData.meanNormal.xyz, tangent, bitangent);
    float3x3 voxelTBN = float3x3(tangent, bitangent, nodeData.meanNormal.xyz);

    int normalIndex = int(nodeInfo.nodeIndex) / pushConst.nodeCount;
    float3 Normal;
    float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
    Normal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
    Normal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
    Normal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

    float3 voxelAabbCenter = (nodeData.aabb.maximum + nodeData.aabb.minimum) * 0.5f;
    float3 voxelAabbHalfSize = (nodeData.aabb.maximum - nodeData.aabb.minimum) * 0.5f;

    {
        LightInjectHitPayload payload;
        payload.randomSeed = xxhash32(uint3(nodeInfo.nodeIndex, pushConst.randomFrameIndex, pushConst.sampleIndex));

        float3 sampleDir = fibSpherePoint(pushConst.sampleIndex, pushConst.spv);
        sampleDir = mul(sampleDir, pushConst.randomRotateMatrix);
        sampleDir = mul(sampleDir, voxelTBN);
        float3 samplePos = getAABBPoint(sampleDir, voxelAabbCenter, voxelAabbHalfSize) + 0.01f * sampleDir;
        sampleDir = -sampleDir;

        ray.Origin = samplePos;
        ray.Direction = sampleDir;

        uint bounceDepth = 0;
        float3 bsdf_cosine_pdf = float3(1.0f);

        [unroll]
        while (bounceDepth < pushConst.bounces) {
            bool hit = traceRay(ray, payload);
            if (hit) {
                if (bounceDepth == 0) {
                    bool reSample = payload.cosineON <= 0.0f;
                    reSample |= !checkPointInAABB(payload.hitPos, nodeData.aabb);
                    reSample |= dot(payload.hitNormal, Normal) <= 0.0f;
                    if (reSample) break;
                }

                float3 bsdf = M_1_PI * payload.albedo;
                float3 radiance_nee = float3(0.0f);
                if (isPosInDirectionLight(payload.hitPos, sceneInfo.lights[0])) {
                    if (bounceDepth > 0) payload.emissive = float3(0.0f);
                } else {
                    initNEE(payload);
                    radiance_nee = payload.lightRadiance * bsdf;
                }
                accumulatedRadiance += (payload.emissive + radiance_nee) * bsdf_cosine_pdf;
                //if (bounceDepth == 0 && any(payload.emissive > 0.0f)) accumulatedRadiance += payload.emissive * 10000.0f;

                float3 incidence = sample_hemisphere_mean(payload.randomSeed);
                payload.incidence = mul(incidence, payload.TBN);
                bsdf_cosine_pdf *= bsdf; // * dot(incidence, payload.hitNormal);

                ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                ray.Direction = payload.incidence;

                ++bounceDepth;
            } else {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                bounceDepth = MISS_DEPTH;
            }
        }
    }
    accumulatedRadiance /= pushConst.spv;
    float3 lengthAxis = nodeData.aabb.maximum - nodeData.aabb.minimum;
    int3 axisSign = sign(lengthAxis);
    lengthAxis += float3(int3(1) - axisSign) * 0.0001f;
    if (all(axisSign == 0)) lengthAxis = float3(0.0f);
    accumulatedRadiance *= (lengthAxis.x * lengthAxis.y + lengthAxis.x * lengthAxis.z + lengthAxis.y * lengthAxis.z) * 2.0f;

    if (any(accumulatedRadiance > 0.0f) && !any(isnan(accumulatedRadiance) || isinf(accumulatedRadiance))) {
        LightNodeData[nodeInfo.nodeIndex].radiance += accumulatedRadiance;
    }
}