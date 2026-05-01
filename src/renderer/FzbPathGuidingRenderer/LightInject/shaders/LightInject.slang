#define NEE

#include "renderer/FzbPathGuidingRenderer/LightInject/LightInjectShaderio_FzbPG.h"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"

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
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eVGB)]] RWStructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eHasGeometryVoxelInfo)]] RWStructuredBuffer<HasGeometryVoxelInfo> HasGeometryVoxelInfoBuffer;
[[vk::binding(StaticBindingPoints_LightInject_FzbPG::eGlobalInfo)]] RWStructuredBuffer<LightInjectGlobalInfo> GlobalInfoBuffer;
//----------------------------------------------getHasGeometryVoxel--------------------------------------------
groupshared uint groupHasGeometryVoxelCount;
groupshared uint groupHasGeometryVoxelLabel_inArray;

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
    hasGeometryVoxelLabel_inGroup = WaveReadLaneFirst(hasGeometryVoxelLabel_inGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd<uint>(GlobalInfoBuffer[0].hasGeometryVoxelCount, groupHasGeometryVoxelCount, groupHasGeometryVoxelLabel_inArray);
    GroupMemoryBarrierWithGroupSync();

    if (hasData) {
        uint hasGeometryVoxelIndex = groupHasGeometryVoxelLabel_inArray + hasGeometryVoxelLabel_inGroup + hasGeometryVoxelLabel_inWarp;
        HasGeometryVoxelInfoBuffer[hasGeometryVoxelIndex].voxelIndex_normalIndex = (voxelIndex << 4) | normalIndex;
    }
}
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_setDispatchIndirectCommand(uint threadIndex: SV_DispatchThreadID) {
    DispatchIndirectCommand cmd;
    cmd.x = (GlobalInfoBuffer[0].hasGeometryVoxelCount + LIGHTINJECT_CS_THREADGROUP_SIZE - 1) / LIGHTINJECT_CS_THREADGROUP_SIZE;
    cmd.y = 1; cmd.z = 1;
    GlobalInfoBuffer[0].cmd = cmd;
}
//-----------------------------------------Struct And Function--------------------------------------------
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
    float3 lightSampleDir;
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
        //if (cosineON <= 0.0f) payload.hitNormal = -payload.hitNormal;
        //if (payload.material.type != MaterialType::Dielectric && payload.material.type != MaterialType::RoughDielectric)
        //    payload.material.emissive *= cosineON > 0.0f ? 1.0f : 0.0f;

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
    //uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    payload.lightRadiance = float3(0.0f);

    Light light = sceneInfo.lights[0];
    if (light.type == uint(LightType::Area)) {
        Quadrilateral quad;
        quad.startPos = light.pos;
        quad.normal = light.direction;
        quad.edge1 = light.edge1;
        quad.edge2 = light.edge2;
        float pdf = 1.0f;
        payload.lightSampleDir = sphereRectangleSample(quad, payload.hitPos, rand(payload.randomSeed), rand(payload.randomSeed), pdf);
        float distance = length(payload.lightSampleDir);
        payload.lightSampleDir = normalize(payload.lightSampleDir);

        if (isnan(pdf) || isinf(pdf)) return;
        if (dot(-payload.lightSampleDir, light.direction) <= 0.001f) return;
        if (dot(payload.lightSampleDir, payload.hitNormal) <= 0.001f) return;

        RayDesc ray;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        ray.Origin = payload.hitPos + 0.0001f * payload.lightSampleDir;
        ray.Direction = payload.lightSampleDir;

        HitTestPayload samplePayload;
        if (traceRay_NEE(ray, samplePayload)) {
            if (samplePayload.distance + 0.002f < distance) return;
            payload.lightRadiance = light.color * light.intensity;
        }
    } else if (light.type == uint(LightType::Direction)) {
        payload.lightSampleDir = -light.direction;

        RayDesc ray;
        ray.TMin = 0.001;
        ray.TMax = INFINITE;
        ray.Origin = payload.hitPos + 0.0001f * payload.lightSampleDir;
        ray.Direction = payload.lightSampleDir;

        HitTestPayload samplePayload;
        if (!traceRay_NEE(ray, samplePayload))
            payload.lightRadiance = light.color * light.intensity;
    }

}
//----------------------------------------------LightInject--------------------------------------------
[numthreads(LIGHTINJECT_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_LightInject(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex == 0) groupHasGeometryVoxelCount = GlobalInfoBuffer[0].hasGeometryVoxelCount;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= groupHasGeometryVoxelCount) return;

    uint voxelInfo = HasGeometryVoxelInfoBuffer[threadIndex].voxelIndex_normalIndex;
    uint normalIndex = voxelInfo & 7u;
    uint voxelIndex = voxelInfo >> 4;
    uint voxelIndex_atAll = normalIndex * pushConst.voxelCount + voxelIndex;
    VGBVoxelData_SVOPG voxelData = VGBs[normalIndex][voxelIndex];
    AABB voxelAabb; {
        voxelAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
        voxelAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
        voxelAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
        voxelAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
        voxelAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
        voxelAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);}
    float3 voxelAabbCenter = (voxelAabb.maximum + voxelAabb.minimum) * 0.5f;
    float3 voxelAabbHalfSize = (voxelAabb.maximum - voxelAabb.minimum) * 0.5f;
    float3 voxelNormal = normalize(voxelData.sumNormal_G.xyz / voxelData.sumNormal_G.w);

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    float3 accumulatedRadiance = float3(0.0f);
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    float3 tangent, bitangent;
    orthonormalBasis(voxelNormal, tangent, bitangent);
    float3x3 voxelTBN = float3x3(tangent, bitangent, voxelNormal);

    float3 VGBNormal;
    float plusMinusSign = float(normalIndex & 1u) * 2.0f - 1.0f;
    VGBNormal.x = (1.0f - sign(normalIndex & 6u)) * plusMinusSign;
    VGBNormal.y = ((normalIndex & 2u) >> 1) * plusMinusSign;
    VGBNormal.z = ((normalIndex & 4u) >> 2) * plusMinusSign;

    LightInjectHitPayload payload;
    for (int sampleIndex = 0; sampleIndex < LIGHTINJECT_SAMPLE_COUNT; ++sampleIndex) {
        payload.randomSeed = xxhash32(uint3(voxelIndex_atAll, pushConst.frameIndex, sampleIndex));

        float3 sampleDir = sample_hemisphere_mean(payload.randomSeed);
        sampleDir = mul(sampleDir, voxelTBN);
        float3 samplePos = getAABBPoint(sampleDir, voxelAabbCenter, voxelAabbHalfSize) + 0.01f * sampleDir;
        sampleDir = -sampleDir;

        ray.Origin = samplePos;
        ray.Direction = sampleDir;

        uint bounceDepth = 0;
        float3 bsdf_cosine_pdf = float3(1.0f);

        const uint maxPathDepth = 1;
        while (bounceDepth < maxPathDepth) {
            bool hit = traceRay(ray, payload);
            if (hit) {
                bool reSample = payload.cosineON <= 0.0f;
                reSample |= !checkPointInAABB(payload.hitPos, voxelAabb);
                reSample |= dot(payload.hitNormal, VGBNormal) <= 0.0f;
                if (reSample) {
                    payload.hitNormal = voxelNormal;
                    payload.incidence = mul(sample_hemisphere_mean(payload.randomSeed), voxelTBN);
                    payload.hitPos = getAABBPoint(sampleDir, voxelAabbCenter, voxelAabbHalfSize) + 0.01f * payload.incidence;
                    payload.incidence = -payload.incidence;
                }
                else {
                    float3 bsdf = M_1_PI * payload.albedo;

                    float3 radiance_nee = float3(0.0f);
                    if (isPosInDirectionLight(payload.hitPos, sceneInfo.lights[0])) {
                        if(bounceDepth > 0) payload.emissive = float3(0.0f);
                    }
                    else {
                        initNEE(payload);
                        if (any(payload.lightRadiance > 0.0f)) {
                            float cosineIN = dot(payload.lightSampleDir, payload.hitNormal);
                            radiance_nee = payload.lightRadiance * bsdf; // * cosineIN;
                        }
                    }
                    accumulatedRadiance += (payload.emissive + radiance_nee) * bsdf_cosine_pdf;
                    if (bounceDepth == 0 && payload.emissive.x > 0.0f) accumulatedRadiance += payload.emissive * 2.0f;

                    float3 incidence = sample_hemisphere_mean(payload.randomSeed);
                    payload.incidence = mul(incidence, payload.TBN);
                    bsdf_cosine_pdf *= payload.albedo * M_1_PI; // * dot(incidence, payload.hitNormal);
                }

                ray.Origin = offsetRay(payload.hitPos, payload.hitNormal);
                ray.Direction = payload.incidence;

                ++bounceDepth;

                //bool printfAble = pushConst.frameIndex == 2;
                //printfAble &= normalIndex == 1;
                //printfAble &= voxelIndex == 1150;
                //printfAble &= any(accumulatedRadiance > 0.0f);
                //if (printfAble) {
                //    printf("sampleIndex: %d\naabb: %f %f %f %f %f %f\nhitPos: %f %f %f, hitNormal: %f %f  %f\nreSample: %d\n\n",
                //        sampleIndex,
                //        voxelAabb.minimum.x, voxelAabb.minimum.y, voxelAabb.minimum.z,
                //        voxelAabb.maximum.x, voxelAabb.maximum.y, voxelAabb.maximum.z,
                //        payload.hitPos.x, payload.hitPos.y, payload.hitPos.z,
                //        payload.hitNormal.x, payload.hitNormal.y, payload.hitNormal.z,
                //        reSample
                //    );
                //}
            } else {
                accumulatedRadiance += RayMiss(ray.Direction) * bsdf_cosine_pdf;
                bounceDepth = MISS_DEPTH;
            }
        }
    }
    accumulatedRadiance /= LIGHTINJECT_SAMPLE_COUNT;

    if (any(accumulatedRadiance > 0.0f) && !any(isnan(accumulatedRadiance) || isinf(accumulatedRadiance))) {
        InterlockedAdd<float>(VGBs[normalIndex][voxelIndex].irradiance.w, 1.0f);
        InterlockedAdd<float>(VGBs[normalIndex][voxelIndex].irradiance.x, accumulatedRadiance.x);
        InterlockedAdd<float>(VGBs[normalIndex][voxelIndex].irradiance.y, accumulatedRadiance.y);
        InterlockedAdd<float>(VGBs[normalIndex][voxelIndex].irradiance.z, accumulatedRadiance.z);
    }
}
#ifndef NDEBUG
//------------------------------------------------debug---------------------------------------------------------
struct VSout_Cube {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 color;
}
struct PSout_Cube
{
    float4 color : SV_Target;
};
[shader("vertex")]
VSout_Cube vertexMain_Cube(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    uint voxelTotalCount = pushConst.voxelCount;
    uint VGBIndex = instanceIndex / voxelTotalCount;
    uint voxelIndex = instanceIndex % voxelTotalCount;

    VGBVoxelData_SVOPG voxelData = VGBs[VGBIndex][voxelIndex];
    VSout_Cube output;
    if (voxelData.irradiance.w == 0.0f) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        output.worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.x) : OrderedIntToFloat(voxelData.aabbI.maximum.x);
        output.worldPos.y = (vertexIndex & 3) <= 1 ? OrderedIntToFloat(voxelData.aabbI.minimum.y) : OrderedIntToFloat(voxelData.aabbI.maximum.y);
        output.worldPos.z = vertexIndex / 4 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.z) : OrderedIntToFloat(voxelData.aabbI.maximum.z);

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        output.sv_position = mul(float4(output.worldPos.xyz, 1.0f), sceneInfo.viewProjMatrix);
        output.color = voxelData.irradiance.xyz / voxelData.irradiance.w;
    }
    if (instanceIndex == 5246) {
        output.worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.x) : OrderedIntToFloat(voxelData.aabbI.maximum.x);
        output.worldPos.y = (vertexIndex & 3) <= 1 ? OrderedIntToFloat(voxelData.aabbI.minimum.y) : OrderedIntToFloat(voxelData.aabbI.maximum.y);
        output.worldPos.z = vertexIndex / 4 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.z) : OrderedIntToFloat(voxelData.aabbI.maximum.z);
    
        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        output.sv_position = mul(float4(output.worldPos.xyz, 1.0f), sceneInfo.viewProjMatrix);
        output.color = float3(1.0f, 0.0f, 0.0f);
    }

    return output;
}
[shader("pixel")]
PSout_Cube fragmentMain_Cube(VSout_Cube input) {
    return { float4(input.color, 1.0f) };
}
#endif