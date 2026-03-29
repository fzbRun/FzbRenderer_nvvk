#define NEE

#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/LightInject/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaderio.h"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_LightInject::eVGB_LightInject)]] RWStructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[];

struct LightInjectHitPayload {
    uint randomSeed;
    
    float3 radiance;

    float3 rayOrigin;
    float3 rayDirection;

    uint bounceDepth;

    bool isExt;
}

[shader("raygeneration")]
void raygenMain()
{
    uint3 samplePointIndex = DispatchRaysIndex();
    uint3 samplePointTotalCount = DispatchRaysDimensions();

    float3 samplePointPos = ((float3)samplePointIndex + float3(0.5f)) / (float3)samplePointTotalCount * pushConst.sceneSize + pushConst.sceneStartPos;
    uint sampleIndex = samplePointIndex.z * samplePointTotalCount.x * samplePointTotalCount.y + samplePointIndex.y * samplePointTotalCount.x + samplePointIndex.x;

    uint seed = xxhash32(uint3(sampleIndex, 0, pushConst.frameIndex));
    float3 sampleDir = sample_sphere_mean(seed);

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    RayDesc ray;
    ray.Origin = samplePointPos; // mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction = sampleDir;
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    LightInjectHitPayload payload;
    payload.randomSeed = seed;
    payload.radiance = float3(0.0f);
    payload.bounceDepth = 0;
    payload.isExt = true; // first closestHit shader check
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;

    const uint maxPathDepth = 3;
    while (payload.bounceDepth < maxPathDepth) {
        #ifdef PathTracingMotionBlur
        TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, pushConst.time, payload);
        #else
        TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
        #endif

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;
    }
}

[shader("closesthit")]
void rayClosestHitMain(inout LightInjectHitPayload payload, in BuiltInTriangleIntersectionAttributes attr) {
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    uint instanceID = InstanceIndex();
    uint meshID = InstanceID();
    uint triID = PrimitiveIndex();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceID];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics, payload.rayDirection);

    /*
    In the previous pt, rayDirection should be an outgoing from hitPos, but now the ray we start from samplePoint should be incidence.
    However, our functions are all written based on path tracing starting from the camera, so here outgoing and incidence are interchangeable
    */
    if (hitState.cosineON < 0.0f) {
        bool isDielectirc = material.type == MaterialType::Dielectric || material.type == MaterialType::RoughDielectric;
        if (payload.bounceDepth == 0) {
            if (isDielectirc) payload.isExt = false;
            else {  //inside the conductor
                payload.bounceDepth = MISS_DEPTH;
                return;
            }
        }

        hitState.cosineON = -hitState.cosineON;
        if (!isDielectirc) material.emissive = 0.0f;
    }

    CallablePayload callablePayload;
    callablePayload.material = material;
    callablePayload.TBN = hitState.TBN;
    callablePayload.randomSeed = payload.randomSeed;
    callablePayload.bsdf = float3(0.0f);
    callablePayload.pdf = 1.0f;
    callablePayload.hitPos = hitState.pos;
    callablePayload.hitNormal = hitState.normal;
    callablePayload.outgoing = -payload.rayDirection;
    callablePayload.isExt = payload.isExt;
    callablePayload.rayOrigin = payload.rayOrigin;

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
        float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
        texCoords.y = 1.0f - texCoords.y;
        getTextueData(callablePayload, texCoords);
    }

    callablePayload.radianceSum_nee = float3(0.0f);
    initNEE(callablePayload, pushConst.sceneInfoAddress[0], 1, pushConst.time);

    CallShader<CallablePayload>(uint(material.type), callablePayload);

    int voxelIndex = getVoxelIndex(hitState.pos, pushConst.VGBStartPos_Size.xyz, pushConst.VGBVoxelSize.xyz, (int)pushConst.VGBStartPos_Size.w);

    float3 geoNormal = payload.isExt ? callablePayload.hitNormal : -callablePayload.hitNormal;
    int VGBIndex = getNormalIndex(geoNormal);

    float3 irradiance = payload.radiance * max(hitState.cosineON, 0.0f) + material.emissive;
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        float3 radiance_nee = callablePayload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;
        irradiance += callablePayload.radiance_nee[i] * max(dot(callablePayload.sampleDir_nee[i], hitState.normal), 0.0f);
    }
    if (any(irradiance > 0.0f) && !any(isnan(irradiance) || isinf(irradiance))) {
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].irradiance.w, 1.0f);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].irradiance.x, irradiance.x);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].irradiance.y, irradiance.y);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].irradiance.z, irradiance.z);

        float3 normal = length(irradiance) * hitState.normal;
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].sumNormal_E.w, 1.0f);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].sumNormal_E.x, normal.x);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].sumNormal_E.y, normal.y);
        InterlockedAdd<float>(VGBs[VGBIndex][voxelIndex].sumNormal_E.z, normal.z);
    }

    payload.randomSeed = callablePayload.randomSeed;
    payload.radiance = payload.radiance * callablePayload.bsdf * hitState.cosineON + callablePayload.radianceSum_nee + material.emissive;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = hitState.pos + 0.001f * payload.rayDirection;
    payload.isExt = callablePayload.isExt;
    ++payload.bounceDepth;

    #ifndef NDEBUG
    if (pushConst.frameIndex == 1 && any(isnan(payload.radiance) || isinf(payload.radiance))) {
        printf("materialType: %d, radianceSum_nee: %f %f %f\n",
            material.type, payload.radiance.x, payload.radiance.y, payload.radiance.z
        );
    }
    #endif
}

[shader("miss")]
void rayMissMain(inout LightInjectHitPayload payload)
{
    payload.bounceDepth = MISS_DEPTH;
}

/*
We need to use MIS to calculate the Monte Carlo integral for each sample point. We consider the starting point of the ray as a bsdf sampling point,
along with several direct light sampling points

The weight calculation formula for each sample point is the pdf under that specific sampling method divided
by the sum of pdfs under all sampling methods

The first thing we can know is that a direct light sampling point cannot be a sample point of another direct light, so the pdf must be 0
So we only need to calculate the sampling probability of the ray starting point in each direct light (if it is not within the direct light,
the pdf is 0, so there must be only one direct light sample with pdf); and the pdf of each direct light sample point in the bsdf sampling

Assuming the pdf of bsdf remains the same after reversing the emission and incidence
*/
[shader("callable")]
void diffuseMaterialMain(inout CallablePayload payload) {
    PopulateCallablePayload_DiffuseMaterial(payload);

    float bsdfSamplePoint_lightSamplePdf = 0.0f;
    bool getDirectionLight = false;
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        if (!getDirectionLight) {
            Light light = sceneInfo.lights[i];
            if (isPosInDirectionLight(payload.rayOrigin, light)) {
                bsdfSamplePoint_lightSamplePdf = getPdf_directionLightSample(payload.rayOrigin, payload.outgoing, light);
                getDirectionLight = true;
            }
        }

        float3 radiance_nee = payload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];
        float cosineIN = dot(sampleDir, payload.hitNormal);
        float3 bsdf = M_1_PI * payload.material.albedo;

        float lightSamplePoint_bsdfSamplePdf = M_1_PI * cosineIN;
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / (lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf);
    }

    payload.bsdf *= 1.0f / (payload.pdf + bsdfSamplePoint_lightSamplePdf);
}
[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);

    float bsdfSamplePoint_lightSamplePdf = 0.0f;
    bool getDirectionLight = false;
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        if (!getDirectionLight) {
            Light light = sceneInfo.lights[i];
            if (isPosInDirectionLight(payload.rayOrigin, light)) {
                bsdfSamplePoint_lightSamplePdf = getPdf_directionLightSample(payload.rayOrigin, payload.outgoing, light);
                getDirectionLight = true;
            }
        }

        float3 radiance_nee = payload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];
        float cosineIN = dot(sampleDir, payload.hitNormal);
        float3 bsdf = float3(0.0f);
        float3 h = normalize(sampleDir + payload.incidence);
        if (abs(dot(h, payload.hitNormal)) < 0.99f) bsdf = float3(0.0f);

        float lightSamplePoint_bsdfSamplePdf = abs(dot(sampleDir, payload.hitNormal)) > 0.99f ? 1.0f : 0.0f;
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / (lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf);
    }

    payload.bsdf *= 1.0f / (payload.pdf + bsdfSamplePoint_lightSamplePdf);
}
[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    PopulateCallablePayload_DielectricMaterial(payload);

    float bsdfSamplePoint_lightSamplePdf = 0.0f;
    bool getDirectionLight = false;
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        if (!getDirectionLight) {
            Light light = sceneInfo.lights[i];
            if (isPosInDirectionLight(payload.rayOrigin, light)) {
                bsdfSamplePoint_lightSamplePdf = getPdf_directionLightSample(payload.rayOrigin, payload.outgoing, light);
                getDirectionLight = true;
            }
        }

        float3 radiance_nee = payload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];
        float cosineIN = dot(sampleDir, payload.hitNormal);
        float cosineON = dot(payload.incidence, payload.hitNormal);
        float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineON);

        float lightSamplePoint_bsdfSamplePdf = 0.0f;
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = float3(0.0f);
        if (dot(sampleDir, payload.hitNormal) > 0.0f) {
            float3 h = normalize(sampleDir + payload.incidence);
            if (abs(dot(h, payload.hitNormal)) > 0.99f) {
                bsdf = F;
                lightSamplePoint_bsdfSamplePdf = 1.0f;
            }
        } else {
            float3 h = normalize(eta * payload.incidence + sampleDir);
            if (abs(dot(h, payload.hitNormal)) > 0.99f) {
                bsdf = 1.0f - F;
                lightSamplePoint_bsdfSamplePdf = 1.0f;
            }
        }

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / (lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf);
    }

    payload.bsdf *= 1.0f / (payload.pdf + bsdfSamplePoint_lightSamplePdf);
}
[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    PopulateCallablePayload_RoughConductorMaterial(payload);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    float bsdfSamplePoint_lightSamplePdf = 0.0f;
    bool getDirectionLight = false;
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        if (!getDirectionLight) {
            Light light = sceneInfo.lights[i];
            if (isPosInDirectionLight(payload.rayOrigin, light)) {
                bsdfSamplePoint_lightSamplePdf = getPdf_directionLightSample(payload.rayOrigin, payload.outgoing, light);
                getDirectionLight = true;
            }
        }

        float3 radiance_nee = payload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];

        float3 h = normalize(sampleDir + payload.incidence);
        float3 h_tangentSpace = mul(payload.TBN, h);
        if (h_tangentSpace.z <= 0.000f) continue;

        float3 outgoing_tangentSpace = mul(payload.TBN, payload.incidence);
        float3 incidence_tangentSpace = mul(payload.TBN, sampleDir);
        if (incidence_tangentSpace.z <= 0.0f || outgoing_tangentSpace.z <= 0.0f) continue;

        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        float cosineIH = dot(incidence_tangentSpace, h_tangentSpace);
        if (cosineOH <= 0.0f || cosineIH <= 0.0f) continue;

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
        float D = D_costhetaNH / h_tangentSpace.z;

        float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);

        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                                         float2(roughness)
            );
        if (G2 <= 0.0f) continue;

        float lightSamplePoint_bsdfSamplePdf = D_costhetaNH / (4.0f * cosineOH);
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = (D * G2 * F) / (4.0f * outgoing_tangentSpace.z * incidence_tangentSpace.z);

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIH / (lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf);
    }

    payload.bsdf *= 1.0f / (payload.pdf + bsdfSamplePoint_lightSamplePdf);
}
[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    PopulateCallablePayload_RoughDielectricMaterial(payload);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    float bsdfSamplePoint_lightSamplePdf = 0.0f;
    bool getDirectionLight = false;
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    for (int i = 0; i < lightNum; ++i) {
        if (!getDirectionLight) {
            Light light = sceneInfo.lights[i];
            if (isPosInDirectionLight(payload.rayOrigin, light)) {
                bsdfSamplePoint_lightSamplePdf = getPdf_directionLightSample(payload.rayOrigin, payload.outgoing, light);
                getDirectionLight = true;
            }
        }

        float3 radiance_nee = payload.radiance_nee[i];
        if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];

        bool refraction = dot(sampleDir, payload.hitNormal) < 0.0f;
        float3 h;
        if (refraction) h = normalize(payload.incidence + sampleDir * eta);
        else h = normalize(sampleDir + payload.incidence);
        float3 h_tangentSpace = mul(payload.TBN, h);
        if (h_tangentSpace.z < 0.0f) h_tangentSpace = -h_tangentSpace;

        float3 outgoing_tangentSpace = mul(payload.TBN, payload.incidence);
        float3 incidence_tangentSpace = mul(payload.TBN, sampleDir);

        h_tangentSpace = faceforward(h_tangentSpace, outgoing_tangentSpace, -h_tangentSpace);

        float cosineOH = dot(outgoing_tangentSpace, h_tangentSpace);
        float cosineIH = dot(incidence_tangentSpace, h_tangentSpace);

        float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
        float D = D_costhetaNH / h_tangentSpace.z;

        float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);

        float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                                         float2(roughness)
            );
        if (G2 <= 0.0f) continue;

        float lightSamplePoint_bsdfSamplePdf = 0.0f;
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = float3(1.0f);
        if (refraction) {
            float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
            lightSamplePoint_bsdfSamplePdf = max(D_costhetaNH * cosineOH / (weight * weight), 0.001f);

            payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / abs(weight * weight * outgoing_tangentSpace.z * incidence_tangentSpace.z + 0.001f);
        } else {
            lightSamplePoint_bsdfSamplePdf = max(D_costhetaNH / (4.0f * cosineOH), 0.001f);
            bsdf = (D * G2 * F) / abs(4.0f * outgoing_tangentSpace.z * incidence_tangentSpace.z + 0.001f);
        }

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIH / (lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf);
    }

    payload.bsdf *= 1.0f / (payload.pdf + bsdfSamplePoint_lightSamplePdf);
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
    uint voxelTotalCount = uint(pow(pushConst.VGBStartPos_Size.w, 3));
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

    return output;
}
[shader("pixel")]
PSout_Cube fragmentMain_Cube(VSout_Cube input) {
    return { float4(input.color, 1.0f) };
}
#endif