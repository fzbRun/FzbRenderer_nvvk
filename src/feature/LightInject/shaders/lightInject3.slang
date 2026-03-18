#define NEE

#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/LightInject/shaderio.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_LightInject::eVGB_LightInject)]] RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;

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

    const uint maxPathDepth = 10;
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

int getVGBVoxelIndex(float3 hitPos) {
    int voxelCount = int(pushConst.VGBStartPos_Size.w);
    int voxelTotalCount = voxelCount * voxelCount * voxelCount;

    int3 voxelIndexXYZ = int3((hitPos - pushConst.VGBStartPos_Size.xyz) / pushConst.VGBVoxelSize.xyz);

    int voxelIndexU = 0;
    while (voxelTotalCount > 1) {
        voxelCount = voxelCount / 2;
        voxelTotalCount = voxelTotalCount / 8;
        if (voxelIndexXYZ.z / voxelCount == 1) {
            voxelIndexU += 4 * voxelTotalCount;
            voxelIndexXYZ.z -= voxelCount;
        }
        if (voxelIndexXYZ.y / voxelCount == 1) {
            voxelIndexU += 2 * voxelTotalCount;
            voxelIndexXYZ.y -= voxelCount;
        }
        if (voxelIndexXYZ.x / voxelCount == 1) {
            voxelIndexU += voxelTotalCount;
            voxelIndexXYZ.x -= voxelCount;
        }
    }
    return voxelIndexU;
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

    payload.randomSeed = callablePayload.randomSeed;
    payload.radiance = payload.radiance * callablePayload.bsdf * hitState.cosineON + callablePayload.radianceSum_nee + material.emissive;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = hitState.pos + 0.001f * payload.rayDirection;
    payload.isExt = callablePayload.isExt;
    ++payload.bounceDepth;

    int voxelIndex = getVGBVoxelIndex(hitState.pos);
    if (all(payload.radiance > 0.0f) && !any(isnan(payload.radiance) || isinf(payload.radiance))) {
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.w, 1.0f);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.x, payload.radiance.x);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.y, payload.radiance.y);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.z, payload.radiance.z);

        float3 normal = length(payload.radiance) * hitState.normal;
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.w, 1.0f);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.x, normal.x);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.y, normal.y);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.z, normal.z);
    }

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

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / max(lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf, 1e-10f);
    }

    payload.bsdf /= max(payload.pdf + bsdfSamplePoint_lightSamplePdf, 1e-10f);
}
[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    float3 incidence = sample_hemisphere_mean(payload.randomSeed);
    payload.incidence = mul(incidence, payload.TBN);

    payload.bsdf = getBSDF_Conductor(payload.material.albedo, payload.outgoing, payload.hitNormal, payload.incidence);
    payload.pdf = getPdf_Conductor(payload.outgoing, payload.hitNormal, payload.incidence);

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
        float3 bsdf = getBSDF_Conductor(payload.material.albedo, sampleDir, payload.hitNormal, payload.incidence);

        float lightSamplePoint_bsdfSamplePdf = getPdf_Conductor(sampleDir, payload.hitNormal, payload.incidence);
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float cosineIN = dot(sampleDir, payload.hitNormal);
        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / max(lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf, 1e-10f);
    }

    payload.bsdf /= max(payload.pdf + bsdfSamplePoint_lightSamplePdf, 1e-10f);
}
[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 incidence = sample_sphere_mean(payload.randomSeed);
    payload.incidence = mul(incidence, payload.TBN);

    payload.bsdf = getBSDF_Dielectric(eta, payload.material.albedo, payload.outgoing, payload.hitNormal, payload.incidence);
    payload.pdf = getPdf_Dielectric(eta, payload.material.albedo, payload.outgoing, payload.hitNormal, payload.incidence);
    if (dot(payload.outgoing, payload.incidence) < 0.0f) payload.isExt = !payload.isExt;

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

        float lightSamplePoint_bsdfSamplePdf = getPdf_Dielectric(eta, payload.material.albedo, sampleDir, payload.hitNormal, payload.incidence);
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = getBSDF_Dielectric(eta, payload.material.albedo, sampleDir, payload.hitNormal, payload.incidence);
        float cosineIN = abs(dot(sampleDir, payload.hitNormal));
        payload.radianceSum_nee += radiance_nee * bsdf * cosineIN / max(lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf, 1e-10f);
    }

    payload.bsdf /= max(payload.pdf + bsdfSamplePoint_lightSamplePdf, 1e-10f);
}
[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    float3 incidence = sample_sphere_mean(payload.randomSeed);
    payload.incidence = mul(incidence, payload.TBN);

    payload.bsdf = getBSDF_RoughConductor(payload.material.albedo, payload.material.roughness, payload.outgoing, payload.hitNormal, payload.incidence, payload.TBN);
    payload.pdf = getPdf_RoughConductor(payload.material.roughness, payload.outgoing, payload.hitNormal, payload.incidence, payload.TBN);

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

        float lightSamplePoint_bsdfSamplePdf = getPdf_RoughConductor(payload.material.roughness, sampleDir, payload.hitNormal, payload.incidence, payload.TBN);
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = getBSDF_RoughConductor(payload.material.albedo, payload.material.roughness, sampleDir, payload.hitNormal, payload.incidence, payload.TBN);
        float cosineIH = dot(sampleDir, normalize(sampleDir + payload.incidence));
        payload.radianceSum_nee += radiance_nee * bsdf * cosineIH / max(lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf, 1e-10f);
    }

    payload.bsdf /= max(payload.pdf + bsdfSamplePoint_lightSamplePdf, 1e-10f);
}
[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 incidence = sample_sphere_mean(payload.randomSeed);
    payload.incidence = mul(incidence, payload.TBN);

    payload.bsdf = getBSDF_RoughDielectric(eta, payload.material.albedo, payload.material.roughness, payload.outgoing, payload.hitNormal, payload.incidence, payload.TBN);
    payload.pdf = getPdf_RoughDielectric(eta, payload.material.albedo, payload.material.roughness, payload.outgoing, payload.hitNormal, payload.incidence, payload.TBN);
    if (dot(payload.outgoing, payload.incidence) < 0.0f) payload.isExt = !payload.isExt;

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

        float lightSamplePoint_bsdfSamplePdf = getPdf_RoughDielectric(eta, payload.material.albedo, payload.material.roughness, sampleDir, payload.hitNormal, payload.incidence, payload.TBN);
        float lightSamplePoint_lightSamplePdf = payload.radiancePdf_nee[i];

        float3 bsdf = getBSDF_RoughDielectric(eta, payload.material.albedo, payload.material.roughness, sampleDir, payload.hitNormal, payload.incidence, payload.TBN);

        bool refraction = dot(sampleDir, payload.incidence) < 0.0f;
        float3 h;
        if (refraction) h = normalize(sampleDir * eta + payload.incidence);
        else h = normalize(sampleDir + payload.incidence);
        float cosineIH = dot(h, sampleDir);

        payload.radianceSum_nee += radiance_nee * bsdf * cosineIH / max(lightSamplePoint_bsdfSamplePdf + lightSamplePoint_lightSamplePdf, 1e-10f);
    }

    payload.bsdf /= max(payload.pdf + bsdfSamplePoint_lightSamplePdf, 1e-10f);
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
int3 getVoxelIndexXYZ(int voxelIndex) {
    int voxelCount = int(pushConst.VGBStartPos_Size.w);
    int voxelGroupTotalCount = voxelCount * voxelCount * voxelCount;
    int3 voxelIndexXYZ = int3(0, 0, 0);
    while (voxelIndex > 0) {
        voxelGroupTotalCount /= 8;
        voxelCount /= 2;
        int groupIndex = voxelIndex / voxelGroupTotalCount;
        int3 groupIndexXYZ = int3(groupIndex & 1, (groupIndex & 2) >> 1, (groupIndex & 4) >> 2);
        voxelIndexXYZ += groupIndexXYZ * voxelCount;

        voxelIndex -= groupIndexXYZ.x * voxelGroupTotalCount + 
                      groupIndexXYZ.y * voxelGroupTotalCount * 2 +
                      groupIndexXYZ.z * voxelGroupTotalCount * 4;
    }
    return voxelIndexXYZ;
}
[shader("vertex")]
VSout_Cube vertexMain_Cube(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VGBVoxelData voxelData = VGB[instanceIndex];
    VSout_Cube output;
    if (voxelData.irradiance.w == 0.0f) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        output.worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? asfloat(voxelData.aabbU.minimum.x) : asfloat(voxelData.aabbU.maximum.x);
        output.worldPos.y = (vertexIndex & 3) <= 1 ? asfloat(voxelData.aabbU.minimum.y) : asfloat(voxelData.aabbU.maximum.y);
        output.worldPos.z = vertexIndex / 4 == 0 ? asfloat(voxelData.aabbU.minimum.z) : asfloat(voxelData.aabbU.maximum.z);

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