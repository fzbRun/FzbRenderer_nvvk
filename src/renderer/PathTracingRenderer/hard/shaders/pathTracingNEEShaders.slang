#define NEE
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

[[vk::push_constant]]                           ConstantBuffer<PathTracingPushConstant, ScalarDataLayout> pushConst;

struct NEEHitPayload {
    uint randomSeed;

    float3 radiance; // emissive + directLightSample
    float pdf_bsdfSample;

    float3 bsdf_cosine;
    float pdf;

    bool isExt;

    float3 rayOrigin;
    float3 rayDirection;

    uint bounceDepth;
}

//-------------------------------------------------主Shader-------------------------------------------------
[shader("raygeneration")]
void raygenMain()
{
    float2 launchID   = (float2)DispatchRaysIndex().xy;
    float2 launchSize = (float2)DispatchRaysDimensions().xy;

    SceneInfo* sceneInfo = pushConst.sceneInfoAddress;

    uint seed = xxhash32(uint3(uint2(launchID.xy), pushConst.frameIndex));
    float r1 = rand(seed);
    float r2 = rand(seed);
    float2 subpixel_jitter = pushConst.frameIndex == 0 ? float2(0.5f, 0.5f) : float2(r1, r2);

    const uint rayFlags = 0;    //RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    const float2 pixelCenter = launchID + subpixel_jitter;
    const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

    RayDesc ray;
    ray.Origin = sceneInfo.cameraPosition; // mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction =  normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    // Initial state
    NEEHitPayload payload;
    payload.randomSeed = seed;
    payload.radiance = float3(0, 0, 0);
    payload.pdf_bsdfSample  = 1.0f;
    payload.bsdf_cosine = float3(1.0f);
    payload.pdf = 1.0f;
    payload.isExt = true;
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;
    payload.bounceDepth = 0;

    float3 accumulatedRadiance = float3(0, 0, 0);
    float RR = 0.8f;

    float weight_bsdfSample = 1.0f;
    float lastBsdfSamplePdf = 1.0f;
    float3 lastRayOrigin = payload.rayOrigin;
    float3 lastRayDirection = payload.rayDirection;

    uint lightNum = min(sceneInfo.numLights, NEE_MAX_SAMPLE_NUM);
    while (payload.bounceDepth < pushConst.maxDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
    #ifdef PathTracingMotionBlur
        TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, pushConst.time, payload);
    #else
        TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
    #endif

        if (payload.bounceDepth == MISS_DEPTH) {}
        else if (payload.bounceDepth > 1) { //except first boucne
            float pdf_directLightSample = 0.0f;
            for (int i = 0; i < lightNum; ++i) {
                Light light = sceneInfo.lights[i];
                if (!isPosInDirectionLight(payload.rayOrigin, light)) continue;
                pdf_directLightSample = getPdf_directionLightSample(lastRayOrigin, lastRayDirection, light);
                break; // bsdf sample only in one direct light, other light's pdf is 0
            }
            weight_bsdfSample *= lastBsdfSamplePdf / (lastBsdfSamplePdf + pdf_directLightSample);
        }
        accumulatedRadiance += payload.radiance * weight_bsdfSample;

        lastBsdfSamplePdf = payload.pdf_bsdfSample;
        lastRayOrigin = payload.rayOrigin;
        lastRayDirection = payload.rayDirection;

        float randomNumber = rand(payload.randomSeed);
        if (randomNumber >= RR) break;
        payload.pdf *= RR;

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;
    }

    // When the maximum frame rate is reached, the original image will decay by ((maxframe-1)/maxframe)^k < 1, and the final image will turn black
    if (pushConst.frameIndex > 0)
    {
        float a = 1.0f / float(pushConst.frameIndex + 1);
        float3 old_color = outImage[int2(launchID)].xyz;
        outImage[int2(launchID)] = float4(lerp(old_color, accumulatedRadiance, a), 1.0f);
    }
    else outImage[int2(launchID)] = float4(accumulatedRadiance, 1.0f);
}

[shader("closesthit")]
void rayClosestHitMain(inout NEEHitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    float3 barycentrics = float3(1 - attr.barycentrics.x - attr.barycentrics.y, attr.barycentrics.x, attr.barycentrics.y);
    uint   instanceID   = InstanceIndex();
    uint   meshID       = InstanceID();
    uint   triID        = PrimitiveIndex();

    SceneInfo         sceneInfo = pushConst.sceneInfoAddress[0];
    Instance          instance  = sceneInfo.instances[instanceID];
    Mesh              mesh      = sceneInfo.meshes[instance.meshIndex];
    BSDFMaterial material  = sceneInfo.materials[instance.materialIndex];

    HitState hitState = getHitStateWithPositionFetch(barycentrics);

    float cosineON = dot(hitState.normal, -payload.rayDirection);
    if (cosineON < 0) hitState.normal = -hitState.normal;
    if (material.type != MaterialType::Dielectric && material.type != MaterialType::RoughDielectric)
        material.emissive *= max(cosineON, 0.0f);

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

    if (material.materialMapIndex.x >= 0 || material.materialMapIndex.y >= 0 || material.materialMapIndex.z >= 0) {
        int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
        float2 texCoords = getTriangleAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, indices, barycentrics);
        texCoords.y = 1.0f - texCoords.y;
        getTextueData(callablePayload, texCoords);
    }

    callablePayload.radianceSum_nee = float3(0.0f);
    if (pushConst.NEEShaderIndex > 0) initNEE(callablePayload, pushConst.sceneInfoAddress[0], pushConst.NEEShaderIndex, pushConst.time);

    CallShader<CallablePayload>(uint(material.type), callablePayload);

    payload.randomSeed = callablePayload.randomSeed;

    payload.radiance = (material.emissive + callablePayload.radianceSum_nee) * payload.bsdf_cosine / payload.pdf;
    payload.pdf_bsdfSample = callablePayload.pdf;

    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = offsetRay(hitState.pos, hitState.normal);
    payload.bsdf_cosine *= callablePayload.bsdf * abs(dot(hitState.normal, payload.rayDirection));;
    payload.pdf *= callablePayload.pdf;
    payload.isExt = callablePayload.isExt;

    payload.bounceDepth++;
}

[shader("miss")]
void rayMissMain(inout NEEHitPayload payload)
{
    SceneInfo* sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1)
    {
        float3 skyColor = evalSimpleSky(sceneInfo.skySimpleParam, WorldRayDirection());
        payload.radiance += skyColor * payload.bsdf_cosine / payload.pdf;
    }
    else payload.radiance += sceneInfo.backgroundColor * payload.bsdf_cosine / payload.pdf;

    payload.bounceDepth = MISS_DEPTH; // Stop
}

//-------------------------------------------------callableShader--------------------------------------------------
[shader("callable")]
void diffuseMaterialMain(inout CallablePayload payload) {
    PopulateCallablePayload_DiffuseMaterial(payload);

    if (pushConst.NEEShaderIndex > 0) {
        uint lightNum = min(pushConst.sceneInfoAddress[0].numLights, NEE_MAX_SAMPLE_NUM);
        for (int i = 0; i < lightNum; ++i) {
            float3 radiance_nee = payload.radiance_nee[i];
            if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];

            float cosine = dot(payload.hitNormal, payload.outgoing);
            float3 bsdf = M_1_PI * payload.material.albedo * cosine;
            radiance_nee = radiance_nee * bsdf * max(dot(sampleDir, payload.hitNormal), 0.0f);

            // first, calc sampleDir's directLightSample pdf and bsdfSample pdf
            float pdf_directLightSample = payload.radiancePdf_nee[i];
            float pdf_bsdfSample = pdf_DiffuseMaterial(sampleDir, payload.hitNormal);

            radiance_nee *= pdf_directLightSample / (pdf_directLightSample + pdf_bsdfSample + 0.0001f);
            payload.radianceSum_nee += radiance_nee;
        }
    }
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);

    if (pushConst.NEEShaderIndex > 0) {
        uint lightNum = min(pushConst.sceneInfoAddress[0].numLights, NEE_MAX_SAMPLE_NUM);
        for (int i = 0; i < lightNum; ++i) {
            float3 radiance_nee = payload.radiance_nee[i];
            if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 h = normalize(sampleDir + payload.outgoing);
            float3 bsdf = float3(0.0f);
            if (abs(dot(h, payload.hitNormal)) > 0.99f) bsdf = float3(1.0f);

            radiance_nee = radiance_nee * bsdf * max(dot(sampleDir, payload.hitNormal), 0.0f);

            float pdf_directLightSample = payload.radiancePdf_nee[i];
            float pdf_bsdfSample = abs(dot(sampleDir, payload.hitNormal)) > 0.99f ? 1.0f : 0.0f;

            radiance_nee *= pdf_directLightSample / (pdf_directLightSample + pdf_bsdfSample + 0.0001f);
            payload.radianceSum_nee += radiance_nee;
        }
    }
}

[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float eta = payload.material.eta.x;
    if (!payload.isExt) eta = 1.0f / eta;

    float3 F;
    PopulateCallablePayload_DielectricMaterial(payload, F, eta);

    if (pushConst.NEEShaderIndex > 0) {
        uint lightNum = min(pushConst.sceneInfoAddress[0].numLights, NEE_MAX_SAMPLE_NUM);
        for (int i = 0; i < lightNum; ++i) {
            float3 radiance_nee = payload.radiance_nee[i];
            if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

            float pdf_bsdfSample = 0.0f;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 bsdf = float3(0.0f);
            if (dot(sampleDir, payload.hitNormal) > 0.0f) {
                float3 h = normalize(sampleDir + payload.outgoing);
                if (abs(dot(h, payload.hitNormal)) > 0.99f) {
                    bsdf = F;
                    pdf_bsdfSample = 1.0f;
                }
            } else {
                float3 h = normalize(eta * payload.outgoing + sampleDir);
                if (abs(dot(h, payload.hitNormal)) > 0.99f) {
                    bsdf = 1.0f - F;
                    pdf_bsdfSample = 1.0f;
                }
            }

            radiance_nee = radiance_nee * bsdf * abs(dot(sampleDir, payload.hitNormal));

            float pdf_directLightSample = payload.radiancePdf_nee[i];
            
            radiance_nee *= pdf_directLightSample / (pdf_directLightSample + pdf_bsdfSample + 0.0001f);
            payload.radianceSum_nee += radiance_nee;
        }
    }
}

[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    PopulateCallablePayload_RoughConductorMaterial(payload, outgoing_tangentSpace);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    if (pushConst.NEEShaderIndex > 0) {
        uint lightNum = min(pushConst.sceneInfoAddress[0].numLights, NEE_MAX_SAMPLE_NUM);
        for (int i = 0; i < lightNum; ++i) {
            float3 radiance_nee = payload.radiance_nee[i];
            if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];

            float3 h = normalize(sampleDir + payload.outgoing);
            float3 h_tangentSpace = mul(payload.TBN, h); // h需要在切线空间中
            // 因为TBN可能不准，导致出现细微的小于0的z，导致出错。上面的h不用这样是因为hvd_ggx_sample_vndf必然生成z大于0的h
            if (h_tangentSpace.z <= 0.001f) continue;
            float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
            float D = D_costhetaNH / h_tangentSpace.z;

            float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);
            float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);

            float3 incidence_tangentSpace = mul(payload.TBN, sampleDir);

            float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(roughness)
            );

            float3 bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
            radiance_nee = radiance_nee * bsdf * max(dot(sampleDir, payload.hitNormal), 0.0f);

            float pdf_directLightSample = payload.radiancePdf_nee[i];
            float pdf_bsdfSample = max(D_costhetaNH / (4.0f * cosineOH), 0.001f);

            radiance_nee *= pdf_directLightSample / (pdf_directLightSample + pdf_bsdfSample + 0.0001f);
            payload.radianceSum_nee += radiance_nee;
        }
    }
}

[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    float eta;
    PopulateCallablePayload_RoughDielectricMaterial(payload, outgoing_tangentSpace, eta);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    if (pushConst.NEEShaderIndex > 0) {
        uint lightNum = min(pushConst.sceneInfoAddress[0].numLights, NEE_MAX_SAMPLE_NUM);
        for (int i = 0; i < lightNum; ++i) {
            float3 radiance_nee = payload.radiance_nee[i];
            if (radiance_nee.x + radiance_nee.y + radiance_nee.z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];

            bool refraction = dot(sampleDir, payload.hitNormal) < 0.0f;
            float3 h;
            if (refraction) h = normalize(sampleDir + payload.outgoing * eta);
            else h = normalize(sampleDir + payload.outgoing);

            // 这里与roughConductor不同，h需要与法线同侧  h需要在切线空间中
            float3 h_tangentSpace = mul(payload.TBN, h);
            if (h_tangentSpace.z < 0.0f) h_tangentSpace = -h_tangentSpace;
            float D_costhetaNH = hvd_ggx_eval(float2(1.0f / roughness), h_tangentSpace);
            float D = D_costhetaNH / max(h_tangentSpace.z, 0.001f);

            float cosineOH = max(dot(outgoing_tangentSpace, h_tangentSpace), 0.001f);
            float3 F = schlickFresnel(payload.material.albedo, float3(1.0F), cosineOH);

            h_tangentSpace = faceforward(h_tangentSpace, outgoing_tangentSpace, -h_tangentSpace); // dot(I, nerf) < 0 return n;else return -n

            float3 incidence_tangentSpace = mul(payload.TBN, sampleDir);

            float G2 = ggx_smith_shadow_mask(
                G1_outgoing,
                G1_incidence,
                outgoing_tangentSpace,
                incidence_tangentSpace,
                float2(roughness)
            );

            float pdf_bsdfSample;

            float3 bsdf = float3(1.0f);
            if (refraction) {
                float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
                pdf_bsdfSample = max(D_costhetaNH * cosineOH / (weight * weight), 0.001f);

                payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
            } else {
                pdf_bsdfSample = max(D_costhetaNH / (4.0f * cosineOH), 0.001f);
                bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
            }
                
            radiance_nee = radiance_nee * bsdf * abs(dot(sampleDir, payload.hitNormal));

            float pdf_directLightSample = payload.radiancePdf_nee[i];

            radiance_nee *= pdf_directLightSample / (pdf_directLightSample + pdf_bsdfSample + 0.0001f);
            payload.radianceSum_nee += radiance_nee;
        }
    }
}