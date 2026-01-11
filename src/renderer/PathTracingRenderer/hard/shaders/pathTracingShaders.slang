/*
slang大括号的左右(包括开始大括号的下一行开始，以及结束大括号的上行结尾)不能有注释，或者说中文注释
*/
#include "feature/PathTracing/shaders/pathTracingCommon.slang"

// clang-format off
//ScalarDataLayout
[[vk::push_constant]]                           ConstantBuffer<PathTracingPushConstant, ScalarDataLayout> pushConst;

// clang-format on
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
    HitPayload payload;
    payload.randomSeed = seed;
    payload.radiance  = float3(0, 0, 0);
    payload.bsdf_cosine = float3(1.0f);
    payload.pdf = 1.0f;
    payload.bounceDepth = 0;
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;
    payload.isExt = true;

    float3 accumulatedRadiance = float3(0, 0, 0);
    float RR = 0.8f;

    // Iterative reflection loop
    while (payload.bounceDepth < pushConst.maxDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
        //TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
        TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, pushConst.time, payload);
        accumulatedRadiance += payload.radiance;

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
void rayClosestHitMain(inout HitPayload payload, in BuiltInTriangleIntersectionAttributes attr)
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

    //int3 indices = getTriangleIndices(mesh.dataBuffer, mesh.triMesh, triID);
    // float3 pos = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, indices, barycentrics);
    //float3 normal = getTriangleAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, indices, barycentrics);
    // hitState.pos = float3(mul(float4(pos, 1.0), ObjectToWorld4x3()));
    // hitState.normal = normalize(mul(WorldToObject4x3(), normal).xyz);
    // float3 tangent, bitangent;
    // orthonormalBasis(hitState.normal, tangent, bitangent);
    // hitState.TBN = float3x3(tangent, bitangent, hitState.normal);

    float cosineON = dot(hitState.normal, -payload.rayDirection);
    if (cosineON < 0) hitState.normal = -hitState.normal;
    cosineON = abs(cosineON);
    float3 emissive = material.emissive * cosineON;

    //float cosineON = dot(hitState.normal, -payload.rayDirection);
    //float3 emissive = material.emissive;
    //if (material.type == uint(MaterialType::Deielectric) || material.type == uint(MaterialType::RoughDeielectric)) {
    //    if (cosineON < 0) hitState.normal = -hitState.normal;
    //    cosineON = abs(cosineON);
    //} else emissive *= max(cosineON, 0.0f);

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
    payload.radiance = (callablePayload.radianceSum_nee + emissive) * payload.bsdf_cosine / payload.pdf;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = offsetRay(hitState.pos, hitState.normal); // hitState.pos + 0.001f * payload.rayDirection;
    payload.bsdf_cosine *= callablePayload.bsdf * cosineON;
    payload.pdf *= callablePayload.pdf;
    payload.isExt = callablePayload.isExt;

    payload.bounceDepth++;
}

[shader("miss")]
void rayMissMain(inout HitPayload payload)
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
    payload.incidence = sampleIncidence_DiffuseMaterial(payload.randomSeed, payload.TBN);

    float cosine = dot(payload.incidence, payload.hitNormal);
    payload.pdf = M_1_PI * cosine;
    payload.bsdf = payload.material.albedo * payload.pdf;

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

            cosine = dot(payload.hitNormal, payload.outgoing);
            float3 bsdf = M_1_PI * payload.material.albedo * cosine;
            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * max(dot(payload.sampleDir_nee[i], payload.hitNormal), 0.0f);
        }
    }
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 h = normalize(sampleDir + payload.outgoing);
            float3 bsdf = float3(1.0f);
            if (abs(dot(h, payload.hitNormal)) < 0.95f) bsdf = float3(0.0f);

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * max(dot(sampleDir, payload.hitNormal), 0.0f);
        }
    }
}

[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float3 F;
    PopulateCallablePayload_DielectricMaterial(payload, F);

    if (pushConst.NEEShaderIndex > 0) {
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

            float3 sampleDir = payload.sampleDir_nee[i];
            float3 bsdf = float3(1.0f);
            if (dot(sampleDir, payload.hitNormal) > 0.0f) {
                float3 h = normalize(sampleDir + payload.outgoing);
                if (abs(dot(h, payload.hitNormal)) < 0.99f) bsdf = F;
            } else {
                float eta = payload.material.eta.x;
                if (!payload.isExt) eta = 1.0f / eta; // 所以NEE要在找到incidence之前

                float3 h = normalize(eta * payload.outgoing + sampleDir);
                if (abs(dot(h, payload.hitNormal)) < 0.99f) bsdf = 1.0f - F;
            }

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
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
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

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
            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * max(dot(sampleDir, payload.hitNormal), 0.0f);
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
        for (int i = 0; i < payload.lightNum; ++i) {
            if(payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

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

            float3 bsdf = float3(1.0f);
            if (refraction) {
                float weight = dot(incidence_tangentSpace, h_tangentSpace) / eta + cosineOH;
                payload.bsdf = D * G2 * (1.0f - F) * abs(dot(incidence_tangentSpace, h_tangentSpace)) * cosineOH
                        / (weight * weight * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);
            } else
                bsdf = (D * G2 * F) / (4.0f * abs(outgoing_tangentSpace.z) * abs(incidence_tangentSpace.z) + 0.001f);

            payload.radianceSum_nee += payload.radiance_nee[i] * bsdf * abs(dot(sampleDir, payload.hitNormal));
        }
    }
}