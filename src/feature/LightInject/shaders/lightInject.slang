#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "feature/LightInject/shaderio.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant, ScalarDataLayout> pushConst;
[[vk::binding(StaticBindingPoints_LightInject::eVGB_LightInject)]] RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;

//------------------------------------------------lightInject------------------------------------------------
struct LightInjectHitPayload {
    uint randomSeed;

    float3 irradiance;
    float3 radiance;
    uint voxelIndex;
    float3 bsdf_cosine;
    float cosine;
    float3 normal;
    float pdf;

    float3 rayOrigin;
    float3 rayDirection;

    uint bounceDepth;

    bool isExt;
}

[shader("raygeneration")]
void raygenMain()
{
    float2 launchID = (float2)DispatchRaysIndex().xy;
    float2 launchSize = (float2)DispatchRaysDimensions().xy;

    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;

    uint seed = xxhash32(uint3(uint2(launchID.xy), pushConst.frameIndex));

    const uint rayFlags = 0; // RAY_FLAG_CULL_BACK_FACING_TRIANGLES;
    const float2 pixelCenter = launchID + float2(0.5f, 0.5f);
    const float2 clipCoords = launchID / launchSize * 2.0 - 1.0;
    const float4 viewCoords = mul(float4(clipCoords, 1.0, 1.0), sceneInfo.projInvMatrix);

    RayDesc ray;
    ray.Origin = sceneInfo.cameraPosition; // mul(float4(0.0, 0.0, 0.0, 1.0), sceneInfo.viewInvMatrix).xyz;
    ray.Direction = normalize(mul(float4(normalize(viewCoords.xyz), 0.0), sceneInfo.viewInvMatrix).xyz);
    ray.TMin = 0.001;
    ray.TMax = INFINITE;

    // Initial state
    LightInjectHitPayload payload;
    payload.randomSeed = seed;
    payload.irradiance = float3(0, 0, 0);
    payload.voxelIndex = 0;
    payload.radiance = float3(0, 0, 0);
    payload.bsdf_cosine = float3(1.0f);
    payload.cosine = 1.0f;
    payload.normal = float3(0.0f);
    payload.pdf = 1.0f;
    payload.bounceDepth = 0;
    payload.rayOrigin = ray.Origin;
    payload.rayDirection = ray.Direction;
    payload.isExt = true;

    const uint maxPathDepth = 2;
    float3 voxelIrradiances[maxPathDepth];
    int voxelIndices[maxPathDepth];
    float3 radiances[maxPathDepth];
    float3 normals[maxPathDepth];
    float3 bsdf_cosines[maxPathDepth];
    float cosines[maxPathDepth];
    float pdfs[maxPathDepth];

    float RR = 0.8f;

    uint32_t pathCount = 0;
    while (payload.bounceDepth < maxPathDepth && (payload.bsdf_cosine.x + payload.bsdf_cosine.y + payload.bsdf_cosine.z > 0.0f))
    {
        //TraceRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, payload);
        TraceMotionRay(topLevelAS, rayFlags, 0xff, 0, 0, 0, ray, pushConst.time, payload);
 
        voxelIrradiances[pathCount] = payload.irradiance;
        voxelIndices[pathCount] = payload.bounceDepth == MISS_DEPTH ? -1 : payload.voxelIndex;
        radiances[pathCount] = payload.radiance;
        normals[pathCount] = payload.normal;
        bsdf_cosines[pathCount] = payload.bsdf_cosine;
        cosines[pathCount] = payload.cosine;
        pdfs[pathCount] = payload.pdf;
        ++pathCount;

        float randomNumber = rand(payload.randomSeed);
        if (randomNumber >= RR) break;
        payload.pdf *= RR;

        ray.Origin = payload.rayOrigin;
        ray.Direction = payload.rayDirection;
    }

    float3 radiance = float3(0.0f);
    for (int i = pathCount - 1; i >= 0; --i) {
        uint voxelIndex = voxelIndices[i];
        if (voxelIndex == -1) {
            radiance = radiances[i];
            continue;
        }

        // Dividing by pdf will result in a lot of noise; however, not dividing by pdf will lead to indirect light
        // not being an estimate of the entire hemisphere, resulting in smaller indirect light values
        float3 irradiance = voxelIrradiances[i] + radiance * cosines[i];    // / pdfs[i];
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.w, 1.0f);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.x, irradiance.x);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.y, irradiance.y);
        InterlockedAdd<float>(VGB[voxelIndex].irradiance.z, irradiance.z);

        radiance = radiance * bsdf_cosines[i] / pdfs[i] + radiances[i];

        float3 normal = length(irradiance) * normals[i];
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.w, 1.0f);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.x, normal.x);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.y, normal.y);
        InterlockedAdd<float>(VGB[voxelIndex].sumNormal_E.z, normal.z);
    }

    outImage[int2(launchID)] = float4(radiance, 1.0f);
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

    HitState hitState = getHitStateWithPositionFetch(barycentrics);

    float cosineON = dot(hitState.normal, -payload.rayDirection);
    if (cosineON < 0) hitState.normal = -hitState.normal;
    cosineON = abs(cosineON);
    float3 emissive = material.emissive * cosineON;
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
    initNEE(callablePayload, pushConst.sceneInfoAddress[0], 1, pushConst.time);

    CallShader<CallablePayload>(uint(material.type), callablePayload);

    payload.randomSeed = callablePayload.randomSeed;
    payload.irradiance = callablePayload.hitPos + emissive;    //sotre irradiance in hitPos
    payload.voxelIndex = getVGBVoxelIndex(hitState.pos);
    payload.radiance = callablePayload.radianceSum_nee + emissive;
    payload.rayDirection = callablePayload.incidence;
    payload.rayOrigin = offsetRay(hitState.pos, hitState.normal); // hitState.pos + 0.001f * payload.rayDirection;
    payload.bsdf_cosine = callablePayload.bsdf * cosineON;
    payload.cosine = cosineON;
    payload.normal = hitState.normal;
    payload.pdf = callablePayload.pdf;
    payload.isExt = callablePayload.isExt;

    payload.bounceDepth++;
}

[shader("miss")]
void rayMissMain(inout LightInjectHitPayload payload)
{
    SceneInfo *sceneInfo = pushConst.sceneInfoAddress;
    if (sceneInfo.useSky == 1)
    {
        float3 skyColor = evalSimpleSky(sceneInfo.skySimpleParam, WorldRayDirection());
        payload.radiance = skyColor;
    }
    else payload.radiance = sceneInfo.backgroundColor;

    payload.bounceDepth = MISS_DEPTH; // Stop
}

[shader("callable")]
void diffuseMaterialMain(inout CallablePayload payload) {
    payload.incidence = sampleIncidence_DiffuseMaterial(payload.randomSeed, payload.TBN);

    float cosine = dot(payload.incidence, payload.hitNormal);
    payload.pdf = M_1_PI * cosine;
    payload.bsdf = payload.material.albedo * payload.pdf;

    float3 irradiance = float3(0.0f);
    for (int i = 0; i < payload.lightNum; ++i) {
        if (payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

        cosine = dot(payload.hitNormal, payload.outgoing);
        float3 bsdf = M_1_PI * payload.material.albedo * cosine;
        
        cosine = dot(payload.sampleDir_nee[i], payload.hitNormal); // coine must > 0
        float3 radiance = payload.radiance_nee[i] * cosine;
        payload.radianceSum_nee += radiance * bsdf;

        irradiance += radiance; // payload.pdf;
    }
    payload.hitPos = irradiance;    // / lightCount; //1/n acc((L * cosTheta) / p)
}

[shader("callable")]
void conductorMaterialMain(inout CallablePayload payload) {
    payload.bsdf = 1.0f;
    payload.pdf = 1.0f;
    payload.incidence = reflect(-payload.outgoing, payload.hitNormal);

    float3 irradiance = float3(0.0f);
    for (int i = 0; i < payload.lightNum; ++i) {
        if (payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

        float3 sampleDir = payload.sampleDir_nee[i];
        float3 h = normalize(sampleDir + payload.outgoing);
        float3 bsdf = float3(1.0f);
        if (abs(dot(h, payload.hitNormal)) < 0.95f) bsdf = float3(0.0f);

        float3 cosine = dot(sampleDir, payload.hitNormal);
        float3 radiance = payload.radiance_nee[i] * cosine;
        payload.radianceSum_nee += radiance * bsdf;

        irradiance += radiance;
    }
    payload.hitPos = irradiance;
}

[shader("callable")]
void dielectricMaterialMain(inout CallablePayload payload) {
    float3 F;
    PopulateCallablePayload_DielectricMaterial(payload, F);

    float3 irradiance = float3(0.0f);
    for (int i = 0; i < payload.lightNum; ++i) {
        if (payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

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

        float3 cosine = abs(dot(sampleDir, payload.hitNormal));
        float3 radiance = payload.radiance_nee[i] * cosine;
        payload.radianceSum_nee += radiance * bsdf;

        irradiance += radiance;
    }
    payload.hitPos = irradiance;
}

[shader("callable")]
void roughConductorMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    PopulateCallablePayload_RoughConductorMaterial(payload, outgoing_tangentSpace);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    float3 irradiance = float3(0.0f);
    for (int i = 0; i < payload.lightNum; ++i) {
        if (payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

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
        float3 cosine = dot(sampleDir, payload.hitNormal);
        float3 radiance = payload.radiance_nee[i] * cosine;
        payload.radianceSum_nee += radiance * bsdf;
        irradiance += radiance;
    }
    payload.hitPos = irradiance;
}

[shader("callable")]
void roughDielectricMaterialMain(inout CallablePayload payload) {
    float3 outgoing_tangentSpace;
    float eta;
    PopulateCallablePayload_RoughDielectricMaterial(payload, outgoing_tangentSpace, eta);

    float roughness = payload.material.roughness;
    float G1_outgoing, G1_incidence;

    float3 irradiance = float3(0.0f);
    for (int i = 0; i < payload.lightNum; ++i) {
        if (payload.radiance_nee[i].x + payload.radiance_nee[i].y + payload.radiance_nee[i].z < 0.001f) continue;

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

        float3 cosine = abs(dot(sampleDir, payload.hitNormal));
        float3 radiance = payload.radiance_nee[i] * cosine;
        payload.radianceSum_nee += radiance * bsdf;

        irradiance += radiance;
    }
    payload.hitPos = irradiance;
}

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