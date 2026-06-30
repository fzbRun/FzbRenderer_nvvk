#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eAlbedoImage)]] Texture2D<float4> albedoImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eEmissiveImage)]] Texture2D<float4> emissiveImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eLightAttenuationEstimatorBuffer)]] StructuredBuffer<float3> lightAttenuationEstimatorBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler)]] Sampler3D<float4> volumetricFogImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eShadowMap)]] Sampler2DShadow shadowMap;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

groupshared float3 groupLightAttenuationEstimator;

float getShadow(float4 worldPos) {
    float4 lightMapNdcPos = mul(worldPos, pushConst.lightVP);
    lightMapNdcPos /= lightMapNdcPos.w;

    //uint2 shadowMapSize;
    //shadowMap.GetDimensions(shadowMapSize.x, shadowMapSize.y);
    //uint2 lightMapIndex = uint2(float2(lightMapNdcPos.x * 0.5f + 0.5f, (lightMapNdcPos.y * 0.5f + 0.5f)) * shadowMapSize);
    //float shadowDepth = shadowMap[lightMapIndex];
    //return lightMapNdcPos.z > shadowDepth;

    float2 screenPos = lightMapNdcPos.xy * 0.5f + 0.5f;
    float shadow = shadowMap.SampleCmpLevel(screenPos, lightMapNdcPos.z, 0);
    return shadow;
}

void getVolumetricFogScatter(float3 startPos, float3 endPos, uint randomSeed, out float3 L, out float3 attenuation, out float3 attenuation2) {
    L = 0.0f;
    attenuation = 1.0f;
    attenuation2 = 1.0f;

    float3 fogStartPos = pushConst.fogStartPos;
    float3 voxelSize = pushConst.fogVoxelSize;
    int3 gridSize = pushConst.fogVoxelGridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + float3(gridSize) * voxelSize;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    float eps = 1e-5 * min(min(voxelSize.x, voxelSize.y), voxelSize.z);
    float tCurrent = tEnter + eps;
    float3 pos = startPos + dir * tCurrent;

    float3 invDir; int3 voxel; int3 step; float3 tDelta; float3 voxelBorder; float3 tMax;
    {
        invDir.x = (abs(dir.x) > 1e-8) ? rcp(dir.x) : 1e30;
        invDir.y = (abs(dir.y) > 1e-8) ? rcp(dir.y) : 1e30;
        invDir.z = (abs(dir.z) > 1e-8) ? rcp(dir.z) : 1e30;

        float3 voxelF = (pos - fogStartPos) / voxelSize;
        voxel = int3(floor(voxelF));
        voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

        step.x = (dir.x > 0.0) ? 1 : ((dir.x < 0.0) ? -1 : 0);
        step.y = (dir.y > 0.0) ? 1 : ((dir.y < 0.0) ? -1 : 0);
        step.z = (dir.z > 0.0) ? 1 : ((dir.z < 0.0) ? -1 : 0);

        tDelta.x = (step.x != 0) ? voxelSize.x * abs(invDir.x) : 1e30;
        tDelta.y = (step.y != 0) ? voxelSize.y * abs(invDir.y) : 1e30;
        tDelta.z = (step.z != 0) ? voxelSize.z * abs(invDir.z) : 1e30;

        voxelBorder.x = (step.x > 0) ? (fogStartPos.x + (voxel.x + 1) * voxelSize.x) : (fogStartPos.x + voxel.x * voxelSize.x);
        voxelBorder.y = (step.y > 0) ? (fogStartPos.y + (voxel.y + 1) * voxelSize.y) : (fogStartPos.y + voxel.y * voxelSize.y);
        voxelBorder.z = (step.z > 0) ? (fogStartPos.z + (voxel.z + 1) * voxelSize.z) : (fogStartPos.z + voxel.z * voxelSize.z);

        // 修正：基于 worldPos 计算全局 tMax
        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        // 极微小修正：若因 eps 推进导致 tMax 略小于 tCurrent，补一整个体素
        if (step.x != 0 && tMax.x < tCurrent) tMax.x += tDelta.x;
        if (step.y != 0 && tMax.y < tCurrent) tMax.y += tDelta.y;
        if (step.z != 0 && tMax.z < tCurrent) tMax.z += tDelta.z;
    }
    
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 E = 1.0f;
    float gridLength = length((gridMax - gridMin));
    float3 lightAttenuationEstimator = groupLightAttenuationEstimator;

    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, tExit);
        float dt = tNext - tCurrent;

        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - pushConst.fogStartPos) / (pushConst.fogVoxelGridSize * pushConst.fogVoxelSize);
            float extinctionCoeff = volumetricFogImage.SampleLevel(uvw, 0).x;

            for (int lightIndex = 0; lightIndex < sceneInfo.numLights; ++lightIndex) {
                Light light = sceneInfo.lights[lightIndex];
                float3 lightColor = light.color * light.intensity;
                float3 lightDirection; float3 lightPos;
                if (light.type == LightType::Direction) {
                    lightDirection = -light.direction;
                    lightPos = samplePoint + 1000.0f * lightDirection;
                } else if (light.type == LightType::Point) {
                    lightDirection = normalize(light.pos - samplePoint);
                    lightPos = light.pos;
                }

                float fp = (1.0f - pushConst.phase * pushConst.phase) /
                           (4.0f * M_PI *
                            pow((1 + pushConst.phase * pushConst.phase -
                            2.0f * pushConst.phase * dot(-dir, lightDirection)), 1.5f));

                float tLightEnter, tLightExit;
                isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit);
                float lightDt = tLightExit - tLightEnter;
                if (lightIndex == 1) lightDt = max(lightDt - 0.1f, 0.0f);

                float3 activeSunColor = lightColor * lerp(1.0f, lightAttenuationEstimator, lightDt / gridLength) * pushConst.lightAttenuationStrength;
                if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f)) ;

                L += activeSunColor * fp * pushConst.scattering * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * dt));
            }

            E *= exp(-extinctionCoeff * dt);
            if (all(E <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
        else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }

    attenuation = E;

    float tLightEnter, tLightExit;
    isHit(endPos - sceneInfo.lights[0].direction * 10.0f, endPos, gridMin, gridMax, tLightEnter, tLightExit);
    float mainLightDt = tLightExit - tLightEnter;
    attenuation2 = lerp(1.0f, lightAttenuationEstimator, mainLightDt / gridLength) * pushConst.lightAttenuationStrength;
}

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_deferredRendering(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupLightAttenuationEstimator = lightAttenuationEstimatorBuffer[0];
    GroupMemoryBarrierWithGroupSync();

    uint2 screenSize;
    albedoImage.GetDimensions(screenSize.x, screenSize.y);
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    float depth = depthImage[threadIndex];
    float2 uv = (float2(threadIndex) + 0.5f) / float2(screenSize);
    float2 ndc = uv * 2.0f - 1.0f;
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos = mul(mul(clipPos, sceneInfo.projInvMatrix), sceneInfo.viewInvMatrix);
    worldPos /= worldPos.w;

    uint randomSeed = xxhash32(uint3(pushConst.frameIndex, threadIndex));

    float3 attenuation, attenuation2;
    float3 L = 0.0f;
    getVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, L, attenuation, attenuation2);

    if (depth >= 1.0f - 1e-30f) {
        renderedImage[threadIndex] = sceneInfo.useSky == 1 ? float4(renderedImage[threadIndex].xyz * attenuation + L, 1.0f) : float4(sceneInfo.backgroundColor * attenuation + L, 1.0f);
        return;
    }
    float shadow = getShadow(worldPos);

    float3 albedo = albedoImage[threadIndex].xyz;
    float3 normal = normalImage[threadIndex].xyz;

    Light light = sceneInfo.lights[0];

    float3 viewDir = normalize(sceneInfo.cameraPosition - worldPos.xyz);
    float3 lightDir = normalize(-light.direction);

    float3 lightColor = light.color * light.intensity;

    float3 ambient = 0.1f * albedo;
    float3 diffuse = max(dot(lightDir, normal), 0.0f) * lightColor * albedo;
    float3 h = normalize(viewDir + lightDir);
    float3 specular = pow(max(dot(h, normal), 0.0f), 32) * lightColor * max(dot(normal, lightDir), 0.0f);

    float3 emissive = emissiveImage[threadIndex].xyz;

    renderedImage[threadIndex] = float4((ambient + (diffuse + specular) * shadow * attenuation2 + emissive) * attenuation + L, 1.0f);
}