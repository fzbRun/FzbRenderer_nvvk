#ifndef GET_VOLUMETRIC_FOG_SLANG
#define GET_VOLUMETRIC_FOG_SLANG

#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#ifndef SHADER_SOURCE
#define SHADER_SOURCE

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages_sampler)]] Sampler3D<float4> volumetricFogFluidVoxelInfoImages_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer)]] StructuredBuffer<NoiseFogInfo, ScalarDataLayout> noiseFogInfoBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eShadowMap)]] Sampler2DShadow shadowMap;

#ifdef COMPUTE_SHADER
groupshared GlobalInfo_VolumetricFog groupGlobalInfo;
groupshared SceneInfo groupSceneInfo;
groupshared VolumetricFogInfo groupVolumetricFogInfos[MAX_VOLUMETRIC_FOG_COUNT];
groupshared HeightFogInfo groupHeightFogInfos[MAX_HEIGHT_FOG_COUNT];
groupshared FluidFogInfo groupFluidFogInfos[MAX_FLUID_FOG_COUNT];
groupshared NoiseFogInfo groupNoiseFogInfos[MAX_NOISE_FOG_COUNT];

void getGroupsharedData(uint groupThreadIndex) {
    if (groupThreadIndex >= MAX_VOLUMETRIC_FOG_COUNT) return;

    if (groupThreadIndex == 0) {
        groupGlobalInfo = globalInfoBuffer[0];
        groupSceneInfo = pushConst.sceneInfoAddress[0];
    }
    if (groupThreadIndex < min(pushConst.volumetricFogCount, MAX_VOLUMETRIC_FOG_COUNT)) groupVolumetricFogInfos[groupThreadIndex] = volumetricFogInfosBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.heightFogCount, MAX_HEIGHT_FOG_COUNT)) groupHeightFogInfos[groupThreadIndex] = heightFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.fluidFogCount, MAX_FLUID_FOG_COUNT)) groupFluidFogInfos[groupThreadIndex] = fluidFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.noiseFogCount, MAX_NOISE_FOG_COUNT)) groupNoiseFogInfos[groupThreadIndex] = noiseFogInfoBuffer[groupThreadIndex];
}
#endif
#endif

float getShadow(float4 worldPos) {
    float4 lightMapNdcPos = mul(worldPos, pushConst.lightVP);
    lightMapNdcPos /= lightMapNdcPos.w;

    // uint2 shadowMapSize;
    // shadowMap.GetDimensions(shadowMapSize.x, shadowMapSize.y);
    // uint2 lightMapIndex = uint2(float2(lightMapNdcPos.x * 0.5f + 0.5f, (lightMapNdcPos.y * 0.5f + 0.5f)) * shadowMapSize);
    // float shadowDepth = shadowMap[lightMapIndex];
    // return lightMapNdcPos.z > shadowDepth;

    float2 screenPos = lightMapNdcPos.xy * 0.5f + 0.5f;
    float shadow = shadowMap.SampleCmpLevel(screenPos, lightMapNdcPos.z, 0);
    return shadow;
}

/*
float Exx1(float x) {
    const float A[5] = { -2.3347335, -0.2506219, 0.0200852, -0.0007219, 0.0000105 };
    const float B[6] = { 3.3306574, 1.9812792, 0.5523182, 0.1080712, 0.0124142, 0.0006312 };
    float num = A[0] + x * (A[1] + x * (A[2] + x * (A[3] + x * A[4])));
    float den = B[0] + x * (B[1] + x * (B[2] + x * (B[3] + x * (B[4] + x * B[5]))));
    return exp(-x) * num / den / x;
}
float ExpIntegralEi(float x) {
    if (x < 0.0f) {
        if (x < -1.0f) return -Exx1(-x);
        const float gamma = 0.5772156649f;
        float sum = gamma + log(-x) + x;
        float xn = x;
        for (int n = 2; n <= 10; n++) {
            xn *= x;
            float term = xn / (float(n) * float(n));
            sum += term;
            if (abs(term) < 1e-8f) break;
        }
        return sum;
    } else if (x > 0.0f) {
        if (x < 1.0f) {
            const float gamma = 0.5772156649f;
            float sum = gamma + log(x) + x;
            float xn = x;
            for (int n = 2; n <= 10; n++) {
                xn *= x;
                float term = xn / (float(n) * float(n));
                sum += term;
                if (abs(term) < 1e-8f) break;
            }
            return sum;
        }
        const float a[4] = { 8.5733287401f, 18.0590169730f, 8.6347608925f, 0.2677737343f };
        const float b[4] = { 9.5733223454f, 25.6329561486f, 21.0996530827f, 3.9584969228f };
        float num = x * x * x + a[0] * x * x + a[1] * x + a[2];
        float den = x * x * x + b[0] * x * x + b[1] * x + b[2];
        return (num / den) * exp(x) / x;
    }
    return -1e10f; // x == 0
}
void getHeightFogScatter(float3 startPos, float3 endPos, inout float3 L, inout float3 attenuation, inout float3 attenuation2) {
    float fogStartY = -69.5f;
    float fogHeight = 32.0f + fogStartY;
    float heightScale = 1.0f;
    float absorption = 0.01f;
    float scattering = 0.03f;
    float phase = -0.7f;
    float phase2 = phase * phase;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Light mainLight = sceneInfo.lights[0];
    float3 mainLightDir = mainLight.direction;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 viewDir = rayDir / maxDist;

    if (startPos.y > fogHeight && viewDir.y >= -1e-6f) return;
    if (endPos.y > fogHeight && viewDir.y <= 1e-6f) return;

    float3 gridMin = float3(-1e30f, fogStartY, -1e30f);
    float3 gridMax = float3(1e30f, fogHeight, 1e30f);

    float tEnter = 0.0f;
    float tExit = maxDist;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;

    float3 fogRayStart = startPos + viewDir * tEnter;
    float3 fogRayEnd = startPos + viewDir * tExit;
    float S = distance(fogRayStart, fogRayEnd);
    if (S < 1e-8) return;

    float cosTheta = dot(-viewDir, -mainLightDir);
    float phaseFunction = 0.25f * M_1_OVER_PI * (1.0f - phase2) / pow(1.0f + phase2 - 2.0f * phase * cosTheta, 1.5f);
    float extinctionCoeff0 = absorption + scattering;
    float cameraY = fogRayStart.y;

    float viewDirY = viewDir.y;
    float lightDirY = mainLightDir.y;
    float3 lightColor = mainLight.color * mainLight.intensity;

    if (lightDirY < -1e-6f) {
        if (abs(viewDirY) < 1e-30f)
            L += phaseFunction * scattering * lightColor * S * exp(-extinctionCoeff0 * exp(-heightScale * (cameraY - fogStartY)) * S);
        else {
            float A = extinctionCoeff0 * (exp(-heightScale * (cameraY - fogStartY)) / (heightScale * viewDirY) + exp(-heightScale * (fogHeight - fogStartY)) / (heightScale * lightDirY));
            float B = extinctionCoeff0 * exp(-heightScale * (cameraY - fogStartY)) / heightScale * (1.0f / viewDirY + 1.0f / lightDirY);
            float scatterIntegral = exp(-A) * (ExpIntegralEi(B) - ExpIntegralEi(B * exp(-heightScale * viewDirY * S))) / (heightScale * viewDirY);
            L += phaseFunction * scattering * lightColor * scatterIntegral;
        }
    }

    float viewY = fogRayEnd.y - fogRayStart.y;
    float E = extinctionCoeff0 * S * exp(-heightScale * (fogRayStart.y - fogStartY));
    E *= abs(viewY) < 1e-6f ? 1.0f : (1.0f - exp(-heightScale * viewY)) / (heightScale * viewY);
    attenuation *= exp(-E);

    float height = fogHeight - fogRayEnd.y;
    float mainLightDt = height / -lightDirY;
    float lightAttenuation = (absorption + scattering) * mainLightDt * exp(-heightScale * (fogHeight - fogStartY));
    attenuation2 *= abs(height) < 1e-6f ? 0.0f : exp(-lightAttenuation * (1.0f - exp(-heightScale * height)) / (heightScale * height));
}
*/
void getHeightFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float3 L, inout float3 attenuation, inout float3 attenuation2
) {
    HeightFogInfo heightFogInfo = { 1.0f };
    if (fogInfo.volumetricFogTypeIndex >= 0){
        #ifdef COMPUTE_SHADER
        heightFogInfo = groupHeightFogInfos[fogInfo.volumetricFogTypeIndex];
        #else
        heightFogInfo = heightFogInfoBuffer[fogInfo.volumetricFogTypeIndex];
        #endif
    }

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize;

    float heightScale = heightFogInfo.heightScale;
    float absorption = fogInfo.absorption.x;
    float scattering = fogInfo.scattering;
    float phase = fogInfo.phase;
    float phase2 = phase * phase;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float E = 1.0f;

    const float distancePower = 2.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float u0 = pow(float(sampleIndex) / float(sampleCount), distancePower);
        float u1 = pow(float(sampleIndex + 1) / float(sampleCount), distancePower);
        float tPart = (u1 - u0) * tTotal;
        float tPartStart = tEnter + u0 * tTotal;
        float tSample = tPartStart + tPart * (pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5f);

        float3 samplePoint = startPos + dir * tSample;

        float fogHeight = samplePoint.y - fogStartPos.y;
        float extinctionCoeff = (absorption + scattering) * exp(-heightScale * fogHeight);
        extinctionCoeff = max(extinctionCoeff, 1e-30f);

        float3 indirectL = 0.0f;
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

            float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

            float tLightEnter, tLightExit;
            isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit);
            float lightDt = tLightExit - tLightEnter;

            float height = -lightDirection.y * lightDt;
            float lightEnterY = lightPos.y + lightDirection.y * tLightEnter;
            float lightAttenuation = (absorption + scattering) * lightDt * exp(-heightScale * (lightEnterY - fogStartPos.y));
            lightAttenuation *= abs(height) < 1e-6f ? 1.0f : (1.0f - exp(-heightScale * height)) / (heightScale * height);
            lightAttenuation = exp(-lightAttenuation);

            float3 activeSunColor = lightColor * lightAttenuation * pushConst.lightAttenuationStrength;
            if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

            indirectL += activeSunColor * fp * fogInfo.color * scattering * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * tPart));
        }

        float a = exp(-extinctionCoeff * tPart);

        // Soft boundary blending based on distance to fluid AABB
        float fluidBlend = 0.0f;
        float3 fluidAABBHalfSize = (globalInfo.fluidAABB.maximum - globalInfo.fluidAABB.minimum) * 0.5f;
        AABB mergeAABB = { globalInfo.fluidAABB.minimum - 0.5f * fluidAABBHalfSize, globalInfo.fluidAABB.maximum + 0.5f * fluidAABBHalfSize };
        if (globalInfo.fluidStartUp == 1 && checkPointInAABB(samplePoint, mergeAABB)) {
            float3 d = min(abs(samplePoint - mergeAABB.minimum), abs(mergeAABB.maximum - samplePoint));
            float3 fluidGridCenter = (mergeAABB.minimum + mergeAABB.maximum) * 0.5f;
            float3 maxD = min(fluidGridCenter - mergeAABB.minimum, mergeAABB.maximum - fluidGridCenter);
            fluidBlend = min(min(d.x, d.y), d.z) / min(min(maxD.x, maxD.y), maxD.z);
        }

        L += indirectL * (1.0f - fluidBlend);
        attenuation *= lerp(a, 1.0f, fluidBlend);

        E *= a;
        if (all(attenuation <= 1e-3f)) break;
    }
    L += fogInfo.color * fogInfo.ambientIntensity;

    float tLightEnter, tLightExit;
    float3 mainLightPos = endPos - sceneInfo.lights[0].direction * 1000.0f;
    if (isHit(mainLightPos, endPos, gridMin, gridMax, tLightEnter, tLightExit)) {
        float mainLightDt = tLightExit - tLightEnter;

        float height = sceneInfo.lights[0].direction.y * mainLightDt;
        float lightEnterY = mainLightPos.y + tLightEnter * sceneInfo.lights[0].direction.y;
        float lightAttenuation = (absorption + scattering) * mainLightDt * exp(-heightScale * (lightEnterY - fogStartPos.y));
        lightAttenuation *= abs(height) < 1e-6f ? 1.0f : (1.0f - exp(-heightScale * height)) / (heightScale * height);
        lightAttenuation = exp(-lightAttenuation);

        attenuation2 *= lightAttenuation;
    }
}

void getFluidFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed,
    VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float3 L, inout float3 attenuation
) {
    #ifdef COMPUTE_SHADER
        FluidFogInfo fluidFogInfo = groupFluidFogInfos[fogInfo.volumetricFogTypeIndex];
    #else
        FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[fogInfo.volumetricFogTypeIndex];
    #endif

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 voxelSize = fogInfo.fogVoxelSize;
    int3 gridSize = fogInfo.fogVoxelGridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + float3(gridSize) * voxelSize;
    float3 gridCenter = (gridMin + gridMax) * 0.5f;

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

        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        if (step.x != 0 && tMax.x < tCurrent) tMax.x += tDelta.x;
        if (step.y != 0 && tMax.y < tCurrent) tMax.y += tDelta.y;
        if (step.z != 0 && tMax.z < tCurrent) tMax.z += tDelta.z;
    }

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 E = 1.0f;
    float gridLength = length((gridMax - gridMin));

    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, tExit);
        float dt = tNext - tCurrent;

        float3 fluidL = 0.0f;
        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
            float4 voxelInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0);
            float extinctionCoeff = voxelInfo.x;
            extinctionCoeff = max(extinctionCoeff, 1e-30f);
            float scattering = voxelInfo.y;
            float phase = voxelInfo.z;
            float phase2 = phase * phase;
            float3 color = voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);

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

                float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

                float tLightEnter, tLightExit;
                isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit);
                float lightDt = tLightExit - tLightEnter;

                float3 activeSunColor = lightColor * lerp(1.0f, fluidFogInfo.lightAttenuationEstimator, lightDt / gridLength) * pushConst.lightAttenuationStrength;
                if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

                fluidL += activeSunColor * fp * scattering * color * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * dt));
                // L += extinctionCoeff * fogInfo.color * fogInfo.ambientIntensity;
            }

            float3 d = min(samplePoint - globalInfo.fluidAABB.minimum, globalInfo.fluidAABB.maximum - samplePoint);
            float3 maxD = min(gridCenter - globalInfo.fluidAABB.minimum, globalInfo.fluidAABB.maximum - gridCenter);
            float fluidBlend = min(min(d.x, d.y), d.z) / min(min(maxD.x, maxD.y), maxD.z);

            float a = exp(-extinctionCoeff * dt);
            E *= lerp(1.0f, a, fluidBlend);
            L += lerp(0.0f, fluidL, fluidBlend);
            if (all(E <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        } else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }
    L += fogInfo.color * fogInfo.ambientIntensity;
    attenuation *= E;

    // float tLightEnter, tLightExit;
    // if (isHit(endPos - sceneInfo.lights[0].direction * 10.0f, endPos, gridMin, gridMax, tLightEnter, tLightExit)) {
    //     float mainLightDt = tLightExit - tLightEnter;
    //     attenuation2 *= lerp(1.0f, fluidFogInfo.lightAttenuationEstimator, mainLightDt / gridLength) * pushConst.lightAttenuationStrength;
    // }
}

void getNoiseFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    VolumetricFogInfo fogInfo, GlobalInfo_VolumetricFog globalInfo,
    inout float3 L, inout float3 attenuation, inout float3 attenuation2
) {
    #ifdef COMPUTE_SHADER
    NoiseFogInfo noiseFogInfo = groupNoiseFogInfos[fogInfo.volumetricFogTypeIndex];
    #else
    NoiseFogInfo noiseFogInfo = noiseFogInfoBuffer[fogInfo.volumetricFogTypeIndex];
    #endif

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 voxelSize = fogInfo.fogVoxelSize;
    int3 gridSize = fogInfo.fogVoxelGridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + float3(gridSize) * voxelSize;

    float3 rayDir = endPos - startPos;
    float maxDist = length(rayDir);
    if (maxDist < 1e-8) return;
    float3 dir = rayDir / maxDist;

    float tEnter, tExit;
    if (!isHit(startPos, endPos, gridMin, gridMax, tEnter, tExit)) return;
    float tTotal = tExit - tEnter;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 E = 1.0f;
    float gridLength = length((gridMax - gridMin));

    float phase = fogInfo.phase;
    float phase2 = phase * phase;

    const float distancePower = 2.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float u0 = pow(float(sampleIndex) / float(sampleCount), distancePower);
        float u1 = pow(float(sampleIndex + 1) / float(sampleCount), distancePower);
        float tPart = (u1 - u0) * tTotal;
        float tPartStart = tEnter + u0 * tTotal;
        float tSample = tPartStart + tPart * (pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5f);

        float3 samplePoint = startPos + dir * tSample;
        if (checkPointInAABB(samplePoint, globalInfo.fluidAABB) && globalInfo.fluidStartUp == 1) continue;

        CloudSetting cloudSetting = { noiseFogInfo.cloudScale, noiseFogInfo.cloudFlowSpeed, noiseFogInfo.cloudCoverage, noiseFogInfo.cloudTypePreference, noiseFogInfo.weatherScale };
        float density = SampleCloudDensity(samplePoint, gridMin, gridMax, pushConst.time, cloudSetting);
        // if (pushConst.frameIndex == 0 && rand(randomSeed) < 0.1f) printf("%f\n", density);
        float extinctionCoeff = (fogInfo.absorption.x + fogInfo.scattering) * density;
        extinctionCoeff = max(extinctionCoeff, 1e-30f);

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

            float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

            float tLightEnter, tLightExit;
            isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit);
            float lightDt = tLightExit - tLightEnter;

            float3 activeSunColor = lightColor * pushConst .lightAttenuationStrength; // * lerp(1.0f, fogInfo.lightAttenuationEstimator, lightDt / gridLength);
            if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

            L += activeSunColor * fp * fogInfo.scattering * fogInfo.color * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * tPart));
        }

        E *= exp(-extinctionCoeff * tPart);
        if (all(E <= 1e-3f)) break;
    }

    L += fogInfo.color * fogInfo.ambientIntensity;
    attenuation *= E;

    float tLightEnter, tLightExit;
    if (isHit(endPos - sceneInfo.lights[0].direction * 10.0f, endPos, gridMin, gridMax, tLightEnter, tLightExit)) {
        float mainLightDt = tLightExit - tLightEnter;
        attenuation2 *= pushConst .lightAttenuationStrength; // lerp(1.0f, fogInfo.lightAttenuationEstimator, mainLightDt / gridLength);
    }
}

bool getCameraFrustumFogData(float3 worldPos, inout float3 hitWorldPos, inout float3 uvw) {
    SceneInfo sceneInfo;
    #ifdef COMPUTE_SHADER
    sceneInfo = groupSceneInfo;
    #else
    sceneInfo = pushConst.sceneInfoAddress[0];
    #endif
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint3 gridSize = pushConst.attenuationGridSize;
    float nearPlane = pushConst.attenuationNearPlane;
    float farPlane  = pushConst.attenuationFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    float3 toPoint = worldPos - cameraPos;
    float viewZ = dot(toPoint, cameraForward);
    float viewX = dot(toPoint, cameraRight);
    float viewY = dot(toPoint, cameraUp);

    float entryDepth = clamp(viewZ, nearPlane, farPlane);

    float halfHeight = entryDepth * tanHalfFov;
    float halfWidth  = halfHeight * aspect;

    float scale = entryDepth / max(viewZ, 1e-8f);
    float entryViewX = viewX * scale;
    float entryViewY = viewY * scale;

    uvw.x = (entryViewX / halfWidth  + 1.0f) * 0.5f;
    uvw.y = (entryViewY / halfHeight + 1.0f) * 0.5f;
    float tLinear = (entryDepth - nearPlane) / (farPlane - nearPlane);
    const float zPower = 3.0f;
    uvw.z = pow(tLinear, 1.0f / zPower);
    uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);

    hitWorldPos = cameraPos + cameraRight * entryViewX + cameraUp * entryViewY + cameraForward * entryDepth;

    return true;
}

void getVolumetricFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount,
    inout float3 L, inout float3 attenuation, inout float3 attenuation2
) {
    #ifdef COMPUTE_SHADER
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    #else
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    #endif

    VolumetricFogInfo fogInfo;
    fogInfo.fogStartPos = { 5084.0f, -69.5f, -4486.0f };
    fogInfo.fogVoxelGridSize = { 16, 16, 16 };
    fogInfo.fogVoxelSize = { 7.0f, 2.0f, 5.0f };
    fogInfo.absorption = 0.01f;
    fogInfo.scattering = 0.03f;
    fogInfo.phase = -0.7f;
    fogInfo.color = 1.0f;
    fogInfo.ambientIntensity = 0.0f;
    fogInfo.volumetricFogTypeIndex = -1;

    getHeightFogScatter(startPos, endPos, randomSeed, sampleCount, fogInfo, globalInfo, L, attenuation, attenuation2);
    for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
        uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];

        #ifdef COMPUTE_SHADER
        fogInfo = groupVolumetricFogInfos[fogIndex];
        #else
        fogInfo = volumetricFogInfosBuffer[fogIndex];
        #endif

        uint typeFogIndex = fogInfo.volumetricFogTypeIndex;
        if (fogInfo.type == VolumetricFogType::Height) {
            getHeightFogScatter(startPos, endPos, randomSeed, sampleCount, fogInfo, globalInfo, L, attenuation, attenuation2);
        } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp == 1) {
            getFluidFogScatter(startPos, endPos, randomSeed, fogInfo, globalInfo, L, attenuation);
        }
        else if (fogInfo.type == VolumetricFogType::Noise) {
            NoiseFogInfo noiseFogInfo = noiseFogInfoBuffer[typeFogIndex];
            getNoiseFogScatter(startPos, endPos, randomSeed, sampleCount, fogInfo, globalInfo, L, attenuation, attenuation2);
        }
    }
}

#endif
