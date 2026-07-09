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

[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVisibleVolumetricFogIndexBuffer)]] RWStructuredBuffer<uint> visibleVolumetricFogIndexBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogExtinctionImages_sampler)]] Sampler3D<float4> volumetricFogExtinctionImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eShadowMap)]] Sampler2DShadow shadowMap;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

groupshared GlobalInfo_VolumetricFog groupGlobalInfo;

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
void getHeightFogScatter(float3 startPos, float3 endPos, inout uint randomSeed, VolumetricFogInfo fogInfo, inout float3 L, inout float3 attenuation, inout float3 attenuation2) {
    float3 fogStartPos = fogInfo.fogStartPos;
    float3 gridMin = fogStartPos;
    float3 gridMax = fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize;

    float heightScale = fogInfo.viscosity;
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
    float3 E = 1.0f;

    uint sampleCount = 10;
    float tPart = tTotal / sampleCount;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float tPartStart = sampleIndex * tPart + tEnter;
        float tSample = tPartStart + tPart * rand(randomSeed);

        float3 samplePoint = startPos + dir * tSample;

        float fogHeight = samplePoint.y - fogStartPos.y;
        float extinctionCoeff = (absorption + scattering) * exp(-heightScale * fogHeight);
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

            float height = -lightDirection.y * lightDt;
            float lightEnterY = lightPos.y + lightDirection.y * tLightEnter;
            float lightAttenuation = (absorption + scattering) * lightDt * exp(-heightScale * (lightEnterY - fogStartPos.y));
            lightAttenuation *= abs(height) < 1e-6f ? 1.0f : (1.0f - exp(-heightScale * height)) / (heightScale * height);
            lightAttenuation = exp(-lightAttenuation);

            float3 activeSunColor = lightColor * lightAttenuation * pushConst.lightAttenuationStrength;
            if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

            L += activeSunColor * fp * scattering * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * tPart));
        }

        E *= exp(-extinctionCoeff * tPart);
        if (all(E <= 1e-3f)) break;
    }

    attenuation *= E;
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

void getFluidFogScatter(float3 startPos, float3 endPos, inout uint randomSeed, uint32_t fogIndex, VolumetricFogInfo fogInfo, inout float3 L, inout float3 attenuation, inout float3 attenuation2) {
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

        // 淇锛氬熀浜?worldPos 璁＄畻鍏ㄥ眬 tMax
        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        // 鏋佸井灏忎慨姝ｏ細鑻ュ洜 eps 鎺ㄨ繘瀵艰嚧 tMax 鐣ュ皬浜?tCurrent锛岃ˉ涓€鏁翠釜浣撶礌
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

        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
            float extinctionCoeff = volumetricFogExtinctionImages_sample[fogIndex].SampleLevel(uvw, 0).x;
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

                float fp = (1.0f - fogInfo.phase * fogInfo.phase) /
                    (4.0f * M_PI *
                     pow((1 + fogInfo.phase * fogInfo.phase -
                     2.0f * fogInfo.phase * dot(-dir, lightDirection)), 1.5f));

                float tLightEnter, tLightExit;
                isHit(lightPos, samplePoint, gridMin, gridMax, tLightEnter, tLightExit);
                float lightDt = tLightExit - tLightEnter;

                float3 activeSunColor = lightColor * lerp(1.0f, fogInfo.lightAttenuationEstimator, lightDt / gridLength) * pushConst.lightAttenuationStrength;
                if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

                L += activeSunColor * fp * fogInfo.scattering * fogInfo.color * E / extinctionCoeff * (1.0f - exp(-extinctionCoeff * dt));
            }

            E *= exp(-extinctionCoeff * dt);
            if (all(E <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }
    L += fogInfo.color * fogInfo.ambientIntensity;
    attenuation *= E;

    float tLightEnter, tLightExit;
    if (isHit(endPos - sceneInfo.lights[0].direction * 10.0f, endPos, gridMin, gridMax, tLightEnter, tLightExit)) {
        float mainLightDt = tLightExit - tLightEnter;
        attenuation2 *= lerp(1.0f, fogInfo.lightAttenuationEstimator, mainLightDt / gridLength) * pushConst.lightAttenuationStrength;
    }
}

void getNoiseFogScatter(float3 startPos, float3 endPos, inout uint randomSeed, VolumetricFogInfo fogInfo, inout float3 L, inout float3 attenuation, inout float3 attenuation2) {
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

    uint sampleCount = 12;
    float tPart = tTotal / sampleCount;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float tPartStart = sampleIndex * tPart + tEnter;
        float tSample = tPartStart + tPart * rand(randomSeed);

        float3 samplePoint = startPos + dir * tSample;

        float density = SampleCloudDensity(samplePoint, gridMin, gridMax, pushConst.time, 0.05f, 0.5);
        //if (pushConst.frameIndex == 0 && rand(randomSeed) < 0.1f) printf("%f\n", density);
        float extinctionCoeff = (fogInfo.absorption.y + fogInfo.scattering) * density;
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

            float3 activeSunColor = lightColor * lerp(1.0f, fogInfo.lightAttenuationEstimator, lightDt / gridLength) * pushConst.lightAttenuationStrength;
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
        attenuation2 *= lerp(1.0f, fogInfo.lightAttenuationEstimator, mainLightDt / gridLength) * pushConst.lightAttenuationStrength;
    }
}

void getVolumetricFogScatter(float3 startPos, float3 endPos, inout uint randomSeed, inout float3 L, inout float3 attenuation, inout float3 attenuation2) {
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    VolumetricFogInfo fogInfo;
    fogInfo.fogStartPos = { 5084.0f, -69.5f, -4486.0f };
    fogInfo.fogVoxelGridSize = { 16, 16, 16 };
    fogInfo.fogVoxelSize = { 7.0f, 2.0f, 5.0f };
    fogInfo.absorption = 0.01f;
    fogInfo.scattering = 0.03f;
    fogInfo.phase = -0.7f;
    fogInfo.viscosity = 1.0f;

    getHeightFogScatter(startPos, endPos, randomSeed, fogInfo, L, attenuation, attenuation2);
    for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
        uint fogIndex = visibleVolumetricFogIndexBuffer[i];
        fogInfo = volumetricFogInfosBuffer[fogIndex];

        if (fogInfo.type == VolumetricFogType::Height) getHeightFogScatter(startPos, endPos, randomSeed, fogInfo, L, attenuation, attenuation2);
        else if (fogInfo.type == VolumetricFogType::Fluid) getFluidFogScatter(startPos, endPos, randomSeed, fogIndex, fogInfo, L, attenuation, attenuation2);
        else if (fogInfo.type == VolumetricFogType::Noise) getNoiseFogScatter(startPos, endPos, randomSeed, fogInfo, L, attenuation, attenuation2);
    }
}

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_deferredRendering(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 screenSize;
    albedoImage.GetDimensions(screenSize.x, screenSize.y);
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0) groupGlobalInfo = globalInfoBuffer[0];
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    float depth = depthImage[threadIndex];
    float2 uv = (float2(threadIndex) + 0.5f) / float2(screenSize);
    float2 ndc = uv * 2.0f - 1.0f;
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos = mul(mul(clipPos, sceneInfo.projInvMatrix), sceneInfo.viewInvMatrix);
    worldPos /= worldPos.w;

    uint randomSeed = xxhash32(uint3(pushConst.frameIndex, threadIndex));

    float3 attenuation = 1.0f, attenuation2 = 1.0f;
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
