#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#ifndef SHADER_SOURCE
#define SHADER_SOURCE

[[vk::binding(StaticBindingPoints_VolumetricFog::eAlbedoImage)]] Texture2D<float4> albedoImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eEmissiveImage)]] Texture2D<float4> emissiveImage;

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

[[vk::binding(StaticBindingPoints_VolumetricFog::eShadowMap)]] Sampler2DShadow shadowMap;

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eHeightFogInfosBuffer)]] RWStructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogInfosBuffer)]] RWStructuredBuffer<GridFogInfo, ScalarDataLayout> gridFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogImages)]] Sampler3D<float4> gridFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfosBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages)]] RWTexture3D<float4> fluidFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages_sample)]] Sampler3D<float4> fluidFogImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccImage)]] RWTexture3D<float4> fogAccImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccImage_sample)]] Sampler3D<float4> fogAccImage_sample;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccHasFogVoxelInfosBuffer)]] RWStructuredBuffer<FogAccHasFogVoxelInfo, ScalarDataLayout> fogAccHasFogVoxelInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccHistoryImage)]] Sampler3D<float4> fogAccHistoryImage;

//#define COMPUTE_SHADER
#ifdef COMPUTE_SHADER
groupshared SceneInfo groupSceneInfo;
groupshared FogGlobalInfo groupFogGlobalInfo;

groupshared VolumetricFogInfo groupVolumetricFogInfos[MAX_VOLUMETRIC_FOG_COUNT];
groupshared HeightFogInfo groupHeightFogInfos[MAX_HEIGHT_FOG_COUNT];
groupshared GridFogInfo grouGridFogInfos[MAX_GRID_FOG_COUNT];
groupshared FluidFogInfo groupFluidFogInfos[MAX_FLUID_FOG_COUNT];

void getGroupsharedData(uint groupThreadIndex, SceneInfo* sceneInfoAddress, FogGlobalInfo* fogGlobalInfoAddress) {
    if (groupThreadIndex >= MAX_VOLUMETRIC_FOG_COUNT) return;

    FogGlobalInfo fogGlobalInfo = fogGlobalInfoAddress[0];
    if (groupThreadIndex == 0) {
        groupFogGlobalInfo = fogGlobalInfo;
        groupSceneInfo = sceneInfoAddress[0];
    }
    if (groupThreadIndex < min(fogGlobalInfo.volumetricFogCount, MAX_VOLUMETRIC_FOG_COUNT)) groupVolumetricFogInfos[groupThreadIndex] = fogInfosBuffer[groupThreadIndex];
    if (groupThreadIndex < min(fogGlobalInfo.heightFogCount, MAX_HEIGHT_FOG_COUNT)) groupHeightFogInfos[groupThreadIndex] = heightFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(fogGlobalInfo.gridFogCount, MAX_GRID_FOG_COUNT)) grouGridFogInfos[groupThreadIndex] = gridFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(fogGlobalInfo.fluidFogCount, MAX_FLUID_FOG_COUNT)) groupFluidFogInfos[groupThreadIndex] = fluidFogInfoBuffer[groupThreadIndex];
}
#endif

#endif

float getShadow(float4 worldPos, float4x4 lightVP) {
    float4 lightMapNdcPos = mul(worldPos, lightVP);
    if (abs(lightMapNdcPos.w) < 1e-6f) return 1.0f;
    lightMapNdcPos /= lightMapNdcPos.w;

    float2 screenPos = lightMapNdcPos.xy * 0.5f + 0.5f;
    if (any(screenPos < 0.0f) || any(screenPos > 1.0f) || lightMapNdcPos.z < 0.0f || lightMapNdcPos.z > 1.0f) return 1.0f;

    float shadow = shadowMap.SampleCmpLevel(screenPos, lightMapNdcPos.z, 0);
    return shadow;
}
//----------------------------------------------------------------------------------------
float radicalInverse(uint index, uint base) {
    float result = 0.0f;
    float invBase = 1.0f / (float)base;
    float f = invBase;
    while (index > 0u) {
        result += f * (float)(index % base);
        index /= base;
        f *= invBase;
    }
    return result;
}

float getFogSliceDither(float2 ditherPos, uint temporalFrameIndex, float strength) {
    float dither = InterleavedGradientNoise(ditherPos, (float)(temporalFrameIndex & 63u));
    return (dither - 0.5f) * 2.0f * strength;
}

float getCellDither(float2 cell, float layer) {
    return InterleavedGradientNoise(cell, layer);
}

float getMarchLayerOffset(uint temporalFrameIndex, float spatialDither, bool isRandom, bool advanceTemporal) {
    if (!isRandom) return 0.5f;
    if (!advanceTemporal) return frac(spatialDither);
    return frac(radicalInverse(temporalFrameIndex % 16u + 1u, 2u) + spatialDither);
}

VolumetricFogInfo getVolumetricFog(int i) {
    #ifdef COMPUTE_SHADER
    VolumetricFogInfo volumetricFogInfo = groupVolumetricFogInfos[i];
    #else
    VolumetricFogInfo volumetricFogInfo = fogInfosBuffer[i];
    #endif
    return volumetricFogInfo;
}

HeightFogInfo getHeightFog(int i) {
    #ifdef COMPUTE_SHADER
    HeightFogInfo heightFogInfo = groupHeightFogInfos[i];
    #else
    HeightFogInfo heightFogInfo = heightFogInfoBuffer[i];
    #endif
    return heightFogInfo;
}
FluidFogInfo getFluidFog(int i) {
    #ifdef COMPUTE_SHADER
    FluidFogInfo fluidFogInfo = groupFluidFogInfos[i];
    #else
    FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[i];
    #endif
    return fluidFogInfo;
}
GridFogInfo getGridFog(int i) {
    #ifdef COMPUTE_SHADER
    GridFogInfo gridFogInfo = grouGridFogInfos[i];
    #else
    GridFogInfo gridFogInfo = gridFogInfoBuffer[i];
    #endif
    return gridFogInfo;
}

void getFogScatter(
    float3 samplePoint, float3 sampleDir, float sampleLength, SceneInfo sceneInfo, float lightAttenuation, float4x4 lightVP,
    float extinctionCoeff, float scattering, float phase, float3 color, float3 ambientIntensity, float3 E,
    inout float a, inout float3 L
) {
    Light light = sceneInfo.lights[0];
    float3 lightColor = light.color * light.intensity;
    float3 lightDirection = -light.direction;
    float3 lightPos = samplePoint + 1000.0f * lightDirection;

    phase = clamp(phase, -0.999f, 0.999f);
    float phase2 = phase * phase;
    float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-sampleDir, lightDirection)), 1.5f));

    float3 activeSunColor = lightColor * lightAttenuation;
    activeSunColor *= getShadow(float4(samplePoint, 1.0f), lightVP);

    float x = extinctionCoeff * sampleLength;
    float factor = (x < 1e-4) ? sampleLength * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
    L = activeSunColor * fp * color * scattering * E * factor;
    L += color * ambientIntensity * scattering * E * factor;
    a = exp(-extinctionCoeff * sampleLength);

    #ifdef MULTI_SCATTERING
    fp = 0.5f * M_1_PI;
    float3 E1 = lightColor * lightAttenuation * fp * color * scattering * E * factor;
    float fms = scattering / max(extinctionCoeff, 1e-10f) * (1.0f - exp(-20.0f * extinctionCoeff));
    float3 multiScattering = E1 * fms / max(1.0f - fms, 1e-10f);
    L += multiScattering;

    //float a_temp = 0.5f, b = 0.5f, c = 0.5f;
    //for (int i = 1; i < 8; ++i) {
    //    float phase_temp = phase * c;
    //    phase2 = phase_temp * phase_temp;
    //    fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase_temp * dot(-sampleDir, lightDirection)), 1.5f));
    //
    //    x = a_temp * extinctionCoeff * sampleLength;
    //    factor = (x < 1e-4) ? sampleLength * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
    //
    //    L += b * activeSunColor * fp * color * scattering * E * factor;
    //
    //    a_temp *= 0.5f; b *= 0.5f; c *= 0.5f;
    //}
    #endif
}
//---------------------------------------------------------------------------------------------------------------------
float EiScaled(float x, float logAbsX) {
    const float eulerGamma = 0.5772156649f;
    if (logAbsX < -13.0f) return eulerGamma + logAbsX;

    float absX = abs(x);
    if (absX > 20.0f) {
        float invX = 1.0f / x;
        float term = invX;
        float series = term;
        for (int order = 1; order <= 24; ++order) {
            term *= float(order) * invX;
            series += term;
            if (abs(term) < 1e-9f * abs(series)) break;
        }
        return series;
    }

    if (x > 0.0f || absX < 1.0f) {
        float term = x;
        float series = term;
        for (int order = 2; order <= 48; ++order) {
            term *= x / float(order);
            series += term / float(order);
        }
        return (eulerGamma + logAbsX + series) * exp(-x);
    }

    float b = -x + 1.0f;
    float c = 1e30f;
    float d = 1.0f / b;
    float h = d;
    for (int order = 1; order <= 64; ++order) {
        float an = -float(order) * float(order);
        b += 2.0f;
        d = 1.0f / (an * d + b);
        c = b + an / c;
        float delta = c * d;
        h *= delta;
        if (abs(delta - 1.0f) < 1e-7f) break;
    }
    return -h;
}

float Ei(float x) {
    float absX = abs(x);
    if (absX < 1e-30f) return -1e30f;
    return EiScaled(x, log(absX)) * exp(x);
}

float oneMinusExpRatio(float t) {
    if (abs(t) < 1e-4f) return 1.0f - 0.5f * t + t * t / 6.0f;
    return (1.0f - exp(-t)) / t;
}

float scatterIntegral(float X, float lnInvB, float transmittanceStart, float transmittanceEnd) {
    float b = exp(-lnInvB);
    float Xb = X * b;
    if (max(abs(X), abs(Xb)) < 1e-3f) {
        float series = lnInvB + X * (1.0f - b) + X * X * (1.0f - b * b) * 0.25f + X * X * X * (1.0f - b * b * b) / 18.0f;
        return transmittanceStart * exp(-X) * series;
    }
    float logAbsX = (X != 0.0f) ? log(abs(X)) : -1e30f;
    return transmittanceStart * EiScaled(X, logAbsX) - transmittanceEnd * EiScaled(Xb, logAbsX - lnInvB);
}

bool getAnalyticHeightFogDirectionalScatter(float3 startPos, float3 endPos, AABB fogAABB, HeightFogInfo fogInfo, Light light, float4x4 lightVP, inout float attenuation, inout float3 L) {
    float tEntry, tExit;
    if (!isHit(startPos, endPos, fogAABB.minimum, fogAABB.maximum, tEntry, tExit)) return false;
    float d = distance(endPos, startPos);
    float3 dir = (endPos - startPos) / max(d, 1e-10f);
    endPos = startPos + tExit * dir;
    startPos = startPos + tEntry * dir;
    float3 midPos = (startPos + endPos) * 0.5f;

    float tLightEntry, tLightExit;
    float3 lightPos = midPos - 10000.0f * light.direction;
    isHit(lightPos, midPos, fogAABB.minimum, fogAABB.maximum, tLightEntry, tLightExit);
    lightPos = lightPos + tLightEntry * light.direction;

    float phase2 = fogInfo.phase * fogInfo.phase;
    float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * fogInfo.phase * dot(-dir, -light.direction)), 1.5f));
    float shadow = getShadow(float4(midPos, 1.0f), lightVP);
    float3 baseScatter = shadow * fp * fogInfo.scattering * light.color * light.intensity * attenuation;

    float y0 = fogAABB.minimum.y;
    float y1 = startPos.y;
    float y2 = endPos.y;
    float yL = lightPos.y;      //assume all pos use same yL
    float wSy = dir.y;
    float wLy = light.direction.y; // from light to pos
    float sigmaT0 = fogInfo.absorption + fogInfo.scattering;
    float k = fogInfo.heightScale;

    float SL = distance(lightPos, midPos);
    float S = distance(startPos, endPos);

    float sigmaStart = sigmaT0 * exp(-k * (y1 - y0));
    float sigmaLight = sigmaT0 * exp(-k * (yL - y0));
    float lnInvB = k * wSy * S;
    float tauView = sigmaStart * S * oneMinusExpRatio(lnInvB);
    bool viewFlat = abs(lnInvB) < 1e-4f;
    bool lightFlat = abs(k * wLy * SL) < 1e-4f;

    attenuation *= exp(-tauView);

    if (viewFlat && lightFlat) {
        L += baseScatter * exp(-sigmaStart * SL) * S * oneMinusExpRatio(sigmaStart * S);
    } else if (viewFlat) {
        float tauLight = sigmaLight * ((y1 - yL) / wLy) * oneMinusExpRatio(k * (y1 - yL));
        L += baseScatter * exp(-tauLight) * S * oneMinusExpRatio(sigmaStart * S);
    } else if (lightFlat) {
        float cL = sigmaStart * SL;
        float X = sigmaStart / (k * wSy) - cL;
        float transmittanceStart = exp(-cL);
        float transmittanceEnd = exp(-tauView - cL * exp(-lnInvB));
        L += baseScatter * scatterIntegral(X, lnInvB, transmittanceStart, transmittanceEnd) / (k * wSy);
    } else {
        float tauLightStart = sigmaLight * ((y1 - yL) / wLy) * oneMinusExpRatio(k * (y1 - yL));
        float tauLightEnd = sigmaLight * ((y2 - yL) / wLy) * oneMinusExpRatio(k * (y2 - yL));
        float X = sigmaStart * (1.0f / (k * wSy) + 1.0f / (k * wLy));
        float transmittanceStart = exp(-tauLightStart);
        float transmittanceEnd = exp(-tauView - tauLightEnd);
        L += baseScatter * scatterIntegral(X, lnInvB, transmittanceStart, transmittanceEnd) / (k * wSy);
    }

    return true;
}

float getHeightFogTransmittance(float extinction, float heightScale, float S, float fogStartY, float startY, float endY) {
    if (heightScale == 0.0f) return exp(-extinction * S);

    float dy = endY - startY;
    float hdy = heightScale * dy;

    if (abs(dy) < 1e-6f || abs(hdy) < 1e-6f) return exp(-extinction * S * exp(-heightScale * (startY - fogStartY)));

    float base = -heightScale * (startY - fogStartY);
    float exp_base = exp(min(base, 88.0f));

    float ratio = (1.0f - exp(-hdy)) / hdy;

    return exp(-extinction * S * exp_base * ratio);
}
//---------------------------------------------------------------------------------------------------------------------
float getTransmittance2(float3 pos, FogGlobalInfo fogGlobalInfo, Light light) {
    float lightTransmittance = 1.0f;
    if (fogGlobalInfo.useGlobalHeightFog == 1 && checkPointInAABB(pos, fogGlobalInfo.globalHeightFogAABB)) {
        HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
        float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (pos.y - fogGlobalInfo.globalHeightFogAABB.minimum.y));
        float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;

        float3 hitPos = getAABBPoint(fogGlobalInfo.globalHeightFogAABB, pos, -light.direction);
        float S = distance(hitPos, pos);
        lightTransmittance *= getHeightFogTransmittance(extinction, globalHeightFogInfo.heightScale, S, fogGlobalInfo.globalHeightFogAABB.minimum.y, pos.y, hitPos.y);
    }

    for (int i = 0; i < fogGlobalInfo.volumetricFogCount; ++i) {
        VolumetricFogInfo fogInfo = getVolumetricFog(i);
        if (!checkPointInAABB(pos, fogInfo.aabb)) continue;

        float3 hitPos = getAABBPoint(fogInfo.aabb, pos, -light.direction);
        float S = distance(hitPos, pos);

        float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
        if (fogInfo.type == VolumetricFogType::Height) {
            HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

            float heightScale = heightFogInfo.heightScale;
            float height = max(pos.y - fogInfo.aabb.minimum.y, 0.0f);
            float heightAttenuation = exp(-heightScale * height);
            scattering = heightFogInfo.scattering * heightAttenuation;
            extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;

            lightTransmittance *= getHeightFogTransmittance(extinction, heightFogInfo.heightScale, S, fogInfo.aabb.minimum.y, pos.y, hitPos.y);
        } else if (fogInfo.type == VolumetricFogType::Grid) {
            GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

            float fogLength = distance(fogInfo.aabb.minimum, fogInfo.aabb.maximum);
            lightTransmittance *= lerp(1.0f, gridFogInfo.transmittance, saturate(S / fogLength));
        }
    }

    return lightTransmittance;
}

void getVolumetricFogScatter_DeltaTracking(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount, float stepPower, bool useGlobalHeight, float layerOffset, 
    float4x4 lightVP, SceneInfo *sceneInfoAddress, FogGlobalInfo *fogGlobalInfoAddress,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    FogGlobalInfo fogGlobalInfo = groupFogGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    FogGlobalInfo fogGlobalInfo = fogGlobalInfoAddress[0];
    SceneInfo sceneInfo = sceneInfoAddress[0];
    #endif

    float3 dir = normalize(endPos - startPos);
    float tTotal = length(endPos - startPos);

    Light light = sceneInfo.lights[0];
    float3 lightColor = light.color * light.intensity;
    float3 lightDirection = -light.direction;

    const float maxExtinction = 1.0f; // this true, dont't worry
    float transmittance = 0.0f;
    float3 L_scattering = 0.0f;
    for (int i = 0; i < sampleCount; ++i) {
        float t = 0.0f;
        int maxIteration = 100;
        for(int j = 0; j < maxIteration; ++j){
            float r = rand(randomSeed);
            t = t - log(r) / maxExtinction;
            if (t > tTotal) break;

            float3 samplePoint = startPos + t * dir;
            float envExtinction = 0.0f;
            float envScattering = 0.0f;
            float envPhase = 0.0f;
            float3 envColor = 0.0f;
            {
                if (useGlobalHeight && fogGlobalInfo.useGlobalHeightFog == 1 && checkPointInAABB(samplePoint, fogGlobalInfo.globalHeightFogAABB)) {
                    HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
                    float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - fogGlobalInfo.globalHeightFogAABB.minimum.y));
                    float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
                    envExtinction += extinction;
                    envScattering += globalHeightFogInfo.scattering * heightAttenuation;
                    envPhase += extinction * globalHeightFogInfo.phase;
                    envColor += extinction * globalHeightFogInfo.color;
                }

                bool inFluid = false;
                for (int i = 0; i < fogGlobalInfo.fluidFogCount; ++i) {
                    VolumetricFogInfo fogInfo = getVolumetricFog(i);
                    if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

                    FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
                    if (fluidFogInfo.startUp == 0) continue;

                    float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
                    uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
                    float4 voxelInfo = fluidFogImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                    // if in fluid, the voxelInfo has include the envionment fog info
                    float extinction = max(voxelInfo.x, 1e-30f);
                    envExtinction += extinction;
                    envScattering += voxelInfo.y;
                    envPhase += extinction * voxelInfo.z;
                    envColor += extinction * voxelInfo.w;

                    inFluid = true;
                    break;
                }

                if (!inFluid) {
                    for (int i = fogGlobalInfo.fluidFogCount; i < fogGlobalInfo.volumetricFogCount; ++i) {
                        VolumetricFogInfo fogInfo = getVolumetricFog(i);
                        if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

                        if (fogInfo.type == VolumetricFogType::Height) {
                            HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

                            float heightScale = heightFogInfo.heightScale;
                            float height = max(samplePoint.y - fogInfo.aabb.minimum.y, 0.0f);
                            float heightAttenuation = exp(-heightScale * height);
                            float extinction = (heightFogInfo.absorption + heightFogInfo.scattering) * heightAttenuation;
                            envExtinction += extinction;
                            envScattering += heightFogInfo.scattering * heightAttenuation;
                            envPhase += extinction * heightFogInfo.phase;
                            envColor += extinction * heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
                        } else if (fogInfo.type == VolumetricFogType::Grid) {
                            GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

                            float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                            float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                            float extinction = voxelInfo.x;
                            envExtinction += extinction;
                            envScattering += voxelInfo.y;
                            envPhase += extinction * voxelInfo.z;
                            envColor += extinction * voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
                        }
                    }
                }
                envPhase /= max(envExtinction, 1e-30f);
                envColor /= max(envExtinction, 1e-30f);
            }
            envExtinction = max(envExtinction, 1e-10f);

            float k = rand(randomSeed);
            if (k <= envExtinction / maxExtinction) {
                envPhase = clamp(envPhase, -0.999f, 0.999f);
                float phase2 = envPhase * envPhase;
                float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * envPhase * dot(-dir, lightDirection)), 1.5f));

                float3 activeSunColor = lightColor * 1.0f;
                activeSunColor *= getShadow(float4(samplePoint, 1.0f), lightVP);

                float3 L_scattering_temp = activeSunColor * fp * envColor;
                //L_scattering_temp += envColor;
                L_scattering += envScattering / envExtinction * luminance(L_scattering_temp);

                break;
            }
        }
        transmittance += t > tTotal ? 1.0f : 0.0f;
    }
    L += luminance(L_scattering / sampleCount);
    attenuation *= transmittance / sampleCount;
    attenuation2 *= 1.0f;
}
void getVolumetricFogScatter_RatioTracking(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount, float stepPower, bool useGlobalHeight, float layerOffset, 
    float4x4 lightVP, SceneInfo *sceneInfoAddress, FogGlobalInfo *fogGlobalInfoAddress,
                                           inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    FogGlobalInfo fogGlobalInfo = groupFogGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    FogGlobalInfo fogGlobalInfo = fogGlobalInfoAddress[0];
    SceneInfo sceneInfo = sceneInfoAddress[0];
    #endif

    float3 dir = normalize(endPos - startPos);
    float tTotal = length(endPos - startPos);

    Light light = sceneInfo.lights[0];
    float3 lightColor = light.color * light.intensity;
    float3 lightDirection = -light.direction;

    const float maxExtinction = 1.0f; // must be an upper bound of sigma_t along the ray
    float transmittance = 0.0f;
    float3 L_scattering = 0.0f;
    float segmentStartTransmittance = attenuation;
    for (uint sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float t = 0.0f;
        float throughput = 1.0f;
        while (true) {
            float r = rand(randomSeed);
            t = t - log(r) / maxExtinction;
            if (t > tTotal) break;

            float3 samplePoint = startPos + t * dir;
            float envExtinction = 0.0f;
            float envScattering = 0.0f;
            float envPhase = 0.0f;
            float3 envColor = 0.0f;
            {
                if (useGlobalHeight && fogGlobalInfo.useGlobalHeightFog == 1 && checkPointInAABB(samplePoint, fogGlobalInfo.globalHeightFogAABB)) {
                    HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
                    float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - fogGlobalInfo.globalHeightFogAABB.minimum.y));
                    float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
                    envExtinction += extinction;
                    envScattering += globalHeightFogInfo.scattering * heightAttenuation;
                    envPhase += extinction * globalHeightFogInfo.phase;
                    envColor += extinction * globalHeightFogInfo.color;
                }

                bool inFluid = false;
                for (int i = 0; i < fogGlobalInfo.fluidFogCount; ++i) {
                    VolumetricFogInfo fogInfo = getVolumetricFog(i);
                    if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

                    FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
                    if (fluidFogInfo.startUp == 0) continue;

                    float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
                    uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
                    float4 voxelInfo = fluidFogImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                    // if in fluid, the voxelInfo has include the envionment fog info
                    float extinction = max(voxelInfo.x, 1e-30f);
                    envExtinction += extinction;
                    envScattering += voxelInfo.y;
                    envPhase += extinction * voxelInfo.z;
                    envColor += extinction * voxelInfo.w;

                    inFluid = true;
                    break;
                }

                if (!inFluid) {
                    for (int i = fogGlobalInfo.fluidFogCount; i < fogGlobalInfo.volumetricFogCount; ++i) {
                        VolumetricFogInfo fogInfo = getVolumetricFog(i);
                        if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

                        if (fogInfo.type == VolumetricFogType::Height) {
                            HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

                            float heightScale = heightFogInfo.heightScale;
                            float height = max(samplePoint.y - fogInfo.aabb.minimum.y, 0.0f);
                            float heightAttenuation = exp(-heightScale * height);
                            float extinction = (heightFogInfo.absorption + heightFogInfo.scattering) * heightAttenuation;
                            envExtinction += extinction;
                            envScattering += heightFogInfo.scattering * heightAttenuation;
                            envPhase += extinction * heightFogInfo.phase;
                            envColor += extinction * heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
                        } else if (fogInfo.type == VolumetricFogType::Grid) {
                            GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

                            float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                            float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                            float extinction = voxelInfo.x;
                            envExtinction += extinction;
                            envScattering += voxelInfo.y;
                            envPhase += extinction * voxelInfo.z;
                            envColor += extinction * voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
                        }
                    }
                }
                envPhase /= max(envExtinction, 1e-30f);
                envColor /= max(envExtinction, 1e-30f);
            }
            envExtinction = max(envExtinction, 1e-10f);

            if (envExtinction > 1e-10f) {
                envPhase = clamp(envPhase, -0.999f, 0.999f);
                float phase2 = envPhase * envPhase;
                float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * envPhase * dot(-dir, lightDirection)), 1.5f));

                float3 activeSunColor = lightColor * 1.0f;
                activeSunColor *= getShadow(float4(samplePoint, 1.0f), lightVP);

                float3 source = activeSunColor * fp * envColor;
                L_scattering += segmentStartTransmittance * throughput * envScattering * luminance(source) / maxExtinction;
            }

            throughput *= max(0.0f, 1.0f - envExtinction / maxExtinction);
        }

        transmittance += throughput;
    }
    L += luminance(L_scattering / sampleCount);
    attenuation *= transmittance / sampleCount;
    attenuation2 *= 1.0f;
}
void getVolumetricFogScatter(
    float3 startPos, float3 endPos, inout uint randomSeed, uint sampleCount, float stepPower, bool useGlobalHeight, float layerOffset, 
    float4x4 lightVP, SceneInfo* sceneInfoAddress, FogGlobalInfo* fogGlobalInfoAddress,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    FogGlobalInfo fogGlobalInfo = groupFogGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    FogGlobalInfo fogGlobalInfo = fogGlobalInfoAddress[0];
    SceneInfo sceneInfo = sceneInfoAddress[0];
    #endif

    float3 dir = normalize(endPos - startPos);
    float tTotal = length(endPos - startPos);

    /*
    {
        float tEntry, tExit;
        if (!isHit(startPos, endPos, fogGlobalInfo.globalHeightFogAABB.minimum, fogGlobalInfo.globalHeightFogAABB.maximum, tEntry, tExit)) return;
        float3 sampleDir = normalize(endPos - startPos);
        endPos = startPos + tExit * sampleDir;
        startPos = startPos + tEntry * sampleDir;
        float sampleLength = distance(startPos, endPos);

        float3 L0 = 0.0f;
        getAnalyticHeightFogDirectionalScatter(startPos, endPos, fogGlobalInfo.globalHeightFogAABB, fogGlobalInfo.globalHeightFogInfo, sceneInfo.lights[0], lightVP, attenuation, L0);
        L = luminance(L0);
        return;
    }
    */

    for (int i = 0; i < fogGlobalInfo.fluidFogCount; ++i) {
        VolumetricFogInfo fogInfo = getVolumetricFog(i);
        FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
        if (fluidFogInfo.startUp == 0) continue;

        float tEntry, tExit;
        if (!isHit(startPos, endPos, fogInfo.aabb.minimum, fogInfo.aabb.maximum, tEntry, tExit)) continue;
        sampleCount *= 2;
        break;
    }

    float tPartStart = 0;
    float tPartEnd = pow(1.0f / sampleCount, stepPower) * tTotal;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        //layerOffset = rand(randomSeed);
        float tPart = tPartEnd - tPartStart;
        float3 samplePoint = startPos + dir * (tPartStart + tPart * layerOffset);

        tPartStart = tPartEnd;
        tPartEnd = pow(float(sampleIndex + 1) / sampleCount, stepPower) * tTotal;

        float envExtinction = 0.0f;
        float envScattering = 0.0f;
        float envPhase = 0.0f;
        float3 envColor = 0.0f;

        float lightTransmittance = 1.0f;
        if (useGlobalHeight && fogGlobalInfo.useGlobalHeightFog == 1 && checkPointInAABB(samplePoint, fogGlobalInfo.globalHeightFogAABB)) {
            HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
            float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - fogGlobalInfo.globalHeightFogAABB.minimum.y));
            float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
            envExtinction = extinction;
            envScattering = globalHeightFogInfo.scattering * heightAttenuation;
            envPhase = extinction * globalHeightFogInfo.phase;
            envColor = extinction * globalHeightFogInfo.color;

            float3 hitPos = getAABBPoint(fogGlobalInfo.globalHeightFogAABB, samplePoint, -sceneInfo.lights[0].direction);
            float S = distance(hitPos, samplePoint);
            lightTransmittance *= getHeightFogTransmittance(extinction, globalHeightFogInfo.heightScale, S, fogGlobalInfo.globalHeightFogAABB.minimum.y, samplePoint.y, hitPos.y);
        }

        bool inFluid = false;
        for (int i = 0; i < fogGlobalInfo.fluidFogCount; ++i) {
            VolumetricFogInfo fogInfo = getVolumetricFog(i);
            if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

            FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
            if (fluidFogInfo.startUp == 0) continue;

            float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
            uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
            float4 voxelInfo = fluidFogImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

            // if in fluid, the voxelInfo has include the envionment fog info
            envExtinction = max(voxelInfo.x, 1e-30f);
            envScattering = voxelInfo.y;
            envPhase = envExtinction * voxelInfo.z;
            envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);

            inFluid = true;
            break;
        }

        for (int i = fogGlobalInfo.fluidFogCount; i < fogGlobalInfo.volumetricFogCount; ++i) {
            VolumetricFogInfo fogInfo = getVolumetricFog(i);
            if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

            float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
            if (fogInfo.type == VolumetricFogType::Height) {
                HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

                float heightScale = heightFogInfo.heightScale;
                if (!inFluid) {
                    float height = max(samplePoint.y - fogInfo.aabb.minimum.y, 0.0f);
                    float heightAttenuation = exp(-heightScale * height);
                    extinction = (heightFogInfo.absorption + heightFogInfo.scattering) * heightAttenuation;
                    scattering = heightFogInfo.scattering * heightAttenuation;
                    phase = heightFogInfo.phase;
                    color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
                }

                float3 hitPos = getAABBPoint(fogInfo.aabb, samplePoint, -sceneInfo.lights[0].direction);
                float S = distance(hitPos, samplePoint);
                lightTransmittance *= getHeightFogTransmittance(extinction, heightScale, S, fogInfo.aabb.minimum.y, samplePoint.y, hitPos.y);
            } else if (fogInfo.type == VolumetricFogType::Grid) {
                GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

                if (!inFluid) {
                    float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                    float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                    extinction = voxelInfo.x;
                    scattering = voxelInfo.y;
                    phase = voxelInfo.z;
                    color = voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
                }

                float3 hitPos = getAABBPoint(fogInfo.aabb, samplePoint, -sceneInfo.lights[0].direction);
                float S = distance(hitPos, samplePoint);
                float fogLength = distance(fogInfo.aabb.minimum, fogInfo.aabb.maximum);
                lightTransmittance *= lerp(1.0f, gridFogInfo.transmittance, saturate(S / fogLength));
            }

            envExtinction += extinction;
            envScattering += scattering;
            envPhase += extinction * phase;
            envColor += extinction * color;
        }

        envPhase /= max(envExtinction, 1e-30f);
        envColor /= max(envExtinction, 1e-30f);

        if (sampleIndex == sampleCount - 1) {
            if (!useGlobalHeight && fogGlobalInfo.useGlobalHeightFog == 1 && checkPointInAABB(samplePoint, fogGlobalInfo.globalHeightFogAABB)) {
                HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
                float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - fogGlobalInfo.globalHeightFogAABB.minimum.y));
                float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;

                float3 hitPos = getAABBPoint(fogGlobalInfo.globalHeightFogAABB, samplePoint, -sceneInfo.lights[0].direction);
                float S = distance(hitPos, samplePoint);
                attenuation2 *= getHeightFogTransmittance(extinction, globalHeightFogInfo.heightScale, S, fogGlobalInfo.globalHeightFogAABB.minimum.y, samplePoint.y, hitPos.y);
            }
            attenuation2 *= lightTransmittance;
        }

        if (envExtinction <= 1e-30f) continue;

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, lightTransmittance, lightVP, envExtinction, envScattering, envPhase, envColor, 0.0f, attenuation, a, indirectL);

        attenuation *= a;
        L += luminance(indirectL);
        if (all(attenuation <= 1e-3f)) break;
    }
}

void getGlobalHeightFogScatter(
    float3 startPos, float3 endPos, float randNumber, SceneInfo sceneInfo, FogGlobalInfo globalInfo, float4x4 lightVP,
    float3 E, inout float a, inout float3 L
) {
    if (globalInfo.useGlobalHeightFog == 0) return;
    float tEntry, tExit;
    if (!isHit(startPos, endPos, globalInfo.globalHeightFogAABB.minimum, globalInfo.globalHeightFogAABB.maximum, tEntry, tExit)) return;
    float3 sampleDir = normalize(endPos - startPos);
    endPos = startPos + tExit * sampleDir;
    startPos = startPos + tEntry * sampleDir;
    float sampleLength = distance(startPos, endPos);
    float3 samplePoint = startPos + (tExit - tEntry) * randNumber * sampleDir;

    HeightFogInfo globalHeightFogInfo = globalInfo.globalHeightFogInfo;
    float heightAttenuation = exp(-globalHeightFogInfo.heightScale * max(min(samplePoint.y, globalInfo.globalHeightFogAABB.maximum.y) - globalInfo.globalHeightFogAABB.minimum.y, 0.0f));
    float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
    float scattering = globalHeightFogInfo.scattering * heightAttenuation;
    float phase = clamp(globalHeightFogInfo.phase, -0.9999f, 0.9999f);
    float3 color = globalHeightFogInfo.color;

    getFogScatter(samplePoint, sampleDir, sampleLength, sceneInfo, globalInfo.lightAttenuationStrength, lightVP, extinction, scattering, phase, color, globalHeightFogInfo.ambientIntensity, E, a, L);
}
/*
sliceDither in [-0.5f, 0.5f]
*/
float3 getMorphedFrustumVoxelCenter(
    float depthWithNear, float u, float v,
    float3 cameraPos, float3 cameraForward, float3 cameraRight, float3 cameraUp,
    FrustumGlobalInfo frustumInfo
) {
    float3 D = cameraForward
             + (v * frustumInfo.tanCameraFov_2) * cameraUp
             + (u * frustumInfo.tanCameraFov_2 * frustumInfo.aspectRatio) * cameraRight;
    float scale = lerp(1.0f, 1.0f / length(D), frustumInfo.sphereFactor);
    return cameraPos + D * (depthWithNear * scale);
}

float getMorphedFrustumDepthWithNear(float3 toVoxel, float3 cameraForward, FrustumGlobalInfo frustumInfo) {
    float viewZ = dot(toVoxel, cameraForward);
    if (viewZ <= 0.0f) return -1.0f;
    float radius = length(toVoxel);
    float radialScale = lerp(radius / viewZ, 1.0f, frustumInfo.sphereFactor);
    return radius / radialScale;
}

float4 getFrustumImageInfo(float3 worldPos, bool linear, float3 sliceDither, Sampler3D<float4> image, inout uint randomSeed, SceneInfo *sceneInfoAddress, FrustumGlobalInfo frustumInfo) {
    #ifdef COMPUTE_SHADER
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    SceneInfo sceneInfo = sceneInfoAddress[0];
    #endif

    float nearPlane = frustumInfo.cameraNearPlane;
    float farPlane = frustumInfo.cameraFarPlane;
    float3 compressionParams = max(frustumInfo.compressionParams, 1e-4f);
    float tanHalfFov = frustumInfo.tanCameraFov_2;
    float aspect = frustumInfo.aspectRatio;
    uint3 gridSize = frustumInfo.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float3 toPoint = worldPos - sceneInfo.cameraPosition;
    float depthWithNear = getMorphedFrustumDepthWithNear(toPoint, cameraForward, frustumInfo);
    if (depthWithNear <= 0.0f) return float4(0, 0, 0, 0);

    float clampedDepthWithNear = clamp(depthWithNear, nearPlane, farPlane);
    float scale = clampedDepthWithNear / depthWithNear;
    float3 toClamped = toPoint * scale;
    float viewZ = dot(toClamped, cameraForward);
    float viewX = dot(toClamped, cameraRight);
    float viewY = dot(toClamped, cameraUp);

    float halfHeight = viewZ * tanHalfFov;
    float halfWidth = halfHeight * aspect;
    #ifdef SAMPLE_JITTER_TAA
    float u = saturate(viewX / halfWidth * 0.5f + 0.5f + sliceDither.x / gridSize.x);
    float v = saturate(viewY / halfHeight * 0.5f + 0.5f + sliceDither.y / gridSize.x);
    #else
    float u = saturate(viewX / halfWidth * 0.5f + 0.5f);
    float v = saturate(viewY / halfHeight * 0.5f + 0.5f);
    #endif

    #ifdef Interpolation_Manual
    uint voxelZ = clamp(int(frustumInvMap(clampedDepthWithNear - nearPlane, compressionParams, (farPlane - nearPlane), gridSize.z) / (farPlane - nearPlane) * gridSize.z), 0, gridSize.z - 1);

    float depth = clampedDepthWithNear - nearPlane;
    float depth0 = frustumMap(float(voxelZ) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
    float depth1 = frustumMap(float(voxelZ + 1) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
    float centerZ = (depth0 + depth1) * 0.5f;
    if (!linear || depth == centerZ) {
        float w = (float(voxelZ) + 0.5f) / gridSize.z;
        return image.SampleLevel(float3(u, v, w), 0);
    }

    int z0 = voxelZ, z1 = voxelZ;
    float blendZ = 0.0f;
    if (depth < centerZ && voxelZ > 0) {
        z0 = voxelZ - 1; z1 = voxelZ;
        float dPrev = frustumMap(float(z0) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
        float centerPrev = (dPrev + depth0) * 0.5f;
        blendZ = saturate((depth - centerPrev) / (centerZ - centerPrev));
    } else if (depth > centerZ && voxelZ < int(gridSize.z) - 1) {
        z1 = voxelZ + 1;
        float dNext = frustumMap(float(z1 + 1) / gridSizeF.z * (farPlane - nearPlane), compressionParams, (farPlane - nearPlane), gridSize.z);
        float centerNext = (depth1 + dNext) * 0.5f;
        blendZ = saturate((depth - centerZ) / (centerNext - centerZ));
    }
    float w = (float(z0) + 0.5f) / gridSize.z;
    float4 s0 = image.SampleLevel(float3(u, v, w), 0);

    w = (float(z1) + 0.5f) / gridSize.z;
    float4 s1 = image.SampleLevel(float3(u, v, w), 0);

    // return blendZ;
    return lerp(s0, s1, blendZ);
    #else
    float zSlice = frustumInvMap(clampedDepthWithNear - nearPlane, compressionParams, farPlane - nearPlane, gridSize.z) / (farPlane - nearPlane) * gridSizeF.z;
    #ifdef SAMPLE_JITTER_TAA
    float w = saturate(zSlice / gridSizeF.z + sliceDither.z / gridSizeF.z);
    #else
    zSlice += sliceDither.z;
    float w = saturate(zSlice / gridSizeF.z);
    #endif
    float3 uvw = float3(u, v, w);

    if (linear) return image.SampleLevel(uvw, 0);
    else return image[uint3(uvw * gridSizeF)];
    #endif
}

void getPreAccFog(
    float3 startPos, float3 endPos,
    inout uint randomSeed, float2 ditherPos, int temporalFrameIndex, float jitterStrength,
    SceneInfo *sceneInfoAddress, FrustumGlobalInfo frustumInfo,
    inout float L, inout float attenuation, inout float attenuation2
) {
    #ifdef COMPUTE_SHADER
    SceneInfo sceneInfo = groupSceneInfo;
    #else
    SceneInfo sceneInfo = sceneInfoAddress[0];
    #endif
    float3 dir = normalize(endPos - startPos);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float nearPlane = frustumInfo.cameraNearPlane;
    float farPlane = frustumInfo.cameraFarPlane;
    float tanHalfFov = frustumInfo.tanCameraFov_2;
    float aspect = frustumInfo.aspectRatio;
    float3 compressionParams = frustumInfo.compressionParams;

    float tEnter, tExit;
    if (!IntersectFrustum(startPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tEnter, tExit)) return;
    endPos = startPos + tExit * dir;
    startPos = startPos + tEnter * dir;

    #ifdef SAMPLE_JITTER_TAA
    float3 Halton_2_3_5[8];
    Halton_2_3_5[0] = { 0.5f, 1.0f / 3, 1.0f / 5 };
    Halton_2_3_5[1] = { 0.25f, 2.0f / 3, 2.0f / 5 };
    Halton_2_3_5[2] = { 0.75f, 1.0f / 9, 3.0f / 5 };
    Halton_2_3_5[3] = { 0.125f, 4.0f / 9, 4.0f / 5 };
    Halton_2_3_5[4] = { 0.625f, 7.0f / 9, 1.0f / 25 };
    Halton_2_3_5[5] = { 0.375f, 2.0f / 9, 6.0f / 25 };
    Halton_2_3_5[6] = { 0.875f, 5.0f / 9, 11.0f / 25 };
    Halton_2_3_5[7] = { 0.0625f, 8.0f / 9, 16.0f / 25 };
    float3 sliceDither = Halton_2_3_5[temporalFrameIndex & 7] * 0.1f;
    #else
    float3 sliceDither = getFogSliceDither(ditherPos, temporalFrameIndex, jitterStrength);
    #endif

    float4 fogInfo = getFrustumImageInfo(endPos, true, sliceDither, fogAccImage_sample, randomSeed, sceneInfoAddress, frustumInfo);

    //float viewZ = dot(endPos - cameraPos, cameraForward);
    //float fadeStart = farPlane * 0.8f;
    //float farClipProtect = 1.0f - saturate((viewZ - fadeStart) / max(farPlane - fadeStart, 1e-4f));

    attenuation *= fogInfo.x;
    L += fogInfo.y;
    attenuation2 *= fogInfo.z;
}
