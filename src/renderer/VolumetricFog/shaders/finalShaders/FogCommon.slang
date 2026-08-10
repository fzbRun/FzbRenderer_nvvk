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
// groupshared FluidFogInfo groupFluidFogInfos[MAX_FLUID_FOG_COUNT];

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
    // if (groupThreadIndex < min(fogGlobalInfo.fluidFogCount, MAX_FLUID_FOG_COUNT)) groupFluidFogInfos[groupThreadIndex] = fluidFogInfoBuffer[groupThreadIndex];
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
//----------------------------------------时域抖动------------------------------------------------
// 时域超采样只作用在“体素内部的光线步进采样点”上（见 getVolumetricFogScatter 的 layerOffset）：
// 每帧让采样点落在体素板内的不同分层位置，再由 step7_5 的时域滤波把多帧积起来，
// 等价于在固定的 160x160x80 分辨率下提高了采样率。
// 注意：绝不能去抖动体素中心本身。消费端 getFrustumImageInfo 是按未抖动的网格映射
// 算 uvw 的，它不知道这个偏移；网格一动，每帧读到的雾就横向错开最多半个体素，
// 那不是超采样，那就是闪烁。
// 用低差异序列（radical inverse）而不是白噪声，是因为前者收敛快且不引入闪烁。
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

// 采样视锥体素时的屏幕空间有序抖动（InterleavedGradientNoise），偏移量以“切片”为单位。
// 相比原来在世界空间viewZ上加白噪声：IGN在屏幕空间是稳定的低频图案，
// 时域滤波和TAA可以干净地把它收敛掉；而白噪声在相机移动时表现为逐像素的闪烁，
// 这正是“波动”的主要来源。参考引擎的做法也是在切片空间做 (dither - 0.5) * 2。
float getFogSliceDither(float2 ditherPos, uint temporalFrameIndex, float strength) {
    float dither = InterleavedGradientNoise(ditherPos, (float)(temporalFrameIndex & 63u));
    return (dither - 0.5f) * 2.0f * strength;
}

// 逐单元、与帧号无关的有序抖动。用IGN而不是hash白噪声：IGN在空间上是有序图案，
// 邻域钳制和TAA都能干净地把它收敛掉，白噪声则会留下颗粒。
float getCellDither(float2 cell, float layer) {
    return InterleavedGradientNoise(cell, layer);
}

// 光线步进的“层内偏移”，由两部分合成：
//   1. spatialDither —— 与帧号无关的逐单元有序抖动。它的作用是打破空间相关性。
//      原来所有体素/像素共用同一个偏移，于是雾在体素边界、雾体AABB边界上的量化位置
//      处处相同：fogAcc开时表现为块状伪影，fogAcc关时表现为整片雾逐帧同步跳动。
//   2. 逐帧推进的低差异序列 —— 时域超采样，由时域滤波/TAA积成更高的有效采样率。
// 两者用 Cranley-Patterson 旋转合成：每个单元走自己相位的同一条低差异序列，
// 既保住了序列的分层性质（不会退化成白噪声），又让邻居之间互不相关。
// advanceTemporal 为 false 时只留空间抖动：没有时域滤波去积分的话，逐帧变化就只是闪烁。
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
//FluidFogInfo getFluidFog(int i) {
//    #ifdef COMPUTE_SHADER
//    FluidFogInfo fluidFogInfo = groupFluidFogInfos[i];
//    #else
//    FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[i];
//    #endif
//    return fluidFogInfo;
//}
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

    float tPartStart = 0;
    float tPartEnd = pow(1.0f / sampleCount, stepPower) * tTotal;
    for (int sampleIndex = 0; sampleIndex <= sampleCount; ++sampleIndex) {
        //layerOffset = rand(randomSeed);
        float tPart = tPartEnd - tPartStart;
        float3 samplePoint = startPos + dir * (tPartStart + tPart * layerOffset);
        tPartStart = tPartEnd;
        tPartEnd = pow(float(sampleIndex + 1) / sampleCount, stepPower) * tTotal;

        float envExtinction = 0.0f;
        float envScattering = 0.0f;
        float envPhase = 0.0f;
        float3 envColor = 0.0f;

        if (useGlobalHeight && fogGlobalInfo.useGlobalHeightFog == 1 && samplePoint.y >= fogGlobalInfo.globalHeightFogY.x && samplePoint.y <= fogGlobalInfo.globalHeightFogY.y) {
            HeightFogInfo globalHeightFogInfo = fogGlobalInfo.globalHeightFogInfo;
            float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - fogGlobalInfo.globalHeightFogY.x));
            float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
            envExtinction = extinction;
            envScattering = extinction * globalHeightFogInfo.scattering;
            envPhase = extinction * globalHeightFogInfo.phase;
            envColor = extinction * globalHeightFogInfo.color;
        }

        for (int i = 0; i < fogGlobalInfo.volumetricFogCount; ++i) {
            VolumetricFogInfo fogInfo = getVolumetricFog(i);
            if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

            float3 fogStartPos = fogInfo.aabb.minimum;
            float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
            if (fogInfo.type == VolumetricFogType::Height) {
                HeightFogInfo heightFogInfo = getHeightFog(fogInfo.typeFogIndex);

                float heightScale = heightFogInfo.heightScale;
                float height = max(samplePoint.y - fogStartPos.y, 0.0f);
                float heightAttenuation = exp(-heightScale * height);
                scattering = heightFogInfo.scattering;
                extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;
                phase = heightFogInfo.phase;
                color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity) * 2.0f;
            } else if (fogInfo.type == VolumetricFogType::Grid) {
                GridFogInfo gridFogInfo = getGridFog(fogInfo.typeFogIndex);

                float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                extinction = voxelInfo.x;
                scattering = voxelInfo.y;
                phase = voxelInfo.z;
                color = voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
            }
            /*
            else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp[fogInfo.typeFogIndex] == 1) {
                FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
                float3 uvw = (samplePoint - fogStartPos) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
                uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
                float4 voxelInfo = FluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                // if in fluid, the voxelInfo has include the envionment fog info
                envExtinction = max(voxelInfo.x, 1e-30f);
                envScattering = envExtinction * voxelInfo.y;
                envPhase = envExtinction * voxelInfo.z;
                envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
                break;
            }
            */
            envExtinction += extinction;
            envScattering += extinction * scattering;
            envPhase += extinction * phase;
            envColor += extinction * color;
        }

        envScattering /= max(envExtinction, 1e-30f);
        envPhase /= max(envExtinction, 1e-30f);
        envColor /= max(envExtinction, 1e-30f);

        if (envExtinction <= 1e-30f) continue;

        float3 indirectL; float a;
        getFogScatter(samplePoint, dir, tPart, sceneInfo, fogGlobalInfo.lightAttenuationStrength, lightVP, envExtinction, envScattering, envPhase, envColor, 0.0f, attenuation, a, indirectL);

        attenuation *= a;
        L += luminance(indirectL);
        if (all(attenuation <= 1e-3f)) break;
    }

    attenuation2 *= fogGlobalInfo.lightAttenuationStrength;
}

void getGlobalHeightFogScatter(
    float3 samplePoint, float3 sampleDir, float sampleLength, SceneInfo sceneInfo, FogGlobalInfo globalInfo, float4x4 lightVP,
    float3 E, inout float a, inout float3 L
) {
    float2 globalHeightFogY = globalInfo.globalHeightFogY;
    if (globalInfo.useGlobalHeightFog == 1 && samplePoint.y >= globalHeightFogY.x && samplePoint.y <= globalHeightFogY.y) {
        HeightFogInfo globalHeightFogInfo = globalInfo.globalHeightFogInfo;
        float heightAttenuation = exp(-globalHeightFogInfo.heightScale * max(min(samplePoint.y, globalHeightFogY.y) - globalHeightFogY.x, 0.0f));
        float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
        float scattering = globalHeightFogInfo.scattering;
        float phase = clamp(globalHeightFogInfo.phase, -0.9999f, 0.9999f);
        float3 color = globalHeightFogInfo.color;

        getFogScatter(samplePoint, sampleDir, sampleLength, sceneInfo, globalInfo.lightAttenuationStrength, lightVP, extinction, scattering, phase, color, globalHeightFogInfo.ambientIntensity, E, a, L);
    }
}

// jitterStrength 的单位是“深度切片”，不再是世界空间距离。
// sliceDither 由调用者算好（屏幕空间IGN），传0表示不抖动。
float4 getFrustumImageInfo(float3 worldPos, bool linear, float sliceDither, Sampler3D<float4> image, inout uint randomSeed, SceneInfo *sceneInfoAddress, FrustumGlobalInfo frustumInfo) {
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
    float viewZ = dot(toPoint, cameraForward);
    float viewX = dot(toPoint, cameraRight);
    float viewY = dot(toPoint, cameraUp);
    if (viewZ <= 0.0f) return float4(0, 0, 0, 0);

    float clampedZ = clamp(viewZ, nearPlane, farPlane);
    float scale = clampedZ / viewZ;
    viewX *= scale; viewY *= scale; viewZ = clampedZ;

    float halfHeight = viewZ * tanHalfFov;
    float halfWidth = halfHeight * aspect;
    float u = saturate(viewX / halfWidth * 0.5f + 0.5f);
    float v = saturate(viewY / halfHeight * 0.5f + 0.5f);
    #ifdef Interpolation_Manual
    uint voxelZ = clamp(int(frustumInvMap(viewZ - nearPlane, compressionParams, (farPlane - nearPlane), gridSize.z) / (farPlane - nearPlane) * gridSize.z), 0, gridSize.z - 1);

    float depth = viewZ - nearPlane;
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
    // 先换算到切片空间，抖动在切片空间加，最后再归一化到[0,1]。
    // 这样抖动幅度在近处和远处都恰好是“一个体素”，不会像世界空间抖动那样
    // 近处过强（跨很多体素）远处过弱（完全不起作用）。
    float zSlice = frustumInvMap(viewZ - nearPlane, compressionParams, farPlane - nearPlane, gridSize.z) / (farPlane - nearPlane) * gridSizeF.z;
    zSlice += sliceDither;
    float w = saturate(zSlice / gridSizeF.z);
    float3 uvw = float3(u, v, w);

    if (linear) return image.SampleLevel(uvw, 0);
    else return image[uint3(uvw * gridSizeF)];
    #endif
}

// ditherPos: 屏幕空间像素坐标，用于IGN有序抖动。传0则不抖动。
void getPreAccFog(
    float3 startPos, float3 endPos,
    inout uint randomSeed, float2 ditherPos, int temporalFrameIndex, float jitterStrength,
    SceneInfo *sceneInfoAddress, FrustumGlobalInfo frustumInfo, float lightAttenuation,
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

    // 透射率和入射光必须来自同一个体素样本，否则两者在体素边界处的跳变位置不一致，
    // 会出现“亮边/暗边”的描边式锯齿。原来的实现用两个不同强度、且随机数状态已经推进过的
    // 抖动分别采样同一张图，等于人为把两者解耦了。
    float sliceDither = getFogSliceDither(ditherPos, temporalFrameIndex, jitterStrength);
    float4 fogInfo = getFrustumImageInfo(endPos, true, sliceDither, fogAccImage_sample, randomSeed, sceneInfoAddress, frustumInfo);

    // 远平面保护：视锥网格在远平面处被硬截断，超出的部分没有数据。
    // 参考引擎用FarClipProtectRange把最后20%的雾效渐隐掉，避免远处出现一条硬边。
    float viewZ = dot(endPos - cameraPos, cameraForward);
    float fadeStart = farPlane * 0.8f;
    float farClipProtect = 1.0f - saturate((viewZ - fadeStart) / max(farPlane - fadeStart, 1e-4f));

    attenuation *= lerp(1.0f, fogInfo.x, farClipProtect);
    L += fogInfo.y * farClipProtect;
    attenuation2 *= lightAttenuation;
}
