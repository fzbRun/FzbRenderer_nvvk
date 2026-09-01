#ifndef SHADER_SOURCE_SLANG
#define SHADER_SOURCE_SLANG

#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#ifndef SHADER_SOURCE
#define SHADER_SOURCE

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eHeightFogInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages_sampler)]] Sampler3D<float4> FluidFogVoxelInfoImages_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogInfoBuffer)]] StructuredBuffer<GridFogInfo, ScalarDataLayout> gridFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogImages)]] Sampler3D<float4> gridFogImages[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccResultImage)]] RWTexture3D<float4> volumetricFogAccResultImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccResultImage_sample)]] Sampler3D<float4> volumetricFogAccResultImage_sample;

[[vk::binding(StaticBindingPoints_VolumetricFog::eEnvFogInfoImage)]] RWTexture3D<float4> envVolumetricFogInfoImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eEnvFogInfoImage_sample)]] Sampler3D<float4> envVolumetricFogInfoImage_sample;

[[vk::binding(StaticBindingPoints_VolumetricFog::eShadowMap)]] Sampler2DShadow shadowMap;

#ifdef COMPUTE_SHADER
groupshared GlobalInfo_VolumetricFog groupGlobalInfo;
groupshared SceneInfo groupSceneInfo;
groupshared VolumetricFogInfo groupVolumetricFogInfos[MAX_VOLUMETRIC_FOG_COUNT];
groupshared HeightFogInfo groupHeightFogInfos[MAX_HEIGHT_FOG_COUNT];
groupshared FluidFogInfo groupFluidFogInfos[MAX_FLUID_FOG_COUNT];
groupshared GridFogInfo grouGridFogInfos[MAX_GRID_FOG_COUNT];

void getGroupsharedData(uint groupThreadIndex) {
    if (groupThreadIndex >= MAX_VOLUMETRIC_FOG_COUNT) return;

    if (groupThreadIndex == 0) {
        groupGlobalInfo = globalInfoBuffer[0];
        groupSceneInfo = pushConst.sceneInfoAddress[0];
    }
    if (groupThreadIndex < min(pushConst.volumetricFogCount, MAX_VOLUMETRIC_FOG_COUNT)) groupVolumetricFogInfos[groupThreadIndex] = fogInfosBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.heightFogCount, MAX_HEIGHT_FOG_COUNT)) groupHeightFogInfos[groupThreadIndex] = heightFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.fluidFogCount, MAX_FLUID_FOG_COUNT)) groupFluidFogInfos[groupThreadIndex] = fluidFogInfoBuffer[groupThreadIndex];
    if (groupThreadIndex < min(pushConst.gridFogCount, MAX_GRID_FOG_COUNT)) grouGridFogInfos[groupThreadIndex] = gridFogInfoBuffer[groupThreadIndex];
}
#endif
#endif

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
float getMarchLayerOffset(uint temporalFrameIndex, float spatialDither, bool advanceTemporal) {
    if (pushConst.randomStepping != 1) return 0.5f;
#ifdef FOG_LEGACY_LAYER_OFFSET
    // 仅用于A/B对比：复现修复前的“全局同相位”偏移（所有体素/像素共用一个值）。
    return radicalInverse(temporalFrameIndex % 16u + 1u, 2u);
#else
    if (!advanceTemporal) return frac(spatialDither);
    return frac(radicalInverse(temporalFrameIndex % 16u + 1u, 2u) + spatialDither);
#endif
}

float getShadow(float4 worldPos) {
    float4 lightMapNdcPos = mul(worldPos, pushConst.lightVP);
    if (abs(lightMapNdcPos.w) < 1e-6f) return 1.0f;
    lightMapNdcPos /= lightMapNdcPos.w;

    float2 screenPos = lightMapNdcPos.xy * 0.5f + 0.5f;
    if (any(screenPos < 0.0f) || any(screenPos > 1.0f) || lightMapNdcPos.z < 0.0f || lightMapNdcPos.z > 1.0f) return 1.0f;

    float shadow = shadowMap.SampleCmpLevel(screenPos, lightMapNdcPos.z, 0);
    return shadow;
}

float getAttenuation2(float3 endPos, GlobalInfo_VolumetricFog globalInfo, SceneInfo sceneInfo) {
    /*
    float tLightEnter, tLightExit;
    float mainLightDt = 1e-6f;
    float3 mainLightPos = endPos - sceneInfo.lights[0].direction * 1000.0f;
    #ifdef Uniform_EnvFog_Grid
    float3 gridMin = globalInfo.envStartPos;
    float3 gridMax = globalInfo.envStartPos + float3(globalInfo.envGridSize) * globalInfo.envVoxelSize;
    if (isHit(mainLightPos, endPos, gridMin, gridMax, tLightEnter, tLightExit))
    #else
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    if (IntersectFrustum(mainLightPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tLightEnter, tLightExit))
    #endif
        mainLightDt = tLightExit - tLightEnter;

    float attenuationLength = max(globalInfo.envFogLightAttenuationLength, 1e-6f);
    float attenuationRatio = max(mainLightDt / attenuationLength, 0.0f);
    float lightAttenuation = pow(max(globalInfo.envFogLightAttenuationEstimator, 1e-6f), attenuationRatio);
    return lightAttenuation * pushConst.lightAttenuationStrength;
    */
    return pushConst.lightAttenuationStrength;
}

void getFogScatter(
    float3 samplePoint, float3 sampleDir, float sampleLength, SceneInfo sceneInfo, GlobalInfo_VolumetricFog globalInfo,
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

    float3 activeSunColor = lightColor * getAttenuation2(samplePoint, globalInfo, sceneInfo);
    activeSunColor *= getShadow(float4(samplePoint, 1.0f));

    float x = extinctionCoeff * sampleLength;
    float factor = (x < 1e-4) ? sampleLength * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
    L = activeSunColor * fp * color * scattering * E * factor;
    L += color * ambientIntensity * scattering * E * factor;
    a = exp(-extinctionCoeff * sampleLength);
}
void getGlobalHeightFogScatter(
    float3 samplePoint, float3 sampleDir, float sampleLength, SceneInfo sceneInfo, GlobalInfo_VolumetricFog globalInfo,
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

        getFogScatter(samplePoint, sampleDir, sampleLength, sceneInfo, globalInfo, extinction, scattering, phase, color, globalHeightFogInfo.ambientIntensity, E, a, L);
    }
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

#endif