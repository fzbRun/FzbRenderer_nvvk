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