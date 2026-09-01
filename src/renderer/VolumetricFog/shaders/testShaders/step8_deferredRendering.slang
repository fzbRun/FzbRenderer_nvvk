#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eAlbedoImage)]] Texture2D<float4> albedoImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eEmissiveImage)]] Texture2D<float4> emissiveImage;

[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
#ifdef BLUR_FOG
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedFogResultImage)]] RWTexture2D<float4> renderedFogResultImage;
#endif

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_deferredRendering(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 screenSize = pushConst.screenSize;
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    uint groupThreadIndexU = groupThreadIndex.y * 16 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = groupSceneInfo;

    float depth = depthImage[threadIndex];
    float2 uv = (float2(threadIndex) + 0.5f) / float2(screenSize);
    float2 ndc = uv * 2.0f - 1.0f;
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos = mul(mul(clipPos, groupGlobalInfo.invProjMatrix_taaJitter), sceneInfo.viewInvMatrix);
    worldPos /= worldPos.w;

    float attenuation = 1.0f, attenuation2 = 1.0f, L = 0.0f;
    // 用一直递增的帧号做种子，而不是墙上时钟。原来的 time * 1000 会让相机静止时噪声
    // 也每帧变化，直接表现为画面“波动”；temporalFrameIndex 在相机移动时也不会被清零，
    // 所以抖动序列始终连续，时域滤波才能收敛。
    uint randomSeed = xxhash32(uint3(groupGlobalInfo.temporalFrameIndex, threadIndex));
    if (pushConst.useAccFog == 1) getPreAccFog(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, float2(threadIndex), L, attenuation, attenuation2);
    else {
        // 不走fogAcc时是逐像素直接步进，没有任何时域滤波在后面收敛（TAA只作用于最终颜色，
        // 且相机移动时会被clamp住），所以层内偏移只能用“与帧号无关”的逐像素有序抖动：
        // 空间上打散量化边界，时间上完全静止。原来这里用的是随帧推进的全局偏移，
        // 所有像素同时跳到下一个偏移量 —— 整片雾逐帧同步明暗变化，就是烟雾抖动的来源。
        float spatialDither = getCellDither(float2(threadIndex), 0.0f);
        #ifdef USE_ENVFOG
        getEnvVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, spatialDither, false, L, attenuation, attenuation2);
        #else
        getVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, 3, true, spatialDither, true, L, attenuation, attenuation2);
        #endif
    }

    #ifdef BLUR_FOG
    if (pushConst.useFogBlur == 1) {
        renderedFogResultImage[threadIndex] = float4(attenuation, L, 1.0f, 1.0f);
        L = 0.0f;
        attenuation = 1.0f;
    }
    #endif

    float3 result = 0.0f;
    if (depth >= 1.0f - 1e-30f)
        result = sceneInfo.useSky == 1 ? (renderedImage[threadIndex].xyz * attenuation + L) : (sceneInfo.backgroundColor * attenuation + L);
    else {
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

        result = (ambient + (diffuse + specular) * shadow * attenuation2 + emissive) * attenuation + L;
    }

    //depth = dot(worldPos.xyz - sceneInfo.cameraPosition, -sceneInfo.viewInvMatrix[2].xyz) - pushConst.cameraNearPlane;
    //int voxelIndex = int(frustumInvMap(depth / (pushConst.cameraFarPlane - pushConst.cameraNearPlane), pushConst.compressionParams, 1.0f) * pushConst.frustumGridSize.z);
    //result = 0.0f;
    //result = voxelIndex & 1;

    // result = attenuation;
    // if (all(threadIndex > (screenSize / 2 - 10)) && all(threadIndex < (screenSize / 2 + 10))) {
    //     if (pushConst.frameIndex == 0) printf("L: %f, a: %f\n", L, attenuation);
    //     result = float3(10.0f, 0.0f, 0.0f);
    // }
    renderedImage[threadIndex] = float4(result, 1.0f);
}
