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
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedFogImage)]] RWTexture2D<float4> renderedFogImage;
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
    float4 worldPos = mul(mul(clipPos, sceneInfo.projInvMatrix), sceneInfo.viewInvMatrix);
    worldPos /= worldPos.w;

    float3 attenuation = 1.0f, attenuation2 = 1.0f, L = 0.0f;
    uint randomSeed = xxhash32(uint3(int(pushConst.time * 1000.0f), threadIndex));
    if (pushConst.useAccFog == 1) getPreAccFog(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, L, attenuation, attenuation2);
    else {
        #ifdef USE_ENVFOG
        getEnvVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, true, 1, L, attenuation, attenuation2);
        #else
        getVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, L, attenuation, attenuation2);
        #endif
    }

    #ifdef BLUR_FOG
    if (pushConst.useFogBlur == 1) {
        renderedFogImage[threadIndex] = float4(L, 1.0f);
        L = 0.0f;
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
    renderedImage[threadIndex] = float4(result, 1.0f);
}
