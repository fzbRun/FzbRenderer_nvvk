#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/finalShaders/FogCommon.slang"

[[vk::push_constant]] ConstantBuffer<renderOpaquePushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedFogResultImage)]] RWTexture2D<float4> renderedFogResultImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_renderOpaqueMaterial(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 screenSize = pushConst.screenSize;
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    uint groupThreadIndexU = groupThreadIndex.y * 16 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress);
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = groupSceneInfo;
    float4x4 projInvMatrix = sceneInfo.projInvMatrix;

    float depth = depthImage[threadIndex];
    if (pushConst.useTAA == 1 && depth < 1.0f - 1e-30f) projInvMatrix = pushConst.projInvMatrix_taa;

    float2 uv = (float2(threadIndex) + 0.5f) / float2(screenSize);
    float2 ndc = uv * 2.0f - 1.0f;
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos = mul(mul(clipPos, projInvMatrix), sceneInfo.viewInvMatrix);
    worldPos /= worldPos.w;

    uint randomSeed = xxhash32(uint3(pushConst.temporalFrameIndex, threadIndex));

    float attenuation = 1.0f, L = 0.0f, attenuation2 = 1.0f;
    if (pushConst.useFogAcc == 1)
        getPreAccFog(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, float2(threadIndex), pushConst.temporalFrameIndex, pushConst.jitterStrength, pushConst.sceneInfoAddress, pushConst.frustumInfo, L, attenuation, attenuation2);
    else {
        float spatialDither = getCellDither(float2(threadIndex), 0.0f);
        float layerOffset = getMarchLayerOffset(pushConst.temporalFrameIndex, spatialDither, true, true);
        getVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, 3, true, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
        //if (depth < 1.0f - 1e-30f) {
        //    //getVolumetricFogScatter_DeltaTracking(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, 3, true, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
        //    getVolumetricFogScatter_RatioTracking(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, pushConst.sampleCount, 3, true, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
        //}
    }

    if (pushConst.useFogBlur == 1) {
        renderedFogResultImage[threadIndex] = float4(attenuation, L, 1.0f, 1.0f);
        L = 0.0f;
        attenuation = 1.0f;
    }

    float3 result = 0.0f;
    if (depth >= 1.0f - 1e-30f)
        result = sceneInfo.useSky == 1 ? (renderedImage[threadIndex].xyz * attenuation + L) : (sceneInfo.backgroundColor * attenuation + L);
    else {
        float shadow = getShadow(worldPos, pushConst.lightVP);

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