#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_createLightAttenuationEstimator(uint threadIndex: SV_DispatchThreadID) {
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;

    float lightAttenuation = 1.0f;
    float lightAttenuationLength = 1.0f;
#ifdef Uniform_EnvFog_Grid
    float tEntry = 0.0f;
    float tExit = length(globalInfo.envGridSize * globalInfo.envVoxelSize);
    float3 startPos = globalInfo.envStartPos;
    float3 endPos = startPos + globalInfo.envGridSize * globalInfo.envVoxelSize;
    float3 dir = normalize(endPos - startPos);

    lightAttenuationLength = tExit - tEntry;
    float tPart = lightAttenuationLength / pushConst.sampleCount;
    uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex, 0));
    for (int sampleIndex = 0; sampleIndex < pushConst.sampleCount; ++sampleIndex) {
        float tSample = tEntry + tPart * (sampleIndex + 0.5f);
        float3 samplePoint = startPos + dir * tSample;

        float3 uvw = (samplePoint - startPos) / (globalInfo.envGridSize * globalInfo.envVoxelSize);
        float4 voxelInfo = envVolumetricFogInfoImage_sample.SampleLevel(uvw, 0);
        lightAttenuation *= exp(-voxelInfo.x * tPart);
        if (lightAttenuation <= 0.0f) break;
    }
#else
    float3 mainLightDir = sceneInfo.lights[0].direction;
    float3 mainLightPos = cameraPos + cameraForward * pushConst.cameraFarPlane - mainLightDir * 10000.0f;

    float tEntry, tExit;
    bool intersection = rayFrustumIntersection(mainLightPos, mainLightDir, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.cameraNearPlane, pushConst.cameraFarPlane, tEntry, tExit);

    if (intersection) {
        lightAttenuationLength = tExit - tEntry;
        float tPart = lightAttenuationLength / pushConst.sampleCount;
        uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex, 0));
        for (int sampleIndex = 0; sampleIndex < pushConst.sampleCount; ++sampleIndex) {
            float tSample = tEntry + tPart * (sampleIndex + 0.5f);
            float3 samplePoint = mainLightPos + mainLightDir * tSample;

            float4 voxelInfo = getFrustumImageInfo(samplePoint, true, false, envVolumetricFogInfoImage_sample, randomSeed);
            lightAttenuation *= exp(-voxelInfo.x * tPart);
            if (lightAttenuation <= 0.0f) break;
        }
    }
#endif
    globalInfoBuffer[0].envFogLightAttenuationEstimator = lightAttenuation;
    globalInfoBuffer[0].envFogLightAttenuationLength = lightAttenuationLength;
}
