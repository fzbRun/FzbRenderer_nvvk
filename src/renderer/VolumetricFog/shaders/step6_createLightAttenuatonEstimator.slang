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

    float3 mainLightDir = sceneInfo.lights[0].direction;
    float3 mainLightPos = cameraPos + cameraForward * pushConst.cameraFarPlane - mainLightDir * 10000.0f;

    float tEntry, tExit;
    bool intersection = rayFrustumIntersection(mainLightPos, mainLightDir, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.cameraNearPlane, pushConst.cameraFarPlane, tEntry, tExit);

    float lightAttenuation = 1.0f;
    float lightAttenuationLength = 1.0f;
    if (intersection) {
        lightAttenuationLength = tExit - tEntry;
        float tPart = lightAttenuationLength / pushConst.sampleCount;
        uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex, 0));
        for (int sampleIndex = 0; sampleIndex < pushConst.sampleCount; ++sampleIndex) {
            float tSample = tEntry + tPart * (sampleIndex + rand(randomSeed));
            float3 samplePoint = mainLightPos + mainLightDir * tSample;

            float3 hitWorldPos, uvw;
            getCameraFrustumData(samplePoint, sceneInfo.cameraPosition, sceneInfo.viewInvMatrix, pushConst.frustumGridSize, 
                pushConst.cameraNearPlane, pushConst.cameraFarPlane, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.compressionPrecision, 0, hitWorldPos, uvw);

            float4 voxelInfo = getFrustumImageInfo(samplePoint, true, envVolumetricFogInfoImage, 0);

            lightAttenuation *= exp(-voxelInfo.x * tPart);
            if (lightAttenuation <= 0.0f) break;
        }
    }

    globalInfoBuffer[0].envFogLightAttenuationEstimator = lightAttenuation;
    globalInfoBuffer[0].envFogLightAttenuationLength = lightAttenuationLength;
    //printf("a: %f, l: %f\n", lightAttenuation, lightAttenuationLength);
}
