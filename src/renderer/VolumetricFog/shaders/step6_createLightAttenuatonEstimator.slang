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
    groupSceneInfo = sceneInfo;

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
    float3 endPos = cameraPos + cameraForward * (pushConst.cameraFarPlane - pushConst.cameraNearPlane) * 0.5f;
    float3 mainLightDir = sceneInfo.lights[0].direction;
    float3 startPos = endPos - mainLightDir * 10000.0f;

    float tEntry, tExit;
    bool intersection = IntersectFrustum(startPos, endPos, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.cameraNearPlane, pushConst.cameraFarPlane, tEntry, tExit);

    if (intersection) {
        lightAttenuationLength = tExit - tEntry;
        float tPart = lightAttenuationLength / pushConst.sampleCount;
        uint randomSeed = xxhash32(uint3(threadIndex, pushConst.frameIndex, 0));
        //if (pushConst.frameIndex < 10) printf("\nframe: %d\n", pushConst.frameIndex);
        for (int sampleIndex = 0; sampleIndex < pushConst.sampleCount; ++sampleIndex) {
            float tSample = tEntry + tPart * (sampleIndex + 0.5f);
            float3 samplePoint = startPos + mainLightDir * tSample;

            float4 voxelInfo = getFrustumImageInfo(samplePoint, true, false, envVolumetricFogInfoImage_sample, randomSeed);
            /*
            {
                float3 worldPos = samplePoint;

                float nearPlane = pushConst.cameraNearPlane;
                float farPlane = pushConst.cameraFarPlane;
                float3 compressionParams = pushConst.compressionParams;
                float tanHalfFov = pushConst.tanCameraFov_2;
                float aspect = pushConst.aspectRatio;
                uint3 gridSize = pushConst.frustumGridSize;
                float3 gridSizeF = float3(gridSize);

                float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
                float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
                float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

                float3 toPoint = worldPos - sceneInfo.cameraPosition;
                float viewZ = dot(toPoint, cameraForward);
                float viewX = dot(toPoint, cameraRight);
                float viewY = dot(toPoint, cameraUp);

                float clampedZ = clamp(viewZ, nearPlane, farPlane);
                float scale = clampedZ / viewZ;
                viewX *= scale; viewY *= scale; viewZ = clampedZ;

                float halfHeight = viewZ * tanHalfFov;
                float halfWidth = halfHeight * aspect;
                float u = viewX / halfWidth * 0.5f + 0.5f;
                float v = viewY / halfHeight * 0.5f + 0.5f;
                float w = frustumInvMap(viewZ - nearPlane, compressionParams, farPlane - nearPlane) / (farPlane - nearPlane);

                uint3 voxelIndex = uint3(float3(u, v, w) * gridSize);
                if (pushConst.frameIndex < 10 && sampleIndex == 0) {
                    printf("uvw: %f %f %f voxelIndex: %d %d %d\n samplePoint: %f %f %f,  ext: %f\n", 
                        u, v, w, voxelIndex.x, voxelIndex.y, voxelIndex.z, 
                        samplePoint.x, samplePoint.y, samplePoint.z,
                        voxelInfo.x
                    );
                    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
                        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                            for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                                uint3 offsetVoxelIndex = voxelIndex + uint3(offsetX, offsetY, offsetZ);
                                float ext = envVolumetricFogInfoImage_sample[offsetVoxelIndex].x;
                                printf("offsetVoxelIndex: %d %d %d, ext: %f\n", offsetVoxelIndex.x, offsetVoxelIndex.y, offsetVoxelIndex.z, ext);
                            }
                        }
                    }
                }      
            }
            */
            lightAttenuation *= exp(-voxelInfo.x * tPart);
            if (lightAttenuation <= 0.0f) break;
        }
    }
#endif
    //printf("%f %f\n", lightAttenuation, lightAttenuationLength);
    globalInfoBuffer[0].envFogLightAttenuationEstimator = lightAttenuation;
    globalInfoBuffer[0].envFogLightAttenuationLength = lightAttenuationLength;
}
