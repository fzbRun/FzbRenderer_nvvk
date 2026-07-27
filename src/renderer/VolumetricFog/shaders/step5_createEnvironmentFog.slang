#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createEnvionmentFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;

#ifdef Uniform_EnvFog_Grid
    if (any(voxelIndex >= globalInfo.envGridSize)) return;
    float3 voxelCenter = globalInfo.envStartPos + (float3(voxelIndex) + 0.5f) * globalInfo.envVoxelSize;
    uint3 gridSize = globalInfo.envGridSize;

    float3 voxelDirX = float3(1.0f, 0.0f, 0.0f) * globalInfo.envVoxelSize;
    float3 voxelDirY = float3(0.0f, 1.0f, 0.0f) * globalInfo.envVoxelSize;
    float3 voxelDirZ = float3(0.0f, 0.0f, 1.0f) * globalInfo.envVoxelSize;
    float3 voxelStartPos = voxelCenter - globalInfo.envVoxelSize * 0.5f;
#else
    if (any(voxelIndex >= pushConst.frustumGridSize)) return;
    
    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionPrecision = pushConst.compressionParams;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);
    
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;
    
    float depth_near = frustumMap(float(voxelIndex.z) / gridSize.z * (farPlane - nearPlane), compressionPrecision, farPlane - nearPlane);
    float depth_far = frustumMap(float(voxelIndex.z + 1) / gridSize.z * (farPlane - nearPlane), compressionPrecision, farPlane - nearPlane);
    
    float depth = (depth_near + depth_far) * 0.5f;
    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWeight = halfHeight * aspect;
    float3 voxelCenter = cameraPos + (depth + nearPlane) * cameraForward +
                         ((float(voxelIndex.x) + 0.5f) / gridSize.x * 2.0f - 1.0f) * halfWeight * cameraRight +
                         ((float(voxelIndex.y) + 0.5f) / gridSize.y * 2.0f - 1.0f) * halfHeight * cameraUp;
    
    float voxelHalfHeight = halfHeight / gridSize.y;
    float voxelHalfWidth = halfWeight / gridSize.x;
    float voxelHalfDepth = (depth_far - depth_near) * 0.5f; 
    
    float3 voxelDirX = cameraRight * (2.0f * voxelHalfWidth);
    float3 voxelDirY = cameraUp * (2.0f * voxelHalfHeight);
    float3 voxelDirZ = cameraForward * (2.0f * voxelHalfDepth);
    
    float3 voxelStartPos = voxelCenter - voxelHalfWidth * cameraRight - voxelHalfHeight * cameraUp - voxelHalfDepth * cameraForward;
#endif

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));
    float3x3 rotateMatrix = float3x3(1.0f); //randomRotationMatrixUniform(randomSeed);

    uint sampleCount = pushConst.sampleCount;
    float4 envVoxelInfo = 0.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float3 samplePoint = voxelCenter;
        if (sampleCount > 1) {
            float3 sampleDir = mul(fibSpherePoint(sampleIndex, sampleCount), rotateMatrix);
            AABB voxelAABB = { voxelStartPos, voxelStartPos + voxelDirX + voxelDirY + voxelDirZ };
            samplePoint = getAABBPoint(voxelAABB, voxelCenter, sampleDir);
            samplePoint = lerp(voxelCenter, samplePoint, rand(randomSeed));
        }
    
        float heightAttenuation = exp(-1.0f * max(samplePoint.y - (-69.5f), 0.0f));
        float extinction = (0.01f + 0.03f) * heightAttenuation;
        float envExtinction = extinction;
        float envScattering = extinction * 0.03f;
        float envPhase = extinction * -0.7f;
        float3 envColor = extinction * 1.0f;
    
        for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
            uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];
            VolumetricFogInfo fogInfo = groupVolumetricFogInfos[fogIndex];
            AABB fogAABB = { fogInfo.fogStartPos, fogInfo.fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
            if (!checkPointInAABB(samplePoint, fogAABB)) continue;
    
            extinction = 0.0f;
            float scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
            if (fogInfo.type == VolumetricFogType::Height) {
               float heightScale = groupHeightFogInfos[fogInfo.volumetricFogTypeIndex].heightScale;
               float height = max(samplePoint.y - fogInfo.fogStartPos.y, 0.0f);
               heightAttenuation = exp(-heightScale * height);
               scattering = fogInfo.scattering;
               extinction = (fogInfo.absorption.x + scattering) * heightAttenuation;
               phase = fogInfo.phase;
               color = fogInfo.color;
            } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp == 1) {
               float3 uvw = (samplePoint - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
               uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
               float4 voxelInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0);

               // if in fluid, the voxelInfo has include the envionment fog info
               envExtinction = max(voxelInfo.x, 1e-30f);
               envScattering = envExtinction * voxelInfo.y;
               envPhase = envExtinction * voxelInfo.z;
               envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
               break;
            }else if (fogInfo.type == VolumetricFogType::Noise) {
                NoiseFogInfo noiseFogInfo = groupNoiseFogInfos[fogInfo.volumetricFogTypeIndex];
                CloudSetting cloudSetting = { noiseFogInfo.cloudScale, noiseFogInfo.cloudFlowSpeed, noiseFogInfo.cloudCoverage, noiseFogInfo.cloudTypePreference, noiseFogInfo.weatherScale };
                float fogDensity = SampleCloudDensity(samplePoint, fogAABB.minimum, fogAABB.maximum, pushConst.time, cloudSetting);
                scattering = fogInfo.scattering;
                extinction = (fogInfo.absorption.x + scattering) * fogDensity;
                phase = fogInfo.phase;
                color = fogInfo.color;
            }
    
            envExtinction += extinction;
            envScattering += extinction * scattering;
            envPhase += extinction * phase;
            envColor += extinction * color;
        }
    
        envScattering /= max(envExtinction, 1e-30f);
        envPhase /= max(envExtinction, 1e-30f);
        envColor /= max(envExtinction, 1e-30f);
    
        envVoxelInfo += float4(envExtinction, envScattering, envPhase, luminance(envColor));
    }
    envVoxelInfo /= sampleCount;

#ifdef Uniform_EnvFog_Grid
   envVolumetricFogInfoImage[voxelIndex] = (envVolumetricFogInfoImage[voxelIndex] + envVoxelInfo) * 0.5f;
#else
   envVolumetricFogInfoImage[voxelIndex] = envVoxelInfo;
#endif
}
