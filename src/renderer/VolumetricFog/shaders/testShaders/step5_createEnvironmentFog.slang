#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

float4 GetLocalFogMask(float3 worldPos, GlobalInfo_VolumetricFog globalInfo) {
    float envExtinction = 0.0f;
    float envScattering = 0.0f;
    float envPhase = 0.0f;
    float3 envColor = 0.0f;

    bool saveGlobalHeightFog = false;
    #ifdef FOG_ACC_DELETE_NOFOGVOXEL
    saveGlobalHeightFog = pushConst.useAccFog == 0;
    #endif
    saveGlobalHeightFog &= globalInfo.useGlobalHeightFog == 1 && worldPos.y >= globalInfo.globalHeightFogY.x && worldPos.y <= globalInfo.globalHeightFogY.y;
    if (saveGlobalHeightFog) {
        HeightFogInfo globalHeightFogInfo = globalInfo.globalHeightFogInfo;
        float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (worldPos.y - globalInfo.globalHeightFogY.x));
        float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
        envExtinction = extinction;
        envScattering = extinction * globalHeightFogInfo.scattering;
        envPhase = extinction * globalHeightFogInfo.phase;
        envColor = extinction * globalHeightFogInfo.color;
    }

    for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
        uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];
        VolumetricFogInfo fogInfo = groupVolumetricFogInfos[fogIndex];
        if (!checkPointInAABB(worldPos, fogInfo.aabb)) continue;

        float3 fogStartPos = fogInfo.aabb.minimum;

        float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
        if (fogInfo.type == VolumetricFogType::Height) {
            HeightFogInfo heightFogInfo = groupHeightFogInfos[fogInfo.typeFogIndex];
            float heightScale = heightFogInfo.heightScale;
            float height = max(worldPos.y - fogStartPos.y, 0.0f);
            float heightAttenuation = exp(-heightScale * height);
            scattering = heightFogInfo.scattering;
            extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;
            phase = heightFogInfo.phase;
            color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
        } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp[fogInfo.typeFogIndex] == 1) {
            FluidFogInfo fluidFogInfo = groupFluidFogInfos[fogInfo.typeFogIndex];
            float3 uvw = (worldPos - fogStartPos) / (fluidFogInfo.gridSize * fluidFogInfo.voxelSize);
            uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
            float4 voxelInfo = FluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

            // if in fluid, the voxelInfo has include the envionment fog info
            envExtinction = max(voxelInfo.x, 1e-30f);
            envScattering = envExtinction * voxelInfo.y;
            envPhase = envExtinction * voxelInfo.z;
            envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
            break;
        }
        else if (fogInfo.type == VolumetricFogType::Grid) {
            GridFogInfo gridFogInfo = gridFogInfoBuffer[fogInfo.typeFogIndex];

            float3 uvw = (worldPos - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
            float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

            extinction = voxelInfo.x;
            scattering = voxelInfo.y;
            phase = voxelInfo.z;
            color = voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
        }

        envExtinction += extinction;
        envScattering += extinction * scattering;
        envPhase += extinction * phase;
        envColor += extinction * color;
    }

    envScattering /= max(envExtinction, 1e-30f);
    envPhase /= max(envExtinction, 1e-30f);
    envColor /= max(envExtinction, 1e-30f);

    return float4(envExtinction, envScattering, envPhase, luminance(envColor));
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createEnvionmentFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;

    if (any(voxelIndex >= pushConst.frustumGridSize)) return;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = pushConst.compressionParams;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float3 voxelCenter = cameraPos, voxelCenter_lastVoxel = cameraPos;
    {
        uint depthIndex = voxelIndex.z;
        float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
        float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);

        float halfHeight = (depth + nearPlane) * tanHalfFov;
        float halfWidth = halfHeight * aspect;
        voxelCenter += (depth + nearPlane) * cameraForward;
        voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
        voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;
        float3 dir = normalize(voxelCenter - cameraPos);

        float linearDepth_lastVoxel = max((float)depthIndex - 0.5f, 0.0f) / gridSize.z * (farPlane - nearPlane);
        float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane, gridSize.z);

        float halfHeight_prev = (depth_lastVoxel + nearPlane) * tanHalfFov;
        float halfWidth_prev = halfHeight_prev * aspect;
        voxelCenter_lastVoxel += (depth_lastVoxel + nearPlane) * cameraForward;
        voxelCenter_lastVoxel += halfHeight_prev * (((float)voxelIndex.y + 0.5f) / gridSize.y * 2.0f - 1.0f) * cameraUp;
        voxelCenter_lastVoxel += halfWidth_prev * (((float)voxelIndex.x + 0.5f) / gridSize.x * 2.0f - 1.0f) * cameraRight;
    }
    float densityFixup = length(voxelCenter - voxelCenter_lastVoxel);

    float4 envVoxelInfo = 0.0f;
    {
        float3 uvw = (float3(voxelIndex) + 0.5f) / gridSize;
        const int N = 3;
        const float invN = 1.0f / N;
        float3 rcpGridSize = 1.0f / gridSize;
        float3 ditherSeed = float3(voxelIndex) + pushConst.frameIndex * 0.618;
        [unroll] for (int sz = 0; sz < N; sz++) {
            float tz = ((float)sz + 0.5f) * invN - 0.5f;
            [unroll] for (int sy = 0; sy < N; sy++) {
                float ty = ((float)sy + 0.5f) * invN - 0.5f;
                [unroll] for (int sx = 0; ((sx) < (N)); (sx++)) {
                    float tx = ((float)sx + 0.5f) * invN - 0.5f;
                    float3 sampleSeed = ditherSeed + float3((float)sx, (float)sy, (float)sz) * 7.31f;
                    float3 jitter = ((frac(((sin(float3(dot(sampleSeed, float3(127.1, 311.7, 74.7)), dot(sampleSeed, float3(269.5, 183.3, 246.1)), dot(sampleSeed, float3(113.5, 271.9, 124.6))))) * (43758.5453)))) - (0.5));
                    float3 subUV = (uvw + (float3(((((tx) + (((((jitter.x) * (invN))) * (1.5))))) * (rcpGridSize.x)), ((((ty) + (((((jitter.y) * (invN))) * (1.5))))) * (rcpGridSize.y)), ((((tz) + (((((jitter.z) * (invN))) * (1.5))))) * (rcpGridSize.z)))));
                    float3 subWorldPos = getFrustumVoxelPos(subUV, compressionParams, gridSize.z, nearPlane, farPlane, tanHalfFov, aspect, cameraPos, cameraForward, cameraUp, cameraRight);
                    envVoxelInfo += GetLocalFogMask(subWorldPos, globalInfo);
                }
            }
        }

        envVoxelInfo /= N * N * N;
    }

    envVolumetricFogInfoImage[voxelIndex] = envVoxelInfo;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createEnvionmentFog2(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
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

    uint sampleCount = pushConst.sampleCount;
#else
    if (any(voxelIndex >= pushConst.frustumGridSize)) return;
    
    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = pushConst.compressionParams;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);
    
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float linearDepth = ((float)voxelIndex.z + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);
    uint sampleCount = pushConst.sampleCount;   //max(int(pushConst.sampleCount * lerp(0.1, 1, depth / (farPlane - nearPlane))), 1);
#endif
    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));

    float4 envVoxelInfo = 0.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float3 localSample = 0.5f;
        if (sampleCount > 1) {
            localSample = float3(rand(randomSeed), rand(randomSeed), rand(randomSeed));
            localSample = lerp(0.5f.xxx, localSample, saturate(pushConst.jitterStrength0 * 0.1f));
        }

#ifdef Uniform_EnvFog_Grid
        float3 samplePoint = voxelStartPos + voxelDirX * localSample.x + voxelDirY * localSample.y + voxelDirZ * localSample.z;
#else
        float sampleU = (float(voxelIndex.x) + localSample.x) / gridSizeF.x;
        float sampleV = (float(voxelIndex.y) + localSample.y) / gridSizeF.y;
        float sampleW = (float(voxelIndex.z) + localSample.z) / gridSizeF.z;

        float sampleDepth = nearPlane + frustumMap(sampleW * (farPlane - nearPlane), compressionParams, farPlane - nearPlane, gridSize.z);
        float sampleHalfHeight = sampleDepth * tanHalfFov;
        float sampleHalfWidth = sampleHalfHeight * aspect;

        float3 samplePoint = cameraPos + cameraRight * ((sampleU * 2.0f - 1.0f) * sampleHalfWidth) + cameraUp * ((sampleV * 2.0f - 1.0f) * sampleHalfHeight) + cameraForward * sampleDepth;
#endif

        float envExtinction = 0.0f;
        float envScattering = 0.0f;
        float envPhase = 0.0f;
        float3 envColor = 0.0f;

#ifdef FOG_ACC_DELETE_NOFOGVOXEL
        bool saveGlobalHeightFog = pushConst.useAccFog == 1;
#else
        bool saveGlobalHeightFog = globalInfo.useGlobalHeightFog == 1 && samplePoint.y >= globalInfo.globalHeightFogY.x && samplePoint.y <= globalInfo.globalHeightFogY.y;
#endif
        if (saveGlobalHeightFog) {
            HeightFogInfo globalHeightFogInfo = globalInfo.globalHeightFogInfo;
            float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (samplePoint.y - globalInfo.globalHeightFogY.x));
            float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
            envExtinction = extinction;
            envScattering = extinction * globalHeightFogInfo.scattering;
            envPhase = extinction * globalHeightFogInfo.phase;
            envColor = extinction * globalHeightFogInfo.color;
        }

        for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
            uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];
            VolumetricFogInfo fogInfo = groupVolumetricFogInfos[fogIndex];
            if (!checkPointInAABB(samplePoint, fogInfo.aabb)) continue;

            float3 fogStartPos = fogInfo.aabb.minimum;

            float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
            if (fogInfo.type == VolumetricFogType::Height) {
                HeightFogInfo heightFogInfo = groupHeightFogInfos[fogInfo.typeFogIndex];
                float heightScale = heightFogInfo.heightScale;
                float height = max(samplePoint.y - fogStartPos.y, 0.0f);
                float heightAttenuation = exp(-heightScale * height);
                scattering = heightFogInfo.scattering;
                extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;
                phase = heightFogInfo.phase;
                color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
            } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp[fogInfo.typeFogIndex] == 1) {
                FluidFogInfo fluidFogInfo = groupFluidFogInfos[fogInfo.typeFogIndex];
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
            else if (fogInfo.type == VolumetricFogType::Grid) {
                GridFogInfo gridFogInfo = gridFogInfoBuffer[fogInfo.typeFogIndex];

                float3 uvw = (samplePoint - fogInfo.aabb.minimum) / (fogInfo.aabb.maximum - fogInfo.aabb.minimum);
                float4 voxelInfo = gridFogImages[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                extinction = voxelInfo.x;
                scattering = voxelInfo.y;
                phase = voxelInfo.z;
                color = voxelInfo.w * (1.0f + gridFogInfo.ambientIntensity);
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
