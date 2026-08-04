#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

#if defined(FOG_ACC_GPU_GRIVEN)
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccSyncBuffer)]] RWStructuredBuffer<DispatchIndirectCommand, ScalarDataLayout> fogAccSyncBuffer;

[numthreads(FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1)]
[shader("compute")]
void computeMain_createFrustumAccFog_Stepping_GPUGriven(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    uint groupThreadIndexU = groupThreadIndex.y * FOG_ACC_THREADGROUP_SIZE + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumGridSize.xy)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = pushConst.instanceIndex;
    uint3 voxelIndex = uint3(threadIndex, depthIndex);

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));
    //---------------------------------------------------------------------------------------------
    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane);

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;

    float3 dir = normalize(voxelCenter - cameraPos);

    float linearDepth_lastVoxel = max((float)depthIndex - 0.5f, 0.0f) / gridSize.z * (farPlane - nearPlane);
    float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane);

    float halfHeight_prev = (depth_lastVoxel + nearPlane) * tanHalfFov;
    float halfWidth_prev = halfHeight_prev * aspect;
    float3 voxelCenter_lastVoxel = cameraPos;
    voxelCenter_lastVoxel += (depth_lastVoxel + nearPlane) * cameraForward;
    voxelCenter_lastVoxel += halfHeight_prev * (((float)voxelIndex.y + 0.5f) / gridSize.y * 2.0f - 1.0f) * cameraUp;
    voxelCenter_lastVoxel += halfWidth_prev * (((float)voxelIndex.x + 0.5f) / gridSize.x * 2.0f - 1.0f) * cameraRight;

    float attenuation = 1.0f, L = 0.0f, attenuation2 = 1.0f;
    if (depthIndex > 0) {
        float u = ((float)voxelIndex.x + 0.5f) / gridSize.x + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.x * pushConst.jitterStrength0;
        float v = ((float)voxelIndex.y + 0.5f) / gridSize.y + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.y * pushConst.jitterStrength0;
        float w = (voxelIndex.z - 0.5f) / gridSize.z;

        // float3 uvw = (float3(voxelIndex.x, voxelIndex.y, voxelIndex.z - 1) + 0.5f) / gridSize;
        float3 uvw = { u, v, w };
        float4 accResult = volumetricFogAccResultImage_sample.SampleLevel(uvw, 0);
        attenuation = accResult.x;
        L = accResult.y;
    }

#ifdef USE_ENVFOG
    getEnvVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, true, 1, L, attenuation, attenuation2);
#else

    uint sampleCount = pushConst.sampleCount;
    getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, L, attenuation, attenuation2);
#endif

    volumetricFogAccResultImage[voxelIndex] = float4(attenuation, L, 1.0f, 1.0f);

    if (all(groupThreadIndex == 0)) {
        const uint groupSize = uint(FOG_ACC_THREADGROUP_SIZE);
        const uint groupsPerRow = (pushConst.frustumGridSize.x + groupSize - 1u) / groupSize;
        const uint threadGroupIndexU = threadGroupIndex.y * groupsPerRow + threadGroupIndex.x;
        fogAccSyncBuffer[threadGroupIndexU] = { (pushConst.frustumGridSize.x + groupSize - 1) / groupSize, (pushConst.frustumGridSize.y + groupSize - 1) / groupSize, 1 };
    }
}
//-----------------------------------------------------------------------------------------------------------------------------
#elif defined(FOG_ACC_ONE_DISPATCH)
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccSyncBuffer)]] RWStructuredBuffer<uint> fogAccSyncBuffer;

[numthreads(FOG_ACC_THREADGROUP_SIZE, FOG_ACC_THREADGROUP_SIZE, 1)]
[shader("compute")]
void computeMain_createFrustumAccFog_Stepping_oneDispatch(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    uint groupThreadIndexU = groupThreadIndex.y * FOG_ACC_THREADGROUP_SIZE + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumGridSize.xy)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = threadGroupIndex.z;
    uint3 voxelIndex = uint3(threadIndex, depthIndex);

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));
    //---------------------------------------------------------------------------------------------
    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane);

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;

    float3 dir = normalize(voxelCenter - cameraPos);

    float linearDepth_lastVoxel = max((float)depthIndex - 0.5f, 0.0f) / gridSize.z * (farPlane - nearPlane);
    float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane);

    float halfHeight_prev = (depth_lastVoxel + nearPlane) * tanHalfFov;
    float halfWidth_prev = halfHeight_prev * aspect;
    float3 voxelCenter_lastVoxel = cameraPos;
    voxelCenter_lastVoxel += (depth_lastVoxel + nearPlane) * cameraForward;
    voxelCenter_lastVoxel += halfHeight_prev * (((float)voxelIndex.y + 0.5f) / gridSize.y * 2.0f - 1.0f) * cameraUp;
    voxelCenter_lastVoxel += halfWidth_prev * (((float)voxelIndex.x + 0.5f) / gridSize.x * 2.0f - 1.0f) * cameraRight;
    //---------------------------------------------------------------------------------------------
    float attenuation = 1.0f, L = 0.0f, attenuation2 = 1.0f;
    if (depthIndex > 0) {
        if (all(groupThreadIndex == 0)) {
            uint curFinishDepthIndex = 0;
            uint hopeFinishDepthIndex = depthIndex - 1;
            const uint groupSize = uint(FOG_ACC_THREADGROUP_SIZE);
            const uint groupsPerRow = (pushConst.frustumGridSize.x + groupSize - 1u) / groupSize;
            const uint threadGroupIndexU = threadGroupIndex.y * groupsPerRow + threadGroupIndex.x;
            InterlockedCompareExchange(fogAccSyncBuffer[threadGroupIndexU], hopeFinishDepthIndex, hopeFinishDepthIndex, curFinishDepthIndex);
            while (curFinishDepthIndex != hopeFinishDepthIndex)
                InterlockedCompareExchange(fogAccSyncBuffer[threadGroupIndexU], hopeFinishDepthIndex, hopeFinishDepthIndex, curFinishDepthIndex);
        }
        GroupMemoryBarrierWithGroupSync();

        float u = ((float)voxelIndex.x + 0.5f) / gridSize.x + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.x * 0.0f; // pushConst.jitterStrength0;
        float v = ((float)voxelIndex.y + 0.5f) / gridSize.y + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.y * 0.0f; // pushConst.jitterStrength0;
        float w = (voxelIndex.z - 0.5f) / gridSize.z;
        float3 uvw = { u, v, w };

        float4 accResult = volumetricFogAccResultImage_sample.SampleLevel(uvw, 0);
        attenuation = accResult.x;
        L = accResult.y;
    }

    #ifdef USE_ENVFOG
    getEnvVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, true, 1, L, attenuation, attenuation2);
    #else
    uint sampleCount = pushConst.sampleCount;
    getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, L, attenuation, attenuation2);
    #endif
    volumetricFogAccResultImage[voxelIndex] = float4(attenuation, L, 1.0f, 1.0f);

    GroupMemoryBarrierWithGroupSync();
    if (all(groupThreadIndex == 0)) {
        const uint groupSize = uint(FOG_ACC_THREADGROUP_SIZE);
        const uint groupsPerRow = (pushConst.frustumGridSize.x + groupSize - 1u) / groupSize;
        const uint threadGroupIndexU = threadGroupIndex.y * groupsPerRow + threadGroupIndex.x;
        InterlockedExchange(fogAccSyncBuffer[threadGroupIndexU], depthIndex);
    }
}
#endif

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createFrustumAccFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (any(voxelIndex >= pushConst.frustumGridSize)) return;

    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = groupSceneInfo;
    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float depthOrigin_near = float(voxelIndex.z) / gridSize.z * (farPlane - nearPlane);
    float depthMapping_near = nearPlane + frustumMap(depthOrigin_near, compressionParams, farPlane - nearPlane);

    float depthOrigin_far = float(voxelIndex.z + 1) / gridSize.z * (farPlane - nearPlane);
    float depthMapping_far = nearPlane + frustumMap(depthOrigin_far, compressionParams, farPlane - nearPlane);

    float depthMapping_center = (depthMapping_far + depthMapping_near) * 0.5f;

    float u = (float(voxelIndex.x) + 0.5f) / float(gridSize.x) * 2.0f - 1.0f;
    float v = (float(voxelIndex.y) + 0.5f) / float(gridSize.y) * 2.0f - 1.0f;

    float3 voxelCenter = cameraPos
      + cameraRight * (u * depthMapping_center * tanHalfFov * aspect)
      + cameraUp * (v * depthMapping_center * tanHalfFov)
      + cameraForward * depthMapping_center;

    uint voxelIndexU = voxelIndex.z * pushConst.frustumGridSize.x * pushConst.frustumGridSize.y + voxelIndex.y * pushConst.frustumGridSize.x + voxelIndex.x;
    uint randomSeed =  xxhash32(uint3(pushConst.frameIndex, voxelIndexU, 0));
    float3 samplePoint = voxelCenter;   // + (float3(rand(randomSeed), rand(randomSeed), rand(randomSeed)) * 2.0f - 1.0f) * float3(halfWidth / gridSize.x, halfHeight / gridSize.y, halfDepth);

    float L = 0.0f; float attenuation = 1.0f; float attenuation2 = 1.0f;
    getVolumetricFogScatter(cameraPos, samplePoint, randomSeed, pushConst.sampleCount, L, attenuation, attenuation2);

    volumetricFogAccResultImage[voxelIndex] = float4(attenuation, L, 1.0f, 1.0f);
}

/*
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_createFrustumAccFog_Stepping(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.y * 32 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumGridSize.xy)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = pushConst.instanceIndex;
    uint3 voxelIndex = uint3(threadIndex, depthIndex);

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float depthOrigin_near = float(depthIndex) / gridSize.z * (farPlane - nearPlane);
    float depthMapping_near = nearPlane + frustumMap(depthOrigin_near, compressionParams, farPlane - nearPlane);

    float depthOrigin_far = float(depthIndex + 1) / gridSize.z * (farPlane - nearPlane);
    float depthMapping_far = nearPlane + frustumMap(depthOrigin_far, compressionParams, farPlane - nearPlane);

    float depthMapping_center = (depthMapping_far + depthMapping_near) * 0.5f;

    float u = (float(threadIndex.x) + 0.5f) / float(gridSize.x) * 2.0f - 1.0f;
    float v = (float(threadIndex.y) + 0.5f) / float(gridSize.y) * 2.0f - 1.0f;

    float3 voxelCenter = cameraPos
      + cameraRight * (u * depthMapping_center * tanHalfFov * aspect)
      + cameraUp * (v * depthMapping_center * tanHalfFov)
      + cameraForward * depthMapping_center;

    float voxelHalfHeight = depthMapping_center * tanHalfFov / gridSize.y;
    float voxelHalfWidth = voxelHalfHeight * aspect;
    float voxelHalfDepth = (depthMapping_far - depthMapping_near) * 0.5f;
    float3 voxelStartPos = voxelCenter - cameraRight * voxelHalfWidth - cameraUp * voxelHalfHeight - cameraForward * voxelHalfDepth;

    float3 voxelDirX = cameraRight * (2.0f * voxelHalfWidth);
    float3 voxelDirY = cameraUp * (2.0f * voxelHalfHeight);
    float3 voxelDirZ = cameraForward * (2.0f * voxelHalfDepth);

    float tPart = voxelHalfDepth * 2.0f;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));
    float3x3 rotateMatrix = randomRotationMatrixUniform(randomSeed);

    uint sampleCount = pushConst.sampleCount;
    uint validSampleCount = 0;
    float extinctionCoeff = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
    bool hasSample = false;
    for (int i = 0; i < sampleCount; ++i) {
        float3 sampleDir = mul(fibSpherePoint(i, sampleCount), rotateMatrix);
        AABB voxelAABB = { voxelStartPos, voxelStartPos + voxelDirX + voxelDirY + voxelDirZ };
        float3 samplePoint = voxelCenter;   //getAABBPoint(voxelAABB, voxelCenter, sampleDir);    

        #ifdef Uniform_EnvFog_Grid
        if(!checkPointInAABB(samplePoint, {globalInfo.envStartPos, globalInfo.envStartPos + globalInfo.envGridSize * globalInfo.envVoxelSize})) continue;
        float3 uvw = (samplePoint - globalInfo.envStartPos) / (globalInfo.envGridSize * globalInfo.envVoxelSize);
        float4 voxelInfo = envVolumetricFogInfoImage_sample.SampleLevel(uvw, 0);
        #else
        float4 voxelInfo = getFrustumImageInfo(samplePoint, true, envVolumetricFogInfoImage);
        #endif
        hasSample = true;
        validSampleCount += 1;
        
        extinctionCoeff += max(voxelInfo.x, 1e-30f);
        scattering += voxelInfo.y;
        phase += voxelInfo.z;
        color += voxelInfo.w;
    }

    float3 attenuation = 1.0f;
    float3 L = 0.0f;
    if (depthIndex > 0) {
        int lastDepthIndex = depthIndex - 1;
        uint3 lastVoxelIndex = uint3(threadIndex, lastDepthIndex);

        float halfHeight = voxelCenter.z * tanHalfFov;
        float halfWidth = halfHeight * aspect;
        float3 uvw = { (float(voxelIndex.x) + 0.5f) / gridSize.x, (float(voxelIndex.y) + 0.5f) / gridSize.y, (float(lastDepthIndex) + 0.5f) / gridSize.z };

        attenuation = volumetricFogAttenuationImage_sample.SampleLevel(uvw, 0).xyz;
        L = volumetricFogLImage_sample.SampleLevel(uvw, 0).xyz;
    }
    float3 attenuation_CurFrame = 1.0f;
    float3 L_CurFrame = 0.0f;

    if (hasSample) {
        extinctionCoeff /= validSampleCount;
        scattering /= validSampleCount;
        phase /= validSampleCount;
        color /= validSampleCount;
        float phase2 = phase * phase;

        float3 L_sum = 0.0f;
        float x = extinctionCoeff * tPart;
        float factor = (x < 1e-4) ? tPart * (1.0 - 0.5 * x) : (1.0 - exp(-x)) / extinctionCoeff;
        float3 voxelLightAttenuation = scattering * color * attenuation * factor;
        for (int i = 0; i < sampleCount; ++i) {
            float3 sampleDir = mul(fibSpherePoint(i, sampleCount), rotateMatrix);
            AABB voxelAABB = { voxelStartPos, voxelStartPos + voxelDirX + voxelDirY + voxelDirZ };
            float3 samplePoint = voxelCenter;   //sgetAABBPoint(voxelAABB, voxelCenter, sampleDir);
            float3 dir = normalize(samplePoint - cameraPos);

            for (int lightIndex = 0; lightIndex < sceneInfo.numLights; ++lightIndex) {
                Light light = sceneInfo.lights[lightIndex];
                float3 lightColor = light.color * light.intensity;
                float3 lightDirection; float3 lightPos;
                if (light.type == LightType::Direction) {
                    lightDirection = -light.direction;
                    lightPos = samplePoint + 1000.0f * lightDirection;
                } else if (light.type == LightType::Point) {
                    lightDirection = normalize(light.pos - samplePoint);
                    lightPos = light.pos;
                }

                float fp = (1.0f - phase2) / (4.0f * M_PI * pow((1 + phase2 - 2.0f * phase * dot(-dir, lightDirection)), 1.5f));

                float tLightEnter, tLightExit;
                float lightDt = 1e-6f;
                #ifdef Uniform_EnvFog_Grid
                if (isHit(lightPos, samplePoint, globalInfo.envStartPos, globalInfo.envStartPos + globalInfo.envGridSize * globalInfo.envVoxelSize, tLightEnter, tLightExit))
                    lightDt = tLightExit - tLightEnter;
                #else
                if (IntersectFrustum(lightPos, samplePoint, cameraPos, cameraForward, cameraRight, cameraUp, tanHalfFov, aspect, nearPlane, farPlane, tLightEnter, tLightExit))
                    lightDt = tLightExit - tLightEnter;
                #endif

                float attenuationLength = max(globalInfo.envFogLightAttenuationLength, 1e-6f);
                float attenuationRatio = max(lightDt / attenuationLength, 0.0f);
                float lightAttenuation = pow(max(globalInfo.envFogLightAttenuationEstimator, 1e-6f), attenuationRatio);
                float3 activeSunColor = lightColor * lightAttenuation * pushConst.lightAttenuationStrength;
                if (lightIndex == 0) activeSunColor *= getShadow(float4(samplePoint, 1.0f));

                // should first get extinction_sum, then cal L,but that condition need double SampleImage, so skip
                L_sum += activeSunColor * fp;
            }
        }

        attenuation_CurFrame = exp(-extinctionCoeff * tPart);
        L_CurFrame = L_sum * voxelLightAttenuation / sampleCount;
    }

    if (pushConst.frameIndex > 0) {
        attenuation = (attenuation * attenuation_CurFrame + volumetricFogAttenuationImage[voxelIndex].xyz) * 0.5f;
        L = (L + L_CurFrame + volumetricFogLImage[voxelIndex].xyz) * 0.5f;
    } else {
        attenuation *= attenuation_CurFrame;
        L += L_CurFrame;
    }
    
    volumetricFogAttenuationImage[voxelIndex].xyz = attenuation;
    volumetricFogLImage[voxelIndex].xyz = L;
}
*/

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_createFrustumAccFog_Stepping(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.y * 32 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumGridSize.xy)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = pushConst.instanceIndex;
    uint3 voxelIndex = uint3(threadIndex, depthIndex);

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.frameIndex, 1));
    //---------------------------------------------------------------------------------------------
    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane);

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;

    float3 dir = normalize(voxelCenter - cameraPos);

    float linearDepth_lastVoxel = max((float)depthIndex - 0.5f, 0.0f) / gridSize.z * (farPlane - nearPlane);
    float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane);

    float halfHeight_prev = (depth_lastVoxel + nearPlane) * tanHalfFov;
    float halfWidth_prev = halfHeight_prev * aspect;
    float3 voxelCenter_lastVoxel = cameraPos;
    voxelCenter_lastVoxel += (depth_lastVoxel + nearPlane) * cameraForward;
    voxelCenter_lastVoxel += halfHeight_prev * (((float)voxelIndex.y + 0.5f) / gridSize.y * 2.0f - 1.0f) * cameraUp;
    voxelCenter_lastVoxel += halfWidth_prev * (((float)voxelIndex.x + 0.5f) / gridSize.x * 2.0f - 1.0f) * cameraRight;

    float attenuation = 1.0f, L = 0.0f, attenuation2 = 1.0f;
    if (depthIndex > 0) {
        float u = ((float)voxelIndex.x + 0.5f) / gridSize.x + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.x * pushConst.jitterStrength0;
        float v = ((float)voxelIndex.y + 0.5f) / gridSize.y + (rand(randomSeed) * 2.0f - 1.0f) / gridSize.y * pushConst.jitterStrength0;
        float w = (voxelIndex.z - 0.5f) / gridSize.z;

        // float3 uvw = (float3(voxelIndex.x, voxelIndex.y, voxelIndex.z - 1) + 0.5f) / gridSize;
        float3 uvw = { u, v, w };
        float4 accResult = volumetricFogAccResultImage_sample.SampleLevel(uvw, 0);
        attenuation = accResult.x;
        L = accResult.y;
    }

    #ifdef USE_ENVFOG
    getEnvVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, true, 1, L, attenuation, attenuation2);
    #else

    uint sampleCount = pushConst.sampleCount;
    //if (depthIndex < gridSize.z / 3) sampleCount = 1;
    //else if (depthIndex < gridSize.z / 1.5) sampleCount = pushConst.sampleCount / 2;

    getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, L, attenuation, attenuation2);

    /*
    int sampleCount = pushConst.sampleCount;
    float4 envVoxelInfo = 0.0f;
    for (int sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex) {
        float3 localSample = 0.5f;
        if (sampleCount > 1) {
            localSample = float3(rand(randomSeed), rand(randomSeed), rand(randomSeed));
            localSample = lerp(0.5f.xxx, localSample, saturate(pushConst.jitterStrength * 0.1f));
        }

        float sampleU = (float(voxelIndex.x) + localSample.x) / gridSizeF.x;
        float sampleV = (float(voxelIndex.y) + localSample.y) / gridSizeF.y;
        float sampleW = (float(voxelIndex.z) + localSample.z) / gridSizeF.z;

        float sampleDepth = nearPlane + frustumMap(sampleW * (farPlane - nearPlane), pushConst.compressionParams, farPlane - nearPlane);
        float sampleHalfHeight = sampleDepth * tanHalfFov;
        float sampleHalfWidth = sampleHalfHeight * aspect;

        float3 samplePoint = cameraPos + cameraRight * ((sampleU * 2.0f - 1.0f) * sampleHalfWidth) + cameraUp * ((sampleV * 2.0f - 1.0f) * sampleHalfHeight) + cameraForward * sampleDepth;
        samplePoint = voxelCenter;
        {
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
                    float heightScale = groupHeightFogInfos[fogInfo.typeFogIndex].heightScale;
                    float height = max(samplePoint.y - fogInfo.fogStartPos.y, 0.0f);
                    heightAttenuation = exp(-heightScale * height);
                    scattering = fogInfo.scattering;
                    extinction = (fogInfo.absorption.x + scattering) * heightAttenuation;
                    phase = fogInfo.phase;
                    color = fogInfo.color;
                } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp == 1) {
                    float3 uvw = (samplePoint - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
                    uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
                    float4 voxelInfo = FluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);

                    // if in fluid, the voxelInfo has include the envionment fog info
                    envExtinction = max(voxelInfo.x, 1e-30f);
                    envScattering = envExtinction * voxelInfo.y;
                    envPhase = envExtinction * voxelInfo.z;
                    envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
                    break;
                } else if (fogInfo.type == VolumetricFogType::Noise) {
                    NoiseFogInfo noiseFogInfo = groupNoiseFogInfos[fogInfo.typeFogIndex];
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
    }
    envVoxelInfo /= sampleCount;

    float3 tPart = (voxelCenter - voxelCenter_lastVoxel) / 10;
    for (int i = 0; i < 10; ++i) {
        float3 tSample = (i + pushConst.randomStepping == 1 ? rand(randomSeed) : 0.5f) * tPart;
        float3 samplePoint = voxelCenter_lastVoxel + tSample;
        float a_part = 1.0f; float3 L_part = 0.0f;
        getFogScatter(samplePoint, dir, length(tPart), sceneInfo, globalInfo, envVoxelInfo.x, envVoxelInfo.y, envVoxelInfo.z, envVoxelInfo.w, attenuation, a_part, L_part);

        attenuation *= a_part;
        L += L_part;
    }
    */
    #endif

    //if (pushConst.frameIndex > 0) {
    //    //attenuation = lerp(attenuation, volumetricFogAttenuationImage[voxelIndex].xyz, saturate(length(pushConst.cameraMoveDir) * pushConst.jitterStrength));
    //    //L = lerp(L, volumetricFogLImage[voxelIndex].xyz, saturate(length(pushConst.cameraMoveDir) * pushConst.jitterStrength));
    //
    //    float mergeRatio = 0.5f;
    //    if (all(abs(pushConst.cameraMoveDir) < 1e-6f)) mergeRatio = 1.0f / pushConst.frameIndex;
    //
    //    float3 attenuation_last = volumetricFogAttenuationImage[voxelIndex].xyz;
    //    attenuation = attenuation * mergeRatio + attenuation_last * (1.0f - mergeRatio);
    //
    //    float3 L_last = volumetricFogLImage[voxelIndex].xyz;
    //    L = L * mergeRatio + L_last * (1.0f - mergeRatio);
    //}

    volumetricFogAccResultImage[voxelIndex] = float4(attenuation, L, 1.0f, 1.0f);
}
