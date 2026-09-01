#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/finalShaders/FogCommon.slang"

[[vk::push_constant]] ConstantBuffer<FogAccPushConstant> pushConst;

groupshared uint groupHasFogVoxelCount; // 8 * 8 * 8 / 32
groupshared uint groupHasFogVoxelOffset_inGlobal;

[numthreads(8, 8, 8)]
[shader("compute")]
void computeMain_getHasFogVoxels(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 64 + groupThreadIndex.y * 8 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress);
    if (groupThreadIndexU == 0) groupHasFogVoxelCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumInfo.frustumGridSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    FogGlobalInfo globalFogInfo = groupFogGlobalInfo;

    int depthIndex = threadIndex.z;
    uint3 voxelIndex = threadIndex;

    float nearPlane = pushConst.frustumInfo.cameraNearPlane;
    float farPlane = pushConst.frustumInfo.cameraFarPlane;
    float3 compressionParams = max(pushConst.frustumInfo.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.frustumInfo.tanCameraFov_2;
    float aspect = pushConst.frustumInfo.aspectRatio;
    uint3 gridSize = pushConst.frustumInfo.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float sliceZ = (float)depthIndex;
    float sliceY = (float)voxelIndex.y;
    float sliceX = (float)voxelIndex.x;

    float3 jitter = pushConst.jitterUVW;
    if (pushConst.blurVoxelFog == 0) jitter = 0.0f;

    float linearDepth = (sliceZ + jitter.z) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);
    float depthWithNear = nearPlane + depth;

    float v = (sliceY + jitter.y) / gridSize.y * 2.0f - 1.0f;
    float u = (sliceX + jitter.x) / gridSize.x * 2.0f - 1.0f;

    float3 voxelCenter = getMorphedFrustumVoxelCenter(depthWithNear, u, v, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.frustumInfo);

    float linearDepth_lastVoxel = max(sliceZ - 1.0f + jitter.z, 0.0f) / gridSize.z * (farPlane - nearPlane);
    float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane, gridSize.z);
    float depthWithNear_lastVoxel = nearPlane + depth_lastVoxel;

    float3 voxelCenter_lastVoxel = getMorphedFrustumVoxelCenter(depthWithNear_lastVoxel, u, v, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.frustumInfo);

    bool hasFog = false;
    for (int i = 0; i < globalFogInfo.volumetricFogCount; ++i) {
        VolumetricFogInfo fogInfo = getVolumetricFog(i);
        float tEntery, tExit;
        if (isHit(voxelCenter_lastVoxel, voxelCenter, fogInfo.aabb.minimum, fogInfo.aabb.maximum, tEntery, tExit)) {
            hasFog = true;
            break;
        }
    }

    uint warpIndex = groupThreadIndexU / 32;
    uint warpLane = groupThreadIndexU & 31;

    uint warpHasFogCount = WaveActiveCountBits(hasFog);
    uint hasFogOffset_inWarp = WavePrefixCountBits(hasFog);
    uint hasFogOffset_inGroup = 0;
    if (warpLane == 0) InterlockedAdd(groupHasFogVoxelCount, warpHasFogCount, hasFogOffset_inGroup);
    hasFogOffset_inGroup = WaveReadLaneFirst(hasFogOffset_inGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndexU == 0) InterlockedAdd(pushConst.fogAccHasFogVoxelCountAddress[0], groupHasFogVoxelCount, groupHasFogVoxelOffset_inGlobal);
    GroupMemoryBarrierWithGroupSync();
    uint hasFogOffset_inGlobal = groupHasFogVoxelOffset_inGlobal;

    if (hasFog) {
        FogAccHasFogVoxelInfo hasFogVoxelInfo;
        hasFogVoxelInfo.voxelIndex = voxelIndex;
        hasFogVoxelInfo.voxelCenter = voxelCenter;
        hasFogVoxelInfo.voxelCenter_lastVoxel = voxelCenter_lastVoxel;

        uint fogOffset = hasFogOffset_inWarp + hasFogOffset_inGroup + groupHasFogVoxelOffset_inGlobal;
        fogAccHasFogVoxelInfoBuffer[fogOffset] = hasFogVoxelInfo;
    }
    //---------------------------------------------------------------------------------------------------------------
    float attenuation = 1.0f; float3 L = 0.0f;
    if (globalFogInfo.useGlobalHeightFog == 1) {
        float spatialDither = getCellDither(float2(voxelIndex.xy), (float)voxelIndex.z * 0.618f);
        float layerOffset = getMarchLayerOffset(pushConst.temporalFrameIndex, spatialDither, true, pushConst.blurVoxelFog == 1);

        uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
        uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.temporalFrameIndex, 1));

        float sampleLength = length(voxelCenter_lastVoxel - voxelCenter);
        float3 sampleDir = (voxelCenter - voxelCenter_lastVoxel) / max(sampleLength, 1e-20f);
        float3 samplePoint = voxelCenter_lastVoxel + sampleDir * sampleLength * rand(randomSeed); // layerOffset;

        // uint sampleCount = 10;
        // sampleLength /= sampleCount;
        // for (int i = 0; i < sampleCount; ++i) {
        //     samplePoint = voxelCenter_lastVoxel + sampleDir * sampleLength * (i + rand(randomSeed));
        //
        //     getGlobalHeightFogScatter(samplePoint, sampleDir, sampleLength, sceneInfo, globalFogInfo, pushConst.lightVP, 1.0f, attenuation, L);
        // }

        getGlobalHeightFogScatter(voxelCenter_lastVoxel, voxelCenter, layerOffset, sceneInfo, globalFogInfo, pushConst.lightVP, 1.0f, attenuation, L);

        // getAnalyticHeightFogDirectionalScatter(voxelCenter_lastVoxel, voxelCenter, globalFogInfo.globalHeightFogAABB, globalFogInfo.globalHeightFogInfo, sceneInfo.lights[0], pushConst.lightVP, attenuation, L);
    }
    fogAccImage[voxelIndex] = float4(attenuation, luminance(L), 1.0f, 0.0f);
}

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getFog(uint threadIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    getGroupsharedData(groupThreadIndex, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress);
    if (groupThreadIndex == 0) groupHasFogVoxelCount = pushConst.fogAccHasFogVoxelCountAddress[0];
    GroupMemoryBarrierWithGroupSync();

    if (threadIndex >= groupHasFogVoxelCount) return;
    FogGlobalInfo globalFogInfo = groupFogGlobalInfo;

    SceneInfo sceneInfo = groupSceneInfo;
    FogAccHasFogVoxelInfo voxelInfo = fogAccHasFogVoxelInfoBuffer[threadIndex];
    uint3 voxelIndex = voxelInfo.voxelIndex;
    float3 voxelCenter = voxelInfo.voxelCenter;
    float3 voxelCenter_lastVoxel = voxelInfo.voxelCenter_lastVoxel;

    uint3 gridSize = pushConst.frustumInfo.frustumGridSize;
    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(voxelIndexU, pushConst.temporalFrameIndex, 1));

    float attenuation = 1.0f, L = 0.0f, attenuation2 = 1.0f;
    if (globalFogInfo.useGlobalHeightFog == 1) {
        float2 globalHeightFogResult = fogAccImage[voxelIndex].xy;
        attenuation = globalHeightFogResult.x;
        L = globalHeightFogResult.y;
    }

    float sampleLength = distance(voxelCenter_lastVoxel, voxelCenter);
    uint sampleCount = max(int(float(pushConst.sampleCount) * saturate(sampleLength)), 1);
    float spatialDither = getCellDither(float2(voxelIndex.xy), (float)voxelIndex.z * 0.618f);
    float layerOffset = getMarchLayerOffset(pushConst.temporalFrameIndex, spatialDither, true, pushConst.blurVoxelFog == 1);
    // layerOffset = rand(randomSeed);
    //  globalHeightFog should affect this, but we pre calc that, so this has a lit bit mistake
    getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, 1, false, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
    // getVolumetricFogScatter_MonteCarlo(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, 1, false, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
    
    //float sampleLength = distance(voxelCenter_lastVoxel, voxelCenter);
    //if (sampleLength > 10.0f) getVolumetricFogScatter_MonteCarlo(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, 1, false, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
    //else getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, 1, false, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
    
    fogAccImage[voxelIndex] = float4(attenuation, L, attenuation2, 1.0f);

    //if (all(voxelIndex <= uint3(76, 101, 32)) && all(voxelIndex >= uint3(71, 97, 32))) {
    //    printf("frameIndex: %d, voxelIndex: %d %d %d, L: %f\n", pushConst.temporalFrameIndex, voxelIndex.x, voxelIndex.y, voxelIndex.z, L);
    //}
}

groupshared float2 groupRawFog[8 * 8 * 8];
[numthreads(8, 8, 8)]
[shader("compute")]
void computeMain_blurFog(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 64 + groupThreadIndex.y * 8 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress);

    uint3 gridSizeEarly = pushConst.frustumInfo.frustumGridSize;
    bool inGrid = all(threadIndex < gridSizeEarly);
    float4 curVoxelInfo = inGrid ? fogAccImage[threadIndex] : float4(1.0f, 0.0f, 0.0f, 0.0f);
    groupRawFog[groupThreadIndexU] = curVoxelInfo.xy;

    GroupMemoryBarrierWithGroupSync();
    if (!inGrid) return;

    SceneInfo sceneInfo = groupSceneInfo;

    int depthIndex = threadIndex.z;
    uint3 voxelIndex = threadIndex;

    float nearPlane = pushConst.frustumInfo.cameraNearPlane;
    float farPlane = pushConst.frustumInfo.cameraFarPlane;
    float3 compressionParams = max(pushConst.frustumInfo.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.frustumInfo.tanCameraFov_2;
    float aspect = pushConst.frustumInfo.aspectRatio;
    uint3 gridSize = gridSizeEarly;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.viewInvMatrix[3].xyz;  //sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);
    float depthWithNear = nearPlane + depth;

    float u = (((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f;
    float v = (((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f;

    float3 voxelCenter = getMorphedFrustumVoxelCenter(depthWithNear, u, v, cameraPos, cameraForward, cameraRight, cameraUp, pushConst.frustumInfo);
    //---------------------------------------------------------------------------------------------------------
    if (pushConst.temporalFrameIndex == 0) return; // 第一帧没有历史

    //-------------------------------- 时域重投影 --------------------------------
    float4x4 camInfo_last = pushConst.viewInvMatrix_lastFrame;
    float3 cameraPos_lastTime = camInfo_last[3].xyz;
    float3 cameraRight_lastTime = camInfo_last[0].xyz;
    float3 cameraUp_lastTime = camInfo_last[1].xyz;
    float3 cameraForward_lastTime = -camInfo_last[2].xyz;

    bool inFluid = false;
    float mergeRatioMin = 0.03f, mergeRatioMax = 0.1f;
    {
        FogGlobalInfo fogGlobalInfo = groupFogGlobalInfo;
        for (int i = 0; i < fogGlobalInfo.fluidFogCount; ++i) {
            VolumetricFogInfo fogInfo = getVolumetricFog(i);
            if (!checkPointInAABB(voxelCenter, fogInfo.aabb)) continue;
    
            FluidFogInfo fluidFogInfo = getFluidFog(fogInfo.typeFogIndex);
            if (fluidFogInfo.startUp == 0) continue;

            inFluid = true;
            mergeRatioMin = pushConst.fluidMergeRatio.x;
            mergeRatioMax = pushConst.fluidMergeRatio.y;
            break;
        }
    }
    if (all(abs(cameraPos_lastTime - cameraPos) < 1e-10f) && !inFluid) {
        fogAccImage[voxelIndex].xyz = fogAccHistoryImage[voxelIndex].xyz;
        return;
    }
    // if (all(voxelIndex == 0)) printf("cameraPos_lastTime: %f %f %f, cameraPos: %f %f %f\n", cameraPos_lastTime.x, cameraPos_lastTime.y, cameraPos_lastTime.z, cameraPos.x, cameraPos.y, cameraPos.z);

    float3 toVoxel = voxelCenter - cameraPos_lastTime;
    float viewZ = dot(toVoxel, cameraForward_lastTime);
    float viewX = dot(toVoxel, cameraRight_lastTime);
    float viewY = dot(toVoxel, cameraUp_lastTime);
    float depthWithNear_lastCamera = getMorphedFrustumDepthWithNear(toVoxel, cameraForward_lastTime, pushConst.frustumInfo);

    bool historyValid = true;
    float3 uvw_lastTime = 0.0f;
    if (depthWithNear_lastCamera < nearPlane || depthWithNear_lastCamera > farPlane) historyValid = false;
    else {
        float halfHeight_lastTime = viewZ * tanHalfFov;
        float halfWidth_lastTime = halfHeight_lastTime * aspect;
        if (abs(viewY) > halfHeight_lastTime || abs(viewX) > halfWidth_lastTime) historyValid = false;
        else {
            float sizeZ = farPlane - nearPlane;
            float3 cellPos;
            cellPos.x = ((viewX / halfWidth_lastTime) * 0.5f + 0.5f) * gridSizeF.x;
            cellPos.y = ((viewY / halfHeight_lastTime) * 0.5f + 0.5f) * gridSizeF.y;
            cellPos.z = frustumInvMap(depthWithNear_lastCamera - nearPlane, compressionParams, sizeZ, gridSize.z) / sizeZ * gridSizeF.z;

            uvw_lastTime = saturate(cellPos / gridSizeF);
        }
    }

    if (!historyValid) return; 
    float3 lastVoxelInfo = fogAccHistoryImage.SampleLevel(uvw_lastTime, 0).xyz;
    //-------------------------------- 邻域钳制 --------------------------------
    //float2 nMin = curVoxelInfo.xy;
    //float2 nMax = curVoxelInfo.xy;
    //int3 localIndex = int3(groupThreadIndex);
    //[unroll]
    //for (int axis = 0; axis < 3; ++axis) {
    //    [unroll]
    //    for (int side = -1; side <= 1; side += 2) {
    //        int3 offset = 0;
    //        offset[axis] = side;
    //        int3 neighbor = clamp(localIndex + offset, int3(0), int3(7)); // 组边界回退到自身
    //        float2 n = groupRawFog[neighbor.z * 64 + neighbor.y * 8 + neighbor.x];
    //        nMin = min(nMin, n);
    //        nMax = max(nMax, n);
    //    }
    //}
    //lastVoxelInfo = clamp(lastVoxelInfo, nMin, nMax);

    //-------------------------------- 自适应混合比例 --------------------------------
    float3 cameraMove = cameraPos - cameraPos_lastTime;
    float cameraSpeed = length(cameraMove) / max(pushConst.dt, 1e-4f);
    float rotationDelta = 1.0f - saturate(dot(cameraForward, cameraForward_lastTime));    // 朝向变化量：用当前和上一帧 forward 的夹角近似

    float motion = saturate(cameraSpeed * 0.02f + rotationDelta * 20.0f);
    float mergeRatio = lerp(mergeRatioMin, mergeRatioMax, motion); // 当前帧的权重

    curVoxelInfo.xyz = lerp(lastVoxelInfo, curVoxelInfo.xyz, mergeRatio);

    fogAccImage[voxelIndex] = curVoxelInfo;
}

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_accFog(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    if (any(threadIndex >= pushConst.frustumInfo.frustumGridSize.xy)) return;

    float2 voxelInfo = fogAccImage[uint3(threadIndex, 0)].xy;
    float attenuation = voxelInfo.x, L = voxelInfo.y;
    for (int depthIndex = 1; depthIndex < pushConst.frustumInfo.frustumGridSize.z; ++depthIndex) {
        float2 curDepthVoxelInfo = fogAccImage[uint3(threadIndex, depthIndex)].xy;
        L += attenuation * curDepthVoxelInfo.y;
        attenuation *= curDepthVoxelInfo.x;

        fogAccImage[uint3(threadIndex, depthIndex)].xy = float2(attenuation, L);
    }
}
