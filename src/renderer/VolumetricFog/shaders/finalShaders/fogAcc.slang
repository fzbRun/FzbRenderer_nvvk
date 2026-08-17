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

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float v = (sliceY + jitter.y) / gridSize.y * 2.0f - 1.0f;
    float u = (sliceX + jitter.x) / gridSize.x * 2.0f - 1.0f;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * v * cameraUp;
    voxelCenter += halfWidth * u * cameraRight;

    float3 dir = normalize(voxelCenter - cameraPos);

    float linearDepth_lastVoxel = max(sliceZ - 1.0f + jitter.z, 0.0f) / gridSize.z * (farPlane - nearPlane);
    float depth_lastVoxel = frustumMap(linearDepth_lastVoxel, compressionParams, farPlane - nearPlane, gridSize.z);

    float halfHeight_prev = (depth_lastVoxel + nearPlane) * tanHalfFov;
    float halfWidth_prev = halfHeight_prev * aspect;
    float3 voxelCenter_lastVoxel = cameraPos;
    voxelCenter_lastVoxel += (depth_lastVoxel + nearPlane) * cameraForward;
    voxelCenter_lastVoxel += halfHeight_prev * v * cameraUp;
    voxelCenter_lastVoxel += halfWidth_prev * u * cameraRight;

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

        getGlobalHeightFogScatter(samplePoint, sampleDir, sampleLength, sceneInfo, globalFogInfo, pushConst.lightVP, 1.0f, attenuation, L);

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

    uint sampleCount = pushConst.sampleCount;   //max(pushConst.sampleCount * lerp(0, 1, saturate(abs(voxelCenter.z - voxelCenter_lastVoxel.z)) / 2), 1);
    float spatialDither = getCellDither(float2(voxelIndex.xy), (float)voxelIndex.z * 0.618f);
    float layerOffset = getMarchLayerOffset(pushConst.temporalFrameIndex, spatialDither, true, pushConst.blurVoxelFog == 1);
    // layerOffset = rand(randomSeed);
    //  globalHeightFog should affect this, but we pre calc that, so this has a lit bit mistake
    getVolumetricFogScatter(voxelCenter_lastVoxel, voxelCenter, randomSeed, sampleCount, 1, false, layerOffset, pushConst.lightVP, pushConst.sceneInfoAddress, pushConst.fogGlobalInfoAddress, L, attenuation, attenuation2);
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

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;
    //---------------------------------------------------------------------------------------------------------
    if (pushConst.temporalFrameIndex == 0) return; // 第一帧没有历史

    //-------------------------------- 时域重投影 --------------------------------
    // 用上一帧的完整相机位姿（位置 + 朝向）把当前体素中心投影回上一帧的视锥网格。
    // 原来的实现只用了上一帧的位置，基向量却取的是当前帧的，
    // 所以相机一旦旋转，重投影就是错的 —— 拿到的是错误体素的历史值，
    // 混合 90% 之后表现为拖影和逐帧跳变，这正是“相机移动时波动”的主因。
    float4x4 camInfo_last = pushConst.viewInvMatrix_lastFrame;
    float3 cameraPos_lastTime = camInfo_last[3].xyz;
    float3 cameraRight_lastTime = camInfo_last[0].xyz;
    float3 cameraUp_lastTime = camInfo_last[1].xyz;
    float3 cameraForward_lastTime = -camInfo_last[2].xyz;

    bool inFluid = false;
    float mergeRatioMin = 0.01f, mergeRatioMax = 0.05f;
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

    bool historyValid = true;
    float3 uvw_lastTime = 0.0f;
    if (viewZ < nearPlane || viewZ > farPlane) historyValid = false;
    else {
        float halfHeight_lastTime = viewZ * tanHalfFov;
        float halfWidth_lastTime = halfHeight_lastTime * aspect;
        if (abs(viewY) > halfHeight_lastTime || abs(viewX) > halfWidth_lastTime) historyValid = false;
        else {
            float sizeZ = farPlane - nearPlane;
            float3 cellPos;
            cellPos.x = ((viewX / halfWidth_lastTime) * 0.5f + 0.5f) * gridSizeF.x;
            cellPos.y = ((viewY / halfHeight_lastTime) * 0.5f + 0.5f) * gridSizeF.y;
            cellPos.z = frustumInvMap(viewZ - nearPlane, compressionParams, sizeZ, gridSize.z) / sizeZ * gridSizeF.z;

            uvw_lastTime = saturate(cellPos / gridSizeF);
        }
    }

    // 历史无效（体素上一帧在视锥外）时必须把当前值写回去。
    // 原来这里直接 return，可是 fogBlur_Voxel 之后会无条件把结果图拷进历史图，
    // 于是这些体素保留的是上一轮的陈旧数据，相机转动时视锥边缘会出现一圈闪烁的脏值。
    if (!historyValid) return; // curVoxelInfo 本来就已经在图里，不用重写
    float3 lastVoxelInfo = fogAccHistoryImage.SampleLevel(uvw_lastTime, 0).xyz;
    //-------------------------------- 邻域钳制 --------------------------------
    // 把历史值限制在当前体素邻域的取值范围内。重投影再准也会有半个体素的误差，
    // 在雾的边界处（比如物体遮挡产生的阴影边）会把完全不同的值混进来 —— 表现为拖影。
    // 钳制之后历史只能在“合理范围”内起作用，既保住了时域超采样的收敛，又不会拖影。
    // 前提是这里读到的必须是“逐板”值：这个pass被安排在 Pass1 和 Pass2 之间就是为了这个。
    // 如果放在 Pass2 之后，图里存的是沿Z累积过的量，它沿Z单调平滑，±1邻域的min/max窗口
    // 宽到几乎不构成约束，错位的历史值原样通过 —— 那就是相机移动后的鬼影。
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
    // 相机静止时多留历史（收敛快、抖动被彻底积掉）；相机快速移动时少留历史
    // （重投影误差大，留太多就是拖影）。原来固定 0.1 意味着永远保留 90% 的历史，
    // 相机一动就是十帧的滞后。
    float3 cameraMove = cameraPos - cameraPos_lastTime;
    float cameraSpeed = length(cameraMove) / max(pushConst.dt, 1e-4f);
    // 朝向变化量：用当前和上一帧 forward 的夹角近似
    float rotationDelta = 1.0f - saturate(dot(cameraForward, cameraForward_lastTime));

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