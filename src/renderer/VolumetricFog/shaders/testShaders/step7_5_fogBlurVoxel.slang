#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccResultHistoryImage_sample)]] Sampler3D<float4> renderedFogResultHistoryImage_sample;

#ifdef BLUR_FOG_VOXEL
// 邻域钳制要读周围体素“混合前”的值，但本次dispatch同时也在往同一张图里写。
// 直接读图会读到别的线程组已经混合过的结果，是一个竞态 —— 而竞态的表现就是闪烁，
// 恰恰是我们要消除的东西。所以先把本组的原始值搬进共享内存，同步之后再钳制。
groupshared float2 groupRawFog[8 * 8 * 8];

[numthreads(8, 8, 8)]
[shader("compute")]
void computeMain_FogBlur_Voxel(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 64 + groupThreadIndex.y * 8 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);

    uint3 gridSizeEarly = pushConst.frustumGridSize;
    bool inGrid = all(threadIndex < gridSizeEarly);
    float4 curVoxelInfo = inGrid ? volumetricFogAccResultImage[threadIndex] : float4(1.0f, 0.0f, 0.0f, 0.0f);
    groupRawFog[groupThreadIndexU] = curVoxelInfo.xy;

    GroupMemoryBarrierWithGroupSync();
    if (!inGrid) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = threadIndex.z;
    uint3 voxelIndex = threadIndex;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = gridSizeEarly;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    // 体素中心用未抖动的位置 —— 和 step7 保持一致。
    // （不要在这里加亚体素抖动：消费端 getFrustumImageInfo 按未抖动网格取uvw，
    //   网格一动，每帧读到的雾就横向错开，反而制造闪烁。抖动只放在体素内部的
    //   步进采样点上。）
    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;
    //---------------------------------------------------------------------------------------------------------
    if (pushConst.time <= 1e-30f) return;   // 第一帧没有历史

    //-------------------------------- 时域重投影 --------------------------------
    // 用上一帧的完整相机位姿（位置 + 朝向）把当前体素中心投影回上一帧的视锥网格。
    // 原来的实现只用了上一帧的位置，基向量却取的是当前帧的，
    // 所以相机一旦旋转，重投影就是错的 —— 拿到的是错误体素的历史值，
    // 混合 90% 之后表现为拖影和逐帧跳变，这正是“相机移动时波动”的主因。
    float4x4 camInfo_last = globalInfo.invViewMatrix_lastFrame;
    float3 cameraPos_lastTime = camInfo_last[3].xyz;
    float3 cameraRight_lastTime = camInfo_last[0].xyz;
    float3 cameraUp_lastTime = camInfo_last[1].xyz;
    float3 cameraForward_lastTime = -camInfo_last[2].xyz;

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
    if (!historyValid) return;   // curVoxelInfo 本来就已经在图里，不用重写

    float2 lastVoxelInfo = renderedFogResultHistoryImage_sample.SampleLevel(uvw_lastTime, 0).xy;

    //-------------------------------- 邻域钳制 --------------------------------
    // 把历史值限制在当前体素邻域的取值范围内。重投影再准也会有半个体素的误差，
    // 在雾的边界处（比如物体遮挡产生的阴影边）会把完全不同的值混进来 —— 表现为拖影。
    // 钳制之后历史只能在“合理范围”内起作用，既保住了时域超采样的收敛，又不会拖影。
    // 前提是这里读到的必须是“逐板”值：这个pass被安排在 Pass1 和 Pass2 之间就是为了这个。
    // 如果放在 Pass2 之后，图里存的是沿Z累积过的量，它沿Z单调平滑，±1邻域的min/max窗口
    // 宽到几乎不构成约束，错位的历史值原样通过 —— 那就是相机移动后的鬼影。
    float2 nMin = curVoxelInfo.xy;
    float2 nMax = curVoxelInfo.xy;
    int3 localIndex = int3(groupThreadIndex);
    [unroll]
    for (int axis = 0; axis < 3; ++axis) {
        [unroll]
        for (int side = -1; side <= 1; side += 2) {
            int3 offset = 0;
            offset[axis] = side;
            int3 neighbor = clamp(localIndex + offset, int3(0), int3(7));    // 组边界回退到自身
            float2 n = groupRawFog[neighbor.z * 64 + neighbor.y * 8 + neighbor.x];
            nMin = min(nMin, n);
            nMax = max(nMax, n);
        }
    }
    lastVoxelInfo = clamp(lastVoxelInfo, nMin, nMax);

    //-------------------------------- 自适应混合比例 --------------------------------
    // 相机静止时多留历史（收敛快、抖动被彻底积掉）；相机快速移动时少留历史
    // （重投影误差大，留太多就是拖影）。原来固定 0.1 意味着永远保留 90% 的历史，
    // 相机一动就是十帧的滞后。
    float3 cameraMove = cameraPos - cameraPos_lastTime;
    float cameraSpeed = length(cameraMove) / max(pushConst.dt, 1e-4f);
    // 朝向变化量：用当前和上一帧 forward 的夹角近似
    float rotationDelta = 1.0f - saturate(dot(cameraForward, cameraForward_lastTime));

    // 运动时的上限只到 0.3（而不是 0.5）：拖影本来是靠上面的邻域钳制挡住的，
    // 钳制之后历史已经不可能偏离邻域范围，所以这里不必再为了防拖影而丢历史。
    // 丢历史的代价是直接的——一帧只有 8 个步进采样点，历史留得少就压不住空间高频，
    // 相机移动时表现为雾里的颗粒/锯齿。实测运动中 0.5 的空间高频明显高于 0.3。
    float motion = saturate(cameraSpeed * 0.02f + rotationDelta * 20.0f);
    float mergeRatio = lerp(0.08f, 0.3f, motion);   // 当前帧的权重

    curVoxelInfo.xy = lerp(lastVoxelInfo, curVoxelInfo.xy, mergeRatio);

    volumetricFogAccResultImage[voxelIndex] = curVoxelInfo;
}
//------------------------------------------------------------------------------------------------------------------------


#endif