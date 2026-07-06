#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVisibleVolumetricFogIndexBuffer)]] RWStructuredBuffer<uint> visibleVolumetricFogIndexBuffer;

// localVolumetricFogCount <= 1024, so one threadGroup can deal with
groupshared uint groupVisibleFogCount;
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getVisibleVolumetricFog(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) groupVisibleFogCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= pushConst.volumetricFogCount) return;

    VolumetricFogInfo info = volumetricFogInfosBuffer[threadIndex];
    float4x4 cameraPV = pushConst.sceneInfoAddress[0].viewProjMatrix;

    FrustumPlane planes[6];
    ExtractFrustumPlanes(cameraPV, planes);

    AABB fogAABB = { info.fogStartPos, info.fogStartPos + info.fogVoxelGridSize * info.fogVoxelSize };

    bool inCameraFrustum = isAABBInsideFrustum(fogAABB, planes);

    uint warpIndex = threadIndex / 32;
    uint warpLane = threadIndex & 31;
    uint warpVisibleFogCount = WaveActiveSum(inCameraFrustum);

    uint visibleFogOffset;
    if (warpLane == 0)
        InterlockedAdd(groupVisibleFogCount, warpVisibleFogCount, visibleFogOffset);
    visibleFogOffset = WaveReadLaneAt(visibleFogOffset, 0);
    GroupMemoryBarrierWithGroupSync();

    visibleFogOffset += WavePrefixCountBits(inCameraFrustum);
    visibleVolumetricFogIndexBuffer[visibleFogOffset] = threadIndex;

    //if (pushConst.frameIndex == 0 && inCameraFrustum) printf("%d %d\n", threadIndex, visibleFogOffset);

    if (threadIndex == 0) globalInfoBuffer[0].visibleVolumetricFogCount = groupVisibleFogCount;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImages)]] RWTexture3D<float4> volumetricFogImages[];

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (threadIndex.x >= fogInfo.fogVoxelGridSize.x || threadIndex.y >= fogInfo.fogVoxelGridSize.y || threadIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * threadIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float scale = (voxelCenter.y - fogInfo.fogStartPos.y) / (fogInfo.fogVoxelGridSize.y * fogInfo.fogVoxelSize.y);
    float absorption = fogInfo.absorption.y - (fogInfo.absorption.y - fogInfo.absorption.x) * pow(scale, 2);

    //uint threadIndexU = threadIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y) + threadIndex.y * pushConst.fogVoxelGridSize.x + threadIndex.x;
    //uint randomSeed = xxhash32(uint3(threadIndexU, pushConst.frameIndex, 1));

    float extinctionCoefficient = absorption + fogInfo.scattering;
    volumetricFogImages[pushConst.instanceIndex][threadIndex] = extinctionCoefficient;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo1Image)]] RWTexture3D<float4> volumetricFogVoxelInfo1Image[]; // xyz: velocity; w:
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfoBuffer)]] RWStructuredBuffer<VolumetricFogVoxelInfo, ScalarDataLayout> volumetricFogVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo1Image_sample)]] Sampler3D<float4> volumetricFogVoxelInfo1Image_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImages_sampler)]] Sampler3D<float4> volumetricFogImages_sample[];

groupshared VolumetricFogInfo groupFogInfo;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initVolumetricFogFluid(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    uint threadIndexU = voxelIndex.z * (fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y) + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(voxelIndex);
    float noise = rand(randomSeed) * 0.05f;

    float absorption = 0.1f;
    if (voxelIndex.y > fogInfo.fogVoxelGridSize.y / 2) absorption = 1.0f;

    volumetricFogImages[pushConst.instanceIndex][voxelIndex] = absorption + noise;

    volumetricFogVoxelInfo1Image[pushConst.volumetricFogFluidIndex][voxelIndex] = 0.0f; //float4(rand(randomSeed), rand(randomSeed), rand(randomSeed), 0.0f);
    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure = { float4(0.0f), float4(0.0f) };

    if (any(voxelIndex == 0) || any(voxelIndex == fogInfo.fogVoxelGridSize - 1)) volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].isBoundary = true;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    float3 velocity = volumetricFogVoxelInfo1Image_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw, 0).xyz;

    float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
    float3 uvw_lastTime = (worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    uvw_lastTime = clamp(uvw_lastTime, 0.0f, 1.0f);
    float3 velocity_lastTime = volumetricFogVoxelInfo1Image_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw_lastTime, 0).xyz;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure = { float4(velocity_lastTime, 0.0f), float4(velocity_lastTime, 0.0f) };

    //float3 absorption = volumetricFogImages_sample[pushConst.instanceIndex] .SampleLevel((worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize), 0) .xyz;
    float3 absorption = volumetricFogImages_sample[pushConst.instanceIndex].SampleLevel(uvw_lastTime, 0).xyz;
    volumetricFogImages[pushConst.instanceIndex][voxelIndex] = float4(absorption, 1.0f);

    //if (all(voxelIndex == uint3(0, fogInfo.fogVoxelGridSize.y - 1, 0))) {
    //    printf("A Velocity: %f %f %f, Velocity_lastTime: %f %f %f\n", velocity.x, velocity.y, velocity.z, velocity_lastTime.x, velocity_lastTime.y, velocity_lastTime.z);
    //}
}

groupshared float4 groupVoxelInfos[6][6][6];
void fillGroupSharedVoxelInfos(uint3 voxelIndex, uint3 threadGroupIndex, uint3 voxelGridSize) {
    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4;
    uint voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;

    uint iteration = pushConst.iteration & 1;

    float4 selfData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
    groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = selfData;

    uint3 groupBase = threadGroupIndex * 4;
    uint3 threadGroupSize = min(voxelGridSize - groupBase, uint3(4, 4, 4));

    if (localVoxelIndex.x == 0) {
        float4 neighborData = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        if (voxelIndex.x > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x - 1;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][0] = neighborData;
    }
    else if (localVoxelIndex.x == threadGroupSize.x - 1) {
        float4 neighborData = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        if (voxelIndex.x < voxelGridSize.x - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x + 1;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][threadGroupSize.x + 1] = neighborData;
    }

    if (localVoxelIndex.y == 0) {
        float4 neighborData = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        if (voxelIndex.y > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y - 1) * voxelGridSize.x + voxelIndex.x;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[localVoxelIndex.z + 1][0][localVoxelIndex.x + 1] = neighborData;
    }
    else if (localVoxelIndex.y == threadGroupSize.y - 1) {
        float4 neighborData = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        if (voxelIndex.y < voxelGridSize.y - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y + 1) * voxelGridSize.x + voxelIndex.x;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[localVoxelIndex.z + 1][threadGroupSize.y + 1][localVoxelIndex.x + 1] = neighborData;
    }

    if (localVoxelIndex.z == 0) {
        float4 neighborData = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        if (voxelIndex.z > 0) {
            voxelIndexU = (voxelIndex.z - 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[0][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
    }
    else if (localVoxelIndex.z == threadGroupSize.z - 1) {
        float4 neighborData = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        if (voxelIndex.z < voxelGridSize.z - 1) {
            voxelIndexU = (voxelIndex.z + 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            neighborData = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration];
        }
        groupVoxelInfos[threadGroupSize.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
    }
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_D(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4 + 1;
    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    float3 x_neighbor = 0.0f;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x - 1].xyz;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x + 1].xyz;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y - 1][localVoxelIndex.x].xyz;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y + 1][localVoxelIndex.x].xyz;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z - 1][localVoxelIndex.y][localVoxelIndex.x].xyz;
    x_neighbor += groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y][localVoxelIndex.x].xyz;

    float alpha = (fogInfo.fogVoxelSize.x * fogInfo.fogVoxelSize.x) / max(fogInfo.viscosity * pushConst.dt, 1e-30f);

    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].xyz =
        (x_neighbor + alpha * groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x].xyz) / (6.0f + alpha);

    //if (all(voxelIndex == uint3(0, fogInfo.fogVoxelGridSize.y - 1, 0))) {
    //    float3 diffuseVelocity = (x_neighbor + alpha * groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x].xyz) / (6.0f + alpha);
    //    printf("D %d: Velocity: %f %f %f\n", pushConst.iteration, diffuseVelocity.x, diffuseVelocity.y, diffuseVelocity.z);
    //}
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_F(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;
    uint fIdx = pushConst.volumetricFogFluidIndex;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint iteration = pushConst.iteration & 1;

    float3 worldPos = fogInfo.fogStartPos + fogInfo.fogVoxelSize * (float3(voxelIndex) + 0.5f);
    float density = volumetricFogImages_sample[pushConst.instanceIndex].SampleLevel((worldPos - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize), 0).x;

    float3 vel = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[iteration].xyz;
    vel.y += (pushConst.ambientFogDensity - density) * 9.8f * pushConst.dt * fogInfo.gravitationalIntensity;
    //if (all(voxelIndex == 0)) printf("ambientFogDensity: %f\n", pushConst.ambientFogDensity);

    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[0].xyz = vel;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[1].xyz = vel;

    //if (all(voxelIndex == uint3(0, fogInfo.fogVoxelGridSize.y - 1, 0))) {
    //    printf("F Velocity: %f %f %f\n", vel.x, vel.y, vel.z);
    //}
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_P(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4 + 1;
    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    uint3 dims = fogInfo.fogVoxelGridSize;
    float3 cellSize = fogInfo.fogVoxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = halfDx; // 2.0f * cellSize.y;
    float halfDz = halfDx; // 2.0f * cellSize.z;

    float3 velL = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x - 1].xyz;
    float pL = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x - 1].w;

    float3 velR = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x + 1].xyz;
    float pR = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x + 1].w;

    float3 velB = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y - 1][localVoxelIndex.x].xyz;
    float pB = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y - 1][localVoxelIndex.x].w;

    float3 velT = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y + 1][localVoxelIndex.x].xyz;
    float pT = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y + 1][localVoxelIndex.x].w;

    float3 velD = groupVoxelInfos[localVoxelIndex.z - 1][localVoxelIndex.y][localVoxelIndex.x].xyz;
    float pD = groupVoxelInfos[localVoxelIndex.z - 1][localVoxelIndex.y][localVoxelIndex.x].w;

    float3 velU = groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y][localVoxelIndex.x].xyz;
    float pU = groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y][localVoxelIndex.x].w;

    float div = (velR.x - velL.x) / halfDx + (velT.y - velB.y) / halfDy + (velU.z - velD.z) / halfDz;
    float alpha = cellSize.x * cellSize.x;
    float sumPressure = pL + pR + pB + pT + pD + pU;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].w = (sumPressure - alpha * div) / 6.0f;

    //if (all(voxelIndex == uint3(0, fogInfo.fogVoxelGridSize.y - 1, 0))) {
    //    float pressure = (sumPressure - alpha * div) / 6.0f;
    //    printf("P %d: pressure: %f\n", pushConst.iteration, pressure);
    //}
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_S(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (groupThreadIndex.x == 0 && groupThreadIndex.y == 0 && groupThreadIndex.z == 0) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4 + 1;

    float3 vel_dirty = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x].xyz;

    uint3 dims = fogInfo.fogVoxelGridSize;
    float3 cellSize = fogInfo.fogVoxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = halfDx; // 2.0f * cellSize.y;
    float halfDz = halfDx; // 2.0f * cellSize.z;

    float pL = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x - 1].w;
    float pR = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y][localVoxelIndex.x + 1].w;
    float pB = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y - 1][localVoxelIndex.x].w;
    float pT = groupVoxelInfos[localVoxelIndex.z][localVoxelIndex.y + 1][localVoxelIndex.x].w;
    float pD = groupVoxelInfos[localVoxelIndex.z - 1][localVoxelIndex.y][localVoxelIndex.x].w;
    float pU = groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y][localVoxelIndex.x].w;

    float3 pg = float3((pR - pL) / halfDx, (pT - pB) / halfDy, (pU - pD) / halfDz);
    volumetricFogVoxelInfo1Image[pushConst.volumetricFogFluidIndex][voxelIndex].xyz = vel_dirty - pg;

    //if (all(voxelIndex == uint3(0, fogInfo.fogVoxelGridSize.y - 1, 0))) {
    //    float3 V = vel_dirty - pg;
    //    printf("S Velocity: %f %f %f\n", V.x, V.y, V.z);
    //}
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_createLightAttenuationEstimator(uint threadIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 voxelSize = fogInfo.fogVoxelSize;
    int3 gridSize = fogInfo.fogVoxelGridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = gridMin + gridSize * voxelSize;

    float3 startPos = gridMax;
    float3 endPos = gridMin;
    float3 dir = normalize(endPos - startPos);

    float tEnter = 0.0f;
    float3 tExit3;
    tExit3.x = (abs(dir.x) > 1e-8) ? (gridMax.x - gridMin.x) / abs(dir.x) : 1e30;
    tExit3.y = (abs(dir.y) > 1e-8) ? (gridMax.y - gridMin.y) / abs(dir.y) : 1e30;
    tExit3.z = (abs(dir.z) > 1e-8) ? (gridMax.z - gridMin.z) / abs(dir.z) : 1e30;
    float tExit = min(min(tExit3.x, tExit3.y), tExit3.z);

    float eps = 1e-5 * min(min(voxelSize.x, voxelSize.y), voxelSize.z);
    float tCurrent = tEnter + eps;
    float3 pos = startPos + dir * tCurrent;

    float3 invDir; int3 voxel; int3 step; float3 tDelta; float3 voxelBorder; float3 tMax;
    {
        invDir.x = (abs(dir.x) > 1e-8) ? rcp(dir.x) : 1e30;
        invDir.y = (abs(dir.y) > 1e-8) ? rcp(dir.y) : 1e30;
        invDir.z = (abs(dir.z) > 1e-8) ? rcp(dir.z) : 1e30;

        float3 voxelF = (pos - fogStartPos) / voxelSize;
        voxel = int3(floor(voxelF));
        voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

        step.x = (dir.x > 0.0) ? 1 : ((dir.x < 0.0) ? -1 : 0);
        step.y = (dir.y > 0.0) ? 1 : ((dir.y < 0.0) ? -1 : 0);
        step.z = (dir.z > 0.0) ? 1 : ((dir.z < 0.0) ? -1 : 0);

        tDelta.x = (step.x != 0) ? voxelSize.x * abs(invDir.x) : 1e30;
        tDelta.y = (step.y != 0) ? voxelSize.y * abs(invDir.y) : 1e30;
        tDelta.z = (step.z != 0) ? voxelSize.z * abs(invDir.z) : 1e30;

        voxelBorder.x = (step.x > 0) ? (fogStartPos.x + (voxel.x + 1) * voxelSize.x) : (fogStartPos.x + voxel.x * voxelSize.x);
        voxelBorder.y = (step.y > 0) ? (fogStartPos.y + (voxel.y + 1) * voxelSize.y) : (fogStartPos.y + voxel.y * voxelSize.y);
        voxelBorder.z = (step.z > 0) ? (fogStartPos.z + (voxel.z + 1) * voxelSize.z) : (fogStartPos.z + voxel.z * voxelSize.z);

        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        if (step.x != 0 && tMax.x < tCurrent) tMax.x += tDelta.x;
        if (step.y != 0 && tMax.y < tCurrent) tMax.y += tDelta.y;
        if (step.z != 0 && tMax.z < tCurrent) tMax.z += tDelta.z;
    }

    float3 attenuation = 1.0f;
    int randomSeed = pushConst.frameIndex;
    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, tExit);
        float dt = tNext - tCurrent;

        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
            float extinctionCoeff = volumetricFogImages_sample[pushConst.instanceIndex].SampleLevel(uvw, 0).x;

            attenuation *= exp(-extinctionCoeff * dt);
            if (all(attenuation <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }

    volumetricFogInfosBuffer[pushConst.instanceIndex].lightAttenuationEstimator = attenuation.x;
}