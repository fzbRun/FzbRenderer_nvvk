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
//-----------------------------------------------------------------------------Height----------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogExtinctionImages)]] RWTexture3D<float4> volumetricFogExtinctionImages[];

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    float3 voxelCenter = fogInfo.fogStartPos + (float3(voxelIndex) + 0.5f) * fogInfo.fogVoxelSize;
    float fogHeight = fogInfo.fogStartPos.y + fogInfo.fogVoxelGridSize.y * fogInfo.fogVoxelSize.y;
    float t = saturate((voxelCenter.y - fogInfo.fogStartPos.y) / (fogHeight - fogInfo.fogStartPos.y));

    float3 normalizedPos = (float3(voxelIndex) + 0.5f) / float3(fogInfo.fogVoxelGridSize);
    float3 cloudPos = float3(voxelCenter.x * 0.09f, voxelCenter.y * 0.24f, voxelCenter.z * 0.09f);

    float3 warp = float3(
        fbm(cloudPos * 0.65f + float3(13.1f, 4.7f, 9.2f)),
        fbm(cloudPos * 0.65f + float3(2.6f, 15.4f, 7.8f)),
        fbm(cloudPos * 0.65f + float3(6.9f, 1.8f, 21.3f))) * 2.0f - 1.0f;
    cloudPos.xz += warp.xz * 4.0f;
    cloudPos.y += warp.y * 0.85f;

    float broadCloud = fbm(cloudPos * 1.45f + 19.0f) * 0.5f + 0.5f;
    float rollingCloud = fbm(cloudPos * 3.2f + 53.0f) * 0.5f + 0.5f;
    float silkDetail = fbm(cloudPos * 8.0f + 97.0f) * 0.5f + 0.5f;

    float cloudBody = smoothstep(0.30f, 0.72f, broadCloud);
    cloudBody = lerp(0.38f, 1.0f, cloudBody);
    cloudBody *= lerp(0.70f, 1.35f, rollingCloud);
    cloudBody *= lerp(0.82f, 1.12f, silkDetail);

    float bottomFeather = smoothstep(0.0f, 0.08f, t);
    float halfBodyFalloff = 1.0f - smoothstep(0.58f, 1.0f, t + warp.y * 0.10f);
    float middleLift = smoothstep(0.08f, 0.34f, t) * (1.0f - smoothstep(0.66f, 0.94f, t));

    float edgeBreakup = smoothstep(0.03f, 0.14f, normalizedPos.x)
        * (1.0f - smoothstep(0.86f, 0.97f, normalizedPos.x))
        * smoothstep(0.03f, 0.14f, normalizedPos.z)
        * (1.0f - smoothstep(0.86f, 0.97f, normalizedPos.z));

    float densityMask = saturate(cloudBody * bottomFeather * halfBodyFalloff * edgeBreakup);
    densityMask = saturate(densityMask + middleLift * rollingCloud * 0.25f);

    float extinction = (fogInfo.absorption.y + fogInfo.scattering) * 1.65f * densityMask;

    volumetricFogExtinctionImages[pushConst.instanceIndex][voxelIndex] = extinction;
}
//--------------------------------------------------------------------------------Fluid-------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelVelocityImage)]] RWTexture3D<float4> volumetricFogVoxelVelocityImage[]; // xyz: velocity; w:
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfoBuffer)]] RWStructuredBuffer<VolumetricFogVoxelInfo, ScalarDataLayout> volumetricFogVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelVelocityImage_sample)]] Sampler3D<float4> volumetricFogVoxelVelocityImage_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogExtinctionImages_sampler)]] Sampler3D<float4> volumetricFogExtinctionImages_sample[];

groupshared VolumetricFogInfo groupFogInfo;

float getRestSmokeDensity(uint3 voxelIndex, VolumetricFogInfo fogInfo) {
    float3 gridSize = float3(fogInfo.fogVoxelGridSize);
    float3 normalizedPos = (float3(voxelIndex) + 0.5f) / gridSize;
    float3 sphereCenter = float3(0.5f, 0.5f, 0.5f);
    float3 spherePos = (normalizedPos - sphereCenter) * 2.0f;

    float radius = 0.72f;
    float dist = length(spherePos);
    float sphereMask = 1.0f - smoothstep(radius * 0.72f, radius, dist);

    float lowFreqNoise = fbm(normalizedPos * 4.0f + 17.0f) * 0.5f + 0.5f;
    float highFreqNoise = fbm(normalizedPos * 12.0f + 31.0f) * 0.5f + 0.5f;
    float noise = saturate(lerp(lowFreqNoise, highFreqNoise, 0.35f));
    float noisyMask = saturate(sphereMask * lerp(0.45f, 1.25f, noise));

    return (fogInfo.absorption.y + fogInfo.scattering) * noisyMask;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFogFluid(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    volumetricFogExtinctionImages[pushConst.instanceIndex][voxelIndex] = getRestSmokeDensity(voxelIndex, fogInfo);

    volumetricFogVoxelVelocityImage[pushConst.volumetricFogFluidIndex][voxelIndex] = 0.0f;
    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure = { float4(0.0f), float4(0.0f) };
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].isBoundary = false;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initVolumetricFogFluid(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    //if (any(voxelIndex == 0) || any(voxelIndex == fogInfo.fogVoxelGridSize - 1)) isBoundary = true;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].isBoundary = false;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].density = 0.0f;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    VolumetricFogVoxelInfo voxelInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    // float3 velocity = volumetricFogVoxelVelocityImage[pushConst.volumetricFogFluidIndex][voxelIndex].xyz;
    float3 uvw = (voxelCenter - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    float3 velocity = volumetricFogVoxelVelocityImage_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw, 0).xyz;

    float3 velocity_lastTime = velocity;
    if (voxelInfo.isBoundary) voxelInfo.density = 0.0f;
    else{
        float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
        float3 uvw_lastTime = (worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
        uvw_lastTime = clamp(uvw_lastTime, 0.0f, 1.0f);
        velocity_lastTime = volumetricFogVoxelVelocityImage_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw_lastTime, 0).xyz;

        voxelInfo.density = volumetricFogExtinctionImages_sample[pushConst.instanceIndex].SampleLevel(uvw_lastTime, 0).xyz;
        // uint3 voxelIndex_lastTime = uint3((worldPos_lastTime - fogInfo.fogStartPos) / fogInfo.fogVoxelSize);
        // voxelIndex_lastTime = clamp(voxelIndex_lastTime, 1, fogInfo.fogVoxelGridSize - 2);
        // voxelInfo.density = volumetricFogImages[pushConst.instanceIndex][voxelIndex_lastTime].xyz;
    }

    voxelInfo.dirtyVelocity_pressure[0].xyz = velocity_lastTime;
    voxelInfo.dirtyVelocity_pressure[1].xyz = velocity_lastTime;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU] = voxelInfo;
}

groupshared float4 groupVoxelInfos[6][6][6];
groupshared bool groupVoxelBoundary[6][6][6];
void fillGroupSharedVoxelInfos(uint3 voxelIndex, uint3 threadGroupIndex, uint3 voxelGridSize) {
    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4;
    uint voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;

    uint iteration = pushConst.iteration & 1;

    VolumetricFogVoxelInfo voxelInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
    float4 selfData = voxelInfo.dirtyVelocity_pressure[iteration];
    groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = selfData;
    groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = voxelInfo.isBoundary;

    uint3 groupBase = threadGroupIndex * 4;
    uint3 threadGroupSize = min(voxelGridSize - groupBase, uint3(4, 4, 4));

    if (localVoxelIndex.x == 0) {
        float4 neighborData = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.x > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x - 1;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][0] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][0] = neighborBoundary;
    }
    if (localVoxelIndex.x == threadGroupSize.x - 1) {
        float4 neighborData = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.x < voxelGridSize.x - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x + 1;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][threadGroupSize.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][threadGroupSize.x + 1] = neighborBoundary;
    }

    if (localVoxelIndex.y == 0) {
        float4 neighborData = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.y > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y - 1) * voxelGridSize.x + voxelIndex.x;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][0][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][0][localVoxelIndex.x + 1] = neighborBoundary;
    }
    if (localVoxelIndex.y == threadGroupSize.y - 1) {
        float4 neighborData = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.y < voxelGridSize.y - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y + 1) * voxelGridSize.x + voxelIndex.x;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][threadGroupSize.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][threadGroupSize.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }

    if (localVoxelIndex.z == 0) {
        float4 neighborData = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.z > 0) {
            voxelIndexU = (voxelIndex.z - 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[0][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[0][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }
    if (localVoxelIndex.z == threadGroupSize.z - 1) {
        float4 neighborData = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        //float4 neighborData = float4(-selfData.xyz, selfData.w);
        bool neighborBoundary = true;
        if (voxelIndex.z < voxelGridSize.z - 1) {
            voxelIndexU = (voxelIndex.z + 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            VolumetricFogVoxelInfo neighborInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[threadGroupSize.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[threadGroupSize.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_D(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    float3 x_neighbor = 0.0f;
    {
        float4 n = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
        //if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) n = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
        //if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) n = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
        //if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) n = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
        //if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) n = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
        //if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) n = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
        //if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) n = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        x_neighbor += n.xyz;
    }

    float alpha = (fogInfo.fogVoxelSize.x * fogInfo.fogVoxelSize.x) / max(fogInfo.viscosity * pushConst.dt, 1e-30f);

    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].xyz = (x_neighbor + alpha * selfData.xyz) / (6.0f + alpha);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_F(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    VolumetricFogVoxelInfo voxelInfo = volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU];
    if (!voxelInfo.isBoundary) {
        uint fIdx = pushConst.volumetricFogFluidIndex;
        uint iteration = pushConst.iteration & 1;

        float3 worldPos = fogInfo.fogStartPos + fogInfo.fogVoxelSize * (float3(voxelIndex) + 0.5f);

        float3 vel = voxelInfo.dirtyVelocity_pressure[iteration].xyz;
        // vel.y += (pushConst.ambientFogDensity - voxelInfo.density.x) * 9.8f * pushConst.dt * fogInfo.FIntensity;

        voxelInfo.dirtyVelocity_pressure[0].xyz = vel;
        voxelInfo.dirtyVelocity_pressure[1].xyz = vel;

        float restDensity = getRestSmokeDensity(voxelIndex, fogInfo);
        float restoreRate = 0.35f;
        float speedDamping = 1.0f - saturate(length(vel) * 2.0f);
        float restoreWeight = saturate(restoreRate * pushConst.dt) * speedDamping * 20.0f;
        voxelInfo.density = lerp(voxelInfo.density, float3(restDensity), restoreWeight);
    }

    volumetricFogExtinctionImages[pushConst.instanceIndex][voxelIndex] = float4(voxelInfo.density, 0.0f);
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU] = voxelInfo;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_P(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    float3 cellSize = fogInfo.fogVoxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = halfDx;
    float halfDz = halfDx;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];

    // L / R (-X / +X)
    float4 nL = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
    // if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) { nL = float4(-selfData.x, selfData.y, selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) nL.w = selfData.w;
    float4 nR = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
    // if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) { nR = float4(-selfData.x, selfData.y, selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) nR.w = selfData.w;

    // B / T (-Y / +Y)
    float4 nB = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
    // if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) { nB = float4(selfData.x, -selfData.y, selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) nB.w = selfData.w;
    float4 nT = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
    // if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) { nT = float4(selfData.x, -selfData.y, selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) nT.w = selfData.w;

    // D / U (-Z / +Z)
    float4 nD = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
    // if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) { nD = float4(selfData.x, selfData.y, -selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) nD.w = selfData.w;
    float4 nU = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
    // if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) { nU = float4(selfData.x, selfData.y, -selfData.z, selfData.w); }
    if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) nU.w = selfData.w;

    float div = (nR.x - nL.x) / halfDx + (nT.y - nB.y) / halfDy + (nU.z - nD.z) / halfDz;
    float alpha = -cellSize.x * cellSize.x;
    float sumPressure = nL.w + nR.w + nB.w + nT.w + nD.w + nU.w;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;
    volumetricFogVoxelInfoBuffer[pushConst.volumetricFogFluidIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].w = (sumPressure + alpha * div) / 6.0f;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_S(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;

    float3 velocity = 0.0f;
    if (!groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) {
        float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
        float3 vel_dirty = selfData.xyz;

        float3 cellSize = fogInfo.fogVoxelSize;
        float halfDx = 2.0f * cellSize.x;
        float halfDy = halfDx;
        float halfDz = halfDx;

        float3 obstV = float3(0, 0, 0);
        float3 vMask = float3(1, 1, 1);

        float4 nL = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
        if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1] || voxelIndex.x == 0) {
            nL.w = selfData.w;
            obstV.x = nL.x; vMask.x = 0;
        }
        float4 nR = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
        if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1] || voxelIndex.x == fogInfo.fogVoxelGridSize.x - 1) {
            nR.w = selfData.w;
            obstV.x = nR.x; vMask.x = 0;
        }
        float4 nB = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
        if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x] || voxelIndex.y == 0) {
            nB.w = selfData.w;
            obstV.y = nB.y; vMask.y = 0;
        }
        float4 nT = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
        if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x] || voxelIndex.y == fogInfo.fogVoxelGridSize.y - 1) {
            nT.w = selfData.w;
            obstV.y = nT.y; vMask.y = 0;
        }
        float4 nD = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
        if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x] || voxelIndex.z == 0) {
            nD.w = selfData.w;
            obstV.z = nD.z; vMask.z = 0;
        }
        float4 nU = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
        if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x] || voxelIndex.z == fogInfo.fogVoxelGridSize.z - 1) {
            nU.w = selfData.w;
            obstV.z = nU.z; vMask.z = 0;
        }

        float3 pg = float3((nR.w - nL.w) / halfDx, (nT.w - nB.w) / halfDy, (nU.w - nD.w) / halfDz);
        velocity = ((vel_dirty - pg) * vMask) + obstV;
    }

    volumetricFogVoxelVelocityImage[pushConst.volumetricFogFluidIndex][voxelIndex].xyz = velocity;
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
            float extinctionCoeff = volumetricFogExtinctionImages_sample[pushConst.instanceIndex].SampleLevel(uvw, 0).x;

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
