#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> volumetricFogFluidInfoBuffer;

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
    uint warpVisibleFogCount = WaveActiveSum(uint(inCameraFrustum));

    uint visibleFogOffset;
    if (warpLane == 0)
        InterlockedAdd(groupVisibleFogCount, warpVisibleFogCount, visibleFogOffset);
    visibleFogOffset = WaveReadLaneAt(visibleFogOffset, 0);
    GroupMemoryBarrierWithGroupSync();

    visibleFogOffset += WavePrefixCountBits(inCameraFrustum);
    if (inCameraFrustum)
        globalInfoBuffer[0].visibleVolumetricFogIndices[visibleFogOffset] = threadIndex;

    if (threadIndex == 0) {
        VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.fluidFogIndex];
        FluidFogInfo fluidFogInfo = volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex];

        globalInfoBuffer[0].visibleVolumetricFogCount = groupVisibleFogCount;
        globalInfoBuffer[0].fluidAABB = { fogInfo.fogStartPos, fogInfo.fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
        globalInfoBuffer[0].fluidStartUp = fluidFogInfo.startUp;
    }
}
//-----------------------------------------------------------------------------Height----------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;


[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    return;
}
//--------------------------------------------------------------------------------Fluid-------------------------------------------------------------------------------------
// BoundaryCondition_Velocity: float3(1.0f, 1.0f, 1.0f): Non; float3(-1.0f, 1.0f, 1.0f): Slip; float3(-1.0f, -1.0f, -1.0f): NoSlip
// BoundaryCondition_Pressure: 0: Non; 1: Slip or NoSlip;
#define BoundaryConditionType 0
#if BoundaryConditionType == 0  //Non
    #define BoundaryCondition_Have false
    #define BoundaryCondition_Velocity float3(1.0f, 1.0f, 1.0f)
    #define BoundaryCondition_Pressure 0
#else
    #define BoundaryCondition_Have true
    #define BoundaryCondition_Pressure 1
    #if BoundaryConditionType == 1 // Slip
        #define BoundaryCondition_Velocity float3(-1.0f, 1.0f, 1.0f)
    #else // NoSlip
        #define BoundaryCondition_Velocity float3(-1.0f, -1.0f, -1.0f)
    #endif
#endif

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage)]] RWTexture3D<float4> volumetricFogFluidVoxelVelocityImage[]; // xyz: velocity; w:
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer)]] RWStructuredBuffer<VolumetricFogFluidVoxelInfo, ScalarDataLayout> volumetricFogFluidVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample)]] Sampler3D<float4> volumetricFogFluidVoxelVelocityImage_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages)]] RWTexture3D<float4> volumetricFogFluidVoxelInfoImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages_sampler)]] Sampler3D<float4> volumetricFogFluidVoxelInfoImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer)]] StructuredBuffer<NoiseFogInfo, ScalarDataLayout> noiseFogInfoBuffer;

// ensure OTHER_TYPE_FOG_COUNT > volumetricFogCount
#define OTHER_TYPE_FOG_COUNT 3
groupshared VolumetricFogInfo groupOtherTypeFogInfos[OTHER_TYPE_FOG_COUNT];

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
void computeMain_initVolumetricFogFluid(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    if (groupThreadIndexU < pushConst.volumetricFogCount) groupOtherTypeFogInfos[groupThreadIndexU] = volumetricFogInfosBuffer[groupThreadIndexU];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupOtherTypeFogInfos[pushConst.instanceIndex];
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    FluidFogInfo fluidFogInfo = volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex];

    float3 voxelCenter = fogInfo.fogStartPos + (float3(voxelIndex) + 0.5f) * fogInfo.fogVoxelSize;
    float heightAttenuation = exp(-1.0f * (voxelCenter.y - (-69.5f)));
    float extinction = (0.01f + 0.03f) * heightAttenuation;
    float envExtinction = extinction;
    float envScattering = extinction * 0.03f;
    float envPhase = extinction * -0.7f;
    float3 envColor = extinction * 1.0f;

    for (int i = 0; i < pushConst.volumetricFogCount; ++i) {
        if (i == pushConst.instanceIndex) continue;
        VolumetricFogInfo otherFogInfo = groupOtherTypeFogInfos[i];
        AABB otherFogAABB = { otherFogInfo.fogStartPos, otherFogInfo.fogStartPos + otherFogInfo.fogVoxelGridSize * otherFogInfo.fogVoxelSize };
        if (!checkPointInAABB(voxelCenter, otherFogAABB)) continue;

        if (otherFogInfo.type == VolumetricFogType::Height) {
            float heightScale = heightFogInfoBuffer[otherFogInfo.volumetricFogTypeIndex].heightScale;
            float height = voxelCenter.y - otherFogInfo.fogStartPos.y;
            heightAttenuation = exp(-heightScale * height);
            extinction = (otherFogInfo.absorption.x + otherFogInfo.scattering) * heightAttenuation;
        } else if (otherFogInfo.type == VolumetricFogType::Fluid) continue;
        else if (otherFogInfo.type == VolumetricFogType::Noise) {
            NoiseFogInfo noiseFogInfo = noiseFogInfoBuffer[otherFogInfo.volumetricFogTypeIndex];
            CloudSetting cloudSetting = { noiseFogInfo.cloudScale, noiseFogInfo.cloudFlowSpeed, noiseFogInfo.cloudCoverage, noiseFogInfo.cloudTypePreference, noiseFogInfo.weatherScale };
            float fogDensity = SampleCloudDensity(voxelCenter, otherFogAABB.minimum, otherFogAABB.maximum, pushConst.time, cloudSetting);
            extinction = (otherFogInfo.absorption.x + otherFogInfo.scattering) * fogDensity;
        }

        envExtinction += extinction;
        envScattering += extinction * otherFogInfo.scattering;
        envPhase += extinction * otherFogInfo.phase;
        envColor += extinction * otherFogInfo.color;
    }
    envScattering /= max(envExtinction, 1e-30f);
    envPhase /= max(envExtinction, 1e-30f);
    envColor /= max(envExtinction, 1e-30f);
    float4 envVoxelInfo = float4(envExtinction, envScattering, envPhase, luminance(envColor));
    float4 voxelInfo = envVoxelInfo;

    int3 voxelIndex_lastTime = int3((voxelCenter - fluidFogInfo.fogStartPos_lastTime) / fogInfo.fogVoxelSize);
    if (any(voxelIndex_lastTime < 0) || any(voxelIndex_lastTime > fogInfo.fogVoxelGridSize - 1) || pushConst.time == 0) {
        volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex].xyz = 0.0f;
    } else {
        voxelInfo = volumetricFogFluidVoxelInfoImages[fogInfo.volumetricFogTypeIndex][voxelIndex_lastTime];

        float3 velocity = volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex_lastTime].xyz;
        //float3 halfGrid = (float3)(fogInfo.fogVoxelGridSize) * 0.5f;
        //float3 distToEdge = halfGrid - abs((float3)voxelIndex - halfGrid);
        //float d = length(distToEdge * fogInfo.fogVoxelSize);
        //float speedDamping = exp(-length(velocity) * d * 3.0f); // spped + distance
        //float restoreRate = 0.5f;
        //float restoreWeight = saturate(restoreRate * pushConst.dt) * speedDamping * fluidFogInfo.restoreSpeed;

        float3 halfGrid = float3(fogInfo.fogVoxelGridSize) * 0.5f;
        float3 distToCenter = abs(float3(voxelIndex) - halfGrid);
        float maxPossibleDist = length(halfGrid);
        float centerDist = length(distToCenter) / maxPossibleDist;

        float edgeFactor = smoothstep(0.0, 1.0, centerDist);
        float restoreRate = 0.5f;
        float restoreWeight = saturate(restoreRate * pushConst.dt) * edgeFactor * fluidFogInfo.restoreSpeed;

        voxelInfo = lerp(voxelInfo, envVoxelInfo, restoreWeight);

        velocity = lerp(velocity, 0.0f, restoreWeight); 
        volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex].xyz = velocity;
    }
    volumetricFogFluidVoxelInfoImages[fogInfo.volumetricFogTypeIndex][voxelIndex] = voxelInfo;

    /*
        if (pushConst.time == 0) {
            float4 voxelInfo;
            voxelInfo.x = getRestSmokeDensity(voxelIndex, fogInfo);
            voxelInfo.y = fogInfo.scattering;
            voxelInfo.z = fogInfo.phase;
            volumetricFogFluidVoxelInfoImages[fogInfo.volumetricFogTypeIndex][voxelIndex] = voxelInfo;

            volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex] = 0.0f;
        }
    */
    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU].isBoundary = false;
}

groupshared VolumetricFogInfo groupFogInfo;
groupshared FluidFogInfo groupFluidFogInfo;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    VolumetricFogFluidVoxelInfo voxelInfo = volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU];

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    float3 velocity = volumetricFogFluidVoxelVelocityImage_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0).xyz;

    float3 velocity_lastTime = velocity;
    if (voxelInfo.isBoundary) voxelInfo.voxelFogInfo = 0.0f;
    else{
        float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
        float3 uvw_lastTime = (worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
        #if BoundaryConditionType != 0
            uvw_lastTime = clamp(uvw_lastTime, 0.0f, 1.0f);
        #endif
        int3 voxelIdx_lastTime = int3((worldPos_lastTime - fogInfo.fogStartPos) / fogInfo.fogVoxelSize);
        voxelIdx_lastTime = clamp(voxelIdx_lastTime, int3(0, 0, 0), int3(fogInfo.fogVoxelGridSize) - 1);
        uint voxelIdxU_lastTime = voxelIdx_lastTime.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIdx_lastTime.y * fogInfo.fogVoxelGridSize.x + voxelIdx_lastTime.x;
        bool backtrackIntoBoundary = volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIdxU_lastTime].isBoundary;
        if (backtrackIntoBoundary) {
            voxelInfo.voxelFogInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0);
        } else {
            velocity_lastTime = volumetricFogFluidVoxelVelocityImage_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw_lastTime, 0).xyz;
            voxelInfo.voxelFogInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw_lastTime, 0);
        }
    }

    voxelInfo.dirtyVelocity_pressure[0].xyz = velocity_lastTime;
    voxelInfo.dirtyVelocity_pressure[1].xyz = velocity_lastTime;
    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU] = voxelInfo;
}

groupshared float4 groupVoxelInfos[6][6][6];
groupshared bool groupVoxelBoundary[6][6][6];
void fillGroupSharedVoxelInfos(uint3 voxelIndex, uint3 threadGroupIndex, uint32_t fluidIndex, uint3 voxelGridSize) {
    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4;
    uint voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;

    uint iteration = pushConst.iteration & 1;

    VolumetricFogFluidVoxelInfo voxelInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
    float4 selfData = voxelInfo.dirtyVelocity_pressure[iteration];
    groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = selfData;
    groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = voxelInfo.isBoundary;

    uint3 groupBase = threadGroupIndex * 4;
    uint3 threadGroupSize = min(voxelGridSize - groupBase, uint3(4, 4, 4));

    if (localVoxelIndex.x == 0) {
        float4 neighborData = float4(BoundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.x > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x - 1;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][0] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][0] = neighborBoundary;
    }
    if (localVoxelIndex.x == threadGroupSize.x - 1) {
        float4 neighborData = float4(BoundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.x < voxelGridSize.x - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x + 1;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][threadGroupSize.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][threadGroupSize.x + 1] = neighborBoundary;
    }

    if (localVoxelIndex.y == 0) {
        //float3 boundaryCondition_Velocity = float3(BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.x, BoundaryCondition_Velocity.z);
        //float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        float4 neighborData = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        bool neighborBoundary = true;// BoundaryCondition_Have;
        if (voxelIndex.y > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y - 1) * voxelGridSize.x + voxelIndex.x;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][0][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][0][localVoxelIndex.x + 1] = neighborBoundary;
    }
    if (localVoxelIndex.y == threadGroupSize.y - 1) {
        float3 boundaryCondition_Velocity = float3(BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.x, BoundaryCondition_Velocity.z);
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.y < voxelGridSize.y - 1) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + (voxelIndex.y + 1) * voxelGridSize.x + voxelIndex.x;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[localVoxelIndex.z + 1][threadGroupSize.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[localVoxelIndex.z + 1][threadGroupSize.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }

    if (localVoxelIndex.z == 0) {
        float3 boundaryCondition_Velocity = float3(BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.x);
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.z > 0) {
            voxelIndexU = (voxelIndex.z - 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_pressure[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[0][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[0][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }
    if (localVoxelIndex.z == threadGroupSize.z - 1) {
        float3 boundaryCondition_Velocity = float3(BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.y, BoundaryCondition_Velocity.x);
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.z < voxelGridSize.z - 1) {
            voxelIndexU = (voxelIndex.z + 1) * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;
            VolumetricFogFluidVoxelInfo neighborInfo = volumetricFogFluidVoxelInfoBuffer[fluidIndex][voxelIndexU];
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

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.volumetricFogTypeIndex, fogInfo.fogVoxelGridSize);
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

    FluidFogInfo fluidFogInfo = volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex];
    float alpha = (fogInfo.fogVoxelSize.x * fogInfo.fogVoxelSize.x) / max(fluidFogInfo.viscosity * pushConst.dt, 1e-30f);

    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].xyz = (x_neighbor + alpha * selfData.xyz) / (6.0f + alpha);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_F(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = volumetricFogFluidInfoBuffer[groupFogInfo.volumetricFogTypeIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    VolumetricFogFluidVoxelInfo voxelInfo = volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU];
    if (!voxelInfo.isBoundary) {
        uint fIdx = fogInfo.volumetricFogTypeIndex;
        uint iteration = pushConst.iteration & 1;

        float3 vel = voxelInfo.dirtyVelocity_pressure[iteration].xyz;
        vel.y += -9.8f * voxelInfo.voxelFogInfo.x * pushConst.dt * groupFluidFogInfo.FIntensity;
        //float3 worldPos = fogInfo.fogStartPos + fogInfo.fogVoxelSize * (float3(voxelIndex) + 0.5f);
        // float height = worldPos.y - (-69.5f);
        //float ambientFogDensity = (0.01f + 0.03f) * exp(-1.0f * height);
        //float buoyancy = -(voxelInfo.density.x - ambientFogDensity) * 9.8f * pushConst.dt * groupFluidFogInfo.FIntensity;
        //vel.y += buoyancy;

        // apply mouse force
        //if (pushConst.mouseForceStrength > 0.0f) {
        //    float3 worldPos = fogInfo.fogStartPos + fogInfo.fogVoxelSize * (float3(voxelIndex) + 0.5f);
        //    float dist = distance(worldPos, pushConst.mouseForcePosition);
        //    if (dist < pushConst.mouseForceRadius) {
        //        float falloff = exp(-dist * dist / (pushConst.mouseForceRadius * pushConst.mouseForceRadius * 0.1f));
        //        float3 forceDir = normalize(worldPos - pushConst.mouseForcePosition + 1e-8f);
        //        vel += forceDir * pushConst.mouseForceStrength * falloff * pushConst.dt * groupFluidFogInfo.FIntensity;
        //
        //        float densityFalloff = exp(-dist * dist / (pushConst.mouseForceRadius * pushConst.mouseForceRadius * 0.3f));
        //        float eraseAmount = densityFalloff * pushConst.mouseForceStrength * pushConst.dt * 0.15f;
        //        voxelInfo.voxelFogInfo.x = max(voxelInfo.voxelFogInfo.x - eraseAmount, 0.0f);
        //    }
        //}

        voxelInfo.dirtyVelocity_pressure[0].xyz = vel;
        voxelInfo.dirtyVelocity_pressure[1].xyz = vel;

        //float restDensity = getRestSmokeDensity(voxelIndex, fogInfo);
        //float restoreRate = 0.5f;
        ////float speedDamping = exp(-length(vel) * 3.0f);
        //float restoreWeight = saturate(restoreRate * pushConst.dt) * 20.0f;
        //voxelInfo.voxelFogInfo.x = lerp(voxelInfo.voxelFogInfo.x, restDensity, restoreWeight);
        //voxelInfo.voxelFogInfo.y = lerp(voxelInfo.voxelFogInfo.y, fogInfo.scattering, restoreWeight);
        //voxelInfo.voxelFogInfo.z = lerp(voxelInfo.voxelFogInfo.z, fogInfo.phase, restoreWeight);
    }

    // Boundary feathering: fade extinction at voxel grid edges to prevent hard seam with environment fog
    //{
    //    float3 halfGrid = float3(fogInfo.fogVoxelGridSize) * 0.5f;
    //    float3 distToEdge = halfGrid - abs(float3(voxelIndex) - halfGrid);
    //    float minDistToEdge = min(min(distToEdge.x, distToEdge.y), distToEdge.z);
    //    float featherWidth = 2.5f; // voxels from edge where blending begins
    //    float edgeFeather = 1.0f - smoothstep(0.0f, featherWidth, minDistToEdge);
    //    voxelInfo.voxelFogInfo.x *= (1.0f - edgeFeather * 0.6f);
    //}

    volumetricFogFluidVoxelInfoImages[fogInfo.volumetricFogTypeIndex][voxelIndex] = voxelInfo.voxelFogInfo;
    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU] = voxelInfo;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_P(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.volumetricFogTypeIndex, fogInfo.fogVoxelGridSize);
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
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) nL.w = selfData.w;
    float4 nR = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) nR.w = selfData.w;

    // B / T (-Y / +Y)
    float4 nB = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
    if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) nB.w = selfData.w;
    float4 nT = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
    if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) nT.w = selfData.w;

    // D / U (-Z / +Z)
    float4 nD = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
    if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) nD.w = selfData.w;
    float4 nU = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
    if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) nU.w = selfData.w;

    float div = (nR.x - nL.x) / halfDx + (nT.y - nB.y) / halfDy + (nU.z - nD.z) / halfDz;
    float alpha = -cellSize.x * cellSize.x;
    float sumPressure = nL.w + nR.w + nB.w + nT.w + nD.w + nU.w;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;
    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU].dirtyVelocity_pressure[nextIteration].w = (sumPressure + alpha * div) / 6.0f;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_S(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    if (any(voxelIndex >= fogInfo.fogVoxelGridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.volumetricFogTypeIndex, fogInfo.fogVoxelGridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    float3 velocity = selfData.xyz;
    if (!groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) {
        float3 vel_dirty = selfData.xyz;

        float3 cellSize = fogInfo.fogVoxelSize;
        float halfDx = 2.0f * cellSize.x;
        float halfDy = halfDx;
        float halfDz = halfDx;

        float3 obstV = float3(0, 0, 0);
        float3 vMask = float3(1, 1, 1);

        float4 nL = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
        if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) {
            nL.w = selfData.w;
            obstV.x = nL.x;
            vMask.x = 0;
        }
        float4 nR = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
        if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) {
            nR.w = selfData.w;
            obstV.x = nR.x; 
            vMask.x = 0;
        }
        float4 nB = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
        if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) {
            nB.w = selfData.w;
            obstV.y = nB.y;
            vMask.y = 0;
        }
        float4 nT = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
        if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) {
            nT.w = selfData.w;
            obstV.y = nT.y; 
            vMask.y = 0;
        }
        float4 nD = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
        if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) {
            nD.w = selfData.w;
            obstV.z = nD.z; 
            vMask.z = 0;
        }
        float4 nU = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
        if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) {
            nU.w = selfData.w;
            obstV.z = nU.z; 
            vMask.z = 0;
        }

        float3 pg = float3((nR.w - nL.w) / halfDx, (nT.w - nB.w) / halfDy, (nU.w - nD.w) / halfDz);
        //#if BoundaryConditionType != 0
        //obstV *= int3(voxelIndex == 0) * int3(voxelIndex == fogInfo.fogVoxelGridSize - 1);
        //#endif
        velocity = ((vel_dirty - pg) * vMask) + obstV;
    }

    volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex].xyz = velocity;
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
            float4 voxelInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0);
            float extinctionCoeff = voxelInfo.x + voxelInfo.y;

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

    volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex].lightAttenuationEstimator = attenuation.x;
}
