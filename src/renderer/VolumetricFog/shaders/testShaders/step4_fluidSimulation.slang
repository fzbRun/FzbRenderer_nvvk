#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> FluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[]; // xyz: velocity; w:
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> fluidFogVoxelVelocityImage_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages)]] RWTexture3D<float4> fluidFogVoxelInfoImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages_sampler)]] Sampler3D<float4> fluidFogVoxelInfoImages_sample[];
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

groupshared VolumetricFogInfo groupFogInfo;
groupshared FluidFogInfo groupFluidFogInfo;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU];

    AABB voxelAABB;
    voxelAABB.minimum = startPos + voxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + voxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - startPos) / (gridSize * voxelSize);
    float3 velocity = fluidFogVoxelVelocityImage_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0).xyz;

    float3 velocity_lastTime = velocity;
    #ifdef FLUID_SIMULATION_OBJECT_SAVE_FOG

    {
        float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
        float3 uvw_lastTime = (worldPos_lastTime - startPos) / (gridSize * voxelSize);

        int3 voxelIdx_lastTime = int3((worldPos_lastTime - startPos) / voxelSize);
        voxelIdx_lastTime = clamp(voxelIdx_lastTime, int3(0, 0, 0), int3(gridSize) - 1);
        uint voxelIdxU_lastTime = voxelIdx_lastTime.z * gridSize.x * gridSize.y + voxelIdx_lastTime.y * gridSize.x + voxelIdx_lastTime.x;
        velocity_lastTime = fluidFogVoxelVelocityImage_sample[fogInfo.typeFogIndex].SampleLevel(uvw_lastTime, 0).xyz;
        voxelInfo.voxelFogInfo = fluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw_lastTime, 0);
    }

    #else

    if (voxelInfo.isBoundary) voxelInfo.voxelFogInfo = 0.0f;
    else {
        float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
        float3 uvw_lastTime = (worldPos_lastTime - startPos) / (gridSize * voxelSize);
        #if BoundaryConditionType != 0

        uvw_lastTime = clamp(uvw_lastTime, 0.0f, 1.0f);

        #endif
        int3 voxelIdx_lastTime = int3((worldPos_lastTime - startPos) / voxelSize);
        voxelIdx_lastTime = clamp(voxelIdx_lastTime, int3(0, 0, 0), int3(gridSize) - 1);
        uint voxelIdxU_lastTime = voxelIdx_lastTime.z * gridSize.x * gridSize.y + voxelIdx_lastTime.y * gridSize.x + voxelIdx_lastTime.x;
        bool backtrackIntoBoundary = fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIdxU_lastTime].isBoundary;
        if (backtrackIntoBoundary) {
            voxelInfo.voxelFogInfo = fluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);
        } else {
            velocity_lastTime = fluidFogVoxelVelocityImage_sample[fogInfo.typeFogIndex].SampleLevel(uvw_lastTime, 0).xyz;
            voxelInfo.voxelFogInfo = fluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw_lastTime, 0);
        }
    }

    #endif

    voxelInfo.dirtyVelocity_p[0].xyz = velocity_lastTime;
    voxelInfo.dirtyVelocity_p[1].xyz = velocity_lastTime;
    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU] = voxelInfo;
}

groupshared float4 groupVoxelInfos[6][6][6];
groupshared bool groupVoxelBoundary[6][6][6];
void fillGroupSharedVoxelInfos(uint3 voxelIndex, uint3 threadGroupIndex, uint32_t fluidIndex, uint3 voxelGridSize) {
    uint3 localVoxelIndex = voxelIndex - threadGroupIndex * 4;
    uint voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x;

    uint iteration = pushConst.iteration & 1;

    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
    float4 selfData = voxelInfo.dirtyVelocity_p[iteration];
    groupVoxelInfos[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = selfData;
    groupVoxelBoundary[localVoxelIndex.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = voxelInfo.isBoundary;

    uint3 groupBase = threadGroupIndex * 4;
    uint3 threadGroupSize = min(voxelGridSize - groupBase, uint3(4, 4, 4));

    if (localVoxelIndex.x == 0) {
        float4 neighborData = float4(BoundaryCondition_Velocity, BoundaryCondition_Pressure) * selfData;
        bool neighborBoundary = BoundaryCondition_Have;
        if (voxelIndex.x > 0) {
            voxelIndexU = voxelIndex.z * voxelGridSize.x * voxelGridSize.y + voxelIndex.y * voxelGridSize.x + voxelIndex.x - 1;
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
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
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
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
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
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
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
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
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
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
            FluidFogVoxelInfo neighborInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];
            neighborData = neighborInfo.dirtyVelocity_p[iteration];
            neighborBoundary = neighborInfo.isBoundary;
        }
        groupVoxelInfos[threadGroupSize.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborData;
        groupVoxelBoundary[threadGroupSize.z + 1][localVoxelIndex.y + 1][localVoxelIndex.x + 1] = neighborBoundary;
    }
}

#ifdef FLUID_SIMPLIFY_VISCOSITY
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_D(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.typeFogIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;
    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];

    float3 nLV = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1].xyz;
    float3 nRV = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1].xyz;
    float3 nBV = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x].xyz;
    float3 nTV = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x].xyz;
    float3 nDV = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x].xyz;
    float3 nUV = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x].xyz;

    float3 V = selfData.xyz;
    float3 newV = V;
    newV += fluidFogInfo.viscosity * pushConst.dt * (nLV + nRV - 2.0f * V) / fluidFogInfo.voxelSize.x / fluidFogInfo.voxelSize.x;
    newV += fluidFogInfo.viscosity * pushConst.dt * (nBV + nTV - 2.0f * V) / fluidFogInfo.voxelSize.y / fluidFogInfo.voxelSize.y;
    newV += fluidFogInfo.viscosity * pushConst.dt * (nDV + nUV - 2.0f * V) / fluidFogInfo.voxelSize.z / fluidFogInfo.voxelSize.z;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].dirtyVelocity_p[0].xyz = newV;
}
#else
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_D(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.typeFogIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    float3 x_neighbor = 0.0f;
    {
        float4 n = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
        // if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) n = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
        // if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) n = float4(-selfData.x, selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
        // if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) n = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
        // if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) n = float4(selfData.x, -selfData.y, selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
        // if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) n = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        x_neighbor += n.xyz;

        n = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
        // if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) n = float4(selfData.x, selfData.y, -selfData.z, selfData.w);
        x_neighbor += n.xyz;
    }

    float alpha = (voxelSize.x * voxelSize.x) / max(fluidFogInfo.viscosity * pushConst.dt, 1e-30f);

    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].dirtyVelocity_p[nextIteration].xyz = (x_neighbor + alpha * selfData.xyz) / (6.0f + alpha);
}
#endif

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_F(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU];
    if (!voxelInfo.isBoundary) {
        uint fIdx = fogInfo.typeFogIndex;
        uint iteration = pushConst.iteration & 1;

        float3 vel = voxelInfo.dirtyVelocity_p[iteration].xyz;
        vel.y += -9.8f * voxelInfo.voxelFogInfo.x * pushConst.dt * groupFluidFogInfo.FIntensity;

        voxelInfo.dirtyVelocity_p[0].xyz = vel;
        voxelInfo.dirtyVelocity_p[1].xyz = vel;
    }

    fluidFogVoxelInfoImages[fogInfo.typeFogIndex][voxelIndex] = voxelInfo.voxelFogInfo;

#ifdef FLUID_SIMPLIFY_PRESSURE
    fluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = voxelInfo.dirtyVelocity_p[0].xyz;
#else
    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU] = voxelInfo;
#endif
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_P(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.typeFogIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    float3 cellSize = voxelSize;
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

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;
    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].dirtyVelocity_p[nextIteration].w = (sumPressure + alpha * div) / 6.0f;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_S(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) {
        groupFogInfo = fogInfosBuffer[pushConst.instanceIndex];
        groupFluidFogInfo = FluidFogInfoBuffer[groupFogInfo.typeFogIndex];
    }
    GroupMemoryBarrierWithGroupSync();

    VolumetricFogInfo fogInfo = groupFogInfo;
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, fogInfo.typeFogIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    float3 velocity = selfData.xyz;
    if (!groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) {
        float3 vel_dirty = selfData.xyz;

        float3 cellSize = voxelSize;
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
        // #if BoundaryConditionType != 0
        // obstV *= int3(voxelIndex == 0) * int3(voxelIndex == gridSize - 1);
        // #endif
        velocity = ((vel_dirty - pg) * vMask) + obstV;
    }

    fluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = velocity;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
/*
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_createLightAttenuationEstimator(uint threadIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = fogInfosBuffer[pushConst.instanceIndex];

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 voxelSize = voxelSize;
    int3 gridSize = gridSize;

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

            float3 uvw = (samplePoint - fogStartPos) / (gridSize * voxelSize);
            float4 voxelInfo = fluidFogVoxelInfoImages_sample[fogInfo.typeFogIndex].SampleLevel(uvw, 0);
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

    FluidFogInfoBuffer[fogInfo.typeFogIndex].lightAttenuationEstimator = attenuation.x;
}
*/
