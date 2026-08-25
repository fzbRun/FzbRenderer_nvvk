#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<FluidSimulationPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfosBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages)]] RWTexture3D<float4> fluidFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages_sample)]] Sampler3D<float4> fluidFogImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfoBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> fluidFogVoxelVelocityImage_sample[];

#if defined(FLUID_A_MACCORMACK) || defined(FLUID_SIMULATION_CPF)
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImage_temp1)]] RWTexture3D<float4> fluidFogImages_temp1[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImage_temp1_sample)]] Sampler3D<float4> fluidFogImages_temp1_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImage_temp2)]] RWTexture3D<float4> fluidFogImages_temp2[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImage_temp2_sample)]] Sampler3D<float4> fluidFogImages_temp2_sample[];
#endif
//--------------------------------------------------------------------------------Fluid-------------------------------------------------------------------------------------
// BoundaryCondition_Velocity: float3(1.0f, 1.0f, 1.0f): Non; float3(-1.0f, 1.0f, 1.0f): Slip; float3(-1.0f, -1.0f, -1.0f): NoSlip
// BoundaryCondition_P: 0: Non; 1: Slip or NoSlip;
#define BoundaryConditionType 0
#if BoundaryConditionType == 0  //Non
    #define BoundaryCondition_Have false
    #define BoundaryCondition_Velocity float3(1.0f, 1.0f, 1.0f)
    #define BoundaryCondition_P 0
#else
    #define BoundaryCondition_Have true
    #define BoundaryCondition_P 1
    #if BoundaryConditionType == 1 // Slip
        #define BoundaryCondition_Velocity float3(-1.0f, 1.0f, 1.0f)
    #else // NoSlip
        #define BoundaryCondition_Velocity float3(-1.0f, -1.0f, -1.0f)
    #endif
#endif

groupshared FluidFogInfo groupFluidFogInfo;

#ifdef FLUID_A_MACCORMACK

#define FLUID_A_PASS_FORWARD 0
#define FLUID_A_PASS_BACKWARD 1
#define FLUID_A_PASS_CORRECT 2
#define FLUID_A_MACCORMACK_MIX 0.65f

float4 clampToFluidFogNodeRange(float4 value, float3 sourceWorldPos, uint fluidIndex, float3 startPos, uint3 gridSize, float3 voxelSize) {
    float3 cellSpace = (sourceWorldPos - startPos) / voxelSize;
    int3 base = int3(floor(cellSpace));
    base = clamp(base, int3(0, 0, 0), int3(gridSize) - 1);

    float4 minValue = float4(1e30f, 1e30f, 1e30f, 1e30f);
    float4 maxValue = -minValue;

    [unroll]for (int z = 0; z <= 1; ++z) {
        [unroll]for (int y = 0; y <= 1; ++y) {
            [unroll]for (int x = 0; x <= 1; ++x) {
                int3 offset = int3(x, y, z);
                uint3 nodeIndex = uint3(clamp(base + offset, int3(0, 0, 0), int3(gridSize) - 1));
                float3 uvw = (float3(nodeIndex) + 0.5f) / float3(gridSize);
                float4 nodeValue = fluidFogImages_sample[fluidIndex].SampleLevel(uvw, 0);
                minValue = min(minValue, nodeValue);
                maxValue = max(maxValue, nodeValue);
            }
        }
    }
    return clamp(value, minValue, maxValue);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    uint fluidIndex = pushConst.fluidIndex;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU];

    float3 voxelCenter = startPos + (float3(voxelIndex) + 0.5f) * voxelSize;
    float3 uvw = (float3(voxelIndex) + 0.5f) / float3(gridSize);
    float3 velocity = fluidFogVoxelVelocityImage_sample[fluidIndex].SampleLevel(uvw, 0).xyz;

    if (pushConst.iteration == FLUID_A_PASS_FORWARD) {
        float4 phi_hat;
        if (!voxelInfo.isBoundary) {
            float3 sourceWorldPos = voxelCenter - pushConst.dt * velocity;
            uvw = (sourceWorldPos - startPos) / (float3(gridSize) * voxelSize);
            phi_hat = fluidFogImages_sample[fluidIndex].SampleLevel(uvw, 0);
        } else phi_hat = fluidFogImages_sample[fluidIndex].SampleLevel(uvw, 0);

        fluidFogImages_temp1[fluidIndex][voxelIndex] = phi_hat;
        return;
    }

    if (pushConst.iteration == FLUID_A_PASS_BACKWARD) {
        float4 phi_hat_hat;
        if (!voxelInfo.isBoundary) {
            float3 sourceWorldPos = voxelCenter + pushConst.dt * velocity;
            uvw = (sourceWorldPos - startPos) / (float3(gridSize) * voxelSize);
            phi_hat_hat = fluidFogImages_temp1_sample[fluidIndex].SampleLevel(uvw, 0);
        } else phi_hat_hat = fluidFogImages_temp1_sample[fluidIndex].SampleLevel(uvw, 0);

        fluidFogImages_temp2[fluidIndex][voxelIndex] = phi_hat_hat;
        return;
    }

    float3 velocity_lastTime = velocity;
    if (!voxelInfo.isBoundary) {
        float3 sourceWorldPos = voxelCenter - pushConst.dt * velocity;
        float3 sourceUvw = (sourceWorldPos - startPos) / (float3(gridSize) * voxelSize);
        velocity_lastTime = fluidFogVoxelVelocityImage_sample[fluidIndex].SampleLevel(sourceUvw, 0).xyz;

        float4 phi_hat = fluidFogImages_temp1_sample[fluidIndex].SampleLevel(uvw, 0);
        float4 phi_hat_hat = fluidFogImages_temp2_sample[fluidIndex].SampleLevel(uvw, 0);
        float4 phi_n = fluidFogImages_sample[fluidIndex].SampleLevel(uvw, 0);
        float4 result = phi_hat + 0.5f * (phi_n - phi_hat_hat);
        float4 macCormack = clampToFluidFogNodeRange(result, sourceWorldPos, fluidIndex, startPos, gridSize, voxelSize);
        float4 semiLagrangian = fluidFogImages_sample[fluidIndex].SampleLevel(sourceUvw, 0);
        voxelInfo.voxelFogInfo = lerp(semiLagrangian, macCormack, FLUID_A_MACCORMACK_MIX);
    } else voxelInfo.voxelFogInfo = fluidFogImages_sample[fluidIndex].SampleLevel(uvw, 0);

    voxelInfo.dirtyVelocity_p[0].xyz = velocity_lastTime;
    voxelInfo.dirtyVelocity_p[1].xyz = velocity_lastTime;
    fluidFogVoxelInfoBuffer[fluidIndex][voxelIndexU] = voxelInfo;
}
#else
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_A(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU];

    AABB voxelAABB;
    voxelAABB.minimum = startPos + voxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + voxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - startPos) / (gridSize * voxelSize);
    float3 velocity = fluidFogVoxelVelocityImage_sample[pushConst.fluidIndex].SampleLevel(uvw, 0).xyz;

    #ifdef FLUID_SIMULATION_OBJECT_SAVE_FOG

    if (!voxelInfo.isBoundary) {
        float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
        float3 uvw_lastTime = (worldPos_lastTime - startPos) / (gridSize * voxelSize);

        int3 voxelIdx_lastTime = int3((worldPos_lastTime - startPos) / voxelSize);
        voxelIdx_lastTime = clamp(voxelIdx_lastTime, int3(0, 0, 0), int3(gridSize) - 1);
        uint voxelIdxU_lastTime = voxelIdx_lastTime.z * gridSize.x * gridSize.y + voxelIdx_lastTime.y * gridSize.x + voxelIdx_lastTime.x;
        float3 velocity_lastTime = fluidFogVoxelVelocityImage_sample[pushConst.fluidIndex].SampleLevel(uvw_lastTime, 0).xyz;

        uvw = uvw_lastTime;
        velocity = velocity_lastTime;
    }
    voxelInfo.voxelFogInfo = fluidFogImages_sample[pushConst.fluidIndex].SampleLevel(uvw, 0);

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
        bool backtrackIntoBoundary = fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIdxU_lastTime].isBoundary;
        if (backtrackIntoBoundary) {
            voxelInfo.voxelFogInfo = fluidFogImages_sample[pushConst.fluidIndex].SampleLevel(uvw, 0);
        } else {
            velocity_lastTime = fluidFogVoxelVelocityImage_sample[pushConst.fluidIndex].SampleLevel(uvw_lastTime, 0).xyz;
            voxelInfo.voxelFogInfo = fluidFogImages_sample[pushConst.fluidIndex].SampleLevel(uvw_lastTime, 0);
        }
    }

    #endif

    voxelInfo.dirtyVelocity_p[0].xyz = velocity;
    voxelInfo.dirtyVelocity_p[1].xyz = velocity;
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU] = voxelInfo;
}
#endif

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
        float4 neighborData = float4(BoundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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
        float4 neighborData = float4(BoundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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
        //float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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
        float4 neighborData = float4(boundaryCondition_Velocity, BoundaryCondition_P) * selfData;
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

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_D(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, pushConst.fluidIndex, gridSize);
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
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[0].xyz = newV;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_D_Iteration(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, pushConst.fluidIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    /*
    {
        float3 x_neighbor = 0.0f;
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

        float alpha = (voxelSize.x * voxelSize.x) / max(fluidFogInfo.viscosity * pushConst.dt, 1e-30f);
        fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[nextIteration].xyz = (x_neighbor + alpha * selfData.xyz) / (6.0f + alpha);
    }
    */
    {
        float3 x_neighbor = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1].xyz + groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1].xyz;
        float3 y_neighbor = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x].xyz + groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x].xyz;
        float3 z_neighbor = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x].xyz + groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x].xyz;
        
        float viscosityDt = max(fluidFogInfo.viscosity * pushConst.dt, 1e-30f);
        float ax = viscosityDt / (voxelSize.x * voxelSize.x);
        float ay = viscosityDt / (voxelSize.y * voxelSize.y);
        float az = viscosityDt / (voxelSize.z * voxelSize.z);
            
        float3 newVelocity = (ax * x_neighbor + ay * y_neighbor + az * z_neighbor + selfData.xyz) / (1.0f + 2.0f * (ax + ay + az));
        fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[nextIteration].xyz = newVelocity;
    }
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_F(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU];
    if (!voxelInfo.isBoundary) {
        uint iteration = pushConst.iteration & 1;

        float3 vel = voxelInfo.dirtyVelocity_p[iteration].xyz;
        vel.y += -9.8f * voxelInfo.voxelFogInfo.x * pushConst.dt * groupFluidFogInfo.FIntensity;

        voxelInfo.dirtyVelocity_p[0].xyz = vel;
        voxelInfo.dirtyVelocity_p[1].xyz = vel;
    }

    fluidFogImages[pushConst.fluidIndex][voxelIndex] = voxelInfo.voxelFogInfo;

    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU] = voxelInfo;
    //if (pushConst.useP == 1) fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU] = voxelInfo;
    //else fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex].xyz = voxelInfo.dirtyVelocity_p[0].xyz;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_P(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    AABB voxelAABB;
    voxelAABB.minimum = startPos + voxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + voxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - startPos) / (gridSize * voxelSize);
    float3 velocity = fluidFogVoxelVelocityImage_sample[pushConst.fluidIndex].SampleLevel(uvw, 0).xyz;

    float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
    float3 uvw_lastTime = (worldPos_lastTime - startPos) / (gridSize * voxelSize);

    int3 voxelIdx_lastTime = int3((worldPos_lastTime - startPos) / voxelSize);
    voxelIdx_lastTime = clamp(voxelIdx_lastTime, int3(0, 0, 0), int3(gridSize) - 1);
    uint voxelIdxU_lastTime = voxelIdx_lastTime.z * gridSize.x * gridSize.y + voxelIdx_lastTime.y * gridSize.x + voxelIdx_lastTime.x;
    float3 velocity_lastTime = fluidFogVoxelVelocityImage_sample[pushConst.fluidIndex].SampleLevel(uvw_lastTime, 0).xyz;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfo voxelInfo = fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU];
    velocity = voxelInfo.dirtyVelocity_p[0].xyz;

    float3 dv = (velocity - velocity_lastTime) / pushConst.dt;
    float3 dp = dv * fluidFogInfo.magicNumber;
    float3 p = dp * voxelSize;
    uint randomSeed = xxhash32(uint3(voxelIndexU, int(pushConst.time * 1000.0f), 1));
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[0].w = length(p);    //length(p);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_P_Iteration(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, pushConst.fluidIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    float3 cellSize = voxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = halfDx;
    float halfDz = halfDx;
    // float halfDy = 2.0f * cellSize.y;
    // float halfDz = 2.0f * cellSize.z;

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
    fluidFogImages_temp1[pushConst.fluidIndex][voxelIndex] = float4(div, div, div, div);

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    uint nextIteration = (pushConst.iteration + 1) & 1;

    // float alpha = -cellSize.x * cellSize.x;
    // float sumP = nL.w + nR.w + nB.w + nT.w + nD.w + nU.w;
    // fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[nextIteration].w = (sumP + alpha * div) / 6.0f;

    float invDx2 = 1.0f / (cellSize.x * cellSize.x);
    float invDy2 = 1.0f / (cellSize.y * cellSize.y);
    float invDz2 = 1.0f / (cellSize.z * cellSize.z);

    float weightedPSum =
        (nL.w + nR.w) * invDx2 +
        (nB.w + nT.w) * invDy2 +
        (nD.w + nU.w) * invDz2;

    float newP = (weightedPSum - div) / (2.0f * (invDx2 + invDy2 + invDz2));
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[nextIteration].w = newP;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_S(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;

    float3 startPos = pushConst.fluidStartPos;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, pushConst.fluidIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];
    float3 velocity = selfData.xyz;
    if (!groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) {
        float3 vel_dirty = selfData.xyz;

        float3 cellSize = voxelSize;
        float halfDx = 2.0f * cellSize.x;
        float halfDy = 2.0f * cellSize.y;   //halfDx;
        float halfDz = 2.0f * cellSize.z;   //halfDx;

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

    fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex] = float4(velocity, 0.0f);
}

#ifdef FLUID_SIMULATION_CPF
#include "renderer/VolumetricFog/shaders/finalShaders/data/convolutionKernel.hlsli"
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_P_Divergence(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID, uint3 threadGroupIndex: SV_GroupID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    if (any(voxelIndex >= gridSize)) return;

    fillGroupSharedVoxelInfos(voxelIndex, threadGroupIndex, pushConst.fluidIndex, gridSize);
    GroupMemoryBarrierWithGroupSync();

    uint3 lIdx = voxelIndex - threadGroupIndex * 4 + 1;
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x]) return;

    float4 selfData = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x];

    float halfDx = 2.0f * voxelSize.x;
    float halfDy = halfDx;  //2.0f * voxelSize.y;
    float halfDz = halfDx;  //2.0f * voxelSize.z;

    float4 nL = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x - 1];
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x - 1]) nL.w = selfData.w;
    float4 nR = groupVoxelInfos[lIdx.z][lIdx.y][lIdx.x + 1];
    if (groupVoxelBoundary[lIdx.z][lIdx.y][lIdx.x + 1]) nR.w = selfData.w;

    float4 nB = groupVoxelInfos[lIdx.z][lIdx.y - 1][lIdx.x];
    if (groupVoxelBoundary[lIdx.z][lIdx.y - 1][lIdx.x]) nB.w = selfData.w;
    float4 nT = groupVoxelInfos[lIdx.z][lIdx.y + 1][lIdx.x];
    if (groupVoxelBoundary[lIdx.z][lIdx.y + 1][lIdx.x]) nT.w = selfData.w;

    float4 nD = groupVoxelInfos[lIdx.z - 1][lIdx.y][lIdx.x];
    if (groupVoxelBoundary[lIdx.z - 1][lIdx.y][lIdx.x]) nD.w = selfData.w;
    float4 nU = groupVoxelInfos[lIdx.z + 1][lIdx.y][lIdx.x];
    if (groupVoxelBoundary[lIdx.z + 1][lIdx.y][lIdx.x]) nU.w = selfData.w;

    float div = (nR.x - nL.x) / halfDx + (nT.y - nB.y) / halfDy + (nU.z - nD.z) / halfDz;
    fluidFogImages_temp1[pushConst.fluidIndex][voxelIndex] = float4(div, div, div, div);
}

bool fluidIsSolid(int3 idx, uint3 gridSize) {
    if (any(idx < 0) || any(idx >= int3(gridSize))) return true;
    uint linear = idx.z * gridSize.x * gridSize.y + idx.y * gridSize.x + idx.x;
    return fluidFogVoxelInfoBuffer[pushConst.fluidIndex][linear].isBoundary;
}
void poissonConvolveAxis(RWTexture3D<float4> src[], RWTexture3D<float4> dst[], uint3 voxelIndex, int axis, uint3 gridSize) {
    const int half = INVERSE_Itr_20_Half_Filter_Size; // 19

    int3 stepPos;
    if (axis == 0) stepPos = int3(1, 0, 0);
    else if (axis == 1) stepPos = int3(0, 1, 0);
    else stepPos = int3(0, 0, 1);
    int3 stepNeg = -stepPos;

    int3 pos = voxelIndex;
    int3 neg = voxelIndex;

    float4 acc = INVERSE_Itr_20_R14_Filters[half] * src[pushConst.fluidIndex][voxelIndex];

    for (int t = 1; t <= half; ++t) {
        float4 wp = INVERSE_Itr_20_R14_Filters[half + t];
        float4 wn = INVERSE_Itr_20_R14_Filters[half - t];

        pos += stepPos;
        if (fluidIsSolid(pos, gridSize)) { stepPos = -stepPos; pos += stepPos; }
        acc += wp * src[pushConst.fluidIndex][pos];

        neg += stepNeg;
        if (fluidIsSolid(neg, gridSize)) { stepNeg = -stepNeg; neg += stepNeg; }
        acc += wn * src[pushConst.fluidIndex][neg];
    }

    dst[pushConst.fluidIndex][voxelIndex] = acc;
}
[numthreads(32, 1, 1)]
[shader("compute")]
void computeMain_fluidSimulation_P_Filter(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize = groupFluidFogInfo.gridSize;

    if (any(voxelIndex >= gridSize)) return;
    if (fluidIsSolid(int3(voxelIndex), gridSize)) return;

    if (pushConst.iteration == 0) poissonConvolveAxis(fluidFogImages_temp1, fluidFogImages_temp2, voxelIndex, 0, gridSize);
    else if (pushConst.iteration == 1) poissonConvolveAxis(fluidFogImages_temp2, fluidFogImages_temp1, voxelIndex, 1, gridSize);
    else poissonConvolveAxis(fluidFogImages_temp1, fluidFogImages_temp2, voxelIndex, 2, gridSize);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_fluidSimulation_P_Final(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (all(groupThreadIndex == 0)) groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
    GroupMemoryBarrierWithGroupSync();
    uint3 gridSize = groupFluidFogInfo.gridSize;
    if (any(voxelIndex >= gridSize)) return;
    if (fluidIsSolid(int3(voxelIndex), gridSize)) return;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    float4 p = fluidFogImages_temp2[pushConst.fluidIndex][voxelIndex];
    float pressure = p.x + p.y + p.z + p.w;

    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[0].w = pressure;
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].dirtyVelocity_p[1].w = pressure;
}
#endif
