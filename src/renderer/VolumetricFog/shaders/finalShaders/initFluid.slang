#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<InitFluidPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eHeightFogInfosBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogInfosBuffer)]] StructuredBuffer<GridFogInfo, ScalarDataLayout> gridFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogImages)]] Sampler3D<float4> gridFogImages[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfosBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages)]] RWTexture3D<float4> fluidFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages_sample)]] Sampler3D<float4> fluidFogImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfoBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> fluidFogVoxelVelocityImage_sample[];

groupshared FogGlobalInfo groupFogGlobalInfo;
groupshared VolumetricFogInfo groupVolumetricFogInfos[MAX_VOLUMETRIC_FOG_COUNT];
groupshared HeightFogInfo groupHeightFogInfos[MAX_HEIGHT_FOG_COUNT];
groupshared GridFogInfo grouGridFogInfos[MAX_GRID_FOG_COUNT];
groupshared FluidFogInfo groupFluidFogInfo;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initFluid(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    if (groupThreadIndexU < MAX_VOLUMETRIC_FOG_COUNT) {
        FogGlobalInfo fogGlobalInfo = pushConst.fogGlobalInfoAddress[0];
        if (groupThreadIndexU == 0) {
            groupFogGlobalInfo = fogGlobalInfo;
            groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
        }
        if (groupThreadIndexU < min(fogGlobalInfo.volumetricFogCount, MAX_VOLUMETRIC_FOG_COUNT)) groupVolumetricFogInfos[groupThreadIndexU] = fogInfosBuffer[groupThreadIndexU];
        if (groupThreadIndexU < min(fogGlobalInfo.heightFogCount, MAX_HEIGHT_FOG_COUNT)) groupHeightFogInfos[groupThreadIndexU] = heightFogInfoBuffer[groupThreadIndexU];
        if (groupThreadIndexU < min(fogGlobalInfo.gridFogCount, MAX_GRID_FOG_COUNT)) grouGridFogInfos[groupThreadIndexU] = gridFogInfoBuffer[groupThreadIndexU];
    }
    GroupMemoryBarrierWithGroupSync();

    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    FogGlobalInfo fogGlobalInfo = groupFogGlobalInfo;

    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;
    float3 voxelCenter = pushConst.fluidStartPos + (float3(voxelIndex) + 0.5f) * voxelSize;

    float envExtinction = 0.0f;
    float envScattering = 0.0f;
    float envPhase = 0.0f;
    float3 envColor = 0.0f;
    //if (fogGlobalInfo.useGlobalHeightFog && voxelCenter.y >= groupGlobalHeightFogY.x && voxelCenter.y <= groupGlobalHeightFogY.y) {
    //    HeightFogInfo globalHeightFogInfo = groupGlobalHeightFogInfo;
    //    float heightAttenuation = exp(-globalHeightFogInfo.heightScale * (voxelCenter.y - groupGlobalHeightFogY.x));
    //    float extinction = (globalHeightFogInfo.absorption + globalHeightFogInfo.scattering) * heightAttenuation;
    //    envExtinction = extinction;
    //    envScattering = extinction * globalHeightFogInfo.scattering;
    //    envPhase = extinction * globalHeightFogInfo.phase;
    //    envColor = extinction * globalHeightFogInfo.color;
    //}

    for (int i = 0; i < fogGlobalInfo.volumetricFogCount; ++i) {
        if (i == pushConst.fogIndex) continue;
        VolumetricFogInfo otherFogInfo = groupVolumetricFogInfos[i];
        AABB otherFogAABB = otherFogInfo.aabb;
        if (!checkPointInAABB(voxelCenter, otherFogAABB)) continue;

        float extinction = 0.0f, scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
        if (otherFogInfo.type == VolumetricFogType::Height) {
            HeightFogInfo heightFogInfo = heightFogInfoBuffer[otherFogInfo.typeFogIndex];
            float heightScale = heightFogInfo.heightScale;
            float height = max(voxelCenter.y - otherFogInfo.aabb.minimum.y, 0.0f);
            float heightAttenuation = exp(-heightScale * height);

            extinction = (heightFogInfo.absorption + heightFogInfo.scattering) * heightAttenuation;
            scattering = heightFogInfo.scattering * heightAttenuation;
            phase = heightFogInfo.phase;
            color = heightFogInfo.color * (1.0f + heightFogInfo.ambientIntensity);
        } else if (otherFogInfo.type == VolumetricFogType::Fluid) continue;
        else if (otherFogInfo.type == VolumetricFogType::Grid) {
            GridFogInfo gridFogInfo = gridFogInfoBuffer[otherFogInfo.typeFogIndex];

            float3 uvw = (voxelCenter - otherFogAABB.minimum) / (otherFogAABB.maximum - otherFogAABB.minimum);
            float4 voxelInfo = gridFogImages[otherFogInfo.typeFogIndex].SampleLevel(uvw, 0);

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
    float4 envVoxelInfo = float4(envExtinction, envScattering, envPhase, luminance(envColor));
    float4 voxelInfo = envVoxelInfo;
    float3 velocity = 0.0f;

    //int3 voxelIndex_lastTime = int3((voxelCenter - fluidFogInfo.fogStartPos_lastTime) / voxelSize);
    //if (all(voxelIndex_lastTime >= 0) && all(voxelIndex_lastTime <= gridSize - 1) && pushConst.time > 0) {
    //    voxelInfo = fluidFogImages[pushConst.fluidIndex][voxelIndex_lastTime];
    //
    //    velocity = fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex_lastTime].xyz;
    //
    //    float3 halfGrid = float3(gridSize) * 0.5f;
    //    float3 distToCenter = abs(float3(voxelIndex) - halfGrid);
    //    float maxPossibleDist = length(halfGrid);
    //    float centerDist = length(distToCenter) / maxPossibleDist;
    //
    //    float edgeFactor = smoothstep(0.0, 1.0, centerDist);
    //    float restoreRate = 0.5f;
    //    float restoreWeight = saturate(restoreRate * pushConst.dt) * edgeFactor * fluidFogInfo.restoreSpeed;
    //
    //    voxelInfo = lerp(voxelInfo, envVoxelInfo, saturate(restoreWeight));
    //    velocity = lerp(velocity, 0.0f, saturate(restoreWeight));
    //}

    if (pushConst.time > 0) {
        float3 voxelCenterOffset = voxelCenter - fluidFogInfo.fogStartPos_lastTime;
        int3 voxelIndex_lastTime = int3(voxelCenterOffset / voxelSize);
        voxelIndex_lastTime = clamp(voxelIndex_lastTime, 0, gridSize - 1);
        voxelInfo = fluidFogImages[pushConst.fluidIndex][voxelIndex_lastTime];

        velocity = fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex_lastTime].xyz;

        float3 halfGrid = float3(gridSize) * 0.5f;
        float3 distToCenter = abs(float3(voxelIndex) - halfGrid);
        float maxPossibleDist = length(halfGrid);
        float centerDist = length(distToCenter) / maxPossibleDist;

        float3 voxelGridSize = float3(gridSize) * voxelSize;
        float3 newVoxelOffset = max(float3(0.0f), max(-voxelCenterOffset, voxelCenterOffset - voxelGridSize));
        float3 distNorm = saturate(newVoxelOffset / (3.0f * voxelSize));
        float distance01 = dot(distNorm, float3(1.0f / 3.0f));
        float offsetFactor = lerp(2.0f, 1.0f, distance01);

        float edgeFactor = smoothstep(0.0, 1.0, centerDist);
        float restoreRate = 0.5f;
        float restoreWeight = saturate(restoreRate * pushConst.dt) * edgeFactor * offsetFactor * fluidFogInfo.restoreSpeed;

        voxelInfo = lerp(voxelInfo, envVoxelInfo, saturate(restoreWeight));
        velocity = lerp(velocity, 0.0f, saturate(restoreWeight));
    }

    fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex] = float4(velocity, 0.0f);
    fluidFogImages[pushConst.fluidIndex][voxelIndex] = voxelInfo;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].isBoundary = false;

    pushConst.dynamicMeshInjectedAddress[pushConst.fluidIndex][voxelIndexU] = 0;
}


/*
groupshared SceneInfo groupSceneInfo;
groupshared AABB groupFluidAABB;
groupshared uint groupInFluidInstanceCount;
groupshared uint groupInFluidOffset_inGlobal;

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getInFluidInstanceInfo(uint instanceIndex: SV_DispatchThreadID, uint groupThreadIndex: SV_GroupThreadID) {
    if (instanceIndex >= pushConst.instanceCount) return;
    if (groupThreadIndex == 0) {
        groupSceneInfo = pushConst.sceneInfoAddress[0];
        groupFluidAABB = fogInfosBuffer[0].aabb;    //only judge the first fluid fog
        groupInFluidInstanceCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    InstanceInfo instanceInfo = pushConst.instanceInfoAddress[instanceIndex];
    float4x4 transformMatrix = groupSceneInfo.instances[instanceInfo.instanceIndex].transform;

    AABB aabb_CurFrame; float3 velocity = 0;
    {
        aabb_CurFrame.minimum = mul(float4(instanceInfo.aabb_local.minimum.x, instanceInfo.aabb_local.minimum.y, instanceInfo.aabb_local.minimum.z, 1.0), transformMatrix).xyz;
        aabb_CurFrame.maximum = aabb_CurFrame.minimum;
        velocity += aabb_CurFrame.minimum - instanceInfo.aabb_lastFrame.minimum;

        for (int i = 1; i < 8; ++i) {
            float4 pos = 1.0f;
            pos.x = ((i & 3) & 1) == 0 ? instanceInfo.aabb_local.minimum.x : instanceInfo.aabb_local.maximum.x;
            pos.y = (i & 3) <= 1 ? instanceInfo.aabb_local.minimum.y : instanceInfo.aabb_local.maximum.y;
            pos.z = i / 4 == 0 ? instanceInfo.aabb_local.minimum.z : instanceInfo.aabb_local.maximum.z;
            pos = mul(pos, transformMatrix);

            float3 pos_lastFrame = 0.0f;
            pos_lastFrame.x = ((i & 3) & 1) == 0 ? instanceInfo.aabb_lastFrame.minimum.x : instanceInfo.aabb_lastFrame.maximum.x;
            pos_lastFrame.y = (i & 3) <= 1 ? instanceInfo.aabb_lastFrame.minimum.y : instanceInfo.aabb_lastFrame.maximum.y;
            pos_lastFrame.z = i / 4 == 0 ? instanceInfo.aabb_lastFrame.minimum.z : instanceInfo.aabb_lastFrame.maximum.z;

            velocity += pos.xyz - pos_lastFrame;

            aabb_CurFrame.minimum = min(aabb_CurFrame.minimum, pos.xyz);
            aabb_CurFrame.maximum = max(aabb_CurFrame.maximum, pos.xyz);
        }
        velocity /= (pushConst.dt * 8);
    }
    //if (abs(velocity.y) > 0.0f)
    //    printf("aabb_local: %f %f %f %f %f %f\naabb_Cur: %f %f %f %f %f %f\naabb_last: %f %f %f %f %f %f\nvelocity: %f %f %f\n\n",
    //           instanceInfo.aabb_local.minimum.x, instanceInfo.aabb_local.minimum.y, instanceInfo.aabb_local.minimum.z,
    //           instanceInfo.aabb_local.maximum.x, instanceInfo.aabb_local.maximum.y, instanceInfo.aabb_local.maximum.z,
    //           aabb_CurFrame.minimum.x, aabb_CurFrame.minimum.y, aabb_CurFrame.minimum.z,
    //           aabb_CurFrame.maximum.x, aabb_CurFrame.maximum.y, aabb_CurFrame.maximum.z,
    //           instanceInfo.aabb_lastFrame.minimum.x, instanceInfo.aabb_lastFrame.minimum.y, instanceInfo.aabb_lastFrame.minimum.z,
    //           instanceInfo.aabb_lastFrame.maximum.x, instanceInfo.aabb_lastFrame.maximum.y, instanceInfo.aabb_lastFrame.maximum.z,
    //           velocity.x, velocity.y, velocity.z,
    //    );

    pushConst.instanceInfoAddress[instanceIndex].aabb_lastFrame = aabb_CurFrame;
    if (pushConst.hasFluidStartUp == 0) return;

    velocity.x = abs(velocity.x) < 0.1f ? 0.0f : velocity.x;
    velocity.y = abs(velocity.y) < 0.1f ? 0.0f : velocity.y;
    velocity.z = abs(velocity.z) < 0.1f ? 0.0f : velocity.z;

    bool inFluid = AABBIntersects(groupFluidAABB, aabb_CurFrame);

    uint warpIndex = groupThreadIndex / 32;
    uint warpLane = groupThreadIndex & 31;

    uint warpInFluidCount = WaveActiveCountBits(inFluid);
    uint inFluidOffset_inWarp = WavePrefixCountBits(inFluid);
    uint inFluidOffset_inGroup = 0;
    if (warpLane == 0) InterlockedAdd(groupInFluidInstanceCount, warpInFluidCount, inFluidOffset_inGroup);
    inFluidOffset_inGroup = WaveReadLaneFirst(inFluidOffset_inGroup);
    GroupMemoryBarrierWithGroupSync();

    if (groupThreadIndex == 0) InterlockedAdd(pushConst.inFluidInstanceCountAddress[0], groupInFluidInstanceCount, groupInFluidOffset_inGlobal);
    GroupMemoryBarrierWithGroupSync();
    uint inFluidOffset_inGlobal = groupInFluidOffset_inGlobal;

    if (inFluid) {
        uint offset = inFluidOffset_inWarp + inFluidOffset_inGroup + inFluidOffset_inGlobal;

        InFluidInstanceInfo inFluidInstanceInfo;
        inFluidInstanceInfo.aabb = aabb_CurFrame;
        inFluidInstanceInfo.velocity = velocity;
        pushConst.inFluidInstanceInfoAddress[offset] = inFluidInstanceInfo;
    }
}
//----------------------------------------------------------------------------------------------------------------------
bool AABBBoundaryIntersects(AABB a, AABB b) {
    // 1. 检查 a 是否完全在 b 内（包含边界）
    if (a.minimum.x >= b.minimum.x && a.maximum.x <= b.maximum.x &&
        a.minimum.y >= b.minimum.y && a.maximum.y <= b.maximum.y &&
        a.minimum.z >= b.minimum.z && a.maximum.z <= b.maximum.z) {
        return true;
    }

    // 2. 检查 b 是否完全在 a 内（严格包含，此时 a 不在 b 内，故不会触发上一条）
    if (b.minimum.x >= a.minimum.x && b.maximum.x <= a.maximum.x &&
        b.minimum.y >= a.minimum.y && b.maximum.y <= a.maximum.y &&
        b.minimum.z >= a.minimum.z && b.maximum.z <= a.maximum.z) {
        return false;
    }

    // 3. 一般相交判断（包括边界接触）
    return (a.minimum.x <= b.maximum.x && a.maximum.x >= b.minimum.x &&
            a.minimum.y <= b.maximum.y && a.maximum.y >= b.minimum.y &&
            a.minimum.z <= b.maximum.z && a.maximum.z >= b.minimum.z);
}

#define SHARED_INSTANCE_COUNT 256
groupshared InFluidInstanceInfo groupInstanceInfos[SHARED_INSTANCE_COUNT];  //9KB

[numthreads(8, 8, 8)]
[shader("compute")]
void computeMain_injectFluidInfo(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 64 + groupThreadIndex.y * 8 + groupThreadIndex.x;
    if (groupThreadIndexU == 0) {
        groupFluidFogInfo = fluidFogInfoBuffer[pushConst.fluidIndex];
        groupInFluidInstanceCount = pushConst.inFluidInstanceCountAddress[0];
    }
    GroupMemoryBarrierWithGroupSync();
    int instanceCount = groupInFluidInstanceCount;
    int sharedInstanceCount = min(SHARED_INSTANCE_COUNT, instanceCount);
    if (groupThreadIndexU < sharedInstanceCount) groupInstanceInfos[groupThreadIndexU] = pushConst.inFluidInstanceInfoAddress[groupThreadIndexU];
    GroupMemoryBarrierWithGroupSync();
    
    FluidFogInfo fluidFogInfo = groupFluidFogInfo;
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;

    AABB voxelAABB;
    voxelAABB.minimum = pushConst.fluidStartPos + voxelIndex * voxelSize;
    voxelAABB.maximum = voxelAABB.minimum + voxelSize;

    float3 velocity = 0.0f; bool isBoundary = false;
    for (int i = 0; i < sharedInstanceCount; ++i) {
        InFluidInstanceInfo instanceInfo = groupInstanceInfos[i];
        if (!AABBBoundaryIntersects(instanceInfo.aabb, voxelAABB)) continue;

        //if (pushConst.time == 0 && voxelIndex.y == 31 && voxelIndex.x == 16 && voxelIndex.z == 14) {
        //    printf("InstanceAABB: %f %f %f %f %f %f\n", 
        //        instanceInfo.aabb.minimum.x, instanceInfo.aabb.minimum.y, instanceInfo.aabb.minimum.z,
        //        instanceInfo.aabb.maximum.x, instanceInfo.aabb.maximum.y, instanceInfo.aabb.maximum.z,
        //    );
        //}

        velocity += instanceInfo.velocity;
        isBoundary = true;
    }
    if (instanceCount > SHARED_INSTANCE_COUNT) {
        for (int i = SHARED_INSTANCE_COUNT; i < instanceCount; ++i) {
            InFluidInstanceInfo instanceInfo = pushConst.inFluidInstanceInfoAddress[i];
            if (!AABBBoundaryIntersects(instanceInfo.aabb, voxelAABB)) continue;

            //if (pushConst.time == 0 && voxelIndex.y == 31 && voxelIndex.x == 16 && voxelIndex.z == 14) {
            //    printf("InstanceAABB: %f %f %f %f %f %f\n", 
            //        instanceInfo.aabb.minimum.x, instanceInfo.aabb.minimum.y, instanceInfo.aabb.minimum.z,
            //        instanceInfo.aabb.maximum.x, instanceInfo.aabb.maximum.y, instanceInfo.aabb.maximum.z,
            //    );
            //}

            velocity += instanceInfo.velocity;
            isBoundary = true;
        }
    }
    if (isBoundary) {
        uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
        fluidFogVoxelInfoBuffer[pushConst.fluidIndex][voxelIndexU].isBoundary = true;
        if (any(abs(velocity) > 0.01f))
            fluidFogVoxelVelocityImage[pushConst.fluidIndex][voxelIndex].xyz = velocity * fluidFogInfo.FIntensity;

        //if (pushConst.time == 0 && voxelIndex.y == 31 && voxelIndex.x == 16 && voxelIndex.z == 14) {
        //    printf("voxelIndex: %d %d %d, velocity: %f %f %f\n", 
        //        voxelIndex.x, voxelIndex.y, voxelIndex.z,
        //        velocity.x, velocity.y, velocity.z,
        //    );
        //}
    } 
    //else {
    //    if (pushConst.time == 0 && voxelIndex.y == 31 && voxelIndex.x == 16 && voxelIndex.z == 14) {
    //        printf("voxelIndex: %d %d %d, velocity: %f %f %f\n",
    //            voxelIndex.x, voxelIndex.y, voxelIndex.z,
    //            velocity.x, velocity.y, velocity.z,
    //        );
    //    }
    //}

    //if (any(abs(velocity) > 0.01f) && pushConst.time == 0) {
    //    printf("voxelIndex: %d %d %d\nvoxelAABB: %f %f %f %f %f %f\n\n", 
    //        voxelIndex.x, voxelIndex.y, voxelIndex.z,
    //        voxelAABB.minimum.x, voxelAABB.minimum.y, voxelAABB.minimum.z,
    //        voxelAABB.maximum.x, voxelAABB.maximum.y, voxelAABB.maximum.z,
    //    );
    //}
}
*/