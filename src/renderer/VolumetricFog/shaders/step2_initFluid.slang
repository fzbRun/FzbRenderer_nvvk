#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> volumetricFogFluidInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage)]] RWTexture3D<float4> volumetricFogFluidVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer)]] RWStructuredBuffer<VolumetricFogFluidVoxelInfo, ScalarDataLayout> volumetricFogFluidVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage_sample)]] Sampler3D<float4> volumetricFogFluidVoxelVelocityImage_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages)]] RWTexture3D<float4> volumetricFogFluidVoxelInfoImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoImages_sampler)]] Sampler3D<float4> volumetricFogFluidVoxelInfoImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogHeightInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogNoiseInfoBuffer)]] StructuredBuffer<NoiseFogInfo, ScalarDataLayout> noiseFogInfoBuffer;

groupshared VolumetricFogInfo groupOtherTypeFogInfos[MAX_VOLUMETRIC_FOG_COUNT];

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
    float heightAttenuation = exp(-1.0f * max(voxelCenter.y - (-69.5f), 0.0f));
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
            float height = max(voxelCenter.y - otherFogInfo.fogStartPos.y, 0.0f);
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
        // float3 halfGrid = (float3)(fogInfo.fogVoxelGridSize) * 0.5f;
        // float3 distToEdge = halfGrid - abs((float3)voxelIndex - halfGrid);
        // float d = length(distToEdge * fogInfo.fogVoxelSize);
        // float speedDamping = exp(-length(velocity) * d * 3.0f); // spped + distance
        // float restoreRate = 0.5f;
        // float restoreWeight = saturate(restoreRate * pushConst.dt) * speedDamping * fluidFogInfo.restoreSpeed;

        float3 halfGrid = float3(fogInfo.fogVoxelGridSize) * 0.5f;
        float3 distToCenter = abs(float3(voxelIndex) - halfGrid);
        float maxPossibleDist = length(halfGrid);
        float centerDist = length(distToCenter) / maxPossibleDist;

        float edgeFactor = smoothstep(0.0, 1.0, centerDist);
        float restoreRate = 0.5f;
        float restoreWeight = saturate(restoreRate * pushConst.dt) * edgeFactor * fluidFogInfo.restoreSpeed;

        voxelInfo = lerp(voxelInfo, envVoxelInfo, saturate(restoreWeight));

        velocity = lerp(velocity, 0.0f, saturate(restoreWeight));
        volumetricFogFluidVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex].xyz = velocity;
    }
    volumetricFogFluidVoxelInfoImages[fogInfo.volumetricFogTypeIndex][voxelIndex] = voxelInfo;

    uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
    volumetricFogFluidVoxelInfoBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU].isBoundary = false;
}
