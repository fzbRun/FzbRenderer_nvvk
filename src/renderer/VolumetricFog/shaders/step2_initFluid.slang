#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> FluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> FluidFogVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> FluidFogVoxelInfoBuffer[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> FluidFogVoxelVelocityImage_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages)]] RWTexture3D<float4> FluidFogVoxelInfoImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoImages_sampler)]] Sampler3D<float4> FluidFogVoxelInfoImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eHeightFogInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogInfoBuffer)]] StructuredBuffer<GridFogInfo, ScalarDataLayout> gridFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogImages)]] Sampler3D<float4> gridFogImages[];

groupshared VolumetricFogInfo groupOtherTypeFogInfos[MAX_VOLUMETRIC_FOG_COUNT];
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initFluidFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    if (groupThreadIndexU < pushConst.volumetricFogCount) groupOtherTypeFogInfos[groupThreadIndexU] = fogInfosBuffer[groupThreadIndexU];
    GroupMemoryBarrierWithGroupSync();

    int fogIndex = pushConst.instanceIndex;
    VolumetricFogInfo fogInfo = groupOtherTypeFogInfos[fogIndex];
    FluidFogInfo fluidFogInfo = FluidFogInfoBuffer[fogInfo.typeFogIndex];
    if (any(voxelIndex >= fluidFogInfo.gridSize)) return;

    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;

    float3 voxelCenter = fogInfo.aabb.minimum + (float3(voxelIndex) + 0.5f) * voxelSize;
    float heightAttenuation = exp(-1.0f * max(voxelCenter.y - (-69.5f), 0.0f));
    float extinction = (0.01f + 0.03f) * heightAttenuation;
    float envExtinction = extinction;
    float envScattering = extinction * 0.03f;
    float envPhase = extinction * -0.7f;
    float3 envColor = extinction * 1.0f;

    for (int i = 0; i < pushConst.volumetricFogCount; ++i) {
        if (i == fogIndex) continue;
        VolumetricFogInfo otherFogInfo = groupOtherTypeFogInfos[i];
        AABB otherFogAABB = otherFogInfo.aabb;
        if (!checkPointInAABB(voxelCenter, otherFogAABB)) continue;

        float scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
        if (otherFogInfo.type == VolumetricFogType::Height) {
            HeightFogInfo heightFogInfo = heightFogInfoBuffer[otherFogInfo.typeFogIndex];
            float heightScale = heightFogInfo.heightScale;
            float height = max(voxelCenter.y - otherFogInfo.aabb.minimum.y, 0.0f);
            heightAttenuation = exp(-heightScale * height);

            scattering = heightFogInfo.scattering;
            extinction = (heightFogInfo.absorption + scattering) * heightAttenuation;
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

    int3 voxelIndex_lastTime = int3((voxelCenter - fluidFogInfo.fogStartPos_lastTime) / voxelSize);
    if (any(voxelIndex_lastTime < 0) || any(voxelIndex_lastTime > gridSize - 1) || pushConst.time == 0) {
        FluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = 0.0f;
    } else {
        voxelInfo = FluidFogVoxelInfoImages[fogInfo.typeFogIndex][voxelIndex_lastTime];

        float3 velocity = FluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex_lastTime].xyz;

        float3 halfGrid = float3(gridSize) * 0.5f;
        float3 distToCenter = abs(float3(voxelIndex) - halfGrid);
        float maxPossibleDist = length(halfGrid);
        float centerDist = length(distToCenter) / maxPossibleDist;

        float edgeFactor = smoothstep(0.0, 1.0, centerDist);
        float restoreRate = 0.5f;
        float restoreWeight = saturate(restoreRate * pushConst.dt) * edgeFactor * fluidFogInfo.restoreSpeed;

        voxelInfo = lerp(voxelInfo, envVoxelInfo, saturate(restoreWeight));

        velocity = lerp(velocity, 0.0f, saturate(restoreWeight));
        FluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = velocity;
    }
    FluidFogVoxelInfoImages[fogInfo.typeFogIndex][voxelIndex] = voxelInfo;

    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    FluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].isBoundary = false;
}
