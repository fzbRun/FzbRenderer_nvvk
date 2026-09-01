#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<GridFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_GridFogSet::eGridFogImages)]] RWTexture3D<float4> gridFogImages[];

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initGridFogImage(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (any(voxelIndex >= pushConst.gridSize)) return;

    float3 voxelCenter = pushConst.fogRange.minimum + (float3(voxelIndex) + 0.5f) * pushConst.voxelSize;

    GridFogGenerationInfo generationInfo = pushConst.generationInfo;
    float4 voxelInfo = { pushConst.absorption + pushConst.scattering, pushConst.scattering, pushConst.phase, luminance(pushConst.color) };
    if (pushConst.mode == GridFogGenerateMode::Cloud) {
        CloudSetting cloudSetting = { generationInfo.cloudInfo.cloudScale, generationInfo.cloudInfo.cloudFlowSpeed, generationInfo.cloudInfo.cloudCoverage, generationInfo.cloudInfo.cloudTypePreference, generationInfo.cloudInfo.weatherScale };

        uint seed = pushConst.randomSeed;
        float time = rand(seed) * 1000.0f;
        float fogDensity = SampleCloudDensity(voxelCenter, pushConst.fogRange.minimum, pushConst.fogRange.maximum, time, cloudSetting);
        voxelInfo.x *= fogDensity;
        voxelInfo.y *= fogDensity;
    }
    gridFogImages[pushConst.index][voxelIndex] = voxelInfo;
}