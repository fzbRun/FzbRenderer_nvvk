#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage)]] RWTexture3D<float4> volumetricFogAttenuationImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuation2Image)]] RWTexture3D<float4> volumetricFogAttenuation2Image;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogLImage)]] RWTexture3D<float4> volumetricFogLImage;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createAttenuationImage(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (any(voxelIndex >= pushConst.attenuationGridSize)) return;

    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 cameraPos = sceneInfo.cameraPosition;

    float3 voxelSize = 1.0f;
    float3 gridStartPos = cameraPos - pushConst.attenuationGridSize * voxelSize * 0.5f;
    float3 voxelCenter = (float3(voxelIndex) + 0.5f) * voxelSize + gridStartPos;

    uint voxelIndexU = voxelIndex.z * pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y + voxelIndex.y * pushConst.attenuationGridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(uint3(pushConst.frameIndex, voxelIndexU, 0));

    float3 L = 0.0f; float3 attenuation = 1.0f; float3 attenuation2 = 1.0f;
    uint sampleCount = lerp(5, 20, (float)voxelIndex.z / pushConst.attenuationGridSize.z);
    getVolumetricFogScatter(cameraPos, voxelCenter, randomSeed, sampleCount, L, attenuation, attenuation2);

    volumetricFogAttenuationImage[voxelIndex] = float4(attenuation, 1.0f);
    volumetricFogAttenuation2Image[voxelIndex] = float4(attenuation2, 1.0f);
    volumetricFogLImage[voxelIndex] = float4(L, 1.0f);
}