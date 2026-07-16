#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
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
    float3 cameraRight   = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp      = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint3 gridSize = pushConst.attenuationGridSize;
    float nearPlane = pushConst.attenuationNearPlane;
    float farPlane  = pushConst.attenuationFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    float u = (float(voxelIndex.x) + 0.5f) / float(gridSize.x) * 2.0f - 1.0f;
    float v = (float(voxelIndex.y) + 0.5f) / float(gridSize.y) * 2.0f - 1.0f;
    float t = (float(voxelIndex.z) + 0.5f) / float(gridSize.z);
    const float zPower = 3.0f;
    t = pow(t, zPower);
    float depth = lerp(nearPlane, farPlane, t);

    float halfHeight = depth * tanHalfFov;
    float halfWidth  = halfHeight * aspect;
    float3 viewPos = float3(u * halfWidth, v * halfHeight, depth);

    float3 voxelCenter = cameraPos + cameraRight * viewPos.x + cameraUp * viewPos.y + cameraForward * viewPos.z;

    uint voxelIndexU = voxelIndex.z * pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y + voxelIndex.y * pushConst.attenuationGridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(voxelIndex);   //xxhash32(uint3(pushConst.frameIndex, voxelIndexU, 0));

    float3 L = 0.0f; float3 attenuation = 1.0f; float3 attenuation2 = 1.0f;
    getVolumetricFogScatter(cameraPos, voxelCenter, randomSeed, pushConst.sampleCount, L, attenuation, attenuation2);

    volumetricFogAttenuationImage[voxelIndex] = float4(attenuation, 1.0f);
    volumetricFogAttenuation2Image[voxelIndex] = float4(attenuation2, 1.0f);
    volumetricFogLImage[voxelIndex] = float4(L, 1.0f);
}