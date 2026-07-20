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

uint4 PackVectors(float3 v0, float3 v1, float3 v2){
    uint3 c0 = uint3(saturate(v0) * 255.0 + 0.5);
    uint3 c1 = uint3(saturate(v1) * 255.0 + 0.5);

    uint h2x = f32tof16(v2.x);
    uint h2y = f32tof16(v2.y);
    uint h2z = f32tof16(v2.z);

    uint4 packed;
    packed.r = c0.x | (c0.y << 8) | (c1.x << 16) | (c1.y << 24);
    packed.g = c0.z | (c1.z << 8) | ((h2x & 0xFFFF) << 16);
    packed.b = (h2x >> 16) | (h2y << 16);
    packed.a = h2z;

    return packed;
}

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
    //float t = (float(voxelIndex.z) + 0.5f) / float(gridSize.z);
    //t = pow(t, pushConst.compressionPrecision);
    //float depth = lerp(nearPlane, farPlane, t);

    float t_last = float(voxelIndex.z) / float(gridSize.z);
    t_last = pow(t_last, pushConst.compressionPrecision);
    float depth_last = lerp(nearPlane, farPlane, t_last);

    float t_next = float(voxelIndex.z + 1) / float(gridSize.z);
    t_next = pow(t_next, pushConst.compressionPrecision);
    float depth_next = lerp(nearPlane, farPlane, t_next);

    float halfDepth = (depth_next - depth_last) * 0.5f;
    float depth = depth_last + halfDepth;

    float halfHeight = depth * tanHalfFov;
    float halfWidth  = halfHeight * aspect;
    float3 viewPos = float3(u * halfWidth, v * halfHeight, depth);
    float3 voxelCenter = cameraPos + cameraRight * viewPos.x + cameraUp * viewPos.y + cameraForward * viewPos.z;

    uint voxelIndexU = voxelIndex.z * pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y + voxelIndex.y * pushConst.attenuationGridSize.x + voxelIndex.x;
    uint randomSeed = xxhash32(voxelIndex); // xxhash32(uint3(pushConst.frameIndex, voxelIndexU, 0));
    float3 samplePoint = voxelCenter;   // + (float3(rand(randomSeed), rand(randomSeed), rand(randomSeed)) * 2.0f - 1.0f) * float3(halfWidth / gridSize.x, halfHeight / gridSize.y, halfDepth);

    float3 L = 0.0f; float3 attenuation = 1.0f; float3 attenuation2 = 1.0f;
    getVolumetricFogScatter(cameraPos, samplePoint, randomSeed, pushConst.sampleCount, L, attenuation, attenuation2);

    volumetricFogAttenuationImage[voxelIndex] = float4(attenuation, 1.0f);
    volumetricFogAttenuation2Image[voxelIndex] = float4(attenuation2, 1.0f);
    volumetricFogLImage[voxelIndex] = float4(L, 1.0f);
}