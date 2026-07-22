#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

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
void computeMain_createFrustumAccFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (any(voxelIndex >= pushConst.frustumGridSize)) return;

    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 voxelCenter = getCameraFrustumVoxelCenter(voxelIndex, cameraPos, sceneInfo.viewInvMatrix, pushConst.frustumGridSize, 
        pushConst.cameraNearPlane, pushConst.cameraFarPlane, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.compressionPrecision);

    uint voxelIndexU = voxelIndex.z * pushConst.frustumGridSize.x * pushConst.frustumGridSize.y + voxelIndex.y * pushConst.frustumGridSize.x + voxelIndex.x;
    uint randomSeed =  xxhash32(uint3(pushConst.frameIndex, voxelIndexU, 0));
    float3 samplePoint = voxelCenter;   // + (float3(rand(randomSeed), rand(randomSeed), rand(randomSeed)) * 2.0f - 1.0f) * float3(halfWidth / gridSize.x, halfHeight / gridSize.y, halfDepth);

    float3 L = 0.0f; float3 attenuation = 1.0f; float3 attenuation2 = 1.0f;
    getEnvVolumetricFogScatter(cameraPos, samplePoint, randomSeed, L, attenuation, attenuation2);

    volumetricFogAttenuationImage[voxelIndex] = float4(attenuation, 1.0f);
    volumetricFogAttenuation2Image[voxelIndex] = float4(attenuation2, 1.0f);
    volumetricFogLImage[voxelIndex] = float4(L, 1.0f);
}