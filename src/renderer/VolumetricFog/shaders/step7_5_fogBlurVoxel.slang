#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eFogAccResultHistoryImage_sample)]] Sampler3D<float4> renderedFogResultHistoryImage_sample;

#ifdef BLUR_FOG_VOXEL
[numthreads(8, 8, 8)]
[shader("compute")]
void computeMain_FogBlur_Voxel(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 64 + groupThreadIndex.y * 8 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();
    if (any(threadIndex >= pushConst.frustumGridSize)) return;

    SceneInfo sceneInfo = groupSceneInfo;
    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;

    int depthIndex = threadIndex.z;
    uint3 voxelIndex = threadIndex;

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionParams = max(pushConst.compressionParams, 1e-4f);
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    float linearDepth = ((float)depthIndex + 0.5f) / gridSize.z * (farPlane - nearPlane);
    float depth = frustumMap(linearDepth, compressionParams, farPlane - nearPlane, gridSize.z);

    float halfHeight = (depth + nearPlane) * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float3 voxelCenter = cameraPos;
    voxelCenter += (depth + nearPlane) * cameraForward;
    voxelCenter += halfHeight * ((((float)voxelIndex.y + 0.5f) / gridSize.y) * 2.0f - 1.0f) * cameraUp;
    voxelCenter += halfWidth * ((((float)voxelIndex.x + 0.5f) / gridSize.x) * 2.0f - 1.0f) * cameraRight;
    //---------------------------------------------------------------------------------------------------------
    float4 curVoxelInfo = volumetricFogAccResultImage[voxelIndex];

    float3 cameraPos_lastTime = pushConst.cameraPos_lastTime;
    //if (all(abs(cameraPos_lastTime - cameraPos) < 1e-6f)) return;

    float3 uvw_lastTime;
    {
        float viewZ = dot(voxelCenter - cameraPos_lastTime, cameraForward);
        float viewX = dot(voxelCenter - cameraPos_lastTime, cameraRight);
        float viewY = dot(voxelCenter - cameraPos_lastTime, cameraUp);

        if (viewZ < nearPlane || viewZ > farPlane) return;

        float halfHight_lastTime = viewZ * tanHalfFov;
        if (abs(viewY) > halfHight_lastTime) return;

        float halfWidth_lastTime = halfHight_lastTime * aspect;
        if (abs(viewX) > halfWidth_lastTime) return;

        uvw_lastTime.x = (viewX / halfWidth_lastTime) * 0.5f + 0.5f;
        uvw_lastTime.y = (viewY / halfHight_lastTime) * 0.5f + 0.5f;

        float sizeZ = farPlane - nearPlane;
        uvw_lastTime.z = frustumInvMap(viewZ - nearPlane, compressionParams, sizeZ, gridSize.z) / sizeZ;

        uvw_lastTime = clamp(uvw_lastTime, 0, 1);
    }
    if (pushConst.time > 1e-30f) {
        float2 lastVoxelInfo = renderedFogResultHistoryImage_sample.SampleLevel(uvw_lastTime, 0).xy;

        float mergeRatio = 0.1f;
        curVoxelInfo.x = curVoxelInfo.x * mergeRatio + lastVoxelInfo.x * (1.0f - mergeRatio);
        curVoxelInfo.y = curVoxelInfo.y * mergeRatio + lastVoxelInfo.y * (1.0f - mergeRatio);

        volumetricFogAccResultImage[voxelIndex] = curVoxelInfo;
    }
}
//------------------------------------------------------------------------------------------------------------------------


#endif