#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createEnvionmentFog(uint3 voxelIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    uint groupThreadIndexU = groupThreadIndex.z * 16 + groupThreadIndex.y * 4 + groupThreadIndex.x;
    getGroupsharedData(groupThreadIndexU);
    GroupMemoryBarrierWithGroupSync();

    if (any(voxelIndex >= pushConst.frustumGridSize)) return;

    GlobalInfo_VolumetricFog globalInfo = groupGlobalInfo;
    SceneInfo sceneInfo = groupSceneInfo;

    float3 cameraPos = sceneInfo.cameraPosition;
    float3 voxelCenter = getCameraFrustumVoxelCenter(voxelIndex, cameraPos, sceneInfo.viewInvMatrix, pushConst.frustumGridSize, 
        pushConst.cameraNearPlane, pushConst.cameraFarPlane, pushConst.tanCameraFov_2, pushConst.aspectRatio, pushConst.compressionPrecision);

    // global height fog
    float heightAttenuation = exp(-1.0f * max(voxelCenter.y - (-69.5f), 0.0f));
    float extinction = (0.01f + 0.03f) * heightAttenuation;
    float envExtinction = extinction;
    float envScattering = extinction * 0.03f;
    float envPhase = extinction * -0.7f;
    float3 envColor = extinction * 1.0f;

    for (int i = 0; i < globalInfo.visibleVolumetricFogCount; ++i) {
        uint fogIndex = globalInfo.visibleVolumetricFogIndices[i];
        VolumetricFogInfo fogInfo = groupVolumetricFogInfos[fogIndex];
        AABB otherFogAABB = { fogInfo.fogStartPos, fogInfo.fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
        if (!checkPointInAABB(voxelCenter, otherFogAABB)) continue;

        extinction = 0.0f;
        float scattering = 0.0f, phase = 0.0f; float3 color = 0.0f;
        if (fogInfo.type == VolumetricFogType::Height) {
            float heightScale = groupHeightFogInfos[fogInfo.volumetricFogTypeIndex].heightScale;
            float height = max(voxelCenter.y - fogInfo.fogStartPos.y, 0.0f);
            heightAttenuation = exp(-heightScale * height);
            scattering = fogInfo.scattering;
            extinction = (fogInfo.absorption.x + scattering) * heightAttenuation;
            phase = fogInfo.phase;
            color = fogInfo.color;
        } else if (fogInfo.type == VolumetricFogType::Fluid && globalInfo.fluidStartUp == 1) {
            float3 uvw = (voxelCenter - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
            uvw = clamp(uvw, 0.0f, 1.0f - 1e-6f);
            float4 voxelInfo = volumetricFogFluidVoxelInfoImages_sample[fogInfo.volumetricFogTypeIndex].SampleLevel(uvw, 0);

            // if in fluid, the voxelInfo has include the envionment fog info
            envExtinction = max(voxelInfo.x, 1e-30f);
            envScattering = envExtinction * voxelInfo.y;
            envPhase = envExtinction * voxelInfo.z;
            envColor = envExtinction * voxelInfo.w; // UnpackFloatToFloat3(voxelInfo.w);
            break;
        }
        else if (fogInfo.type == VolumetricFogType::Noise) {
            NoiseFogInfo noiseFogInfo = groupNoiseFogInfos[fogInfo.volumetricFogTypeIndex];
            CloudSetting cloudSetting = { noiseFogInfo.cloudScale, noiseFogInfo.cloudFlowSpeed, noiseFogInfo.cloudCoverage, noiseFogInfo.cloudTypePreference, noiseFogInfo.weatherScale };
            float fogDensity = SampleCloudDensity(voxelCenter, otherFogAABB.minimum, otherFogAABB.maximum, pushConst.time, cloudSetting);
            scattering = fogInfo.scattering;
            extinction = (fogInfo.absorption.x + scattering) * fogDensity;
            phase = fogInfo.phase;
            color = fogInfo.color;
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

    envVolumetricFogInfoImage[voxelIndex] = envVoxelInfo;

    //if (pushConst.frameIndex == 0 && all(voxelIndex == uint3(10, 10, 10))) {
    //    printf("voxelIndex: %d %d %d, voxelCenter: %f %f %f, envVoxelInfo: %f %f %f %f\n", 
    //        voxelIndex.x, voxelIndex.y, voxelIndex.z, 
    //        voxelCenter.x, voxelCenter.y, voxelCenter.z, 
    //        envVoxelInfo.x, envVoxelInfo.y, envVoxelInfo.z, envVoxelInfo.w
    //    );
    //}
}
