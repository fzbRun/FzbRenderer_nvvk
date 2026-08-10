#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#define COMPUTE_SHADER
#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_test(uint threadIndex: SV_DispatchThreadID) {
    GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float3 compressionPrecision = pushConst.compressionParams;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;
    uint3 gridSize = pushConst.frustumGridSize;
    float3 gridSizeF = float3(gridSize);

    float3 uvw = float3(0.501191, 0.938060, 0.585981);
    uint3 voxelIndex = uint3(uvw * gridSize);
    if (pushConst.frameIndex < 10) {
        float data = envVolumetricFogInfoImage_sample.SampleLevel(uvw, 0).x;
        printf("uvw: %f %f %f voxelIndex: %d %d %d\n ext: %f\n", 
            uvw.x, uvw.y, uvw.z, voxelIndex.x, voxelIndex.y, voxelIndex.z, 
            data
        );
        for (int offsetX = -1; offsetX <= 1; ++offsetX) {
            for (int offsetY = -1; offsetY <= 1; ++offsetY) {
                for (int offsetZ = -1; offsetZ <= 1; ++offsetZ) {
                    uint3 offsetVoxelIndex = voxelIndex + uint3(offsetX, offsetY, offsetZ);
                    float ext = envVolumetricFogInfoImage_sample[offsetVoxelIndex].x;
                    printf("offsetVoxelIndex: %d %d %d, ext: %f\n", offsetVoxelIndex.x, offsetVoxelIndex.y, offsetVoxelIndex.z, ext);
                }
            }
        }
    }
}