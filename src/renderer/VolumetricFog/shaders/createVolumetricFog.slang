#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImage)]] RWTexture3D<float4> volumetricFogImage;

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    if (threadIndex.x >= pushConst.fogVoxelGridSize.x || threadIndex.y >= pushConst.fogVoxelGridSize.y || threadIndex.z >= pushConst.fogVoxelGridSize.z) return;

    AABB voxelAABB;
    voxelAABB.minimum = pushConst.fogStartPos + pushConst.fogVoxelSize * threadIndex;
    voxelAABB.maximum = voxelAABB.minimum + pushConst.fogVoxelSize;

    float3 volumetricFogCenter = pushConst.fogStartPos + pushConst.fogVoxelSize * pushConst.fogVoxelGridSize * 0.5f;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float scale = distance(voxelCenter, volumetricFogCenter) / length(pushConst.fogVoxelSize * pushConst.fogVoxelGridSize * 0.5f);
    float extinctionCoefficient = lerp(pushConst.extinctionCoefficient.y, pushConst.extinctionCoefficient.x, scale);

    //uint threadIndexU = threadIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y) + threadIndex.y * pushConst.fogVoxelGridSize.x + threadIndex.x;
    //uint randomSeed = xxhash32(uint3(threadIndexU, pushConst.frameIndex, 1));
    //
    //float angle = (float(pushConst.frameIndex % 1000) / 999 * 2.0f - 1.0f) * M_PI_2;
    //float randomNumber = 1.0f - rand(randomSeed) * scale * scale * 0.2 * cos(angle);
    //extinctionCoefficient *= randomNumber;

    volumetricFogImage[threadIndex] = extinctionCoefficient;
}