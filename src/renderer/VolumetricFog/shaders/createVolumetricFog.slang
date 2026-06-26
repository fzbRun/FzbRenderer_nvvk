#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImage)]] RWTexture3D<float4> volumetricFogImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImage_sampler)]] Sampler3D<float4> volumetricFogImage_sample;
[[vk::binding(StaticBindingPoints_VolumetricFog::eLightAttenuationEstimatorBuffer)]] RWStructuredBuffer<float3> lightAttenuationEstimatorBuffer;

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

    uint threadIndexU = threadIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y) + threadIndex.y * pushConst.fogVoxelGridSize.x + threadIndex.x;
    uint randomSeed = xxhash32(uint3(threadIndexU, pushConst.frameIndex, 1));

    // float angle = (float(pushConst.frameIndex % 1000) / 999 * 2.0f - 1.0f) * M_PI_2;
    // float randomNumber = 1.0f - rand(randomSeed) * scale * scale * 0.2 * cos(angle);
    //float randomNumber = rand(randomSeed);
    //extinctionCoefficient *= randomNumber;

    volumetricFogImage[threadIndex] = extinctionCoefficient;
}

[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_createLightAttenuationEstimator(uint threadIndex: SV_DispatchThreadID) {
    float3 fogStartPos = pushConst.fogStartPos;
    float3 voxelSize = pushConst.fogVoxelSize;
    int3 gridSize = pushConst.fogVoxelGridSize;

    float3 gridMin = pushConst.fogStartPos;
    float3 gridMax = gridMin + pushConst.fogVoxelGridSize * pushConst.fogVoxelSize;

    float3 startPos = gridMax;
    float3 endPos = gridMin;
    float3 dir = normalize(endPos - startPos);

    float tEnter = 0.0f;
    float3 tExit3;
    tExit3.x = (abs(dir.x) > 1e-8) ? (gridMax.x - gridMin.x) / abs(dir.x) : 1e30;
    tExit3.y = (abs(dir.y) > 1e-8) ? (gridMax.y - gridMin.y) / abs(dir.y) : 1e30;
    tExit3.z = (abs(dir.z) > 1e-8) ? (gridMax.z - gridMin.z) / abs(dir.z) : 1e30;
    float tExit = min(min(tExit3.x, tExit3.y), tExit3.z);

    float eps = 1e-5 * min(min(voxelSize.x, voxelSize.y), voxelSize.z);
    float tCurrent = tEnter + eps;
    float3 pos = startPos + dir * tCurrent;

    float3 invDir; int3 voxel; int3 step; float3 tDelta; float3 voxelBorder; float3 tMax;
    {
        invDir.x = (abs(dir.x) > 1e-8) ? rcp(dir.x) : 1e30;
        invDir.y = (abs(dir.y) > 1e-8) ? rcp(dir.y) : 1e30;
        invDir.z = (abs(dir.z) > 1e-8) ? rcp(dir.z) : 1e30;

        float3 voxelF = (pos - fogStartPos) / voxelSize;
        voxel = int3(floor(voxelF));
        voxel = clamp(voxel, int3(0, 0, 0), gridSize - 1);

        step.x = (dir.x > 0.0) ? 1 : ((dir.x < 0.0) ? -1 : 0);
        step.y = (dir.y > 0.0) ? 1 : ((dir.y < 0.0) ? -1 : 0);
        step.z = (dir.z > 0.0) ? 1 : ((dir.z < 0.0) ? -1 : 0);

        tDelta.x = (step.x != 0) ? voxelSize.x * abs(invDir.x) : 1e30;
        tDelta.y = (step.y != 0) ? voxelSize.y * abs(invDir.y) : 1e30;
        tDelta.z = (step.z != 0) ? voxelSize.z * abs(invDir.z) : 1e30;

        voxelBorder.x = (step.x > 0) ? (fogStartPos.x + (voxel.x + 1) * voxelSize.x) : (fogStartPos.x + voxel.x * voxelSize.x);
        voxelBorder.y = (step.y > 0) ? (fogStartPos.y + (voxel.y + 1) * voxelSize.y) : (fogStartPos.y + voxel.y * voxelSize.y);
        voxelBorder.z = (step.z > 0) ? (fogStartPos.z + (voxel.z + 1) * voxelSize.z) : (fogStartPos.z + voxel.z * voxelSize.z);

        // 修正：基于 worldPos 计算全局 tMax
        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

        // 极微小修正：若因 eps 推进导致 tMax 略小于 tCurrent，补一整个体素
        if (step.x != 0 && tMax.x < tCurrent) tMax.x += tDelta.x;
        if (step.y != 0 && tMax.y < tCurrent) tMax.y += tDelta.y;
        if (step.z != 0 && tMax.z < tCurrent) tMax.z += tDelta.z;
    }

    float3 attenuation = 1.0f;
    int randomSeed = pushConst.frameIndex;
    int maxIter = gridSize.x + gridSize.y + gridSize.z;
    for (int i = 0; i < maxIter; i++) {
        if (any(voxel < 0) || any(voxel >= gridSize)) break;

        float tNext = min(min(tMax.x, tMax.y), tMax.z);
        tNext = min(tNext, tExit);
        float dt = tNext - tCurrent;

        if (dt > 0.0) {
            float tSample = tCurrent + dt * rand(randomSeed);
            float3 samplePoint = startPos + dir * tSample;

            float3 uvw = (samplePoint - pushConst.fogStartPos) / (pushConst.fogVoxelGridSize * pushConst.fogVoxelSize);
            float extinctionCoeff = volumetricFogImage_sample.SampleLevel(uvw, 0).x;

            attenuation *= exp(-extinctionCoeff * dt);
            if (all(attenuation <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
        else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }

    lightAttenuationEstimatorBuffer[0] = attenuation;
}