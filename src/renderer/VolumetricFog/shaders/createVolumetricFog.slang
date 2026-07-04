#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVisibleVolumetricFogIndexBuffer)]] RWStructuredBuffer<uint> visibleVolumetricFogIndexBuffer;

// localVolumetricFogCount <= 1024, so one threadGroup can deal with
groupshared uint groupVisibleFogCount;
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getVisibleVolumetricFog(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) groupVisibleFogCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= pushConst.volumetricFogCount) return;

    VolumetricFogInfo info = volumetricFogInfosBuffer[threadIndex];
    float4x4 cameraPV = pushConst.sceneInfoAddress[0].viewProjMatrix;

    FrustumPlane planes[6];
    ExtractFrustumPlanes(cameraPV, planes);

    AABB fogAABB = { info.fogStartPos, info.fogStartPos + info.fogVoxelGridSize * info.fogVoxelSize };

    bool inCameraFrustum = isAABBInsideFrustum(fogAABB, planes);

    uint warpIndex = threadIndex / 32;
    uint warpLane = threadIndex & 31;
    uint warpVisibleFogCount = WaveActiveSum(inCameraFrustum);

    uint visibleFogOffset;
    if (warpLane == 0)
        InterlockedAdd(groupVisibleFogCount, warpVisibleFogCount, visibleFogOffset);
    visibleFogOffset = WaveReadLaneAt(visibleFogOffset, 0);
    GroupMemoryBarrierWithGroupSync();

    visibleFogOffset += WavePrefixCountBits(inCameraFrustum);
    visibleVolumetricFogIndexBuffer[visibleFogOffset] = threadIndex;

    //if (pushConst.frameIndex == 0 && inCameraFrustum) printf("%d %d\n", threadIndex, visibleFogOffset);

    if (threadIndex == 0) globalInfoBuffer[0].visibleVolumetricFogCount = groupVisibleFogCount;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImages)]] RWTexture3D<float4> volumetricFogImages[];

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog(uint3 threadIndex: SV_DispatchThreadID, uint3 groupThreadIndex: SV_GroupThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (threadIndex.x >= fogInfo.fogVoxelGridSize.x || threadIndex.y >= fogInfo.fogVoxelGridSize.y || threadIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * threadIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float scale = (voxelCenter.y - fogInfo.fogStartPos.y) / (fogInfo.fogVoxelGridSize.y * fogInfo.fogVoxelSize.y);
    float absorption = fogInfo.absorption.y - (fogInfo.absorption.y - fogInfo.absorption.x) * pow(scale, 2);

    //uint threadIndexU = threadIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y) + threadIndex.y * pushConst.fogVoxelGridSize.x + threadIndex.x;
    //uint randomSeed = xxhash32(uint3(threadIndexU, pushConst.frameIndex, 1));

    float extinctionCoefficient = absorption + fogInfo.scattering;
    volumetricFogImages[pushConst.instanceIndex][threadIndex] = extinctionCoefficient;
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo1Image)]] RWTexture3D<float4> volumetricFogVoxelInfo1Image[]; // xyz: velocity; w:
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo2Image)]] RWTexture3D<float4> volumetricFogVoxelInfo2Image[]; // xyz: dirtyVelocity1; w: pressure1
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo3Image)]] RWTexture3D<float4> volumetricFogVoxelInfo3Image[]; // xyz: dirtyVelocity2; w: pressure2

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogVoxelInfo1Image_sample)]] Sampler3D<float4> volumetricFogVoxelInfo1Image_sample[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogImages_sampler)]] Sampler3D<float4> volumetricFogImages_sample[];

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_initVolumetricFogFluid(uint3 voxelIndex: SV_DispatchThreadID){
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    float absorption = 0.0f;
    if (voxelIndex.y > fogInfo.fogVoxelGridSize.y / 2) absorption = 3.0f;
    volumetricFogImages[pushConst.instanceIndex][voxelIndex] = absorption;

    volumetricFogVoxelInfo1Image[pushConst.volumetricFogFluidIndex][voxelIndex] = 0.0f;
    volumetricFogVoxelInfo2Image[pushConst.volumetricFogFluidIndex][voxelIndex] = 0.0f;
    volumetricFogVoxelInfo3Image[pushConst.volumetricFogFluidIndex][voxelIndex] = 0.0f;
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_A(uint3 voxelIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;

    AABB voxelAABB;
    voxelAABB.minimum = fogInfo.fogStartPos + fogInfo.fogVoxelSize * voxelIndex;
    voxelAABB.maximum = voxelAABB.minimum + fogInfo.fogVoxelSize;
    float3 voxelCenter = (voxelAABB.minimum + voxelAABB.maximum) * 0.5f;

    float3 uvw = (voxelCenter - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    float3 velocity = volumetricFogVoxelInfo1Image_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw, 0).xyz;

    float3 worldPos_lastTime = voxelCenter - pushConst.dt * velocity;
    float3 uvw_lastTime = (worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
    float3 velocity_lastTime = volumetricFogVoxelInfo1Image_sample[pushConst.volumetricFogFluidIndex].SampleLevel(uvw_lastTime, 0).xyz;

    volumetricFogVoxelInfo1Image[pushConst.volumetricFogFluidIndex][voxelIndex].xyz = velocity_lastTime;
    volumetricFogVoxelInfo2Image[pushConst.volumetricFogFluidIndex][voxelIndex].xyz = velocity_lastTime;

    float3 absorption = volumetricFogImages_sample[pushConst.instanceIndex].SampleLevel((worldPos_lastTime - fogInfo.fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize), 0).xyz;
    volumetricFogImages[pushConst.instanceIndex][voxelIndex] = float4(absorption, 1.0f);
}

// --- Helper kernels: receive texture refs as params, avoid OpStore on image types ---
void diffusionKernel(uint3 voxelIndex, VolumetricFogInfo fogInfo, RWTexture3D<float4> cur, RWTexture3D<float4> nxt){
    float3 x_neighbor = 0.0f;
    if (voxelIndex.x > 0) x_neighbor += cur[uint3(voxelIndex.x - 1, voxelIndex.y, voxelIndex.z)].xyz;
    if (voxelIndex.x < fogInfo.fogVoxelGridSize.x - 1) x_neighbor += cur[uint3(voxelIndex.x + 1, voxelIndex.y, voxelIndex.z)].xyz;
    if (voxelIndex.y > 0) x_neighbor += cur[uint3(voxelIndex.x, voxelIndex.y - 1, voxelIndex.z)].xyz;
    if (voxelIndex.y < fogInfo.fogVoxelGridSize.y - 1) x_neighbor += cur[uint3(voxelIndex.x, voxelIndex.y + 1, voxelIndex.z)].xyz;
    if (voxelIndex.z > 0) x_neighbor += cur[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z - 1)].xyz;
    if (voxelIndex.z < fogInfo.fogVoxelGridSize.z - 1) x_neighbor += cur[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z + 1)].xyz;
    float3 b = cur[voxelIndex].xyz;
    float alpha = (fogInfo.fogVoxelSize.x * fogInfo.fogVoxelSize.x) / max(fogInfo.viscosity * pushConst.dt, 1e-30f);
    nxt[voxelIndex].xyz = (x_neighbor + alpha * b) / (6.0f + alpha);
}
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_D(uint3 voxelIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;
    uint fIdx = pushConst.volumetricFogFluidIndex;
    if ((pushConst.iteration & 1) == 0)
        diffusionKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo2Image[fIdx], volumetricFogVoxelInfo3Image[fIdx]);
    else
        diffusionKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo3Image[fIdx], volumetricFogVoxelInfo2Image[fIdx]);
}

[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_F(uint3 voxelIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;
    uint fIdx = pushConst.volumetricFogFluidIndex;

    float3 vel;
    if ((pushConst.iteration & 1) == 0) vel = volumetricFogVoxelInfo3Image[fIdx][voxelIndex].xyz;
    else vel = volumetricFogVoxelInfo2Image[fIdx][voxelIndex].xyz;

    float density = volumetricFogImages[pushConst.instanceIndex][voxelIndex].x;
    vel.y -= density * 5.0f * pushConst.dt;

    if ((pushConst.iteration & 1) == 0) volumetricFogVoxelInfo3Image[fIdx][voxelIndex].xyz = vel;
    else volumetricFogVoxelInfo2Image[fIdx][voxelIndex].xyz = vel;
}

void pressureKernel(uint3 voxelIndex, VolumetricFogInfo fogInfo, RWTexture3D<float4> cur, RWTexture3D<float4> nxt){
    uint3 dims = fogInfo.fogVoxelGridSize;
    float3 cellSize = fogInfo.fogVoxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = 2.0f * cellSize.y;
    float halfDz = 2.0f * cellSize.z;
    float4 self = cur[voxelIndex];
    float p_self = self.w;
    float3 velL = 0.0f, velR = 0.0f, velB = 0.0f, velT = 0.0f, velD = 0.0f, velU = 0.0f;
    float pL = p_self, pR = p_self, pB = p_self, pT = p_self, pD = p_self, pU = p_self;
    if (voxelIndex.x > 0) { float4 n = cur[uint3(voxelIndex.x - 1, voxelIndex.y, voxelIndex.z)]; pL = n.w; velL = n.xyz; }
    if (voxelIndex.x < dims.x - 1) { float4 n = cur[uint3(voxelIndex.x + 1, voxelIndex.y, voxelIndex.z)]; pR = n.w; velR = n.xyz; }
    if (voxelIndex.y > 0) { float4 n = cur[uint3(voxelIndex.x, voxelIndex.y - 1, voxelIndex.z)]; pB = n.w; velB = n.xyz; }
    if (voxelIndex.y < dims.y - 1) { float4 n = cur[uint3(voxelIndex.x, voxelIndex.y + 1, voxelIndex.z)]; pT = n.w; velT = n.xyz; }
    if (voxelIndex.z > 0) { float4 n = cur[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z - 1)]; pD = n.w; velD = n.xyz; }
    if (voxelIndex.z < dims.z - 1) { float4 n = cur[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z + 1)]; pU = n.w; velU = n.xyz; }
    float div = (velR.x - velL.x) / halfDx + (velT.y - velB.y) / halfDy + (velU.z - velD.z) / halfDz;
    float alpha = cellSize.x * cellSize.x;
    float sumPressure = pL + pR + pB + pT + pD + pU;
    nxt[voxelIndex].w = (sumPressure - alpha * div) / 6.0f;
}
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_P(uint3 voxelIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;
    uint fIdx = pushConst.volumetricFogFluidIndex;
    if ((pushConst.iteration & 1) == 0)
        pressureKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo2Image[fIdx], volumetricFogVoxelInfo3Image[fIdx]);
    else
        pressureKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo3Image[fIdx], volumetricFogVoxelInfo2Image[fIdx]);
}

void gradientKernel(uint3 voxelIndex, VolumetricFogInfo fogInfo, RWTexture3D<float4> pTex, uint fluidIdx){
    float4 cur = pTex[voxelIndex];
    float3 vel_dirty = cur.xyz;
    float p_self = cur.w;
    uint3 dims = fogInfo.fogVoxelGridSize;
    float3 cellSize = fogInfo.fogVoxelSize;
    float halfDx = 2.0f * cellSize.x;
    float halfDy = 2.0f * cellSize.y;
    float halfDz = 2.0f * cellSize.z;
    float pL, pR, pB, pT, pD, pU;
    if (voxelIndex.x > 0) pL = pTex[uint3(voxelIndex.x - 1, voxelIndex.y, voxelIndex.z)].w; else pL = p_self;
    if (voxelIndex.x < dims.x - 1) pR = pTex[uint3(voxelIndex.x + 1, voxelIndex.y, voxelIndex.z)].w; else pR = p_self;
    if (voxelIndex.y > 0) pB = pTex[uint3(voxelIndex.x, voxelIndex.y - 1, voxelIndex.z)].w; else pB = p_self;
    if (voxelIndex.y < dims.y - 1) pT = pTex[uint3(voxelIndex.x, voxelIndex.y + 1, voxelIndex.z)].w; else pT = p_self;
    if (voxelIndex.z > 0) pD = pTex[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z - 1)].w; else pD = p_self;
    if (voxelIndex.z < dims.z - 1) pU = pTex[uint3(voxelIndex.x, voxelIndex.y, voxelIndex.z + 1)].w; else pU = p_self;
    float3 pg = float3((pR - pL) / halfDx, (pT - pB) / halfDy, (pU - pD) / halfDz);
    volumetricFogVoxelInfo1Image[fluidIdx][voxelIndex].xyz = vel_dirty - pg;
}
[numthreads(4, 4, 4)]
[shader("compute")]
void computeMain_createVolumetricFog_Fluid_S(uint3 voxelIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];
    if (voxelIndex.x >= fogInfo.fogVoxelGridSize.x || voxelIndex.y >= fogInfo.fogVoxelGridSize.y || voxelIndex.z >= fogInfo.fogVoxelGridSize.z) return;
    uint fIdx = pushConst.volumetricFogFluidIndex;
    if ((pushConst.iteration & 1) == 0)
        gradientKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo2Image[fIdx], fIdx);
    else
        gradientKernel(voxelIndex, fogInfo, volumetricFogVoxelInfo3Image[fIdx], fIdx);
}
//---------------------------------------------------------------------------------------------------------------------------------------------------------------------
[numthreads(1, 1, 1)]
[shader("compute")]
void computeMain_createLightAttenuationEstimator(uint threadIndex: SV_DispatchThreadID) {
    VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.instanceIndex];

    float3 fogStartPos = fogInfo.fogStartPos;
    float3 voxelSize = fogInfo.fogVoxelSize;
    int3 gridSize = fogInfo.fogVoxelGridSize;

    float3 gridMin = fogStartPos;
    float3 gridMax = gridMin + gridSize * voxelSize;

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

        tMax.x = (step.x != 0) ? (voxelBorder.x - startPos.x) * invDir.x : 1e30;
        tMax.y = (step.y != 0) ? (voxelBorder.y - startPos.y) * invDir.y : 1e30;
        tMax.z = (step.z != 0) ? (voxelBorder.z - startPos.z) * invDir.z : 1e30;

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

            float3 uvw = (samplePoint - fogStartPos) / (fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize);
            float extinctionCoeff = volumetricFogImages_sample[pushConst.instanceIndex].SampleLevel(uvw, 0).x;

            attenuation *= exp(-extinctionCoeff * dt);
            if (all(attenuation <= 0.0f)) break;
        }

        tCurrent = tNext;
        if (tCurrent >= tExit - 1e-6) break;

        if (tMax.x < tMax.y) {
            if (tMax.x < tMax.z) { voxel.x += step.x; tMax.x += tDelta.x; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }else {
            if (tMax.y < tMax.z) { voxel.y += step.y; tMax.y += tDelta.y; }
            else { voxel.z += step.z; tMax.z += tDelta.z; }
        }
    }

    volumetricFogInfosBuffer[pushConst.instanceIndex].lightAttenuationEstimator = attenuation.x;
}