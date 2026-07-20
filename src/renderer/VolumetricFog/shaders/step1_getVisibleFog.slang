#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> volumetricFogFluidInfoBuffer;

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
    uint warpVisibleFogCount = WaveActiveSum(uint(inCameraFrustum));

    uint visibleFogOffset;
    if (warpLane == 0)
        InterlockedAdd(groupVisibleFogCount, warpVisibleFogCount, visibleFogOffset);
    visibleFogOffset = WaveReadLaneAt(visibleFogOffset, 0);
    GroupMemoryBarrierWithGroupSync();

    visibleFogOffset += WavePrefixCountBits(inCameraFrustum);
    if (inCameraFrustum)
        globalInfoBuffer[0].visibleVolumetricFogIndices[visibleFogOffset] = threadIndex;

    if (threadIndex == 0) {
        VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[pushConst.fluidFogIndex];
        FluidFogInfo fluidFogInfo = volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex];

        globalInfoBuffer[0].visibleVolumetricFogCount = groupVisibleFogCount;
        globalInfoBuffer[0].fluidAABB = { fogInfo.fogStartPos, fogInfo.fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
        globalInfoBuffer[0].fluidStartUp = fluidFogInfo.startUp;
    }
}