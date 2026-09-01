#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;

// localVolumetricFogCount <= 1024, so one threadGroup can deal with
groupshared uint groupVisibleFogCount;
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_getVisibleVolumetricFog(uint threadIndex: SV_DispatchThreadID) {
    if (threadIndex == 0) groupVisibleFogCount = 0;
    GroupMemoryBarrierWithGroupSync();
    if (threadIndex >= pushConst.volumetricFogCount) return;

    VolumetricFogInfo info = fogInfosBuffer[threadIndex];
    float4x4 cameraPV = pushConst.sceneInfoAddress[0].viewProjMatrix;

    FrustumPlane planes[6];
    ExtractFrustumPlanes(cameraPV, planes);
    bool inCameraFrustum = true;    //isAABBInsideFrustum(info.aabb, planes);
    uint warpIndex = threadIndex / 32;
    uint warpLane = threadIndex & 31;
    uint warpVisibleFogCount = WaveActiveSum(uint(inCameraFrustum));

    //if (threadIndex == 1) {
    //    printf("aabb: %f %f %f %f %f %f, inCameraFrustum: %d\n",
    //        info.aabb.minimum.x, info.aabb.minimum.y, info.aabb.minimum.z,
    //        info.aabb.maximum.x, info.aabb.maximum.y, info.aabb.maximum.z,
    //        inCameraFrustum,
    //    );
    //}

    uint visibleFogOffset;
    if (warpLane == 0) InterlockedAdd(groupVisibleFogCount, warpVisibleFogCount, visibleFogOffset);
    visibleFogOffset = WaveReadLaneAt(visibleFogOffset, 0);
    GroupMemoryBarrierWithGroupSync();

    visibleFogOffset += WavePrefixCountBits(inCameraFrustum);
    if (inCameraFrustum) globalInfoBuffer[0].visibleVolumetricFogIndices[visibleFogOffset] = threadIndex;

    if (info.type == VolumetricFogType::Fluid) {
        FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[info.typeFogIndex];

        globalInfoBuffer[0].fluidAABB[info.typeFogIndex] = info.aabb;
        globalInfoBuffer[0].fluidStartUp[info.typeFogIndex] = fluidFogInfo.startUp;
    }

    if (threadIndex == 0) globalInfoBuffer[0].visibleVolumetricFogCount = groupVisibleFogCount;
}