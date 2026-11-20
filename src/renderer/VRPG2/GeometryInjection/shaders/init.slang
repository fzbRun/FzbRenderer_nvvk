#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"
#include "renderer/VRPG2/GeometryInjection/GeometryInjectionShaderio.h"

[[vk::push_constant]] ConstantBuffer<GeometryInjectionPushConstant_VRPG> pushConst;
[[vk::binding(GeometryInjectionBinding_VRPG::eVGB)]] RWStructuredBuffer<VoxelData_VRPG, ScalarDataLayout> VGBs[];

[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_clearVGB(uint threadIndex: SV_DispatchThreadID) {
    uint voxelCount = uint(pushConst.voxelSize_Count.w * pushConst.voxelSize_Count.w * pushConst.voxelSize_Count.w);
    uint totalVoxelCount = voxelCount * 6;
    if (threadIndex >= totalVoxelCount) return;

    VoxelData_VRPG voxelData;
    voxelData.aabbI.minimum = int4(FloatToOrderedInt(3.402823466e+38f));
    voxelData.aabbI.maximum = int4(FloatToOrderedInt(-3.402823466e+38f));
    voxelData.irradiance = float4(0.0f);
    voxelData.sumNormal_G = float4(0.0f);

    uint normalIndex = threadIndex / voxelCount;
    uint voxelIndex = threadIndex & (voxelCount - 1);
    VGBs[normalIndex][voxelIndex] = voxelData;
}