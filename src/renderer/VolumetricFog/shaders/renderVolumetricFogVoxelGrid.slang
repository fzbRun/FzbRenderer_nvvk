#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eHeightFogInfoBuffer)]] StructuredBuffer<HeightFogInfo, ScalarDataLayout> heightFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGridFogInfoBuffer)]] StructuredBuffer<GridFogInfo, ScalarDataLayout> gridFogInfoBuffer;

struct VSout {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    float3 voxelSize, startPos; uint3 gridSize;
    if (pushConst.instanceIndex == -100) {
        #ifdef USE_ENVFOG
        GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
        gridSize = globalInfo.envGridSize;
        voxelSize = globalInfo.envVoxelSize;
        startPos = globalInfo.envStartPos;
        #endif
    } else {
        VolumetricFogInfo info = fogInfosBuffer[pushConst.instanceIndex];
        if (info.type == VolumetricFogType::Height) {
            HeightFogInfo heightFoginfo = heightFogInfoBuffer[info.typeFogIndex];
            gridSize = 1;
            voxelSize = info.aabb.maximum - info.aabb.minimum;
        } else if (info.type == VolumetricFogType::Fluid) {
            FluidFogInfo fluidFoginfo = fluidFogInfoBuffer[info.typeFogIndex];
            gridSize = fluidFoginfo.gridSize;
            voxelSize = fluidFoginfo.voxelSize;
        } else if (info.type == VolumetricFogType::Grid) {
            GridFogInfo gridFoginfo = gridFogInfoBuffer[info.typeFogIndex];
            gridSize = gridFoginfo.gridSize;
            voxelSize = gridFoginfo.voxelSize;
        }
        startPos = info.aabb.minimum;
    }

    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (gridSize.x * gridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (gridSize.x * gridSize.y)) / gridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (gridSize.x * gridSize.y) - voxelIndex.y * gridSize.x;

    float3 worldPos;
    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : voxelSize.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? 0.0f : voxelSize.y;
    worldPos.z = vertexIndex / 4 == 0 ? 0.0f : voxelSize.z;
    worldPos += voxelIndex * voxelSize + startPos;

    VSout output;
    output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    return output;
}

struct PSout
{
    float4 color : SV_Target;
};
[shader("pixel")]
PSout fragmentMain(VSout input, float4 position: SV_Position) {
    return { float4(1.0f, 0.0f, 0.0f, 1.0f) };
}