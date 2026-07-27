#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;

struct VSout {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    float3 voxelSize, startPos; uint3 gridSize;
    if (pushConst.instanceIndex == -100) {
        GlobalInfo_VolumetricFog globalInfo = globalInfoBuffer[0];
        gridSize = globalInfo.envGridSize;
        voxelSize = globalInfo.envVoxelSize;
        startPos = globalInfo.envStartPos;
    } else {
        VolumetricFogInfo info = volumetricFogInfosBuffer[pushConst.instanceIndex];
        gridSize = info.fogVoxelGridSize;
        voxelSize = info.fogVoxelSize;
        startPos = info.fogStartPos;
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