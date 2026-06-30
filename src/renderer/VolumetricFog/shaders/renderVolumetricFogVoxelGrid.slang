#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eLocalVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> localVolumetricFogInfosBuffer;

struct VSout {
    float4 sv_position : SV_Position;
}

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    float3 worldPos;
    uint3 fogVoxelGridSize;
    float3 fogStartPos;
    float3 fogVoxelSize;
    if (pushConst.showVoxelGridIndex == -1) {
        fogVoxelGridSize = pushConst.fogVoxelGridSize;
        fogStartPos = pushConst.fogStartPos;
        fogVoxelSize = pushConst.fogVoxelSize;
    } else {
        VolumetricFogInfo info = localVolumetricFogInfosBuffer[pushConst.showVoxelGridIndex];
        fogVoxelGridSize = info.fogVoxelGridSize;
        fogStartPos = info.fogStartPos;
        fogVoxelSize = info.fogVoxelSize;
    }

    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (fogVoxelGridSize.x * fogVoxelGridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (fogVoxelGridSize.x * fogVoxelGridSize.y)) / fogVoxelGridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (fogVoxelGridSize.x * fogVoxelGridSize.y) - voxelIndex.y * fogVoxelGridSize.x;

    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : fogVoxelSize.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? 0.0f : fogVoxelSize.y;
    worldPos.z = vertexIndex / 4 == 0 ? 0.0f : fogVoxelSize.z;
    worldPos += voxelIndex * fogVoxelSize + fogStartPos;

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