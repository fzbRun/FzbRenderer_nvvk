#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;

struct VSout {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VolumetricFogInfo info = volumetricFogInfosBuffer[pushConst.instanceIndex];

    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (info.fogVoxelGridSize.x * info.fogVoxelGridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (info.fogVoxelGridSize.x * info.fogVoxelGridSize.y)) / info.fogVoxelGridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (info.fogVoxelGridSize.x * info.fogVoxelGridSize.y) - voxelIndex.y * info.fogVoxelGridSize.x;

    float3 worldPos;
    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : info.fogVoxelSize.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? 0.0f : info.fogVoxelSize.y;
    worldPos.z = vertexIndex / 4 == 0 ? 0.0f : info.fogVoxelSize.z;
    worldPos += voxelIndex * info.fogVoxelSize + info.fogStartPos;

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