#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVoxelGridImage)]] RWTexture2D<float4> voxelGridImage;

struct VSout {
    float4 sv_position : SV_Position;
}

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y)) / pushConst.fogVoxelGridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (pushConst.fogVoxelGridSize.x * pushConst.fogVoxelGridSize.y) - voxelIndex.y * pushConst.fogVoxelGridSize.x;

    float3 worldPos;
    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : pushConst.fogVoxelSize.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? 0.0f : pushConst.fogVoxelSize.y;
    worldPos.z = vertexIndex / 4 == 0 ? 0.0f : pushConst.fogVoxelSize.z;
    worldPos += voxelIndex * pushConst.fogVoxelSize + pushConst.fogStartPos;

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

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_mergeVoxelGridWithRendered(uint2 threadIndex: SV_DispatchThreadID) {
    uint2 screenSize;
    renderedImage.GetDimensions(screenSize.x, screenSize.y);
    if (threadIndex.x >= screenSize.x || threadIndex.y >= screenSize.y) return;

    float4 voxelGridColor = voxelGridImage[threadIndex];
    if (all(voxelGridColor == 0.0f)) voxelGridImage[threadIndex] = renderedImage[threadIndex];
}