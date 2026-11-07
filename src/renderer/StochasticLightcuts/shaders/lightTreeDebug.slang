#include "renderer/StochasticLightcuts/StochasticLightcutsShaderio.h"

[[vk::push_constant]] ConstantBuffer<LightTreeDebugPushConstant_StochasticLightcuts> pushConst;
//------------------------------------------------------------------------------------------
struct VSout {
    float4 sv_position : SV_Position;
    bool debug_color;
    nointerpolation float4 color;
};
struct PSout
{
    float4 color0 : SV_Target0;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    LightNode node = pushConst.lightTreeAddress[pushConst.curLayerNodeStartIndex + instanceIndex];
    bool hasData = any(node.color > 0.0f);

    VSout output;
    output.color = float4(node.color, 1.0f);
    if (hasData) {
        float3 aabbSize = node.maximum - node.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += node.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    } else output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    return output;
}

[shader("pixel")]
PSout fragmentMain(VSout input) {
    return { input.color };
}
//------------------------------------------------------------------------------------------
[shader("vertex")]
VSout vertexMain2(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    LightNode node = pushConst.lightTreeAddress[pushConst.curLayerNodeStartIndex + instanceIndex];
    bool hasData = any(node.color > 0.0f);

    VSout output;
    output.color = float4(1.0f, 0.0f, 0.0f, 1.0f);
    if (hasData) {
        float3 aabbSize = node.maximum - node.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += node.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    } else output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    return output;
}

[shader("pixel")]
PSout fragmentMain2(VSout input) {
    return { input.color };
}