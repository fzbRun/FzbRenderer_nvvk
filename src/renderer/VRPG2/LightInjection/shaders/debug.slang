#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/LightInjection/LightInjectionShaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

// Debug view of the light injection result, ported from VRPG's LightInject::debug_Cube.
//
// The injection now writes the leaf layer of the light tree (16^3 light cells), so this pass
// draws one solid cube per non-empty light cell, colored by the radiance accumulated there
// (VRPG shows the same quantity as voxelData.irradiance.xyz / irradiance.w).

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant_VRPG> pushConst;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eLightNodeData)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData;

struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float3 color;
};
struct PSout {
    float4 color0 : SV_Target0;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    LightNodeData_Octree_VRPG node = LightNodeData[instanceIndex];

    VSout output;
    output.color = float3(0.0f);
    if (any(node.radiance > 0.0f)) {
        float3 worldPos;
        worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? node.aabb.minimum.x : node.aabb.maximum.x;
        worldPos.y = (vertexIndex & 3) <= 1 ? node.aabb.minimum.y : node.aabb.maximum.y;
        worldPos.z = vertexIndex / 4 == 0 ? node.aabb.minimum.z : node.aabb.maximum.z;

        output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
        output.color = node.radiance;
    } else {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    }

    return output;
}

[shader("fragment")]
PSout fragmentMain(VSout input) {
    PSout output;
    output.color0 = float4(input.color, 1.0f);
    return output;
}
