#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/LightClustering/LightClusteringShaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

// Debug view of the light tree, ported from VRPG's Octree_VRPG::lightTreeDebug
// (Octree/shaders/Debug.slang, mode 1).
//
// Wireframe box per light node of the selected layer / normal partition. Only nodes that
// received radiance are drawn; the color is the node radiance scaled by 50 like in VRPG.

[[vk::push_constant]] ConstantBuffer<LightClusteringDebugPushConstant_VRPG> pushConst;
[[vk::binding(LightClusteringBindingPoints_VRPG::eLightNodeData_Octree)]] RWStructuredBuffer<LightNodeData_Octree_VRPG, ScalarDataLayout> LightNodeData_Octree[];

struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float4 color;
};
struct PSout {
    float4 color0 : SV_Target0;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    int nodeIndex = pushConst.normalIndex_E * pushConst.curLayerNodeCount_E + instanceIndex;
    LightNodeData_Octree_VRPG node = LightNodeData_Octree[pushConst.curLayer_E][nodeIndex];

    VSout output;
    output.color = float4(node.radiance * 50.0f, 1.0f);

    if (any(node.radiance > 0.0f)) {
        float3 aabbSize = node.aabb.maximum - node.aabb.minimum;

        float3 worldPos;
        worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        worldPos.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        worldPos.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        worldPos += node.aabb.minimum;

        output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    } else {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    }

    return output;
}

[shader("fragment")]
PSout fragmentMain(VSout input) {
    PSout output;
    output.color0 = input.color;
    return output;
}
