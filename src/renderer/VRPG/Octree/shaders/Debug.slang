#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<OctreeDebugPushConstant_VRPG> pushConst;

[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_G)]] RWStructuredBuffer<OctreeNodeData_G_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_G[];
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeInfo_G)]] RWStructuredBuffer<OctreeNodeInfo_G_VRPG, ScalarDataLayout> OctreeNodeInfoBuffer_G[];
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];

[[vk::binding(BindingPoints_Octree_VRPG::eIndivisibleNodeInfos_G)]] RWStructuredBuffer<uint2> IndivisibleNodeInfoBuffer_G;

[[vk::binding(BindingPoints_Octree_VRPG::eGlobalInfo)]] RWStructuredBuffer<OctreeGlobalInfo_VRPG, ScalarDataLayout> GlobalInfoBuffer;
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
    bool show = false;
    AABB nodeAABB;
    float3 color = float3(1.0f, 0.0f, 0.0f);
    if (pushConst.mode == 0) {
        int nodeIndex = pushConst.normalIndex_G * pushConst.curLayerNodeCount_G + instanceIndex;
        OctreeNodeData_G_VRPG node = OctreeNodeDataBuffer_G[pushConst.curLayer_G][nodeIndex];
        show = node.indivisible == 1 && node.meanNormal.w > 0.0f;
        nodeAABB = node.aabb;
    } else if(pushConst.mode == 1) {
        int nodeIndex = pushConst.normalIndex_E * pushConst.curLayerNodeCount_E + instanceIndex;
        OctreeNodeData_E_VRPG node = OctreeNodeDataBuffer_E[pushConst.curLayer_E][nodeIndex];
        // OctreeNodeData_E_VRPG node = OctreeNodeDataBuffer_E[pushConst.maxLayer][instanceIndex];
        show = any(node.radiance > 0.0f);
        nodeAABB = node.aabb;
        color = node.radiance * 50.0f;
    } else {
        int nodeIndex = pushConst.normalIndex_G_2 * pushConst.curLayerNodeCount_G_2 + instanceIndex;
        OctreeNodeInfo_G_VRPG nodeInfo = OctreeNodeInfoBuffer_G[pushConst.curLayer_G_2][nodeIndex];
        int label = nodeInfo.label_indivisible >> 1;
        int indivisible = nodeInfo.label_indivisible & 1;

        show = indivisible == 1 && label >= 1;

        for (int i = 0; i < pushConst.curLayer_G_2; ++i)
            label += GlobalInfoBuffer[0].layerInfos_G[i].indivisibleNodeCount;

        uint2 indivisibleNodeInfo = IndivisibleNodeInfoBuffer_G[label - 1];
        OctreeNodeData_G_VRPG node = OctreeNodeDataBuffer_G[indivisibleNodeInfo.x][indivisibleNodeInfo.y];
        show = show && indivisibleNodeInfo.x == pushConst.curLayer_G_2 && indivisibleNodeInfo.y == nodeIndex;
        nodeAABB = node.aabb;

        color = float3(rand(instanceIndex), rand(instanceIndex), rand(instanceIndex));
    }

    VSout output;
    output.color = float4(color, 1.0f);
    if (show) {
        float3 aabbSize = nodeAABB.maximum - nodeAABB.minimum;

        output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
        output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
        output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
        output.sv_position.xyz += nodeAABB.minimum;

        output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    } else output.sv_position = float4(2.0, 2.0, 2.0, 1.0);

    return output;
}

[shader("pixel")]
PSout fragmentMain(VSout input) {
    return { input.color };
}
