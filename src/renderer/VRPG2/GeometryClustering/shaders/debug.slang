#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryClustering/GeometryClusteringShaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

// Debug visualization of the geometry octree and of the SVO, ported from
// VRPG/Octree/shaders/Debug.slang. One instance per node of the selected layer / normal partition,
// the cube wireframe (MeshSet::createWireframe) is drawn as an instanced line list.
//
//   mode 0 : geometry octree  -> indivisible nodes of ClusterData_Octree[layer]
//   mode 2 : SVO              -> nodes registered in ClusterInfo_SVO with their layer color

[[vk::push_constant]] ConstantBuffer<GeometryClusteringDebugPushConstant_VRPG> pushConst;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eGlobalInfo)]] RWStructuredBuffer<GlobalInfo_VRPG2, ScalarDataLayout> GlobalInfoBuffer;
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterData_Octree)]] RWStructuredBuffer<ClusterData_Octree_VRPG, ScalarDataLayout> ClusterData_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_Octree)]] RWStructuredBuffer<ClusterInfo_Octree_VRPG, ScalarDataLayout> ClusterInfo_Octree[];
[[vk::binding(GeometryClusteringBindingPoints_VRPG::eClusterInfo_SVO)]] RWStructuredBuffer<OctreeNodeInfo_VRPG2, ScalarDataLayout> ClusterInfo_SVO;

struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float4 color;
};
struct PSout {
    float4 color0 : SV_Target0;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    bool show = false;
    AABB nodeAABB;
    float3 color = float3(1.0f, 0.0f, 0.0f);

    if (pushConst.mode == 0) {
        int nodeIndex = pushConst.normalIndex_G * pushConst.curLayerNodeCount_G + instanceIndex;
        ClusterData_Octree_VRPG node = ClusterData_Octree[pushConst.curLayer_G][nodeIndex];
        show = node.indivisible == 1 && node.meanNormal.w > 0.0f;
        nodeAABB = node.aabb;
        // brightness encodes the surface coverage of the node
        color = float3(1.0f, 0.25f + 0.75f * clamp(node.fillRate, 0.0f, 1.0f), 0.25f);
    } else {
        int nodeIndex = pushConst.normalIndex_G_2 * pushConst.curLayerNodeCount_G_2 + instanceIndex;
        ClusterInfo_Octree_VRPG nodeInfo = ClusterInfo_Octree[pushConst.curLayer_G_2][nodeIndex];
        int label = int(nodeInfo.label_indivisible >> 1);
        bool indivisible = (nodeInfo.label_indivisible & 1u) == 1u;

        // global label = prefix over the previous layers + layer-local label - 1
        int globalLabel = label - 1;
        for (int i = 0; i < pushConst.curLayer_G_2; ++i)
            globalLabel += GlobalInfoBuffer[0].layerInfos_G[i].indivisibleNodeCount;

        show = indivisible && label >= 1 && globalLabel >= 0 && globalLabel < int(GlobalInfoBuffer[0].ClusterCount);
        if (show) {
            // only nodes that really made it into the SVO are drawn (same as VRPG)
            OctreeNodeInfo_VRPG2 svo = ClusterInfo_SVO[globalLabel];
            show = svo.layerIndex == pushConst.curLayer_G_2 && svo.nodeIndex == nodeIndex;
        }
        nodeAABB = ClusterData_Octree[pushConst.curLayer_G_2][nodeIndex].aabb;

        int layer = pushConst.curLayer_G_2;
        if (layer == 5) color = float3(1.0f, 0.0f, 0.0f);
        else if (layer == 4) color = float3(0.0f, 1.0f, 0.0f);
        else if (layer == 3) color = float3(0.0f, 0.6f, 1.0f);
        else if (layer == 2) color = float3(1.0f, 1.0f, 0.0f);
        else color = float3(1.0f, 1.0f, 1.0f);
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
    } else {
        // degenerate clip position: the whole instance collapses and is culled
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
