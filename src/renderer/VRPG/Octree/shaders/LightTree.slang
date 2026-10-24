#include "common/Shader/Slang/commonFunction.slang"
#include "feature/PathTracing/shaders/pathTracingCommon.slang"
#include "renderer/VRPG/VRPGShaderio.h"
#include "renderer/VRPG/RasterVoxelization/RasterVoxelizationShaderio_VRPG.h"
#include "renderer/VRPG/Octree/OctreeShaderio_VRPG.h"

[[vk::push_constant]] ConstantBuffer<OctreePushConstant_VRPG> pushConst;
[[vk::binding(BindingPoints_Octree_VRPG::eOctreeNodeData_E)]] RWStructuredBuffer<OctreeNodeData_E_VRPG, ScalarDataLayout> OctreeNodeDataBuffer_E[];
[[vk::binding(BindingPoints_Octree_VRPG::eLightVGB)]] StructuredBuffer<VGBVoxelData_VRPG, ScalarDataLayout> LightVGBs[];

AABB emptyAABB() {
    AABB aabb;
    aabb.minimum = float3(3.402823466e+38f);
    aabb.maximum = float3(-3.402823466e+38f);
    return aabb;
}

// Build the leaves of the E-tree directly from the light VGB. The leaf layer is
// lightOctreeMaxLayer, so the E-tree has its own resolution independent of G.
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_initLightTreeLeaves(uint threadIndex: SV_DispatchThreadID) {
    uint lightMaxLayer = pushConst.lightOctreeMaxLayer;
    uint lightVoxelTotalCount = 1u << (3 * lightMaxLayer);
    uint total = lightVoxelTotalCount * 6;
    if (threadIndex >= total) return;

    uint normalIndex = threadIndex / lightVoxelTotalCount;
    uint voxelIndex = threadIndex - normalIndex * lightVoxelTotalCount;
    VGBVoxelData_VRPG lightVoxel = LightVGBs[normalIndex][voxelIndex];

    OctreeNodeData_E_VRPG node;
    node.aabb = emptyAABB();
    node.radiance = float3(0.0f);
    node.halfAngle = M_PI;
    node.meanNormal = float4(0.0f);

    if (lightVoxel.irradiance.w > 0.0f) {
        node.aabb.minimum.x = OrderedIntToFloat(lightVoxel.aabbI.minimum.x);
        node.aabb.minimum.y = OrderedIntToFloat(lightVoxel.aabbI.minimum.y);
        node.aabb.minimum.z = OrderedIntToFloat(lightVoxel.aabbI.minimum.z);
        node.aabb.maximum.x = OrderedIntToFloat(lightVoxel.aabbI.maximum.x);
        node.aabb.maximum.y = OrderedIntToFloat(lightVoxel.aabbI.maximum.y);
        node.aabb.maximum.z = OrderedIntToFloat(lightVoxel.aabbI.maximum.z);

        float3 lengthAxis = node.aabb.maximum - node.aabb.minimum;
        int3 axisSign = sign(lengthAxis);
        lengthAxis += float3(int3(1) - axisSign) * 0.0001f;
        if (all(axisSign == 0)) lengthAxis = float3(0.0f);
        float aabbVolume = lengthAxis.x * lengthAxis.y + lengthAxis.x * lengthAxis.z + lengthAxis.y * lengthAxis.z;

        node.radiance = aabbVolume * lightVoxel.irradiance.xyz / lightVoxel.irradiance.w;
        node.meanNormal.xyz = normalize(lightVoxel.sumNormal_G.xyz);
        node.meanNormal.w = 1.0f;
    }

    OctreeNodeDataBuffer_E[lightMaxLayer][threadIndex] = node;
}

// Bottom-up aggregation of the light tree. currentLayer runs from
// lightOctreeMaxLayer - 1 down to 0.
[numthreads(CREATEOCTREE_CS_THREADGROUP_SIZE, 1, 1)]
[shader("compute")]
void computeMain_aggregateLightTree(uint threadIndex: SV_DispatchThreadID) {
    uint layer = pushConst.currentLayer;
    uint layerNodeCount = 6u << (3 * layer);
    if (threadIndex >= layerNodeCount) return;

    OctreeNodeData_E_VRPG parent;
    parent.aabb = emptyAABB();
    parent.radiance = float3(0.0f);
    parent.halfAngle = M_PI;
    parent.meanNormal = float4(0.0f);

    uint childBase = threadIndex * 8;
    uint childCount = 0;
    float3 normalSum = float3(0.0f);
    for (uint i = 0; i < 8; ++i) {
        OctreeNodeData_E_VRPG child = OctreeNodeDataBuffer_E[layer + 1][childBase + i];
        if (all(child.radiance == 0.0f)) continue;

        parent.aabb.minimum = min(parent.aabb.minimum, child.aabb.minimum);
        parent.aabb.maximum = max(parent.aabb.maximum, child.aabb.maximum);
        parent.radiance += child.radiance;
        normalSum += child.meanNormal.xyz;
        ++childCount;
    }

    if (childCount > 0) {
        parent.meanNormal.xyz = normalize(normalSum);
        parent.meanNormal.w = 1.0f;
    }
    OctreeNodeDataBuffer_E[layer][threadIndex] = parent;
}
