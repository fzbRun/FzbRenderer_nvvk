//#define LPV_DEBUG
#include "renderer/LPV/shaders/LPVCommon.slang"

#ifdef LPV_DEBUG
struct VSout {
    float4 sv_position : SV_Position;
    bool debug_color;
    nointerpolation float4 color;
};
struct PSout{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout output;
    if (instanceIndex >= pushConst.debugVoxelCount) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    int voxelIndexU = instanceIndex;
    int gridIndex = pushConst.gridIndex;
    if (pushConst.gridIndex == pushConst.cascadeCount) {
        for (int i = 0; i < pushConst.cascadeCount; ++i) {
            int gridVoxelCount = pow(pushConst.gridInfo[i].resolution, 3);
            if (voxelIndexU < gridVoxelCount) break;
            voxelIndexU -= gridVoxelCount;
            gridIndex = i;
        }
    }
    int gridResolution = pushConst.gridInfo[gridIndex].resolution;
    float3 gridStartPos = pushConst.gridInfo[gridIndex].startPos;
    float gridVoxelSize = pushConst.gridInfo[gridIndex].voxelSize;

    int3 voxelIndex;
    voxelIndex.z = voxelIndexU / (gridResolution * gridResolution);
    voxelIndex.y = (voxelIndexU - voxelIndex.z * (gridResolution * gridResolution)) / gridResolution;
    voxelIndex.x = voxelIndexU % gridResolution;

    AABB voxelAabb;
    voxelAabb.minimum = gridStartPos + voxelIndex * gridVoxelSize;
    voxelAabb.maximum = voxelAabb.minimum + gridVoxelSize;

    float3 aabbSize = voxelAabb.maximum - voxelAabb.minimum;

    output.sv_position.x = ((vertexIndex & 3) * 2) % 6 == 0 ? 0.0f : aabbSize.x;
    output.sv_position.y = (vertexIndex & 3) <= 1 ? 0.0f : aabbSize.y;
    output.sv_position.z = vertexIndex / 4 == 0 ? 0.0f : aabbSize.z;
    output.sv_position.xyz += voxelAabb.minimum;

    output.sv_position = mul(float4(output.sv_position.xyz, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);

    return output;
}

[shader("pixel")]
PSout fragmentMain(VSout input) {
    return { input.color };
}
#endif