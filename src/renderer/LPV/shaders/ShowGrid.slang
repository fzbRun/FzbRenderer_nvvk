#ifndef LPV_DEBUG
#define LPV_DEBUG
#endif
//#include "renderer/LPV/shaders/LPVCommon.slang"
#include "renderer/LPV/shaders/RSM.slang"

#ifdef LPV_DEBUG
[numthreads(1024, 1, 1)]
[shader("compute")] void injectGridMain(uint threadIndex: SV_DispatchThreadID) {
    int n = LPV_CascadeResolution(pushConst.cascadeIndex);
    int cascade = pushConst.cascadeIndex;
    float3 gridStartPos = LPV_CascadeOrigin(cascade);
    float s = LPV_CascadeCellSize(cascade);

    if (pushConst.gridDataType == 0) {
        uint pixelIndex = threadIndex;
        if (pixelIndex >= pushConst.screenSize.x * pushConst.screenSize.y) return;

        LPVPixelData data = pixelData[pixelIndex];
        if (data.positionValid.w == 0.0f) return;

        int3 cell = int3(floor((data.positionValid.xyz - gridStartPos) / s));
        if (!LPV_CellInside(cell, n)) return;

        uint index = uint((cell.z * n + cell.y) * n + cell.x);
        InterlockedOr(gridDebugData[index].hasData, 1);
    } else if (pushConst.gridDataType == 1){
        uint pixelIndexU = threadIndex;
        if (pixelIndexU >= pushConst.rsmResolution * pushConst.rsmResolution) return;
        int2 pixelIndex = { pixelIndexU / pushConst.rsmResolution, pixelIndexU % pushConst.rsmResolution };

        float3 flux;
        float surfelArea;
        LPVHit hit;
        if (!LPV_RsmTexel(pixelIndex, pushConst.lightIndex, pushConst.rsmResolution, flux, surfelArea, hit)) return;

        int3 voxelIndex = int3(floor((hit.position - gridStartPos) / s));
        if (!LPV_CellInside(voxelIndex, n)) return;

        uint index = uint((voxelIndex.z * n + voxelIndex.y) * n + voxelIndex.x);
        InterlockedOr(gridDebugData[index].hasData, 1);
    } else {
        if (threadIndex >= n * n * n) return;

        int3 voxelIndex;
        voxelIndex.z = threadIndex / (n * n);
        voxelIndex.y = (threadIndex - voxelIndex.z * n * n) / n;
        voxelIndex.x = threadIndex % n;

        float4 R = LPV_LoadAccumSH(cascade, voxelIndex, 0);
        float4 G = LPV_LoadAccumSH(cascade, voxelIndex, 1);
        float4 B = LPV_LoadAccumSH(cascade, voxelIndex, 2);
        if(all(R == 0.0f) && all(G == 0.0f) && all(B == 0.0f)) return;

        uint index = uint((voxelIndex.z * n + voxelIndex.y) * n + voxelIndex.x);
        InterlockedOr(gridDebugData[index].hasData, 1);
    }
}

struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float4 color;
};
struct PSout{
    float4 color0 : SV_Target0;
};
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout output;
    output.color = float4(0.25f, 1.0f, 0.35f, 1.0f);
    if (instanceIndex >= pow(float(LPV_CascadeResolution(pushConst.cascadeIndex)), 3)) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    int voxelIndexU = instanceIndex;
    int cascadeIndex = pushConst.cascadeIndex;
    int gridResolution = pushConst.gridInfo[cascadeIndex].resolution;
    float3 gridStartPos = pushConst.gridInfo[cascadeIndex].startPos;
    float gridVoxelSize = pushConst.gridInfo[cascadeIndex].voxelSize;

    int3 voxelIndex;
    voxelIndex.z = voxelIndexU / (gridResolution * gridResolution);
    voxelIndex.y = (voxelIndexU - voxelIndex.z * (gridResolution * gridResolution)) / gridResolution;
    voxelIndex.x = voxelIndexU % gridResolution;

    if (gridDebugData[voxelIndexU].hasData == 0) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }
    if (any(voxelIndex < pushConst.showGridVoxelRangeMin) || any(voxelIndex > pushConst.showGridVoxelRangeMax)) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return output;
    }

    AABB voxelAabb;
    voxelAabb.minimum = gridStartPos + voxelIndex * gridVoxelSize;
    voxelAabb.maximum = voxelAabb.minimum + gridVoxelSize;

    float3 aabbSize = voxelAabb.maximum - voxelAabb.minimum;

    /* 24 vertices = 12 edges x 2 endpoints, generated from SV_VertexID: no mesh, no index buffer, and
       no polygonMode = LINE (which this renderer cannot enable). Edge e = v / 2 picks the axis it
       runs along (e / 4) and which of the four parallel edges it is (e % 4). */
    int e        = int(vertexIndex) / 2;
    int endpoint = int(vertexIndex) % 2;
    int axis     = e / 4;  // 0 = x, 1 = y, 2 = z
    int pair     = e % 4;  // corner bits of the two other axes
    int a0       = pair & 1;
    int a1       = (pair >> 1) & 1;

    int bx = 0, by = 0, bz = 0;
    if (axis == 0) { bx = endpoint; by = a0; bz = a1; }
    else if (axis == 1) { by = endpoint; bx = a0; bz = a1; }
    else { bz = endpoint; bx = a0; by = a1; }

    float3 worldPos = voxelAabb.minimum + float3(bx, by, bz) * aabbSize;
    output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);

    return output;
}

[shader("pixel")]
PSout fragmentMain(VSout input) {
    return { input.color };
}
#endif