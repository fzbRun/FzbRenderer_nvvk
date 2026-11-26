// This file is always compiled as a debug pass: define the switch itself so it also works when the
// renderer does not put LPV_DEBUG on the command line (and to avoid a redefinition warning when it does).
#ifndef LPV_DEBUG
#define LPV_DEBUG
#endif
#include "renderer/LPV/shaders/LPVCommon.slang"

#ifdef LPV_DEBUG

struct VSout {
    float4 sv_position : SV_Position;
    float3 pos;
    float3 normal;
    nointerpolation int3 voxelIndex;
};
struct PSout {
    float4 color0 : SV_Target0;
};
//-------------------------------------------------------------------------------------------------------
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout out;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    // Sphere tessellation: 8 sectors x 4 stacks x 2 triangles x 3 vertices = 192 vertices.
    const int sectors = 8;
    const int stacks  = 4;

    int  tri    = int(vertexIndex) / 3;
    int  corner = int(vertexIndex) % 3;
    int  quad   = tri / 2;
    bool second = (tri & 1) != 0;
    int  i      = quad / sectors;
    int  j      = quad % sectors;

    int2 uv = int2(j, i);
    if (corner == 1) uv += second ? int2(1, 1) : int2(1, 0);
    else if (corner == 2) uv += second ? int2(0, 1) : int2(1, 1);

    float  phi   = (float(uv.x) / float(sectors)) * 6.28318530718;
    float  theta = (float(uv.y) / float(stacks)) * 3.14159265359;
    float3 dir   = normalize(float3(sin(theta) * cos(phi), cos(theta), sin(theta) * sin(phi)));

    int cascade = pushConst.cascadeIndex;
    int n = pushConst.gridInfo[cascade].resolution;

    int3 voxelIndex;
    voxelIndex.z = instanceIndex / (n * n);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (n * n)) / n;
    voxelIndex.x = instanceIndex % n;
    out.voxelIndex = voxelIndex;

    if (any(voxelIndex < pushConst.showGridVoxelRangeMin) || any(voxelIndex > pushConst.showGridVoxelRangeMax)) {
        out.sv_position = float4(2.0, 2.0, 2.0, 1.0);
        return out;
    }

    float voxelSize = LPV_CascadeCellSize(cascade);
    float3 gridStartPos = pushConst.showSHType < 3 ? LPV_GvOrigin(cascade) : LPV_CascadeOrigin(cascade);
    float3 center = gridStartPos + (float3(voxelIndex) + 0.5f) * voxelSize;
    float  radius    = 0.5f * voxelSize;

    out.pos = dir * radius;
    out.normal      = dir;
    out.sv_position = mul(float4(center + out.pos, 1.0f), sceneInfo.viewProjMatrix);

    return out;
}
[shader("pixel")]
PSout fragmentMain(VSout input) {
    PSout out;

    int cascade = pushConst.cascadeIndex;
    int3 voxelIndex = input.voxelIndex;
    if (pushConst.showSHType == 0 || pushConst.showSHType == 1 || pushConst.showSHType == 2) {
        float3 pos = input.pos + (float3(voxelIndex) + 0.5f) * LPV_CascadeCellSize(cascade) + LPV_GvOrigin(cascade);

        float4 cBlocking;
        if (pushConst.showSHType == 2) cBlocking = LPV_LoadGvMerged(cascade, voxelIndex);
        else cBlocking = LPV_LoadGvAny(pushConst.gvSource, cascade, voxelIndex);
        float blockProb = saturate(dot(cBlocking, LPV_SH_Basis(input.normal)));
        out.color0 = blockProb <= 0.001f ? float4(10.0f, 0.0f, 0.0f, 1.0f) : float4(blockProb, blockProb, blockProb, 1.0f);
    } else if (pushConst.showSHType == 3) {
        int sourceIndex = pushConst.iterationCount & 1;

        float4 cR = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 0);
        float4 cG = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 1);
        float4 cB = LPV_LoadVolumeSH(sourceIndex, cascade, voxelIndex, 2);

        float4 basisSH = LPV_SH_Basis(input.normal);
        float intensityR = max(0.0, dot(cR, basisSH));
        float intensityG = max(0.0, dot(cG, basisSH));
        float intensityB = max(0.0, dot(cB, basisSH));

        out.color0 = { intensityR, intensityG, intensityB, 1.0f };
        out.color0 = abs(out.color0);
        out.color0 = all(out.color0.xyz <= 0.001f) ? float4(10.0f, 0.0f, 0.0f, 1.0f) : out.color0;
    } else if (pushConst.showSHType == 4) {
        float4 cR = LPV_LoadAccumSH(cascade, voxelIndex, 0);
        float4 cG = LPV_LoadAccumSH(cascade, voxelIndex, 1);
        float4 cB = LPV_LoadAccumSH(cascade, voxelIndex, 2);

        float4 basisSH = LPV_SH_Basis(input.normal);
        float intensityR = max(0.0, dot(cR, basisSH));
        float intensityG = max(0.0, dot(cG, basisSH));
        float intensityB = max(0.0, dot(cB, basisSH));

        out.color0 = { intensityR, intensityG, intensityB, 1.0f };
        out.color0 = abs(out.color0);
        out.color0 = all(out.color0.xyz == 0.0f) ? float4(10.0f, 0.0f, 0.0f, 1.0f) : out.color0;
    }

    return out;
}
#endif