#define LPV_DEBUG
#include "renderer/LPV/shaders/LPVCommon.slang"

#ifdef LPV_DEBUG
struct VSout {
    float4 sv_position : SV_Position;
    float3 pos;
    float3 normal;
};
struct PSout {
    float4 color0 : SV_Target0;
};
//-------------------------------------------------------------------------------------------------------
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    VSout out;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[0];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];
    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);

    out.pos = posMesh;
    out.normal = normalize(posMesh);
    out.sv_position = mul(float4(posMesh, 1.0f), sceneInfo.viewProjMatrix);
}
[shader("pixel")]
PSout fragmentMain(VSout input) {
    PSout out;

    if (pushConst.showGvSH == 1) {
        float3 pos = input.pos + pushConst.voxelIndex_showSH * pushConst.gridInfo[pushConst.cascadeIndex_showSH].voxelSize + 
            pushConst.gridInfo[pushConst.cascadeIndex_showSH].startPos;

        float4 cBlocking = LPV_SampleGvMerged(pushConst.cascadeIndex_showSH, pos);
        float blockProb = saturate(dot(cBlocking, LPV_SH_Basis(-input.normal)));
        out.color0 = float4(blockProb, blockProb, blockProb, 1.0f);
    } else {
        float4 cR = LPV_LoadVolumeSH(0, pushConst.cascadeIndex_showSH, pushConst.voxelIndex_showSH, 0);
        float4 cG = LPV_LoadVolumeSH(0, pushConst.cascadeIndex_showSH, pushConst.voxelIndex_showSH, 1);
        float4 cB = LPV_LoadVolumeSH(0, pushConst.cascadeIndex_showSH, pushConst.voxelIndex_showSH, 2);

        float4 basisSH = LPV_SH_Basis(input.normal);
        float intensityR = max(0.0, dot(cR, basisSH));
        float intensityG = max(0.0, dot(cG, basisSH));
        float intensityB = max(0.0, dot(cB, basisSH));

        out.color0 = { intensityR, intensityG, intensityB, 1.0f };
        out.color0 = abs(out.color0);
    }

    return out;
}
#endif