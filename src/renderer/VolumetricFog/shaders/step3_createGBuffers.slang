#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eTextures)]] Sampler2D textures[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eGlobalInfoBuffer)]] RWStructuredBuffer<GlobalInfo_VolumetricFog> globalInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfosBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[]; // xyz: velocity; w:

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex){
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(1); // Error case
}

struct VSin{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSout{
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
    float2 worldTexCoord : TEXCOORD0;

    float2 screenPos_Current;
    float2 screenPos_Last;
};

static float2 Halton_2_3[8] = {
    float2(0.0f, -1.0f / 3.0f),
    float2(-1.0f / 2.0f, 1.0f / 3.0f),
    float2(1.0f / 2.0f, -7.0f / 9.0f),
    float2(-3.0f / 4.0f, -1.0f / 9.0f),
    float2(1.0f / 4.0f, 5.0f / 9.0f),
    float2(-1.0f / 4.0f, -5.0f / 9.0f),
    float2(3.0f / 4.0f, 1.0f / 9.0f),
    float2(-7.0f / 8.0f, 7.0f / 9.0f)
};

[shader("vertex")]
VSout vertexMain(VSin input, uint vertexIndex: SV_VertexID){
    int instanceIndex = pushConst.instanceIndex;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh meshIo = sceneInfo.meshes[instance.meshIndex];

    // Retrieve the data
    float3 posMesh = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.positions, vertexIndex);
    float3 normal = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.normals, vertexIndex);
    float2 texCoord = getAttribute<float2>(meshIo.dataBuffer, meshIo.triMesh.texCoords, vertexIndex);

    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    float4x4 jitterMat = sceneInfo.projMatrix;
    float2 jitter = float2(Halton_2_3[pushConst.iteration % 8].x * 1.0f / pushConst.screenSize.x, Halton_2_3[pushConst.iteration % 8].y * 1.0f / pushConst.screenSize.y);
    jitterMat[2][0] += jitter.x;
    jitterMat[2][1] += jitter.y;

    VSout output;
    output.sv_position = mul(pos, sceneInfo.viewProjMatrix);
    output.worldPos = pos.xyz;
    output.worldNormal = normalize(mul(normal, float3x3(pushConst.normalMatrix)));
    output.worldTexCoord = texCoord;

    output.screenPos_Current = output.sv_position.xy / output.sv_position.w * 0.5f + 0.5f;
    float4 clipPos_Last = mul(float4(posMesh, 1.0), pushConst.lightVP * globalInfoBuffer[0].VPMatrix_lastFrame);
    output.screenPos_Last = clipPos_Last.xy / clipPos_Last.w * 0.5f + 0.5f;

    return output;
}

struct PSout{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 emissive : SV_Target2;
    float4 velocity : SV_Target3;
    float4 vertexInfo : SV_Target4;
};

[shader("pixel")]
PSout fragmentMain(VSout input){
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    for (int i = 0; i < pushConst.volumetricFogCount; ++i) {
        VolumetricFogInfo fogInfo = fogInfosBuffer[i];
        if (fogInfo.type != VolumetricFogType::Fluid) continue;
        if (!checkPointInAABB(input.worldPos.xyz, fogInfo.aabb)) continue;
        FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[fogInfo.typeFogIndex];
        if (fluidFogInfo.startUp == 0) continue;

        float3 fogStartPos = fogInfo.aabb.minimum;
        uint3 gridSize = fluidFogInfo.gridSize;
        float3 voxelSize = fluidFogInfo.voxelSize;

        uint3 voxelIndex = min(uint3((input.worldPos.xyz - fogStartPos) / voxelSize), gridSize - 1);
        uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;

        fluidFogVoxelInfosBuffer[fogInfo.typeFogIndex][voxelIndexU].isBoundary = true;
        fluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = pushConst.instanceVelocity * fluidFogInfo.FIntensity;
    }
    if (material.type == MaterialType::RoughDielectric) discard;

    PSout out;
    out.albedo = float4(material.albedo, 1.0f);
    out.normal = float4(normalize(input.worldNormal), 1.0f);
    out.emissive = float4(material.emissive, 1.0f);
    out.velocity = float4(input.screenPos_Current - input.screenPos_Last, 0.0f, 1.0f);
    out.vertexInfo = float4(pushConst.instanceIndex, instance.meshIndex, instance.materialIndex, 1.0f);
    return out;
}
