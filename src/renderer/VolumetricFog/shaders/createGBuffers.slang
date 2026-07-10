#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eTextures)]] Sampler2D textures[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> volumetricFogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidInfoBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> volumetricFogFluidInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelInfoBuffer)]] RWStructuredBuffer<VolumetricFogFluidVoxelInfo, ScalarDataLayout> volumetricFogVoxelInfosBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogFluidVoxelVelocityImage)]] RWTexture3D<float4> volumetricFogVoxelVelocityImage[]; // xyz: velocity; w:

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(1); // Error case
}

struct VSin
{
    float3 position : POSITION;
    float3 normal : NORMAL;
};

struct VSout
{
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
    float2 worldTexCoord : TEXCOORD0;
};

[shader("vertex")]
VSout vertexMain(VSin input, uint vertexIndex: SV_VertexID)
{
    int instanceIndex = pushConst.instanceIndex;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh meshIo = sceneInfo.meshes[instance.meshIndex];

    // Retrieve the data
    float3 posMesh = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.positions, vertexIndex);
    float3 normal = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.normals, vertexIndex);
    float2 texCoord = getAttribute<float2>(meshIo.dataBuffer, meshIo.triMesh.texCoords, vertexIndex);

    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    VSout output;
    output.sv_position = mul(pos, sceneInfo.viewProjMatrix);
    output.worldPos = pos.xyz;
    output.worldNormal = normalize(mul(normal, float3x3(pushConst.normalMatrix)));
    output.worldTexCoord = texCoord;

    return output;
}

struct PSout
{
    float4 albedo : SV_Target0;
    float4 normal : SV_Target1;
    float4 emissive : SV_Target2;
};

[shader("pixel")]
PSout fragmentMain(VSout input){
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    //if (pushConst.volumetricFogFluidIndex >= 0) {
        for (int i = 0; i < pushConst.volumetricFogCount; ++i) {
            VolumetricFogInfo fogInfo = volumetricFogInfosBuffer[i];
            if (fogInfo.type != VolumetricFogType::Fluid) continue;
            AABB fogAABB = { fogInfo.fogStartPos, fogInfo.fogStartPos + fogInfo.fogVoxelGridSize * fogInfo.fogVoxelSize };
            if (!checkPointInAABB(input.worldPos.xyz, fogAABB)) continue;
            FluidFogInfo fluidFogInfo = volumetricFogFluidInfoBuffer[fogInfo.volumetricFogTypeIndex];
            if (fluidFogInfo.startUp == 0) continue;

            uint3 voxelIndex = min(uint3((input.worldPos.xyz - fogInfo.fogStartPos) / fogInfo.fogVoxelSize), fogInfo.fogVoxelGridSize - 1);
            uint voxelIndexU = voxelIndex.z * fogInfo.fogVoxelGridSize.x * fogInfo.fogVoxelGridSize.y + voxelIndex.y * fogInfo.fogVoxelGridSize.x + voxelIndex.x;
            volumetricFogVoxelInfosBuffer[fogInfo.volumetricFogTypeIndex][voxelIndexU].isBoundary = true;
            volumetricFogVoxelVelocityImage[fogInfo.volumetricFogTypeIndex][voxelIndex].xyz = pushConst.instanceVelocity;
        }
    //}

    return { float4(material.albedo, 1.0f), float4(normalize(input.worldNormal), 1.0f), float4(material.emissive, 1.0f) };
    //return { float4(material.albedo, 1.0f), float4(normalize(input.worldNormal), 1.0f), float4(0.0f, 0.0f, 0.0f, 1.0f) };
}
