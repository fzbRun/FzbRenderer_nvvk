#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<CreateGBuffersPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eTextures)]] Sampler2D textures[];

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

    float4x4 projMatrix = sceneInfo.projMatrix;
    if (pushConst.useTAA == 1) projMatrix = pushConst.projMatrix_taa;

    VSout output;
    output.sv_position = mul(mul(pos, sceneInfo.viewMatrix), projMatrix);
    output.worldPos = pos.xyz;
    output.worldNormal = normalize(mul(normal, float3x3(pushConst.normalMatrix)));
    output.worldTexCoord = texCoord;

    float4 clipPos_Cur = mul(pos, sceneInfo.viewProjMatrix);
    output.screenPos_Current = clipPos_Cur.xy / clipPos_Cur.w * 0.5f + 0.5f;
    float4 clipPos_Last = mul(mul(float4(posMesh, 1.0), pushConst.tansfromMatrix_lastFrame), pushConst.vpMatrix_lastFrame);
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
    if (material.type == MaterialType::RoughDielectric) discard;

    PSout out;
    out.albedo = float4(material.albedo, 1.0f);
    out.normal = float4(normalize(input.worldNormal), 1.0f);
    out.emissive = float4(material.emissive, 1.0f);
    out.velocity = float4(input.screenPos_Current - input.screenPos_Last, 0.0f, 1.0f);
    out.vertexInfo = float4(pushConst.instanceIndex, instance.meshIndex, instance.materialIndex, 1.0f);
    return out;
}
