#include "renderer/StochasticLightcuts/StochasticLightcutsShaderio.h"

[[vk::push_constant]] ConstantBuffer<CreateGBufferPushConstant_StochasticLightcuts, ScalarDataLayout> pushConst;

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

    float3 normal;
    nointerpolation int materialIndex;
};

[shader("vertex")]
VSout vertexMain(VSin input, uint vertexIndex: SV_VertexID){
    int instanceIndex = pushConst.instanceIndex;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    Instance instance = sceneInfo.instances[instanceIndex];
    Mesh meshIo = sceneInfo.meshes[instance.meshIndex];

    // Retrieve the data
    float3 posMesh = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.positions, vertexIndex);
    float3 normalMesh = getAttribute<float3>(meshIo.dataBuffer, meshIo.triMesh.normals, vertexIndex);
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);
    float4x4 projMatrix = sceneInfo.projMatrix;

    VSout output;
    output.sv_position = mul(mul(pos, sceneInfo.viewMatrix), projMatrix);
    output.normal = normalize(mul(normalMesh, float3x3(pushConst.normalMatrix)));
    output.materialIndex = instance.materialIndex;

    return output;
}

struct PSout{
    float4 normal_materialIndex : SV_Target0;
};

[shader("pixel")]
PSout fragmentMain(VSout input) {
    PSout out;
    out.normal_materialIndex = float4(input.normal, input.materialIndex);
    return out;
}