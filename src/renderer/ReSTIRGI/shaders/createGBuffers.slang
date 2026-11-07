#include "renderer/ReSTIRGI/ReSTIR_GIShaderio.h"

[[vk::push_constant]] ConstantBuffer<CreateGBufferPushConstant_ReSTIR_GI, ScalarDataLayout> pushConst;

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
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);
    float4x4 projMatrix = sceneInfo.projMatrix;

    VSout output;
    output.sv_position = mul(mul(pos, sceneInfo.viewMatrix), projMatrix);

    float4 clipPos_Cur = mul(pos, sceneInfo.viewProjMatrix);
    output.screenPos_Current = clipPos_Cur.xy / clipPos_Cur.w * 0.5f + 0.5f;

    float4 worldPos_lastFrame = mul(float4(posMesh, 1.0f), pushConst.tansfromMatrix_lastFrame);
    float4 clipPos_Last = mul(worldPos_lastFrame, pushConst.vpMatrix_lastFrame);
    output.screenPos_Last = clipPos_Last.xy / clipPos_Last.w * 0.5f + 0.5f;

    return output;
}

struct PSout{
    float4 velocity : SV_Target0;
};

[shader("pixel")]
PSout fragmentMain(VSout input) {
    PSout out;
    out.velocity = float4(input.screenPos_Current - input.screenPos_Last, 0.0f, 1.0f);
    return out;
}
