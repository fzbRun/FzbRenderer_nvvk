#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<InjectFluidPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfosBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages)]] RWTexture3D<float4> fluidFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages_sample)]] Sampler3D<float4> fluidFogImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfoBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> fluidFogVoxelVelocityImage_sample[];

//----------------------------------------------------------------------------------------------------------------------
struct VSout
{
    float3 worldPos : POSITION;
    //float3 worldNormal : NORMAL;
    float3 velocity;
};

struct GSout {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 velocity;
};

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(0); // Error case
}

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID) {
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = pushConst.meshes_lowPoly[instance.meshIndex];

    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    //float3 normalMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, vertexIndex);
    //float3 normal = mul(normalMesh, pushConst.normalMatrix);

    float4 worldPos_lastFrame = mul(float4(posMesh, 1.0f), pushConst.tansfromMatrix_lastFrame);

    VSout output;
    output.worldPos = pos.xyz;
    //output.worldNormal = normal;
    output.velocity = (pos.xyz - worldPos_lastFrame.xyz) / pushConst.dt;
    return output;
}
[shader("geometry")]
[maxvertexcount(3)]
void geometryMain(
    triangle VSout input[3],
    inout TriangleStream<GSout> outputStream
) {
    //float3 normal0 = input[0].worldNormal;
    //float3 normal1 = input[1].worldNormal;
    //float3 normal2 = input[2].worldNormal;
    //float3 normal = normalize(normal0 + normal1 + normal2);

    float3 p0 = input[0].worldPos;
    float3 p1 = input[1].worldPos;
    float3 p2 = input[2].worldPos;
    float3 tangent = p1 - p0;
    float3 bitangent = p2 - p0;
    float3 normal = normalize(cross(tangent, bitangent));

    float3 absNormal = abs(normal);
    int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

    GSout output;
    for (int i = 0; i < 3; i++) {
        output.sv_position = mul(float4(input[i].worldPos, 1.0f), pushConst.fluidVPMatrixAddress[pushConst.fluidIndex * 3 + viewIndex]);
        output.worldPos = input[i].worldPos;
        output.velocity = input[i].velocity;
        outputStream.Append(output);
    }

    outputStream.RestartStrip();
}
[shader("pixel")]
void fragmentMain(GSout input) {
    VolumetricFogInfo fogInfo = fogInfosBuffer[pushConst.fluidIndex];
    if (!checkPointInAABB(input.worldPos.xyz, fogInfo.aabb)) return;
    FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[fogInfo.typeFogIndex];
    if (fluidFogInfo.startUp == 0) return;

    float3 fogStartPos = fogInfo.aabb.minimum;
    uint3 gridSize = fluidFogInfo.gridSize;
    float3 voxelSize = fluidFogInfo.voxelSize;

    uint3 voxelIndex = min(uint3((input.worldPos.xyz - fogStartPos) / voxelSize), gridSize - 1);
    uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;

    fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].isBoundary = true;
    if (any(abs(input.velocity) > 0)) {
        uint storeValue = 0;
        InterlockedCompareExchange(pushConst.dynamicMeshInjectedAddress[pushConst.fluidIndex][voxelIndexU], 0, 1, storeValue);
        if (storeValue == 0)
            fluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = input.velocity * fluidFogInfo.FIntensity;
    }  
}