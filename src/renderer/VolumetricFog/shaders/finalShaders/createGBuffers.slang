#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<CreateGBuffersPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eTextures)]] Sampler2D textures[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogInfosBuffer)]] RWStructuredBuffer<VolumetricFogInfo, ScalarDataLayout> fogInfosBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogInfosBuffer)]] RWStructuredBuffer<FluidFogInfo, ScalarDataLayout> fluidFogInfoBuffer;
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages)]] RWTexture3D<float4> fluidFogImages[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogImages_sample)]] Sampler3D<float4> fluidFogImages_sample[];

[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelInfoBuffer)]] RWStructuredBuffer<FluidFogVoxelInfo, ScalarDataLayout> fluidFogVoxelInfoBuffer[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage)]] RWTexture3D<float4> fluidFogVoxelVelocityImage[];
[[vk::binding(StaticBindingPoints_VolumetricFog::eFluidFogVoxelVelocityImage_sample)]] Sampler3D<float4> fluidFogVoxelVelocityImage_sample[];

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

    float3 albedo;
    float3 emissive;
    int meshIndex;
    int materialIndex;

#ifdef FLUID_GBUFFER_INJECT
    MaterialType materialType;
    nointerpolation float3 velocity;
#endif

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

    float4 worldPos_lastFrame = mul(float4(posMesh, 1.0f), pushConst.tansfromMatrix_lastFrame);
    float4 clipPos_Last = mul(worldPos_lastFrame, pushConst.vpMatrix_lastFrame);
    output.screenPos_Last = clipPos_Last.xy / clipPos_Last.w * 0.5f + 0.5f;

    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];
    output.albedo = material.albedo;
    output.emissive = material.emissive;
    output.meshIndex = instance.meshIndex;
    output.materialIndex = instance.materialIndex;

#ifdef FLUID_GBUFFER_INJECT
    output.velocity = (pos.xyz - worldPos_lastFrame.xyz) / pushConst.dt;
    output.materialType = material.type;
#endif

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
PSout fragmentMain(VSout input) {
#ifdef FLUID_GBUFFER_INJECT
    for (int i = 0; i < pushConst.fluidFogCount; ++i) {
        VolumetricFogInfo fogInfo = fogInfosBuffer[i];
        if (!checkPointInAABB(input.worldPos.xyz, fogInfo.aabb)) continue;
        FluidFogInfo fluidFogInfo = fluidFogInfoBuffer[fogInfo.typeFogIndex];
        if (fluidFogInfo.startUp == 0) continue;
    
        float3 fogStartPos = fogInfo.aabb.minimum;
        uint3 gridSize = fluidFogInfo.gridSize;
        float3 voxelSize = fluidFogInfo.voxelSize;
    
        uint3 voxelIndex = min(uint3((input.worldPos.xyz - fogStartPos) / voxelSize), gridSize - 1);
        uint voxelIndexU = voxelIndex.z * gridSize.x * gridSize.y + voxelIndex.y * gridSize.x + voxelIndex.x;
    
        fluidFogVoxelInfoBuffer[fogInfo.typeFogIndex][voxelIndexU].isBoundary = true;
        if (any(abs(input.velocity) > 0))
            fluidFogVoxelVelocityImage[fogInfo.typeFogIndex][voxelIndex].xyz = input.velocity * fluidFogInfo.FIntensity;
    }
    if (input.materialType == MaterialType::RoughDielectric) discard;
 #endif

    PSout out;
    out.albedo = float4(input.albedo, 1.0f);
    out.normal = float4(normalize(input.worldNormal), 1.0f);
    out.emissive = float4(input.emissive, 1.0f);
    out.velocity = float4(input.screenPos_Current - input.screenPos_Last, 0.0f, 1.0f);
    out.vertexInfo = float4(pushConst.instanceIndex, input.meshIndex, input.materialIndex, 1.0f);
    return out;
}
