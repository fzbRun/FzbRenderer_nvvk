#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "feature/ShadowMap/ShadowMapShaderio.h"

[[vk::push_constant]] ConstantBuffer<ShadowMapPushConstant> pushConst;

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(0); // Error case
}
//----------------------------------------------DirectionLight-------------------------------------------
struct VSout
{
    float4 sv_position : SV_Position;
    float3 worldPos;
    float3 albedo;
};

[shader("vertex")]
VSout vertexMain_directionLight(uint vertexIndex: SV_VertexID)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    // Retrieve the data
    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);
    float4 worldPos = mul(float4(posMesh, 1.0), instance.transform);

    VSout output;
    output.sv_position = mul(worldPos, pushConst.lightVP);
    #ifndef NDEBUG
    output.worldPos = worldPos.xyz;
    output.albedo = sceneInfo.materials[instance.materialIndex].albedo;
    #endif
    return output;
}
#ifndef NDEBUG
struct PSout
{
    float4 color0 : SV_Target0;
};
[shader("pixel")]
PSout fragmentMain_directionLight(VSout input) {
    float3 edge0 = ddx(input.worldPos);
    float3 edge1 = ddy(input.worldPos);
    float3 normal = normalize(cross(edge0, edge1));

    Light light = pushConst.sceneInfoAddress.lights[pushConst.lightIndex];
    float3 ambient = 0.1f * input.albedo;

    float3 o = normalize(pushConst.sceneInfoAddress.cameraPosition - input.worldPos);
    float3 i = -light.direction;
    float3 h = normalize(o + i);
    float3 diffuse = input.albedo * max(dot(normal, h), 0.0f);

    float3 reflectDir = reflect(i, normal);
    float3 specular = 0.0f; //pow(max(dot(o, reflectDir), 0.0), 8.0);

    return { float4(ambient + diffuse + specular, 1.0f) };
}
#else
[shader("pixel")]
void fragmentMain_directionLight() {}
#endif
//----------------------------------------------Debug-------------------------------------------
#ifndef NDEBUG
[[vk::binding(BindingPoints_ShadowMap::eShadowMaps, 0)]] Sampler2D<float> shadowMaps[];
[[vk::binding(BindingPoints_ShadowMap::eDepthRestructResultMaps, 0)]] RWTexture2D<float4> depthRestructResultMaps[];

[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_debug(uint2 threadIndex: SV_DispatchThreadID, uint2 groupThreadIndex: SV_GroupThreadID) {
    uint2 shadowMapSize;
    shadowMaps[pushConst.lightIndex].GetDimensions(shadowMapSize.x, shadowMapSize.y);
    if (threadIndex.x >= shadowMapSize.x || threadIndex.y >= shadowMapSize.y) return;

    float depth = shadowMaps[pushConst.lightIndex][threadIndex];
    float2 uv = (float2(threadIndex) + 0.5f) / float2(shadowMapSize);
    float2 ndc = uv * 2.0f - 1.0f;
    float4 clipPos = float4(ndc, depth, 1.0f);
    float4 worldPos = mul(clipPos, pushConst.inverse_lightVP);
    worldPos /= worldPos.w;

    depthRestructResultMaps[0][int2(threadIndex)] = float4(normalize(worldPos.xyz), 1.0f);   // ,
}
#endif