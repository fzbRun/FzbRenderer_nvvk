#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

#include "renderer/VolumetricFog/shaders/getVolumetricFog.slang"

[[vk::binding(StaticBindingPoints_VolumetricFog::eAlbedoImage)]] Texture2D<float4> albedoImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eNormalImage)]] Texture2D<float4> normalImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eDepthImage)]] Texture2D<float> depthImage;
[[vk::binding(StaticBindingPoints_VolumetricFog::eEmissiveImage)]] Texture2D<float4> emissiveImage;

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
    float3 posMesh;
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

    VSout output;
    output.sv_position = mul(mul(pos, sceneInfo.viewMatrix), globalInfoBuffer[0].projMatrix_taaJitter);
    output.worldPos = pos.xyz;
    output.worldNormal = normalize(mul(normal, float3x3(pushConst.normalMatrix)));
    output.worldTexCoord = texCoord;
    output.posMesh = posMesh;

    return output;
}

struct PSout{
    float4 result : SV_Target;
};

[shader("pixel")]
PSout fragmentMain(VSout input) {
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];

    float3 worldPos = input.worldPos;

    float attenuation = 1.0f, attenuation2 = 1.0f, L = 0.0f;
    // 逐像素的种子（原来只按实例/网格变化，同一个物体的所有像素共用一个抖动，等于没抖动）
    uint2 pixelPos = uint2(input.sv_position.xy);
    uint randomSeed = xxhash32(uint3(globalInfoBuffer[0].temporalFrameIndex, pixelPos));
    uint sampleCount = pushConst.useAccFog == 1 ? 0 : 10;
    if (pushConst.useAccFog == 1) getPreAccFog(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, input.sv_position.xy, L, attenuation, attenuation2);
    else {
        // 同step8：非fogAcc路径后面没有体素时域滤波，抖动必须只在空间上（与帧号无关），
        // 否则所有像素每帧同步换偏移，表现为整片雾的同步抖动。
        float spatialDither = getCellDither(float2(pixelPos), 0.0f);
        #ifdef USE_ENVFOG
        getEnvVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, sampleCount, spatialDither, false, L, attenuation, attenuation2);
        #else
        getVolumetricFogScatter(sceneInfo.cameraPosition, worldPos.xyz, randomSeed, sampleCount, 3, true, spatialDither, false, L, attenuation, attenuation2);
        #endif
    }

    float shadow = getShadow(float4(worldPos, 1.0f));

    float3 albedo = material.albedo;
    float3 normal = input.worldNormal;

    Light light = sceneInfo.lights[0];

    float3 viewDir = normalize(sceneInfo.cameraPosition - worldPos.xyz);
    float3 lightDir = normalize(-light.direction);

    float3 lightColor = light.color * light.intensity;

    float3 ambient = 0.1f * albedo;
    float3 diffuse = max(dot(lightDir, normal), 0.0f) * lightColor * albedo;
    float3 h = normalize(viewDir + lightDir);
    float3 specular = pow(max(dot(h, normal), 0.0f), 32) * lightColor * max(dot(normal, lightDir), 0.0f);

    float3 emissive = material.emissive;

    return { float4((ambient + (diffuse + specular) * shadow * attenuation2 + emissive) * attenuation + L, 0.5f) };
}
