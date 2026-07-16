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

[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuationImage_sample)]] Sampler3D<float4> volumetricFogAttenuationImage_sample;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogAttenuation2Image_sample)]] Sampler3D<float4> volumetricFogAttenuation2Image_sample;
[[vk::binding(StaticBindingPoints_VolumetricFog::eVolumetricFogLImage_sample)]] Sampler3D<float4> volumetricFogLImage_sample;

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
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
    output.sv_position = mul(pos, sceneInfo.viewProjMatrix);
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

    float3 hitWorldPos = sceneInfo.cameraPosition; float3 uvw;
    float3 attenuation = 1.0f, attenuation2 = 1.0f, L = 0.0f;
    if (getCameraFrustumFogData(worldPos.xyz, hitWorldPos, uvw, {0, 0})) {
        attenuation = volumetricFogAttenuationImage_sample.SampleLevel(uvw, 0).xyz;
        attenuation2 = volumetricFogAttenuation2Image_sample.SampleLevel(uvw, 0).xyz;
        L = volumetricFogLImage_sample.SampleLevel(uvw, 0).xyz;
    }

    if (length(worldPos.xyz - hitWorldPos) > 1e-5f) {
        uint randomSeed = xxhash32(uint3(pushConst.frameIndex, pushConst.instanceIndex, instance.meshIndex));
        getVolumetricFogScatter(hitWorldPos, worldPos.xyz, randomSeed, 10, L, attenuation, attenuation2);
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

    return { float4((ambient + (diffuse + specular) * shadow * attenuation2 + emissive) * attenuation + L, 1.0f) };
}
