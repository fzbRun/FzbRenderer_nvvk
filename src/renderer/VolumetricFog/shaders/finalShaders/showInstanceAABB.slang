#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<renderInstanceAABBPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

struct VSout {
    float4 sv_position : SV_Position;
};

uint getFrustumVoxelIndex(float viewZ, float3 CompressParams, float sizeZ, uint gridZ){
    float increasingD = (sizeZ - gridZ * CompressParams.x) / (gridZ * 0.5f * (1.0f + gridZ));
    float n = sqrt(2.0f * viewZ / increasingD + (0.5f + CompressParams.x / increasingD) * (0.5f + CompressParams.x / increasingD)) - (0.5f + CompressParams.x / increasingD);
    return int(n);
}

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    InstanceInfo instanceInfo = pushConst.instanceInfoAddress[pushConst.instanceIndex];
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[instanceInfo.instanceIndex];
    float4x4 transformMatrix = instance.transform;

    float3 worldPos;
    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? instanceInfo.aabb_local.minimum.x : instanceInfo.aabb_local.maximum.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? instanceInfo.aabb_local.minimum.y : instanceInfo.aabb_local.maximum.y;
    worldPos.z = vertexIndex / 4 == 0 ? instanceInfo.aabb_local.minimum.z : instanceInfo.aabb_local.maximum.z;
    worldPos = mul(float4(worldPos, 1.0f), transformMatrix).xyz;

    VSout output;
    output.sv_position = mul(float4(worldPos, 1.0f), sceneInfo.viewProjMatrix);
    return output;
}

struct PSout
{
    float4 color : SV_Target;
};
[shader("pixel")]
PSout fragmentMain(VSout input, float4 position: SV_Position) {
    return { float4(1.0f, 0.0f, 0.0f, 1.0f) };
}
