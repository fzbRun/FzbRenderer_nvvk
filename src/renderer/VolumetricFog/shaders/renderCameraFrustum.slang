#include "renderer/VolumetricFog/VolumetricFogShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<VolumetricFogPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_VolumetricFog::eRenderedImage)]] RWTexture2D<float4> renderedImage;

struct VSout {
    float4 sv_position : SV_Position;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    SceneInfo sceneInfo = pushConst.showCameraInfoAddress[0];
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint3 gridSize = pushConst.attenuationGridSize;
    float nearPlane = pushConst.attenuationNearPlane;
    float farPlane = pushConst.attenuationFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y)) / pushConst.attenuationGridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (pushConst.attenuationGridSize.x * pushConst.attenuationGridSize.y) - voxelIndex.y * pushConst.attenuationGridSize.x;
    float u = (float(voxelIndex.x) + 0.5f) / float(gridSize.x) * 2.0f - 1.0f;
    float v = (float(voxelIndex.y) + 0.5f) / float(gridSize.y) * 2.0f - 1.0f;
    //float t = (float(voxelIndex.z) + 0.5f) / float(gridSize.z);
    //t = pow(t, pushConst.compressionPrecision);
    //float depth = lerp(nearPlane, farPlane, t);

    float t_last = float(voxelIndex.z) / float(gridSize.z);
    t_last = pow(t_last, pushConst.compressionPrecision);
    float depth_last = lerp(nearPlane, farPlane, t_last);

    float t_next = float(voxelIndex.z + 1) / float(gridSize.z);
    t_next = pow(t_next, pushConst.compressionPrecision);
    float depth_next = lerp(nearPlane, farPlane, t_next);

    float halfDepth = (depth_next - depth_last) * 0.5f;
    float depth = depth_last + halfDepth;

    float halfHeight = depth * tanHalfFov;
    float halfWidth = halfHeight * aspect;
    float3 viewPos = float3(u * halfWidth, v * halfHeight, depth);
    float3 voxelCenter = cameraPos + cameraRight * viewPos.x + cameraUp * viewPos.y + cameraForward * viewPos.z;

    float3 worldPos;
    worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? -halfWidth / gridSize.x : halfWidth / gridSize.x;
    worldPos.y = (vertexIndex & 3) <= 1 ? -halfHeight / gridSize.y : halfHeight / gridSize.y;
    worldPos.z = vertexIndex / 4 == 0 ? -halfDepth : halfDepth;
    worldPos += voxelCenter;

    VSout output;
    output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
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