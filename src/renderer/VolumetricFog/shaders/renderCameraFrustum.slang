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
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID){
    SceneInfo sceneInfo = pushConst.showCameraInfoAddress[0];
    float3 cameraPos = sceneInfo.cameraPosition;
    float3 cameraRight = sceneInfo.viewInvMatrix[0].xyz;
    float3 cameraUp = sceneInfo.viewInvMatrix[1].xyz;
    float3 cameraForward = -sceneInfo.viewInvMatrix[2].xyz;

    uint3 gridSize = pushConst.frustumGridSize;
    float nearPlane = pushConst.cameraNearPlane;
    float farPlane = pushConst.cameraFarPlane;
    float tanHalfFov = pushConst.tanCameraFov_2;
    float aspect = pushConst.aspectRatio;

    /*
    uint3 voxelIndex;
    voxelIndex.z = instanceIndex / (pushConst.frustumGridSize.x * pushConst.frustumGridSize.y);
    voxelIndex.y = (instanceIndex - voxelIndex.z * (pushConst.frustumGridSize.x * pushConst.frustumGridSize.y)) / pushConst.frustumGridSize.x;
    voxelIndex.x = instanceIndex - voxelIndex.z * (pushConst.frustumGridSize.x * pushConst.frustumGridSize.y) - voxelIndex.y * pushConst.frustumGridSize.x;

    VSout output;
    if (voxelIndex.z > 3) {
        output.sv_position = float4(2.0f, 2.0f, 2.0f, 1.0f);
        return output;
    }

    float u = (float(voxelIndex.x) + 0.5f) / float(gridSize.x) * 2.0f - 1.0f;
    float v = (float(voxelIndex.y) + 0.5f) / float(gridSize.y) * 2.0f - 1.0f;

    float depth_near = nearPlane + frustumMap(float(voxelIndex.z) / gridSize.z * (farPlane - nearPlane), pushConst.compressionParams, farPlane - nearPlane);
    float depth_far = nearPlane + frustumMap(float(voxelIndex.z + 1) / gridSize.z * (farPlane - nearPlane), pushConst.compressionParams, farPlane - nearPlane);

    float depth = (depth_far + depth_near) * 0.5f;
    
    float halfHeight = depth * tanHalfFov;
    float halfWidth = halfHeight * aspect;
    float3 viewPos = float3(u * halfWidth, v * halfHeight, depth);
    float3 voxelCenter = cameraPos + cameraRight * viewPos.x + cameraUp * viewPos.y + cameraForward * viewPos.z;

    float dx = ((vertexIndex & 3) * 2) % 6 == 0 ? -halfWidth / gridSize.x : halfWidth / gridSize.x;
    float dy = (vertexIndex & 3) <= 1 ? -halfHeight / gridSize.y : halfHeight / gridSize.y;
    float dz = vertexIndex / 4 == 0 ? depth_near : depth_far;

    float3 worldPos = cameraPos + cameraRight * (viewPos.x + dx) + cameraUp * (viewPos.y + dy) + cameraForward * dz;

    output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
    return output;
    */

    uint depthIndex = instanceIndex;
    float depth_near = nearPlane + frustumMap(float(depthIndex) / gridSize.z * (farPlane - nearPlane), pushConst.compressionParams, farPlane - nearPlane);
    float depth_far = nearPlane + frustumMap(float(depthIndex + 1) / gridSize.z * (farPlane - nearPlane), pushConst.compressionParams, farPlane - nearPlane);
    float depth = (depth_far + depth_near) * 0.5f;

    float halfHeight = depth * tanHalfFov;
    float halfWidth = halfHeight * aspect;

    float dx = ((vertexIndex & 3) * 2) % 6 == 0 ? -halfWidth : halfWidth;
    float dy = (vertexIndex & 3) <= 1 ? -halfHeight : halfHeight;
    float dz = vertexIndex / 4 == 0 ? depth_near : depth_far;

    float3 worldPos = cameraPos + cameraRight * dx + cameraUp * dy + cameraForward * dz;
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
