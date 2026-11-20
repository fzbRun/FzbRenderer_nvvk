#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/VRPG2/shaders/VRPGCommon.slang"
#include "renderer/VRPG2/GeometryInjection/GeometryInjectionShaderio.h"

[[vk::push_constant]] ConstantBuffer<GeometryInjectionPushConstant_VRPG> pushConst;
[[vk::binding(GeometryInjectionBinding_VRPG::eTextures)]] Sampler2D textures[];
[[vk::binding(GeometryInjectionBinding_VRPG::eVGB)]] RWStructuredBuffer<VoxelData_VRPG, ScalarDataLayout> VGBs[];
//-------------------------------------------------------------------------------------------------
void atomicAABB(float3 aabbMin, float3 aabbMax, int voxelIndexU, int VGBIndex) {
    int minX = FloatToOrderedInt(aabbMin.x);
    int minY = FloatToOrderedInt(aabbMin.y);
    int minZ = FloatToOrderedInt(aabbMin.z);
    int maxX = FloatToOrderedInt(aabbMax.x);
    int maxY = FloatToOrderedInt(aabbMax.y);
    int maxZ = FloatToOrderedInt(aabbMax.z);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.x, minX);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.y, minY);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.z, minZ);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.x, maxX);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.y, maxY);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.z, maxZ);
}
void atomicNormalG(float3 normal, int voxelIndexU, int VGBIndex) {
    float3 normal_G = normalize(normal);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.w, 1.0f);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.x, normal_G.x);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.y, normal_G.y);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.z, normal_G.z);
}
//-------------------------------------------------------------------------------------------------
struct VSout {
    float3 worldPos : POSITION;
    float3 emissive;
};

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex) {
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(0); // Error case
}

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    VSout output;
    output.worldPos = pos.xyz;
    output.emissive = sceneInfo.materials[instance.materialIndex].emissive;

    return output;
}
//-------------------------------------------------------------------------------------------------
struct GSout {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
};

[shader("geometry")]
[maxvertexcount(3)]
void geometryMain(
    triangle VSout input[3],
    inout TriangleStream<GSout> outputStream
) {
    float3 e0 = input[1].worldPos - input[0].worldPos;
    float3 e1 = input[2].worldPos - input[0].worldPos;

    float3 normal = normalize(cross(e0, e1));
    float3 absNormal = abs(normal);
    int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

    GSout output;
    for (int i = 0; i < 3; i++) {
        output.sv_position = mul(float4(input[i].worldPos, 1.0f), pushConst.VP[viewIndex]);
        output.worldPos = input[i].worldPos;
        output.worldNormal = normal;
        outputStream.Append(output);
    }

    outputStream.RestartStrip();
}

[shader("pixel")]
void fragmentMain(GSout input) {
    int voxelIndexU = getVoxelIndex(input.worldPos, pushConst.voxelGroupStartPos, pushConst.voxelSize_Count.xyz, (int)pushConst.voxelSize_Count.w);
    int VGBIndex = getNormalIndex(input.worldNormal);

    float3 halfPixelSizeX = abs(ddx(input.worldPos)) * 0.5f;
    float3 halfPixelSizeY = abs(ddy(input.worldPos)) * 0.5f;
    float3 maximum = input.worldPos + halfPixelSizeX + halfPixelSizeY;
    float3 minimum = input.worldPos - halfPixelSizeX - halfPixelSizeY;

    int3 voxelIndex = int3((input.worldPos - pushConst.voxelGroupStartPos) / pushConst.voxelSize_Count.xyz);
    AABB voxelRange;
    voxelRange.minimum = voxelIndex * pushConst.voxelSize_Count.xyz + pushConst.voxelGroupStartPos;
    voxelRange.maximum = voxelRange.minimum + pushConst.voxelSize_Count.xyz;

    // One pixel-level AABB update instead of up to three 6-atomic updates.
    float3 aabbMin = max(minimum, voxelRange.minimum);
    float3 aabbMax = min(maximum, voxelRange.maximum);
    atomicAABB(aabbMin, aabbMax, voxelIndexU, VGBIndex);

    atomicNormalG(input.worldNormal, voxelIndexU, VGBIndex);
}
//--------------------------------------------threeView---------------------------------------
struct GSout_ThreeView {
    float4 sv_position : SV_Position;
    nointerpolation uint viewportIndex : SV_ViewportArrayIndex;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
    float3 emissive;
};
struct PSout_ThreeView{
    float4 color : SV_Target;
};

[shader("geometry")]
[maxvertexcount(3)]
void geometryMain_ThreeView(
    triangle VSout input[3],
    inout TriangleStream<GSout_ThreeView> outputStream
) {
    float3 e0 = input[1].worldPos - input[0].worldPos;
    float3 e1 = input[2].worldPos - input[0].worldPos;

    float3 normal = normalize(cross(e0, e1));
    float3 absNormal = abs(normal);
    int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

    GSout_ThreeView output;
    output.viewportIndex = viewIndex;
    for (int i = 0; i < 3; i++) {
        output.worldPos = input[i].worldPos;
        output.sv_position = mul(float4(output.worldPos, 1.0f), pushConst.VP[viewIndex]);
        output.worldNormal = normal;
        output.emissive = input[i].emissive;
        outputStream.Append(output);
    }
    outputStream.RestartStrip();
}

[shader("pixel")]
PSout_ThreeView fragmentMain_ThreeView(GSout_ThreeView input) {
    int voxelIndexU = getVoxelIndex(input.worldPos, pushConst.voxelGroupStartPos, pushConst.voxelSize_Count.xyz, (int)pushConst.voxelSize_Count.w);
    int VGBIndex = getNormalIndex(input.worldNormal);

    float3 halfPixelSizeX = abs(ddx(input.worldPos)) * 0.5f;
    float3 halfPixelSizeY = abs(ddy(input.worldPos)) * 0.5f;
    float3 maximum = input.worldPos + halfPixelSizeX + halfPixelSizeY;
    float3 minimum = input.worldPos - halfPixelSizeX - halfPixelSizeY;

    int3 voxelIndex = int3((input.worldPos - pushConst.voxelGroupStartPos) / pushConst.voxelSize_Count.xyz);
    AABB voxelRange;
    voxelRange.minimum = voxelIndex * pushConst.voxelSize_Count.xyz + pushConst.voxelGroupStartPos;
    voxelRange.maximum = voxelRange.minimum + pushConst.voxelSize_Count.xyz;

    // One pixel-level AABB update instead of up to three 6-atomic updates.
    float3 aabbMin = max(minimum, voxelRange.minimum);
    float3 aabbMax = min(maximum, voxelRange.maximum);
    atomicAABB(aabbMin, aabbMax, voxelIndexU, VGBIndex);

    atomicNormalG(input.worldNormal, voxelIndexU, VGBIndex);

    PSout_ThreeView output;
    output.color = float4(abs(input.worldNormal), 1.0f);
    return output;
}