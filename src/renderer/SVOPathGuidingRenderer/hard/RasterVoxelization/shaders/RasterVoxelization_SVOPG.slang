#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/SVOPathGuidingRenderer/hard/shaders/SVOPGCommon.slang"
#include "renderer/SVOPathGuidingRenderer/hard/RasterVoxelization/shaderio.h"

[[vk::push_constant]] ConstantBuffer<RasterVoxelizationPushConstant> pushConst;
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eTextures_SVOPG)]] Sampler2D textures[];
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eVGB_SVOPG)]] RWStructuredBuffer<VGBVoxelData_SVOPG, ScalarDataLayout> VGBs[]; // globallycoherent volatile
#ifdef CLUSTER_WITH_MATERIAL
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eVGBMaterialInfo_SVOPG)]] RWStructuredBuffer<VGBMaterialInfo_SVOPG> VGBMaterialInfos[];
#endif
#ifndef NDEBUG
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eFragmentCountBuffer_SVOPG)]] RWStructuredBuffer<uint32_t> fragmentCountBuffer;
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eWireframeMap_SVOPG)]] RWTexture2D<float4> wireframeMap;
[[vk::binding(RasterVoxelizationBindingPoints_SVOPG::eBaseMap_SVOPG)]] Texture2D<float4> baseMap;
#endif

//----------------------------------------------clearVGB--------------------------------------------
[numthreads(1024, 1, 1)]
[shader("compute")]
void computeMain_clearVGB(uint threadIndex: SV_DispatchThreadID) {
    uint voxelCount = uint(pushConst.voxelSize_Count.w * pushConst.voxelSize_Count.w * pushConst.voxelSize_Count.w);
    uint totalVoxelCount = voxelCount  * 6;
    if (threadIndex >= totalVoxelCount) return;

    VGBVoxelData_SVOPG voxelData;
    voxelData.aabbI.minimum = int4(FloatToOrderedInt(3.402823466e+38f));
    voxelData.aabbI.maximum = int4(FloatToOrderedInt(-3.402823466e+38f));
    voxelData.irradiance = float4(0.0f);
    voxelData.sumNormal_G = float4(0.0f);
    voxelData.sumNormal_E = float4(0.0f);

    uint normalIndex = threadIndex / voxelCount;
    uint voxelIndex = threadIndex & (voxelCount - 1);
    VGBs[normalIndex][voxelIndex] = voxelData;

    #ifdef CLUSTER_WITH_MATERIAL
    uint materialCount[MAX_MATERIAL_COUNT];
    for (int i = 0; i < MAX_MATERIAL_COUNT; i++) materialCount[i] = 0;
    VGBMaterialInfos[normalIndex][voxelIndex].materialCount = materialCount;
    #endif
}
//----------------------------------------------createVGB--------------------------------------------
struct VSout
{
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
};

struct GSout {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
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
VSout vertexMain(uint vertexIndex: SV_VertexID)
{
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instance = sceneInfo.instances[pushConst.instanceIndex];
    Mesh mesh = sceneInfo.meshes[instance.meshIndex];

    // Retrieve the data
    float3 posMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.positions, vertexIndex);
    float3 normalMesh = getAttribute<float3>(mesh.dataBuffer, mesh.triMesh.normals, vertexIndex);

    //BSDFMaterial material = sceneInfo.materials[instance.materialIndex];
    //int normalMapIndex = material.materialMapIndex.x;
    //if (normalMapIndex >= 0) {
    //    float2 texCoords = getAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, vertexIndex);
    //    float3 tangent, bitangent;
    //    orthonormalBasis(normalMesh, tangent, bitangent);
    //    float3x3 TBN = float3x3(tangent, bitangent, normalMesh);
    //
    //    float3 localNormal = textures[normalMapIndex].SampleLevel(texCoords, 0).xyz;
    //    normalMesh = mul(localNormal, TBN);
    //}

    float3 normal = mul(normalMesh, transpose(Inverse3x3(float3x3(instance.transform))));
    float4 pos = mul(float4(posMesh, 1.0), instance.transform);

    VSout output;
    output.worldPos = pos.xyz;
    output.worldNormal = normal;

    return output;
}

[shader("geometry")]
[maxvertexcount(3)]
void geometryMain(
    triangle VSout input[3],
    inout TriangleStream<GSout> outputStream
) {
    float3 normal0 = input[0].worldNormal;
    float3 normal1 = input[1].worldNormal;
    float3 normal2 = input[2].worldNormal;
    float3 normal = normalize(normal0 + normal1 + normal2);
    float3 absNormal = abs(normal);
    int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

    GSout output;
    for (int i = 0; i < 3; i++) {
        output.sv_position = mul(float4(input[i].worldPos, 1.0f), pushConst.VP[viewIndex]);
        output.worldPos = input[i].worldPos;
        output.worldNormal = input[i].worldNormal;
        outputStream.Append(output);
    }

    outputStream.RestartStrip();
}

void atomicAABB(float3 worldPos, int voxelIndexU, int VGBIndex) {
    int worldPosX = FloatToOrderedInt(worldPos.x);
    int worldPosY = FloatToOrderedInt(worldPos.y);
    int worldPosZ = FloatToOrderedInt(worldPos.z);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.x, worldPosX);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.y, worldPosY);
    InterlockedMin<int>(VGBs[VGBIndex][voxelIndexU].aabbI.minimum.z, worldPosZ);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.x, worldPosX);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.y, worldPosY);
    InterlockedMax<int>(VGBs[VGBIndex][voxelIndexU].aabbI.maximum.z, worldPosZ);
}
void atomicNormalG(float3 normal, int voxelIndexU, int VGBIndex) {
    float3 normal_G = normalize(normal);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.w, 1.0f);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.x, normal_G.x);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.y, normal_G.y);
    InterlockedAdd<float>(VGBs[VGBIndex][voxelIndexU].sumNormal_G.z, normal_G.z);
}
/*
void atomicMaterialTypeCount(uint materialIndex, int voxelIndexU, int VGBIndex) {
    uint preVal = 0;
    uint newVal = uint(materialIndex) << 26; ++newVal;
    uint curVal;
    InterlockedCompareExchange(VGBs[VGBIndex][voxelIndexU].materialIndex_Count, preVal, newVal, curVal);
    while (curVal != preVal) {
        preVal = curVal;

        uint voxelMaterialIndex = preVal >> 26;
        uint voxelMaterialCount = preVal & 0x07FFFFFF;
        newVal = preVal;
        if (materialIndex == voxelMaterialIndex) ++newVal;
        else {
            if (voxelMaterialCount == 0) {
                newVal = uint(materialIndex) << 26;
                ++newVal;       //2^26 is too big to exceed
            }
            else --newVal;
        }

        InterlockedCompareExchange(VGBs[VGBIndex][voxelIndexU].materialIndex_Count, preVal, newVal, curVal);
    }
}
*/
#ifdef CLUSTER_WITH_MATERIAL
void atomicMaterialTypeCount(uint materialIndex, int voxelIndexU, int VGBIndex) {
    InterlockedAdd<uint>(VGBMaterialInfos[VGBIndex][voxelIndexU].materialCount[materialIndex], 1);
}
#endif

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

    bool pixelXInVoxel = checkPointInAABB2(maximum, voxelRange);
    if (pixelXInVoxel) atomicAABB(maximum, voxelIndexU, VGBIndex);
    bool pixelYInVoxel = checkPointInAABB2(minimum, voxelRange);
    if (pixelYInVoxel) atomicAABB(minimum, voxelIndexU, VGBIndex);
    if (!pixelXInVoxel || !pixelYInVoxel) atomicAABB(input.worldPos, voxelIndexU, VGBIndex);
    /*
    AABBI voxelAabbI = VGBs[VGBIndex][voxelIndexU].aabbI;
    AABB voxelAabb;
    voxelAabb.minimum.x = OrderedIntToFloat(voxelAabbI.minimum.x);
    voxelAabb.minimum.y = OrderedIntToFloat(voxelAabbI.minimum.y);
    voxelAabb.minimum.z = OrderedIntToFloat(voxelAabbI.minimum.z);
    voxelAabb.maximum.x = OrderedIntToFloat(voxelAabbI.maximum.x);
    voxelAabb.maximum.y = OrderedIntToFloat(voxelAabbI.maximum.y);
    voxelAabb.maximum.z = OrderedIntToFloat(voxelAabbI.maximum.z);

    bool pixelXInVoxel = checkPointInAABB(maximum, voxelRange);
    if (pixelXInVoxel && !checkPointInAABB2(maximum, voxelAabb)) atomicAABB(maximum, voxelIndexU, VGBIndex);

    bool pixelYInVoxel = checkPointInAABB(minimum, voxelRange);
    if (pixelYInVoxel && !checkPointInAABB2(minimum, voxelAabb)) atomicAABB(minimum, voxelIndexU, VGBIndex);

    if ((!pixelXInVoxel || !pixelYInVoxel) && !checkPointInAABB2(input.worldPos, voxelAabb)) atomicAABB(input.worldPos, voxelIndexU, VGBIndex);
    */
    atomicNormalG(input.worldNormal, voxelIndexU, VGBIndex);

    #ifdef CLUSTER_WITH_MATERIAL
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instanceInfo = sceneInfo.instances[pushConst.instanceIndex];
    atomicMaterialTypeCount(instanceInfo.materialIndex, voxelIndexU, VGBIndex);
    #endif
}

#ifndef NDEBUG
//--------------------------------------------threeView---------------------------------------
struct GSout_ThreeView {
    float4 sv_position : SV_Position;
    nointerpolation uint viewportIndex : SV_ViewportArrayIndex;
    float3 worldPos : POSITION;
    float3 worldNormal : NORMAL;
};
struct PSout_ThreeView
{
    float4 color : SV_Target;
};

[shader("geometry")]
[maxvertexcount(3)]
void geometryMain_ThreeView(
    triangle VSout input[3],
    inout TriangleStream<GSout_ThreeView> outputStream
) {
    float3 normal0 = input[0].worldNormal;
    float3 normal1 = input[1].worldNormal;
    float3 normal2 = input[2].worldNormal;
    float3 normal = normalize(normal0 + normal1 + normal2);

    //float3 pos0 = input[0].worldPos;
    //float3 pos1 = input[1].worldPos;
    //float3 pos2 = input[2].worldPos;
    //float3 e0 = pos1 - pos0;
    //float3 e1 = pos2 - pos0;
    //float3 normal = normalize(cross(e0, e1));
    float3 absNormal = abs(normal);

    int viewIndex = absNormal.x > absNormal.y ? (absNormal.x > absNormal.z ? 1 : 0) : (absNormal.y > absNormal.z ? 2 : 0);

    GSout_ThreeView output;
    output.viewportIndex = viewIndex;
    for (int i = 0; i < 3; i++) {
        output.worldPos = input[i].worldPos;
        output.sv_position = mul(float4(output.worldPos, 1.0f), pushConst.VP[viewIndex]);
        output.worldNormal = input[i].worldNormal;
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

    bool pixelXInVoxel = checkPointInAABB2(maximum, voxelRange);
    if (pixelXInVoxel) atomicAABB(maximum, voxelIndexU, VGBIndex);
    bool pixelYInVoxel = checkPointInAABB2(minimum, voxelRange);
    if (pixelYInVoxel) atomicAABB(minimum, voxelIndexU, VGBIndex);
    if (!pixelXInVoxel || !pixelYInVoxel) atomicAABB(input.worldPos, voxelIndexU, VGBIndex);

    /*
    AABBI voxelAabbI = VGBs[VGBIndex][voxelIndexU].aabbI;
    AABB voxelAabb;
    voxelAabb.minimum.x = OrderedIntToFloat(voxelAabbI.minimum.x);
    voxelAabb.minimum.y = OrderedIntToFloat(voxelAabbI.minimum.y);
    voxelAabb.minimum.z = OrderedIntToFloat(voxelAabbI.minimum.z);
    voxelAabb.maximum.x = OrderedIntToFloat(voxelAabbI.maximum.x);
    voxelAabb.maximum.y = OrderedIntToFloat(voxelAabbI.maximum.y);
    voxelAabb.maximum.z = OrderedIntToFloat(voxelAabbI.maximum.z);

    bool pixelXInVoxel = checkPointInAABB(maximum, voxelRange);
    if (pixelXInVoxel && !checkPointInAABB2(maximum, voxelAabb)) atomicAABB(maximum, voxelIndexU, VGBIndex);

    bool pixelYInVoxel = checkPointInAABB(minimum, voxelRange);
    if (pixelYInVoxel && !checkPointInAABB2(minimum, voxelAabb)) atomicAABB(minimum, voxelIndexU, VGBIndex);

    if ((!pixelXInVoxel || !pixelYInVoxel) && !checkPointInAABB2(input.worldPos, voxelAabb)) atomicAABB(input.worldPos, voxelIndexU, VGBIndex);
    */

    atomicNormalG(input.worldNormal, voxelIndexU, VGBIndex);

    #ifdef CLUSTER_WITH_MATERIAL
    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instanceInfo = sceneInfo.instances[pushConst.instanceIndex];
    atomicMaterialTypeCount(instanceInfo.materialIndex, voxelIndexU, VGBIndex);
    #endif

    if (pushConst.frameIndex == 0) InterlockedAdd<uint>(fragmentCountBuffer[0], 1);

    PSout_ThreeView output;
    output.color = float4(abs(input.worldNormal), 1.0f);
    return output;
}
//---------------------------------------------cube--------------------------------------------
struct VSout_Cube {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
    float3 color;
}
struct PSout_Cube
{
    float4 color : SV_Target;
};

[shader("vertex")]
VSout_Cube vertexMain_Cube(uint vertexIndex: SV_VertexID, int instanceIndex : SV_InstanceID)
{
    uint voxelTotalCount = uint(pow(pushConst.voxelSize_Count.w, 3));
    uint VGBIndex = instanceIndex / voxelTotalCount;
    uint voxelIndex = instanceIndex % voxelTotalCount;

    VGBVoxelData_SVOPG voxelData = VGBs[VGBIndex][voxelIndex];

    AABB nodeAabb;
    nodeAabb.minimum.x = OrderedIntToFloat(voxelData.aabbI.minimum.x);
    nodeAabb.minimum.y = OrderedIntToFloat(voxelData.aabbI.minimum.y);
    nodeAabb.minimum.z = OrderedIntToFloat(voxelData.aabbI.minimum.z);
    nodeAabb.maximum.x = OrderedIntToFloat(voxelData.aabbI.maximum.x);
    nodeAabb.maximum.y = OrderedIntToFloat(voxelData.aabbI.maximum.y);
    nodeAabb.maximum.z = OrderedIntToFloat(voxelData.aabbI.maximum.z);

    //int3 voxelIndexXYZ = int3((nodeAabb.minimum - pushConst.voxelGroupStartPos) / pushConst.voxelSize_Count.xyz);
    //float3 voxelStartPos = (float3)voxelIndexXYZ * pushConst.voxelSize_Count.xyz + pushConst.voxelGroupStartPos;
    //float3 voxelEndPos = voxelStartPos + pushConst.voxelSize_Count.xyz;
    //nodeAabb.minimum = nodeAabb.minimum - (nodeAabb.minimum - voxelStartPos) * sign(nodeAabb.maximum - nodeAabb.minimum);
    //nodeAabb.maximum = nodeAabb.maximum + (voxelEndPos - nodeAabb.maximum) * sign(nodeAabb.maximum - nodeAabb.minimum);
    //nodeAabb.minimum = max(nodeAabb.minimum, voxelStartPos);
    //nodeAabb.maximum = min(nodeAabb.maximum, voxelEndPos);

    if (VGBIndex != pushConst.normalIndex) voxelData.sumNormal_G.w = 0.0f;

    VSout_Cube output;
    if (voxelData.sumNormal_G.w == 0.0f) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        output.worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? nodeAabb.minimum.x : nodeAabb.maximum.x;
        output.worldPos.y = (vertexIndex & 3) <= 1 ? nodeAabb.minimum.y : nodeAabb.maximum.y;
        output.worldPos.z = vertexIndex / 4 == 0 ? nodeAabb.minimum.z : nodeAabb.maximum.z;

        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        //Slightly reduce the depth to prevent depth conflicts in subsequent wireframes
        output.sv_position = mul(float4(output.worldPos.xy, output.worldPos.z * 0.99f, 1.0f), sceneInfo.viewProjMatrix);
        output.color = normalize(max(voxelData.sumNormal_G.xyz / voxelData.sumNormal_G.w, 0.0f));
    }
    return output;
}
[shader("pixel")]
PSout_Cube fragmentMain_Cube(VSout_Cube input) {
    return { float4(input.color, 1.0f) };
}
//---------------------------------------------wireframe----------------------------------------
[shader("pixel")]
PSout_Cube fragmentMain_Wireframe(VSout_Cube input, float4 position: SV_Position) {
    return { float4(1.0f, 0.0f, 0.0f, 1.0f) };
}
//----------------------------------------------postProcess-------------------------------------
[numthreads(32, 32, 1)]
[shader("compute")]
void computeMain_postProcess(uint3 dispatchId: SV_DispatchThreadID) {
    uint2 wireframeMapSize;
    wireframeMap.GetDimensions(wireframeMapSize.x, wireframeMapSize.y);
    if (dispatchId.x >= wireframeMapSize.x || dispatchId.y >= wireframeMapSize.y) return;

    float4 result = float4(1.0f);
    float4 wireframeColor = wireframeMap[dispatchId.xy];
    if (wireframeColor.w == 0.0f) {
        uint2 baseMapSize;
        baseMap.GetDimensions(baseMapSize.x, baseMapSize.y);

        float2 texCoords = (float2)dispatchId.xy / wireframeMapSize;
        uint2 baseMapTexCoords = uint2(texCoords * baseMapSize);

        float4 baseColor = baseMap[baseMapTexCoords];
        result = baseColor;
    }
    else result = wireframeColor;
    wireframeMap[dispatchId.xy] = result;
}
#endif