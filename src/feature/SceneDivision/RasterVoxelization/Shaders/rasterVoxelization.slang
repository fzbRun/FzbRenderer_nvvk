#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<RasterVoxelizationPushConstant> pushConst;
[[vk::binding(RasterVoxelizationBindingPoints::eTextures_RV)]] Sampler2D textures[];
[[vk::binding(RasterVoxelizationBindingPoints::eVGB_RV)]] globallycoherent volatile RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;
#ifndef NDEBUG
[[vk::binding(RasterVoxelizationBindingPoints::eFragmentCountBuffer_RV)]] RWStructuredBuffer<uint32_t> fragmentCountBuffer;
[[vk::binding(RasterVoxelizationBindingPoints::eWireframeMap_RV)]] RWTexture2D<float4> wireframeMap;
[[vk::binding(RasterVoxelizationBindingPoints::eBaseMap)]] Texture2D<float4> baseMap;
#endif

//----------------------------------------------clearVGB--------------------------------------------
[numthreads(512, 1, 1)]
[shader("compute")]
void computeMain_clearVGB(uint3 dispatchId: SV_DispatchThreadID) {
    uint totalVoxelCount = uint(pow(pushConst.voxelSize_Count.w, 3));
    if (dispatchId.x >= totalVoxelCount) return;

    VGBVoxelData voxelData;
    voxelData.aabbU.minimum = uint3(asuint(3.402823466e+38F));
    voxelData.aabbU.maximum = uint3(asuint(-3.402823466e+38F));
    voxelData.irradiance = float4(0.0f);
    voxelData.sumNormal_G = float4(0.0f);
    voxelData.sumNormal_E = float4(0.0f);
    voxelData.materialType_Count = 0;

    VGB[dispatchId.x] = voxelData;
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

    BSDFMaterial material = sceneInfo.materials[instance.materialIndex];
    int normalMapIndex = material.materialMapIndex.x;
    if (normalMapIndex >= 0) {
        float2 texCoords = getAttribute<float2>(mesh.dataBuffer, mesh.triMesh.texCoords, vertexIndex);
        float3 tangent, bitangent;
        orthonormalBasis(normalMesh, tangent, bitangent);
        float3x3 TBN = float3x3(tangent, bitangent, normalMesh);

        float3 localNormal = textures[normalMapIndex].SampleLevel(texCoords, 0).xyz;
        normalMesh = mul(localNormal, TBN);
    }

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

int getVoxelIndex(float3 worldPos) {
    int3 voxelIndex = int3((worldPos - pushConst.voxelGroupStartPos) / pushConst.voxelSize_Count.xyz);
    int voxelCount = uint(pushConst.voxelSize_Count.w);
    int voxelTotalCount = voxelCount * voxelCount * voxelCount;
    int voxelIndexU = 0;
    while (voxelTotalCount > 1) {
        voxelCount = voxelCount / 2;
        voxelTotalCount = voxelTotalCount / 8;
        if (voxelIndex.z / voxelCount == 1) {
            voxelIndexU += 4 * voxelTotalCount;
            voxelIndex.z -= voxelCount;
        }
        if (voxelIndex.y / voxelCount == 1) {
            voxelIndexU += 2 * voxelTotalCount;
            voxelIndex.y -= voxelCount;
        }
        if (voxelIndex.x / voxelCount == 1) {
            voxelIndexU += voxelTotalCount;
            voxelIndex.x -= voxelCount;
        }
    }
    return voxelIndexU;
}
void atomicAABB(float3 worldPos, int voxelIndexU) {
    AABBU aabbU = VGB[voxelIndexU].aabbU;
    float leftX = asfloat(aabbU.minimum.x);
    float leftY = asfloat(aabbU.minimum.y);
    float leftZ = asfloat(aabbU.minimum.z);
    float rightX = asfloat(aabbU.maximum.x);
    float rightY = asfloat(aabbU.maximum.y);
    float rightZ = asfloat(aabbU.maximum.z);

    uint preVal;
    uint newVal;
    uint curVal;
    if (worldPos.x < leftX) {
        preVal = aabbU.minimum.x;
        newVal = asuint(worldPos.x);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.x, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= worldPos.x) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.x, preVal, newVal, curVal);
        }
    } 
    if (worldPos.x > rightX) {
        preVal = aabbU.maximum.x;
        newVal = asuint(worldPos.x);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.x, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= worldPos.x) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.x, preVal, newVal, curVal);
        }
    }
    if (worldPos.y < leftY) {
        preVal = aabbU.minimum.y;
        newVal = asuint(worldPos.y);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.y, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= worldPos.y) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.y, preVal, newVal, curVal);
        }
    } 
    if (worldPos.y > rightY) {
        preVal = aabbU.maximum.y;
        newVal = asuint(worldPos.y);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.y, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= worldPos.y) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.y, preVal, newVal, curVal);
        }
    }
    if (worldPos.z < leftZ) {
        preVal = aabbU.minimum.z;
        newVal = asuint(worldPos.z);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.z, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= worldPos.z) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.z, preVal, newVal, curVal);
        }
    } 
    if (worldPos.z > rightZ) {
        preVal = aabbU.maximum.z;
        newVal = asuint(worldPos.z);
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.z, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= worldPos.z) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.z, preVal, newVal, curVal);
        }
    }
}
void atomicNormalG(float3 normal, int voxelIndexU) {
    float3 normal_G = normalize(normal);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.w, 1.0f);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.x, normal_G.x);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.y, normal_G.y);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.z, normal_G.z);
}
void atomicMaterialTypeCount(MaterialType materialType, int voxelIndexU) {
    uint preVal = 0;
    uint newVal = uint(materialType) << 27; ++newVal;
    uint curVal;
    InterlockedCompareExchange(VGB[voxelIndexU].materialType_Count, preVal, newVal, curVal);
    while (curVal != preVal) {
        preVal = curVal;

        MaterialType voxelMaterialType = MaterialType(preVal >> 27);
        uint voxelMaterialCount = preVal & 0x07FFFFFF;
        newVal = preVal;
        if (materialType == voxelMaterialType) ++newVal;
        else {
            if (voxelMaterialCount == 0) {
                newVal = uint(materialType) << 27;
                ++newVal;       //2^27 is too big to exceed
            }
            else --newVal;
        }

        InterlockedCompareExchange(VGB[voxelIndexU].materialType_Count, preVal, newVal, curVal);
    }
}

[shader("pixel")]
void fragmentMain(GSout input) {
    int voxelIndexU = getVoxelIndex(input.worldPos);
    atomicAABB(input.worldPos, voxelIndexU);
    atomicNormalG(input.worldNormal, voxelIndexU);

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instanceInfo = sceneInfo.instances[pushConst.instanceIndex];
    MaterialType instanceMaterialType = sceneInfo.materials[instanceInfo.materialIndex].type;
    atomicMaterialTypeCount(instanceMaterialType, voxelIndexU);
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
    int voxelIndexU = getVoxelIndex(input.worldPos);

    atomicAABB(input.worldPos, voxelIndexU);
    atomicNormalG(input.worldNormal, voxelIndexU);

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
    Instance instanceInfo = sceneInfo.instances[pushConst.instanceIndex];
    MaterialType instanceMaterialType = sceneInfo.materials[instanceInfo.materialIndex].type;
    atomicMaterialTypeCount(instanceMaterialType, voxelIndexU);

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
    VGBVoxelData voxelData = VGB[instanceIndex];
    VSout_Cube output;
    if (voxelData.sumNormal_G.w == 0.0f) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        output.worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? asfloat(voxelData.aabbU.minimum.x) : asfloat(voxelData.aabbU.maximum.x);
        output.worldPos.y = (vertexIndex & 3) <= 1 ? asfloat(voxelData.aabbU.minimum.y) : asfloat(voxelData.aabbU.maximum.y);
        output.worldPos.z = vertexIndex / 4 == 0 ? asfloat(voxelData.aabbU.minimum.z) : asfloat(voxelData.aabbU.maximum.z);

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