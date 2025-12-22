#include "common/Shader/shaderStructType.h"
#include "../shaderio.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<RasterVoxelizationPushConstant> pushConst;
[[vk::binding(RasterVoxelizationBindingPoints::eTextures_RV)]] Sampler2D textures[];
[[vk::binding(RasterVoxelizationBindingPoints::eVGB_RV)]] RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;

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

struct PSout
{
    float4 color : SV_Target;
};

__generic<T : IFloat> T getAttribute(uint8_t *dataBufferAddress, BufferView bufferView, uint attributeIndex)
{
    if (bufferView.count > 0)
    {
        T *ptr = (T *)(dataBufferAddress + bufferView.offset + attributeIndex * bufferView.byteStride);
        return ptr[0];
    }

    return T(1); // Error case
}

// Vertex  Shader
[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID)
{
    int instanceIndex = pushConst.instanceIndex;

    SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];

    Instance instance = sceneInfo.instances[instanceIndex];
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
    output.worldNormal = normalize(normal);

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
    float3 normal = (normal0 + normal1 + normal2) * 0.33333f;
    int viewIndex = normal.x > normal.y ? (normal.x > normal.z ? 1 : 0) : (normal.y > normal.z ? 2 : 0);

    GSout output;
    for (int i = 0; i < 3; i++) {
        output.sv_position = mul(float4(input[i].worldPos, 1.0f), pushConst.VP[viewIndex]);
        output.worldNormal = input[i].worldNormal;
        outputStream.Append(output);
    }

    outputStream.RestartStrip();
}

[shader("pixel")]
PSout fragmentMain(GSout input) {
    int3 voxelIndex = int3((input.worldPos - pushConst.voxelGroupStartPos.xyz) / pushConst.voxelSize_Count.xyz);
    int voxelCount = pushConst.voxelSize_Count.w;
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

    AABBU aabbU = VGB[voxelIndexU].aabbU;
    float leftX = asfloat(aabbU.minimum.x);
    float leftY = asfloat(aabbU.minimum.y);
    float leftZ = asfloat(aabbU.minimum.z);
    float rightX = asfloat(aabbU.maximum.x);
    float rightY = asfloat(aabbU.maximum.y);
    float rightZ = asfloat(aabbU.maximum.z);

    if (input.worldPos.x < leftX) {
        uint preVal = aabbU.minimum.x;
        uint newVal = asuint(input.worldPos.x);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.x, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= input.worldPos.x) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.x, preVal, newVal, curVal);
        }
    } else if (input.worldPos.x > rightX) {
        uint preVal = aabbU.maximum.x;
        uint newVal = asuint(input.worldPos.x);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.x, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= input.worldPos.x) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.x, preVal, newVal, curVal);
        }
    }
    if (input.worldPos.y < leftY) {
        uint preVal = aabbU.minimum.y;
        uint newVal = asuint(input.worldPos.y);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.y, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= input.worldPos.y) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.y, preVal, newVal, curVal);
        }
    } else if (input.worldPos.y > rightY) {
        uint preVal = aabbU.maximum.y;
        uint newVal = asuint(input.worldPos.y);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.y, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= input.worldPos.y) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.y, preVal, newVal, curVal);
        }
    }
    if (input.worldPos.z < leftZ) {
        uint preVal = aabbU.minimum.z;
        uint newVal = asuint(input.worldPos.z);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.z, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) <= input.worldPos.z) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.minimum.z, preVal, newVal, curVal);
        }
    } else if (input.worldPos.z > rightZ) {
        uint preVal = aabbU.maximum.z;
        uint newVal = asuint(input.worldPos.z);
        uint curVal;
        InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.z, preVal, newVal, curVal);
        while (curVal != preVal) {
            if (asfloat(curVal) >= input.worldPos.z) break;
            preVal = curVal;
            InterlockedCompareExchange(VGB[voxelIndexU].aabbU.maximum.z, preVal, newVal, curVal);
        }
    }

    float3 normal_G = normalize(input.worldNormal);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.w, 1.0f);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.x, normal_G.x);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.y, normal_G.y);
    InterlockedAdd<float>(VGB[voxelIndexU].sumNormal_G.z, normal_G.z);
}