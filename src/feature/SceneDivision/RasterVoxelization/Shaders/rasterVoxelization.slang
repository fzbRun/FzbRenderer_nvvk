#include "common/Shader/shaderStructType.h"
#include "feature/SceneDivision/RasterVoxelization/shaderio.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<RasterVoxelizationPushConstant> pushConst;
[[vk::binding(RasterVoxelizationBindingPoints::eTextures_RV)]] Sampler2D textures[];
[[vk::binding(RasterVoxelizationBindingPoints::eVGB_RV)]] globallycoherent volatile RWStructuredBuffer<VGBVoxelData, ScalarDataLayout> VGB;
[[vk::binding(RasterVoxelizationBindingPoints::eFragmentCountBuffer_RV)]] RWStructuredBuffer<uint32_t> fragmentCountBuffer;

//----------------------------------------------clearVGB--------------------------------------------
[numthreads(512, 1, 1)]
[shader("compute")]
void computeMain_clearVGB(uint3 dispatchId: SV_DispatchThreadID) {
    uint totalVoxelCount = pow(pushConst.voxelSize_Count.w, 3);
    if (dispatchId.x >= totalVoxelCount) return;

    VGBVoxelData voxelData;
    voxelData.aabbU.minimum = uint3(asuint(3.402823466e+38F));
    voxelData.aabbU.maximum = uint3(asuint(-3.402823466e+38F));
    voxelData.radiance = float4(0.0f);
    voxelData.sumNormal_G = float4(0.0f);
    voxelData.sumNormal_E = float4(0.0f);

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

[shader("pixel")]
void fragmentMain(GSout input) {
    int voxelIndexU = getVoxelIndex(input.worldPos);
    atomicAABB(input.worldPos, voxelIndexU);
    atomicNormalG(input.worldNormal, voxelIndexU);
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
struct VSout_Wireframe {
    float4 sv_position : SV_Position;
    float3 worldPos : POSITION;
}
struct PSout_Wireframe
{
    float4 color : SV_Target;
};

[shader("vertex")]
VSout_Cube vertexMain_Wireframe(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID)
{
    VSout_Cube output;
    VGBVoxelData voxelData = VGB[instanceIndex];
    if (voxelData.sumNormal_G.w < 0.1f) output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    else {
        SceneInfo sceneInfo = pushConst.sceneInfoAddress[0];
        // Slightly reduce the depth to prevent depth conflicts in subsequent wireframes
        output.sv_position = mul(float4(output.worldPos.xy, output.worldPos.z * 0.99f, 1.0f), sceneInfo.viewProjMatrix);
        output.color = normalize(max(voxelData.sumNormal_G.xyz / voxelData.sumNormal_G.w, 0.0f));
    }
    return output;
}
#endif