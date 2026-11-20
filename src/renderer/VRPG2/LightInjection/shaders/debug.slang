#include "renderer/VRPG2/VRPG2Shaderio.h"
#include "renderer/VRPG2/GeometryInjection/GeometryInjectionShaderio.h"
#include "renderer/VRPG2/LightInjection/LightInjectionShaderio.h"
#include "common/Shader/Slang/commonFunction.slang"

// Debug view of the light injection result, ported from VRPG's LightInject::debug_Cube
// (shaders/lightInject2.slang: vertexMain_Cube / fragmentMain_Cube).
//
// One solid cube per geometry VGB voxel, colored by the average irradiance of the voxel.
// Voxels that were never injected (irradiance.w == 0) are collapsed to a degenerate clip
// position, so they cost no fragments.

[[vk::push_constant]] ConstantBuffer<LightInjectPushConstant_VRPG> pushConst;
[[vk::binding(StaticBindingPoints_LightInject_VRPG::eVGB)]] RWStructuredBuffer<VoxelData_VRPG, ScalarDataLayout> VGBs[];

struct VSout {
    float4 sv_position : SV_Position;
    nointerpolation float3 color;
};
struct PSout {
    float4 color0 : SV_Target0;
};

[shader("vertex")]
VSout vertexMain(uint vertexIndex: SV_VertexID, int instanceIndex: SV_InstanceID) {
    uint voxelTotalCount = uint(max(pushConst.voxelCount, 1));
    uint VGBIndex = uint(instanceIndex) / voxelTotalCount;
    uint voxelIndex = uint(instanceIndex) % voxelTotalCount;

    VoxelData_VRPG voxelData = VGBs[VGBIndex][voxelIndex];

    VSout output;
    output.color = float3(0.0f);
    if (voxelData.irradiance.w == 0.0f) {
        output.sv_position = float4(2.0, 2.0, 2.0, 1.0);
    } else {
        float3 worldPos;
        worldPos.x = ((vertexIndex & 3) * 2) % 6 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.x) : OrderedIntToFloat(voxelData.aabbI.maximum.x);
        worldPos.y = (vertexIndex & 3) <= 1 ? OrderedIntToFloat(voxelData.aabbI.minimum.y) : OrderedIntToFloat(voxelData.aabbI.maximum.y);
        worldPos.z = vertexIndex / 4 == 0 ? OrderedIntToFloat(voxelData.aabbI.minimum.z) : OrderedIntToFloat(voxelData.aabbI.maximum.z);

        output.sv_position = mul(float4(worldPos, 1.0f), pushConst.sceneInfoAddress[0].viewProjMatrix);
        output.color = voxelData.irradiance.xyz / voxelData.irradiance.w;
    }

    return output;
}

[shader("fragment")]
PSout fragmentMain(VSout input) {
    PSout output;
    output.color0 = float4(input.color, 1.0f);
    return output;
}
