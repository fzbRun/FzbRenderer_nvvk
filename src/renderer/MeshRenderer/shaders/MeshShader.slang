#include "renderer/MeshRenderer/MeshRendererShaderio.h"
#include "common/Shader/nvvk/Slang/pbr.h.slang"
#include <nvshaders/slang_types.h>
#include "common/Shader/shaderStructType.h"
#include "common/Shader/Slang/commonFunction.slang"

[[vk::push_constant]] ConstantBuffer<MeshRendererPushConstant> pushConst;

struct MeshOutput {
    float4 Position : SV_POSITION;
    float3 Color : COLOR;
};

[shader("mesh")]
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void meshMain(out indices uint3 triangles[1], out vertices MeshOutput vertices[3]) {
    // Must be called before writing the geometry output
    SetMeshOutputCounts(3, 1); // 3 vertices, 1 primitive

    triangles[0] = uint3(0, 1, 2);

    vertices[0].Position = float4(-0.5, 0.5, 0.0, 1.0);
    vertices[0].Color = float3(1.0, 0.0, 0.0);

    vertices[1].Position = float4(0.5, 0.5, 0.0, 1.0);
    vertices[1].Color = float3(0.0, 1.0, 0.0);

    vertices[2].Position = float4(0.0, -0.5, 0.0, 1.0);
    vertices[2].Color = float3(0.0, 0.0, 1.0);
}

[shader("pixel")]
float4 fragmentMain(MeshOutput input) : SV_TARGET{
    return float4(input.Color, 1);
}