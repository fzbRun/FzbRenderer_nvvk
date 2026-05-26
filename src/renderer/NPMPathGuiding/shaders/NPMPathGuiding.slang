#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/NPMPathGuiding/NPMPathGuidingShaderio.h"

[[vk::push_constant]] ConstantBuffer<NPMPathGuidingPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_NPMPG::eFlowerImage), vk::image_format("rgba8")]] RWTexture2D<float4> flowerImage;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_reversalColor(uint2 threadIndex: SV_DispatchThreadID) {
    uint2 textureSize;
    flowerImage.GetDimensions(textureSize.x, textureSize.y);
    if (threadIndex.x >= textureSize.x || threadIndex.y >= textureSize.y) return;

    float4 color = flowerImage[threadIndex];
    if (pushConst.time == 0) color = 1.0 - color;
    else color *= 2.0f; 
    flowerImage[threadIndex] = color;
}