#include "common/Shader/shaderStructType.h"
#include "nvshaders/functions.h.slang"
#include "common/Shader/Slang/commonFunction.slang"
#include "renderer/NPMPathGuiding/NPMPathGuidingShaderio.h"

[[vk::push_constant]] ConstantBuffer<NPMPathGuidingPushConstant> pushConst;
[[vk::binding(StaticBindingPoints_NPMPG::eOutImage)]] RWTexture2D<float4> outImage;
[[vk::binding(StaticBindingPoints_NPMPG::eFlowerImage), vk::image_format("rgba8")]] Sampler2D<float4> flowerImage;
[[vk::binding(StaticBindingPoints_NPMPG::eColorImageWrite), vk::image_format("rgba8")]] RWTexture2D<float4> colorImageWrite;
[[vk::binding(StaticBindingPoints_NPMPG::eColorImageRead), vk::image_format("rgba8")]] Sampler2D<float4> colorImageRead;

[numthreads(16, 16, 1)]
[shader("compute")]
void computeMain_reversalColor(uint2 threadIndex: SV_DispatchThreadID) {
    if (threadIndex.x >= pushConst.screenSize.x || threadIndex.y >= pushConst.screenSize.y) return;

    if (pushConst.time == 0) {
        float4 color = flowerImage.SampleLevel((float2(threadIndex) + float2(0.5f)) / pushConst.screenSize, 0);
        color = (float(pushConst.frameIndex % 100) / 100) * color;
        colorImageWrite[threadIndex] = color;
    }
    else outImage[threadIndex] = colorImageRead.SampleLevel((float2(threadIndex) + float2(0.5f)) / pushConst.screenSize, 0);

    //if(threadIndex.x == 0 && threadIndex.y == 0) printf("vulkan frameIndex: %d time: %d\n", pushConst.frameIndex, pushConst.time);
}